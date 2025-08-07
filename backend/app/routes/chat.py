import asyncio
import json
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from pydantic import BaseModel, Field
from datetime import datetime

from app.database.database import get_db
from app.database.models import Conversation, Message, User, MessageRole, ConversationStatus
from app.models.ai_orchestrator import AIOrchestrator, ModelRequest
from app.core.logger import LoggerMixin
from app.utils.cache import ConversationCache

router = APIRouter()

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: Optional[int] = None
    system_prompt: Optional[str] = Field(None, max_length=5000)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(2048, ge=1, le=8192)
    stream: bool = False
    model_preference: Optional[str] = None

class MessageResponse(BaseModel):
    id: int
    content: str
    role: str
    created_at: datetime
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
    cost: Optional[float] = None

class ConversationResponse(BaseModel):
    id: int
    title: str
    status: str
    created_at: datetime
    updated_at: Optional[datetime]
    message_count: int

class ChatResponse(BaseModel):
    message: MessageResponse
    conversation_id: int
    total_tokens: Optional[int] = None
    total_cost: Optional[float] = None

class WebSocketManager(LoggerMixin):
    """WebSocket connection manager for real-time chat"""
    
    def __init__(self):
        self.active_connections: Dict[int, List[WebSocket]] = {}
        
    async def connect(self, websocket: WebSocket, user_id: int):
        """Accept WebSocket connection"""
        await websocket.accept()
        
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        
        self.active_connections[user_id].append(websocket)
        self.logger.info(f"WebSocket connected for user {user_id}")
    
    def disconnect(self, websocket: WebSocket, user_id: int):
        """Remove WebSocket connection"""
        if user_id in self.active_connections:
            try:
                self.active_connections[user_id].remove(websocket)
                if not self.active_connections[user_id]:
                    del self.active_connections[user_id]
                self.logger.info(f"WebSocket disconnected for user {user_id}")
            except ValueError:
                pass
    
    async def send_personal_message(self, message: str, user_id: int):
        """Send message to specific user"""
        if user_id in self.active_connections:
            for connection in self.active_connections[user_id][:]:  # Copy to avoid modification during iteration
                try:
                    await connection.send_text(message)
                except Exception as e:
                    self.logger.warning(f"Failed to send message to user {user_id}: {e}")
                    self.active_connections[user_id].remove(connection)
    
    async def broadcast_to_user(self, data: Dict[str, Any], user_id: int):
        """Broadcast data to all user connections"""
        message = json.dumps(data)
        await self.send_personal_message(message, user_id)

# Global WebSocket manager
websocket_manager = WebSocketManager()

async def get_ai_orchestrator() -> AIOrchestrator:
    """Dependency to get AI orchestrator"""
    from main import app
    if hasattr(app.state, 'ai_orchestrator'):
        return app.state.ai_orchestrator
    raise HTTPException(status_code=503, detail="AI services not available")

async def get_conversation_cache() -> ConversationCache:
    """Dependency to get conversation cache"""
    from main import app
    if hasattr(app.state, 'cache'):
        return ConversationCache(app.state.cache.redis_url)
    raise HTTPException(status_code=503, detail="Cache not available")

@router.post("/send", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
    orchestrator: AIOrchestrator = Depends(get_ai_orchestrator),
    cache: ConversationCache = Depends(get_conversation_cache)
):
    """Send a chat message and get AI response"""
    
    # Get user from request state (set by auth middleware)
    if not hasattr(http_request.state, 'user'):
        raise HTTPException(status_code=401, detail="Authentication required")
    
    user = http_request.state.user
    
    try:
        # Get or create conversation
        conversation = await get_or_create_conversation(
            db, user.id, request.conversation_id, request.message[:50]
        )
        
        # Save user message
        user_message = Message(
            conversation_id=conversation.id,
            role=MessageRole.USER,
            content=request.message
        )
        db.add(user_message)
        await db.flush()  # Get the ID
        
        # Prepare AI request
        ai_request = ModelRequest(
            prompt=request.message,
            system_prompt=request.system_prompt or conversation.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=False,
            model_preference=request.model_preference,
            user_id=user.id,
            conversation_id=conversation.id
        )
        
        # Get AI response
        ai_response = await orchestrator.generate(ai_request)
        
        # Save AI message
        ai_message = Message(
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT,
            content=ai_response.content,
            model_used=ai_response.model_used,
            tokens_used=ai_response.tokens_used,
            cost=ai_response.cost,
            response_time=ai_response.response_time
        )
        db.add(ai_message)
        
        # Update conversation
        conversation.updated_at = datetime.utcnow()
        
        await db.commit()
        
        # Cache the conversation and messages
        await cache_conversation_data(cache, conversation.id, db)
        
        # Send WebSocket update
        await websocket_manager.broadcast_to_user({
            "type": "message",
            "conversation_id": conversation.id,
            "message": {
                "id": ai_message.id,
                "content": ai_message.content,
                "role": "assistant",
                "created_at": ai_message.created_at.isoformat(),
                "model_used": ai_message.model_used
            }
        }, user.id)
        
        return ChatResponse(
            message=MessageResponse(
                id=ai_message.id,
                content=ai_message.content,
                role="assistant",
                created_at=ai_message.created_at,
                model_used=ai_message.model_used,
                tokens_used=ai_message.tokens_used,
                cost=ai_message.cost
            ),
            conversation_id=conversation.id,
            total_tokens=ai_response.tokens_used,
            total_cost=ai_response.cost
        )
        
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")

@router.post("/stream")
async def stream_message(
    request: ChatRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
    orchestrator: AIOrchestrator = Depends(get_ai_orchestrator)
):
    """Stream a chat message response"""
    
    if not hasattr(http_request.state, 'user'):
        raise HTTPException(status_code=401, detail="Authentication required")
    
    user = http_request.state.user
    
    async def generate_stream():
        try:
            # Get or create conversation
            conversation = await get_or_create_conversation(
                db, user.id, request.conversation_id, request.message[:50]
            )
            
            # Save user message
            user_message = Message(
                conversation_id=conversation.id,
                role=MessageRole.USER,
                content=request.message
            )
            db.add(user_message)
            await db.flush()
            
            # Prepare AI request
            ai_request = ModelRequest(
                prompt=request.message,
                system_prompt=request.system_prompt or conversation.system_prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
                model_preference=request.model_preference,
                user_id=user.id,
                conversation_id=conversation.id
            )
            
            # Stream AI response
            full_content = ""
            async for chunk in orchestrator.stream_generate(ai_request):
                full_content += chunk
                
                # Send chunk as Server-Sent Event
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                
                # Also send via WebSocket
                await websocket_manager.broadcast_to_user({
                    "type": "stream_chunk",
                    "conversation_id": conversation.id,
                    "content": chunk
                }, user.id)
            
            # Save complete AI message
            ai_message = Message(
                conversation_id=conversation.id,
                role=MessageRole.ASSISTANT,
                content=full_content,
                model_used=ai_request.model_preference or "unknown"
            )
            db.add(ai_message)
            
            conversation.updated_at = datetime.utcnow()
            await db.commit()
            
            # Send completion event
            yield f"data: {json.dumps({'type': 'complete', 'message_id': ai_message.id})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: int,
    orchestrator: AIOrchestrator = Depends(get_ai_orchestrator)
):
    """WebSocket endpoint for real-time chat"""
    
    await websocket_manager.connect(websocket, user_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "chat":
                # Process chat message
                prompt = message_data.get("message", "")
                conversation_id = message_data.get("conversation_id")
                
                if prompt:
                    # Create AI request
                    ai_request = ModelRequest(
                        prompt=prompt,
                        stream=True,
                        user_id=user_id,
                        conversation_id=conversation_id
                    )
                    
                    # Stream response
                    full_content = ""
                    async for chunk in orchestrator.stream_generate(ai_request):
                        full_content += chunk
                        
                        await websocket.send_text(json.dumps({
                            "type": "stream_chunk",
                            "content": chunk,
                            "conversation_id": conversation_id
                        }))
                    
                    # Send completion
                    await websocket.send_text(json.dumps({
                        "type": "stream_complete",
                        "full_content": full_content,
                        "conversation_id": conversation_id
                    }))
            
            elif message_data.get("type") == "ping":
                # Respond to ping
                await websocket.send_text(json.dumps({"type": "pong"}))
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket, user_id)
    except Exception as e:
        websocket_manager.logger.error(f"WebSocket error for user {user_id}: {e}")
        websocket_manager.disconnect(websocket, user_id)

@router.get("/conversations", response_model=List[ConversationResponse])
async def get_conversations(
    request: Request,
    db: AsyncSession = Depends(get_db),
    limit: int = 50,
    offset: int = 0
):
    """Get user's conversations"""
    
    if not hasattr(request.state, 'user'):
        raise HTTPException(status_code=401, detail="Authentication required")
    
    user = request.state.user
    
    # Query conversations with message count
    from sqlalchemy import func
    
    stmt = (
        select(
            Conversation,
            func.count(Message.id).label('message_count')
        )
        .outerjoin(Message)
        .where(Conversation.user_id == user.id)
        .where(Conversation.status == ConversationStatus.ACTIVE)
        .group_by(Conversation.id)
        .order_by(desc(Conversation.updated_at))
        .limit(limit)
        .offset(offset)
    )
    
    result = await db.execute(stmt)
    conversations = result.all()
    
    return [
        ConversationResponse(
            id=conv.id,
            title=conv.title,
            status=conv.status.value,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            message_count=message_count
        )
        for conv, message_count in conversations
    ]

@router.get("/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_conversation_messages(
    conversation_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
    cache: ConversationCache = Depends(get_conversation_cache),
    limit: int = 100,
    offset: int = 0
):
    """Get messages for a conversation"""
    
    if not hasattr(request.state, 'user'):
        raise HTTPException(status_code=401, detail="Authentication required")
    
    user = request.state.user
    
    # Check if user owns the conversation
    conv_stmt = select(Conversation).where(
        Conversation.id == conversation_id,
        Conversation.user_id == user.id
    )
    result = await db.execute(conv_stmt)
    conversation = result.scalar_one_or_none()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Try to get from cache first
    cached_messages = await cache.get_messages(conversation_id)
    if cached_messages and offset == 0:
        return [MessageResponse(**msg) for msg in cached_messages[-limit:]]
    
    # Get from database
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
        .limit(limit)
        .offset(offset)
    )
    
    result = await db.execute(stmt)
    messages = result.scalars().all()
    
    message_responses = [
        MessageResponse(
            id=msg.id,
            content=msg.content,
            role=msg.role.value,
            created_at=msg.created_at,
            model_used=msg.model_used,
            tokens_used=msg.tokens_used,
            cost=msg.cost
        )
        for msg in messages
    ]
    
    # Cache the messages if this is the first page
    if offset == 0:
        await cache.set_messages(
            conversation_id, 
            [msg.dict() for msg in message_responses]
        )
    
    return message_responses

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
    cache: ConversationCache = Depends(get_conversation_cache)
):
    """Delete a conversation"""
    
    if not hasattr(request.state, 'user'):
        raise HTTPException(status_code=401, detail="Authentication required")
    
    user = request.state.user
    
    # Check if user owns the conversation
    stmt = select(Conversation).where(
        Conversation.id == conversation_id,
        Conversation.user_id == user.id
    )
    result = await db.execute(stmt)
    conversation = result.scalar_one_or_none()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Soft delete - mark as deleted
    conversation.status = ConversationStatus.DELETED
    conversation.updated_at = datetime.utcnow()
    
    await db.commit()
    
    # Clear cache
    await cache.clear_conversation(conversation_id)
    
    return {"message": "Conversation deleted successfully"}

# Helper functions

async def get_or_create_conversation(
    db: AsyncSession,
    user_id: int,
    conversation_id: Optional[int],
    title_hint: str
) -> Conversation:
    """Get existing conversation or create new one"""
    
    if conversation_id:
        # Get existing conversation
        stmt = select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
            Conversation.status == ConversationStatus.ACTIVE
        )
        result = await db.execute(stmt)
        conversation = result.scalar_one_or_none()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return conversation
    
    else:
        # Create new conversation
        conversation = Conversation(
            user_id=user_id,
            title=title_hint or "New Conversation",
            status=ConversationStatus.ACTIVE,
            model_provider=settings.DEFAULT_MODEL_PROVIDER,
            model_name=settings.DEFAULT_MODEL_NAME,
            temperature=settings.DEFAULT_TEMPERATURE,
            max_tokens=settings.DEFAULT_MAX_TOKENS
        )
        
        db.add(conversation)
        await db.flush()  # Get the ID
        
        return conversation

async def cache_conversation_data(
    cache: ConversationCache,
    conversation_id: int,
    db: AsyncSession
):
    """Cache conversation and its messages"""
    
    try:
        # Get conversation
        conv_stmt = select(Conversation).where(Conversation.id == conversation_id)
        conv_result = await db.execute(conv_stmt)
        conversation = conv_result.scalar_one_or_none()
        
        if conversation:
            # Cache conversation
            await cache.set_conversation(conversation_id, {
                "id": conversation.id,
                "title": conversation.title,
                "status": conversation.status.value,
                "created_at": conversation.created_at.isoformat(),
                "updated_at": conversation.updated_at.isoformat() if conversation.updated_at else None
            })
            
            # Get and cache recent messages
            msg_stmt = (
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at)
                .limit(100)  # Cache last 100 messages
            )
            msg_result = await db.execute(msg_stmt)
            messages = msg_result.scalars().all()
            
            message_data = [
                {
                    "id": msg.id,
                    "content": msg.content,
                    "role": msg.role.value,
                    "created_at": msg.created_at.isoformat(),
                    "model_used": msg.model_used,
                    "tokens_used": msg.tokens_used,
                    "cost": msg.cost
                }
                for msg in messages
            ]
            
            await cache.set_messages(conversation_id, message_data)
            
    except Exception as e:
        cache.logger.warning(f"Failed to cache conversation data: {e}")
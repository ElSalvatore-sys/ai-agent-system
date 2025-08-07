"""
Enhanced WebSocket Chat Routes
Comprehensive real-time chat system with rooms, typing indicators, file uploads,
message queuing, delivery confirmation, and advanced connection management
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from pydantic import BaseModel, Field

from app.database.database import get_db
from app.database.models import User, Conversation, Message, MessageRole
from app.middleware.auth import get_current_user
from app.models.ai_orchestrator import AIOrchestrator, ModelRequest
from app.services.websocket_manager import websocket_manager, WSMessage, MessageType, ConnectionStatus
from app.services.websocket_auth import websocket_auth
from app.core.logger import LoggerMixin

logger = logging.getLogger(__name__)
router = APIRouter()

class WebSocketStats(BaseModel):
    """WebSocket statistics model"""
    active_connections: int
    total_users: int
    active_rooms: int
    queued_messages: int
    typing_indicators: int

class RoomInfo(BaseModel):
    """Room information model"""
    room_id: str
    name: str
    room_type: str
    participants: List[str]
    participants_count: int
    created_at: str
    last_activity: str
    typing_users: List[str] = Field(default_factory=list)

class ConnectionInfo(BaseModel):
    """Connection information model"""
    connection_id: str
    user_id: str
    status: str
    rooms: List[str]
    last_activity: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class WebSocketTokenResponse(BaseModel):
    """WebSocket token response model"""
    token: str
    expires_in: int

class WebSocketChatHandler(LoggerMixin):
    """Enhanced WebSocket chat message handler"""
    
    def __init__(self):
        self.ai_orchestrator: Optional[AIOrchestrator] = None
    
    async def initialize(self):
        """Initialize the chat handler"""
        try:
            # Get AI orchestrator from app state
            from main import app
            if hasattr(app.state, 'ai_orchestrator'):
                self.ai_orchestrator = app.state.ai_orchestrator
            else:
                self.ai_orchestrator = AIOrchestrator()
                await self.ai_orchestrator.initialize()
            
            self.logger.info("WebSocket chat handler initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize chat handler: {e}")
            raise

    async def handle_ai_chat_message(
        self, 
        connection_id: str, 
        ws_message: WSMessage, 
        user: User
    ):
        """Handle AI chat message with streaming response"""
        try:
            message_content = ws_message.data.get("message", "")
            conversation_id = ws_message.data.get("conversation_id")
            model_preference = ws_message.data.get("model_preference")
            system_prompt = ws_message.data.get("system_prompt")
            temperature = ws_message.data.get("temperature", 0.7)
            max_tokens = ws_message.data.get("max_tokens", 2048)
            
            if not message_content:
                await websocket_manager._send_error(connection_id, "Message content required")
                return
            
            # Get database session
            async for db in get_db():
                # Get or create conversation
                conversation = await self._get_or_create_conversation(
                    db, user.id, conversation_id, model_preference
                )
                
                # Save user message
                user_message = Message(
                    conversation_id=conversation.id,
                    role=MessageRole.USER,
                    content=message_content,
                    created_at=datetime.utcnow()
                )
                db.add(user_message)
                await db.flush()  # Get the ID
                
                # Send user message to room if applicable
                if ws_message.room_id:
                    await websocket_manager._broadcast_to_room(
                        ws_message.room_id,
                        WSMessage(
                            type=MessageType.CHAT,
                            data={
                                "role": "user",
                                "content": message_content,
                                "user_id": user.id,
                                "username": user.username,
                                "message_id": user_message.id,
                                "conversation_id": conversation.id,
                                "timestamp": datetime.utcnow().isoformat()
                            },
                            room_id=ws_message.room_id,
                            message_id=str(user_message.id)
                        ),
                        exclude_user=user.id
                    )
                
                # Get conversation context
                context_messages = await self._get_conversation_context(db, conversation.id)
                
                # Prepare AI request
                ai_request = ModelRequest(
                    prompt=message_content,
                    system_prompt=system_prompt or conversation.system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    model_preference=model_preference,
                    user_id=user.id,
                    conversation_id=conversation.id,
                    context_messages=context_messages
                )
                
                # Stream AI response
                full_response = ""
                tokens_used = 0
                model_used = ""
                cost = 0.0
                response_start_time = datetime.utcnow()
                
                try:
                    # Send typing indicator to room
                    if ws_message.room_id:
                        await websocket_manager._broadcast_to_room(
                            ws_message.room_id,
                            WSMessage(
                                type=MessageType.TYPING,
                                data={
                                    "user_id": "assistant",
                                    "username": "AI Assistant",
                                    "is_typing": True,
                                    "room_id": ws_message.room_id
                                },
                                room_id=ws_message.room_id
                            )
                        )
                    
                    async for chunk in self.ai_orchestrator.stream_generate(ai_request):
                        chunk_content = chunk.get("content", "")
                        
                        if chunk_content:
                            full_response += chunk_content
                            
                            # Send chunk to user
                            await websocket_manager._send_to_connection(connection_id, WSMessage(
                                type=MessageType.CHAT,
                                data={
                                    "role": "assistant",
                                    "content": chunk_content,
                                    "streaming": True,
                                    "conversation_id": conversation.id,
                                    "message_id": ws_message.message_id,
                                    "timestamp": datetime.utcnow().isoformat(),
                                    **chunk
                                }
                            ))
                            
                            # Broadcast chunk to room if applicable
                            if ws_message.room_id:
                                await websocket_manager._broadcast_to_room(
                                    ws_message.room_id,
                                    WSMessage(
                                        type=MessageType.CHAT,
                                        data={
                                            "role": "assistant",
                                            "content": chunk_content,
                                            "streaming": True,
                                            "user_id": "assistant",
                                            "username": "AI Assistant",
                                            "conversation_id": conversation.id,
                                            "timestamp": datetime.utcnow().isoformat()
                                        },
                                        room_id=ws_message.room_id
                                    ),
                                    exclude_user=user.id
                                )
                        
                        # Check for completion
                        if chunk.get("done"):
                            tokens_used = chunk.get("tokens_used", 0)
                            model_used = chunk.get("model_used", "unknown")
                            cost = chunk.get("cost", 0.0)
                            break
                    
                    # Stop typing indicator
                    if ws_message.room_id:
                        await websocket_manager._broadcast_to_room(
                            ws_message.room_id,
                            WSMessage(
                                type=MessageType.TYPING,
                                data={
                                    "user_id": "assistant",
                                    "username": "AI Assistant",
                                    "is_typing": False,
                                    "room_id": ws_message.room_id
                                },
                                room_id=ws_message.room_id
                            )
                        )
                    
                    # Save AI response
                    response_time = (datetime.utcnow() - response_start_time).total_seconds()
                    ai_message = Message(
                        conversation_id=conversation.id,
                        role=MessageRole.ASSISTANT,
                        content=full_response,
                        model_used=model_used,
                        tokens_used=tokens_used,
                        cost=cost,
                        response_time=response_time,
                        created_at=datetime.utcnow()
                    )
                    db.add(ai_message)
                    
                    # Update conversation
                    conversation.updated_at = datetime.utcnow()
                    
                    await db.commit()
                    
                    # Send completion message
                    completion_data = {
                        "role": "assistant",
                        "content": "",
                        "streaming": False,
                        "done": True,
                        "message_id": ai_message.id,
                        "conversation_id": conversation.id,
                        "tokens_used": tokens_used,
                        "model_used": model_used,
                        "cost": cost,
                        "response_time": response_time,
                        "timestamp": ai_message.created_at.isoformat()
                    }
                    
                    await websocket_manager._send_to_connection(connection_id, WSMessage(
                        type=MessageType.CHAT,
                        data=completion_data
                    ))
                    
                    # Broadcast completion to room
                    if ws_message.room_id:
                        await websocket_manager._broadcast_to_room(
                            ws_message.room_id,
                            WSMessage(
                                type=MessageType.CHAT,
                                data={
                                    **completion_data,
                                    "user_id": "assistant",
                                    "username": "AI Assistant"
                                },
                                room_id=ws_message.room_id
                            ),
                            exclude_user=user.id
                        )
                    
                except Exception as e:
                    self.logger.error(f"Error in AI streaming: {e}")
                    await websocket_manager._send_error(connection_id, f"AI processing error: {str(e)}")
                    
                    # Stop typing indicator on error
                    if ws_message.room_id:
                        await websocket_manager._broadcast_to_room(
                            ws_message.room_id,
                            WSMessage(
                                type=MessageType.TYPING,
                                data={
                                    "user_id": "assistant",
                                    "username": "AI Assistant",
                                    "is_typing": False,
                                    "room_id": ws_message.room_id
                                },
                                room_id=ws_message.room_id
                            )
                        )
                
                break  # Exit the db session loop
                
        except Exception as e:
            self.logger.error(f"Error handling AI chat message: {e}")
            await websocket_manager._send_error(connection_id, "Failed to process AI message")

    async def _get_or_create_conversation(
        self,
        db: AsyncSession,
        user_id: int,
        conversation_id: Optional[int] = None,
        model_preference: Optional[str] = None
    ) -> Conversation:
        """Get existing conversation or create new one"""
        if conversation_id:
            result = await db.execute(
                select(Conversation).where(
                    Conversation.id == conversation_id,
                    Conversation.user_id == user_id
                )
            )
            conversation = result.scalar_one_or_none()
            if conversation:
                return conversation
        
        # Create new conversation
        from app.core.config import settings
        conversation = Conversation(
            user_id=user_id,
            title="New WebSocket Conversation",
            model_provider=settings.DEFAULT_MODEL_PROVIDER,
            model_name=settings.DEFAULT_MODEL_NAME,
            temperature=settings.DEFAULT_TEMPERATURE,
            max_tokens=settings.DEFAULT_MAX_TOKENS
        )
        db.add(conversation)
        await db.flush()
        
        return conversation

    async def _get_conversation_context(self, db: AsyncSession, conversation_id: int, limit: int = 10) -> List[Dict[str, str]]:
        """Get recent messages for conversation context"""
        result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(desc(Message.created_at))
            .limit(limit)
        )
        messages = result.scalars().all()
        
        # Reverse to get chronological order
        messages = list(reversed(messages))
        
        return [
            {
                "role": "user" if msg.role == MessageRole.USER else "assistant",
                "content": msg.content
            }
            for msg in messages
        ]

# Global chat handler
chat_handler = WebSocketChatHandler()

# WebSocket endpoints
@router.websocket("/ws")
async def websocket_chat_enhanced(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    room_id: Optional[str] = Query(None),
    room_type: Optional[str] = Query("private"),
    room_name: Optional[str] = Query(None)
):
    """
    Enhanced WebSocket endpoint for real-time chat
    Supports authentication, rooms, typing indicators, file uploads, and more
    """
    connection_id = None
    user = None
    
    try:
        # Initialize chat handler if needed
        if not chat_handler.ai_orchestrator:
            await chat_handler.initialize()
        
        # Authenticate user
        user = await websocket_auth.authenticate_websocket(websocket, token)
        if not user:
            await websocket.close(code=4001, reason="Authentication required")
            return
        
        # Connect to WebSocket manager
        connection_id = await websocket_manager.connect(
            websocket, 
            str(user.id),
            {
                "user_agent": websocket.headers.get("user-agent", "unknown"),
                "ip_address": websocket.client.host if websocket.client else "unknown",
                "username": user.username
            }
        )
        
        # Auto-join room if specified
        if room_id:
            # Validate room access
            if await websocket_auth.validate_room_access(user, room_id, room_type):
                await websocket_manager.join_room(
                    connection_id, 
                    room_id, 
                    room_name or f"Room {room_id}",
                    room_type
                )
            else:
                await websocket_manager._send_error(connection_id, "Access denied to room")
        
        # Log connection activity
        await websocket_auth.log_websocket_activity(
            str(user.id), "websocket_connect", 
            {"connection_id": connection_id, "room_id": room_id}
        )
        
        # Message handling loop
        while True:
            try:
                # Receive message
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                # Create WebSocket message
                ws_message = WSMessage(
                    type=MessageType(message_data.get("type", "chat")),
                    data=message_data.get("data", {}),
                    room_id=message_data.get("room_id"),
                    message_id=message_data.get("message_id"),
                    user_id=str(user.id)
                )
                
                # Handle AI chat messages specifically
                if ws_message.type == MessageType.CHAT:
                    await chat_handler.handle_ai_chat_message(
                        connection_id, ws_message, user
                    )
                else:
                    # Send to WebSocket manager for other message types
                    await websocket_manager.send_message(connection_id, ws_message)
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket_manager._send_error(connection_id, "Invalid JSON format")
            except ValueError as e:
                await websocket_manager._send_error(connection_id, f"Invalid message type: {e}")
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                await websocket_manager._send_error(connection_id, "Message processing error")
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user {user.id if user else 'unknown'}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close(code=4000, reason="Internal server error")
        except:
            pass
    finally:
        # Cleanup
        if connection_id:
            await websocket_manager.disconnect(connection_id)
        
        # Log disconnection
        if user:
            await websocket_auth.log_websocket_activity(
                str(user.id), "websocket_disconnect",
                {"connection_id": connection_id}
            )

# WebSocket management endpoints
@router.get("/ws/stats", response_model=WebSocketStats)
async def websocket_stats(current_user: User = Depends(get_current_user)):
    """Get WebSocket connection statistics"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    stats = websocket_manager.get_stats()
    return WebSocketStats(**stats)

@router.get("/ws/rooms/{room_id}", response_model=RoomInfo)
async def get_room_info(
    room_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get room information"""
    room_info = websocket_manager.get_room_info(room_id)
    if not room_info:
        raise HTTPException(status_code=404, detail="Room not found")
    
    # Check if user has access to room info
    if str(current_user.id) not in room_info["participants"] and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return RoomInfo(**room_info)

@router.get("/ws/connections/{connection_id}", response_model=ConnectionInfo)
async def get_connection_info(
    connection_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get connection information"""
    connection_info = websocket_manager.get_connection_info(connection_id)
    if not connection_info:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    # Check if user owns the connection or is admin
    if connection_info["user_id"] != str(current_user.id) and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return ConnectionInfo(**connection_info)

@router.post("/ws/token", response_model=WebSocketTokenResponse)
async def generate_websocket_token(current_user: User = Depends(get_current_user)):
    """Generate a WebSocket authentication token"""
    token = websocket_auth.generate_websocket_token(str(current_user.id))
    return WebSocketTokenResponse(token=token, expires_in=86400)  # 24 hours

@router.post("/ws/rooms")
async def create_room(
    room_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Create a new chat room"""
    room_id = room_data.get("room_id") or str(uuid4())
    room_name = room_data.get("room_name", f"Room {room_id}")
    room_type = room_data.get("room_type", "private")
    
    # Validate room type
    if room_type not in ["private", "shared", "group"]:
        raise HTTPException(status_code=400, detail="Invalid room type")
    
    # Check if room already exists
    if websocket_manager.get_room_info(room_id):
        raise HTTPException(status_code=409, detail="Room already exists")
    
    return {
        "room_id": room_id,
        "room_name": room_name,
        "room_type": room_type,
        "creator": current_user.id,
        "created_at": datetime.utcnow().isoformat()
    }

@router.delete("/ws/rooms/{room_id}")
async def delete_room(
    room_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a chat room"""
    room_info = websocket_manager.get_room_info(room_id)
    if not room_info:
        raise HTTPException(status_code=404, detail="Room not found")
    
    # Only room creator or admin can delete room
    # For now, any authenticated user can delete
    # In production, implement proper room ownership
    
    # Remove all participants
    for participant_id in list(room_info["participants"]):
        # Find connections for this participant
        if participant_id in websocket_manager.user_connections:
            for connection_id in list(websocket_manager.user_connections[participant_id]):
                await websocket_manager.leave_room(connection_id, room_id)
    
    # Remove room
    if room_id in websocket_manager.rooms:
        del websocket_manager.rooms[room_id]
    
    return {"message": "Room deleted successfully"}

@router.get("/ws/user/{user_id}/connections")
async def get_user_connections(
    user_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get all connections for a user"""
    # Check permission
    if str(current_user.id) != user_id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if user_id not in websocket_manager.user_connections:
        return {"connections": []}
    
    connection_ids = list(websocket_manager.user_connections[user_id])
    connections = []
    
    for connection_id in connection_ids:
        if connection_id in websocket_manager.active_connections:
            connection = websocket_manager.active_connections[connection_id]
            connections.append({
                "connection_id": connection.connection_id,
                "status": connection.status,
                "rooms": list(connection.rooms),
                "last_activity": connection.last_activity.isoformat(),
                "metadata": connection.metadata
            })
    
    return {"connections": connections}
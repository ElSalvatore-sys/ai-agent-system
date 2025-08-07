"""
WebSocket Connection Manager Service
Handles real-time bidirectional communication with advanced features
"""
import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from uuid import uuid4
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
from pydantic import BaseModel, Field
import redis.asyncio as redis

from app.core.config import settings
from app.database.models import User, Conversation, Message

logger = logging.getLogger(__name__)

class MessageType(str, Enum):
    """WebSocket message types"""
    CHAT = "chat"
    TYPING = "typing"
    CONNECT = "connect"
    DISCONNECT = "disconnect" 
    JOIN_ROOM = "join_room"
    LEAVE_ROOM = "leave_room"
    DELIVERY_CONFIRMATION = "delivery_confirmation"
    READ_RECEIPT = "read_receipt"
    FILE_UPLOAD = "file_upload"
    ERROR = "error"
    SYSTEM = "system"

class ConnectionStatus(str, Enum):
    """Connection status types"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    TYPING = "typing"
    IDLE = "idle"

class WSMessage(BaseModel):
    """WebSocket message model"""
    type: MessageType
    data: Dict[str, Any] = Field(default_factory=dict)
    room_id: Optional[str] = None
    message_id: Optional[str] = None
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None

class WSConnection(BaseModel):
    """WebSocket connection model"""
    connection_id: str
    user_id: str
    websocket: WebSocket = Field(exclude=True)
    rooms: Set[str] = Field(default_factory=set)
    status: ConnectionStatus = ConnectionStatus.CONNECTED
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True

class ChatRoom(BaseModel):
    """Chat room model"""
    room_id: str
    name: str
    room_type: str = "private"  # private, shared, group
    participants: Set[str] = Field(default_factory=set)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class WebSocketManager:
    """Advanced WebSocket connection manager"""
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.REDIS_URL
        self.redis_client: Optional[redis.Redis] = None
        
        # Connection management
        self.active_connections: Dict[str, WSConnection] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        self.rooms: Dict[str, ChatRoom] = {}
        
        # Message queuing
        self.message_queue: Dict[str, List[WSMessage]] = {}  # user_id -> messages
        self.pending_deliveries: Dict[str, WSMessage] = {}  # message_id -> message
        
        # Rate limiting
        self.rate_limits: Dict[str, List[float]] = {}  # connection_id -> timestamps
        self.rate_limit_window = 60  # seconds
        self.rate_limit_max = 100  # messages per window
        
        # Typing indicators
        self.typing_users: Dict[str, Dict[str, float]] = {}  # room_id -> {user_id: timestamp}
        self.typing_timeout = 5.0  # seconds
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def initialize(self):
        """Initialize the WebSocket manager"""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            logger.info("WebSocket manager initialized with Redis")
            
            # Start background tasks
            self._cleanup_task = asyncio.create_task(self._cleanup_inactive_connections())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
            
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket manager: {e}")
            raise

    async def cleanup(self):
        """Cleanup WebSocket manager resources"""
        # Cancel background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            
        # Close all connections
        for connection in list(self.active_connections.values()):
            await self.disconnect(connection.connection_id)
            
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
            
        logger.info("WebSocket manager cleaned up")

    async def connect(self, websocket: WebSocket, user_id: str, 
                     connection_metadata: Dict[str, Any] = None) -> str:
        """Connect a new WebSocket client"""
        await websocket.accept()
        
        connection_id = str(uuid4())
        connection = WSConnection(
            connection_id=connection_id,
            user_id=user_id,
            websocket=websocket,
            metadata=connection_metadata or {}
        )
        
        # Store connection
        self.active_connections[connection_id] = connection
        
        # Track user connections
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection_id)
        
        # Send connection confirmation
        await self._send_to_connection(connection_id, WSMessage(
            type=MessageType.CONNECT,
            data={
                "connection_id": connection_id,
                "status": "connected",
                "server_time": datetime.utcnow().isoformat()
            }
        ))
        
        # Deliver queued messages
        await self._deliver_queued_messages(user_id)
        
        logger.info(f"WebSocket connected: {connection_id} for user {user_id}")
        return connection_id

    async def disconnect(self, connection_id: str):
        """Disconnect a WebSocket client"""
        if connection_id not in self.active_connections:
            return
            
        connection = self.active_connections[connection_id]
        user_id = connection.user_id
        
        # Leave all rooms
        for room_id in list(connection.rooms):
            await self.leave_room(connection_id, room_id)
        
        # Remove from user connections
        if user_id in self.user_connections:
            self.user_connections[user_id].discard(connection_id)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        # Close WebSocket if still open
        if connection.websocket.client_state == WebSocketState.CONNECTED:
            try:
                await connection.websocket.close()
            except Exception:
                pass
        
        # Remove connection
        del self.active_connections[connection_id]
        
        logger.info(f"WebSocket disconnected: {connection_id} for user {user_id}")

    async def join_room(self, connection_id: str, room_id: str, 
                       room_name: str = None, room_type: str = "private") -> bool:
        """Join a chat room"""
        if connection_id not in self.active_connections:
            return False
            
        connection = self.active_connections[connection_id]
        user_id = connection.user_id
        
        # Create room if it doesn't exist
        if room_id not in self.rooms:
            self.rooms[room_id] = ChatRoom(
                room_id=room_id,
                name=room_name or room_id,
                room_type=room_type
            )
        
        room = self.rooms[room_id]
        
        # Add user to room
        room.participants.add(user_id)
        connection.rooms.add(room_id)
        room.last_activity = datetime.utcnow()
        
        # Notify room participants
        await self._broadcast_to_room(room_id, WSMessage(
            type=MessageType.SYSTEM,
            data={
                "action": "user_joined",
                "user_id": user_id,
                "room_id": room_id,
                "participants_count": len(room.participants)
            }
        ), exclude_user=user_id)
        
        # Send room info to user
        await self._send_to_connection(connection_id, WSMessage(
            type=MessageType.JOIN_ROOM,
            data={
                "room_id": room_id,
                "room_name": room.name,
                "room_type": room.room_type,
                "participants": list(room.participants),
                "joined_at": datetime.utcnow().isoformat()
            }
        ))
        
        logger.info(f"User {user_id} joined room {room_id}")
        return True

    async def leave_room(self, connection_id: str, room_id: str) -> bool:
        """Leave a chat room"""
        if connection_id not in self.active_connections:
            return False
            
        connection = self.active_connections[connection_id]
        user_id = connection.user_id
        
        if room_id not in self.rooms or room_id not in connection.rooms:
            return False
            
        room = self.rooms[room_id]
        
        # Remove user from room
        room.participants.discard(user_id)
        connection.rooms.discard(room_id)
        
        # Clean up typing indicators
        if room_id in self.typing_users:
            self.typing_users[room_id].pop(user_id, None)
        
        # Notify room participants
        await self._broadcast_to_room(room_id, WSMessage(
            type=MessageType.SYSTEM,
            data={
                "action": "user_left",
                "user_id": user_id,
                "room_id": room_id,
                "participants_count": len(room.participants)
            }
        ), exclude_user=user_id)
        
        # Remove room if empty
        if not room.participants:
            del self.rooms[room_id]
            if room_id in self.typing_users:
                del self.typing_users[room_id]
        
        logger.info(f"User {user_id} left room {room_id}")
        return True

    async def send_message(self, connection_id: str, message: WSMessage) -> bool:
        """Send a message through WebSocket"""
        if not await self._check_rate_limit(connection_id):
            await self._send_error(connection_id, "Rate limit exceeded")
            return False
            
        if connection_id not in self.active_connections:
            return False
            
        connection = self.active_connections[connection_id]
        message.user_id = connection.user_id
        message.message_id = message.message_id or str(uuid4())
        
        # Update last activity
        connection.last_activity = datetime.utcnow()
        
        # Handle different message types
        if message.type == MessageType.CHAT:
            await self._handle_chat_message(connection_id, message)
        elif message.type == MessageType.TYPING:
            await self._handle_typing_indicator(connection_id, message)
        elif message.type == MessageType.JOIN_ROOM:
            room_data = message.data
            await self.join_room(
                connection_id, 
                room_data.get("room_id"),
                room_data.get("room_name"),
                room_data.get("room_type", "private")
            )
        elif message.type == MessageType.LEAVE_ROOM:
            await self.leave_room(connection_id, message.data.get("room_id"))
        elif message.type == MessageType.FILE_UPLOAD:
            await self._handle_file_upload(connection_id, message)
        
        return True

    async def _handle_chat_message(self, connection_id: str, message: WSMessage):
        """Handle chat message"""
        connection = self.active_connections[connection_id]
        room_id = message.room_id
        
        if room_id and room_id in self.rooms:
            # Broadcast to room
            await self._broadcast_to_room(room_id, message, exclude_user=connection.user_id)
            
            # Update room activity
            self.rooms[room_id].last_activity = datetime.utcnow()
        
        # Send delivery confirmation
        await self._send_to_connection(connection_id, WSMessage(
            type=MessageType.DELIVERY_CONFIRMATION,
            data={
                "message_id": message.message_id,
                "status": "delivered",
                "timestamp": datetime.utcnow().isoformat()
            }
        ))
        
        # Store message for offline users (if applicable)
        await self._store_message_for_offline_users(message)

    async def _handle_typing_indicator(self, connection_id: str, message: WSMessage):
        """Handle typing indicator"""
        connection = self.active_connections[connection_id]
        room_id = message.room_id
        user_id = connection.user_id
        
        if not room_id or room_id not in self.rooms:
            return
            
        is_typing = message.data.get("is_typing", False)
        
        if room_id not in self.typing_users:
            self.typing_users[room_id] = {}
        
        if is_typing:
            self.typing_users[room_id][user_id] = time.time()
        else:
            self.typing_users[room_id].pop(user_id, None)
        
        # Broadcast typing status to room
        await self._broadcast_to_room(room_id, WSMessage(
            type=MessageType.TYPING,
            data={
                "user_id": user_id,
                "is_typing": is_typing,
                "room_id": room_id
            }
        ), exclude_user=user_id)

    async def _handle_file_upload(self, connection_id: str, message: WSMessage):
        """Handle file upload through WebSocket"""
        try:
            from app.services.file_upload_service import file_upload_service
            await file_upload_service.handle_file_upload_message(connection_id, message)
        except Exception as e:
            logger.error(f"Error handling file upload: {e}")
            await self._send_error(connection_id, f"File upload error: {str(e)}")

    async def _broadcast_to_room(self, room_id: str, message: WSMessage, 
                                exclude_user: str = None):
        """Broadcast message to all users in a room"""
        if room_id not in self.rooms:
            return
            
        room = self.rooms[room_id]
        message.room_id = room_id
        
        for user_id in room.participants:
            if exclude_user and user_id == exclude_user:
                continue
                
            await self._send_to_user(user_id, message)

    async def _send_to_user(self, user_id: str, message: WSMessage):
        """Send message to all connections of a user"""
        if user_id not in self.user_connections:
            # Queue message for offline user
            await self._queue_message(user_id, message)
            return
            
        connections = list(self.user_connections[user_id])
        for connection_id in connections:
            await self._send_to_connection(connection_id, message)

    async def _send_to_connection(self, connection_id: str, message: WSMessage):
        """Send message to a specific connection"""
        if connection_id not in self.active_connections:
            return
            
        connection = self.active_connections[connection_id]
        
        try:
            if connection.websocket.client_state == WebSocketState.CONNECTED:
                message_data = message.model_dump()
                message_data["timestamp"] = message_data["timestamp"].isoformat()
                await connection.websocket.send_text(json.dumps(message_data))
        except Exception as e:
            logger.error(f"Error sending message to {connection_id}: {e}")
            await self.disconnect(connection_id)

    async def _send_error(self, connection_id: str, error_message: str):
        """Send error message to connection"""
        await self._send_to_connection(connection_id, WSMessage(
            type=MessageType.ERROR,
            data={"error": error_message}
        ))

    async def _queue_message(self, user_id: str, message: WSMessage):
        """Queue message for offline user"""
        if user_id not in self.message_queue:
            self.message_queue[user_id] = []
        
        self.message_queue[user_id].append(message)
        
        # Limit queue size
        if len(self.message_queue[user_id]) > 100:
            self.message_queue[user_id] = self.message_queue[user_id][-100:]

    async def _deliver_queued_messages(self, user_id: str):
        """Deliver queued messages to user"""
        if user_id not in self.message_queue:
            return
            
        messages = self.message_queue.pop(user_id, [])
        for message in messages:
            await self._send_to_user(user_id, message)

    async def _check_rate_limit(self, connection_id: str) -> bool:
        """Check rate limit for connection"""
        now = time.time()
        
        if connection_id not in self.rate_limits:
            self.rate_limits[connection_id] = []
        
        # Remove old timestamps
        cutoff = now - self.rate_limit_window
        self.rate_limits[connection_id] = [
            ts for ts in self.rate_limits[connection_id] if ts > cutoff
        ]
        
        # Check limit
        if len(self.rate_limits[connection_id]) >= self.rate_limit_max:
            return False
        
        # Add current timestamp
        self.rate_limits[connection_id].append(now)
        return True

    async def _store_message_for_offline_users(self, message: WSMessage):
        """Store message for offline users in room"""
        # This would integrate with the database to store messages
        # for users who are not currently connected
        pass

    async def _cleanup_inactive_connections(self):
        """Background task to cleanup inactive connections"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                now = datetime.utcnow()
                inactive_connections = []
                
                for connection_id, connection in self.active_connections.items():
                    # Check if connection is inactive (no activity for 30 minutes)
                    if now - connection.last_activity > timedelta(minutes=30):
                        inactive_connections.append(connection_id)
                    
                    # Check if WebSocket is still connected
                    if connection.websocket.client_state != WebSocketState.CONNECTED:
                        inactive_connections.append(connection_id)
                
                # Cleanup inactive connections
                for connection_id in inactive_connections:
                    await self.disconnect(connection_id)
                
                # Cleanup old typing indicators
                current_time = time.time()
                for room_id in list(self.typing_users.keys()):
                    typing_room = self.typing_users[room_id]
                    expired_users = [
                        user_id for user_id, timestamp in typing_room.items()
                        if current_time - timestamp > self.typing_timeout
                    ]
                    for user_id in expired_users:
                        typing_room.pop(user_id, None)
                    
                    if not typing_room:
                        del self.typing_users[room_id]
                
                logger.debug(f"Cleaned up {len(inactive_connections)} inactive connections")
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")

    async def _heartbeat_monitor(self):
        """Background task to monitor connection health"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Send heartbeat to all connections
                heartbeat_message = WSMessage(
                    type=MessageType.SYSTEM,
                    data={
                        "heartbeat": True,
                        "server_time": datetime.utcnow().isoformat()
                    }
                )
                
                failed_connections = []
                for connection_id in list(self.active_connections.keys()):
                    try:
                        await self._send_to_connection(connection_id, heartbeat_message)
                    except Exception:
                        failed_connections.append(connection_id)
                
                # Remove failed connections
                for connection_id in failed_connections:
                    await self.disconnect(connection_id)
                
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")

    def get_room_info(self, room_id: str) -> Optional[Dict[str, Any]]:
        """Get room information"""
        if room_id not in self.rooms:
            return None
            
        room = self.rooms[room_id]
        return {
            "room_id": room.room_id,
            "name": room.name,
            "room_type": room.room_type,
            "participants": list(room.participants),
            "participants_count": len(room.participants),
            "created_at": room.created_at.isoformat(),
            "last_activity": room.last_activity.isoformat(),
            "typing_users": list(self.typing_users.get(room_id, {}).keys())
        }

    def get_connection_info(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get connection information"""
        if connection_id not in self.active_connections:
            return None
            
        connection = self.active_connections[connection_id]
        return {
            "connection_id": connection.connection_id,
            "user_id": connection.user_id,
            "status": connection.status,
            "rooms": list(connection.rooms),
            "last_activity": connection.last_activity.isoformat(),
            "metadata": connection.metadata
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket manager statistics"""
        return {
            "active_connections": len(self.active_connections),
            "total_users": len(self.user_connections),
            "active_rooms": len(self.rooms),
            "queued_messages": sum(len(msgs) for msgs in self.message_queue.values()),
            "typing_indicators": sum(len(users) for users in self.typing_users.values())
        }

# Global WebSocket manager instance
websocket_manager = WebSocketManager()
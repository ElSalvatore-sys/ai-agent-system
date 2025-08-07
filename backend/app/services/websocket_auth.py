"""
WebSocket Authentication Service
Handles user authentication for WebSocket connections
"""
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import WebSocket, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.database.models import User
from app.database.database import get_db

logger = logging.getLogger(__name__)

class WebSocketAuthenticator:
    """WebSocket authentication handler"""
    
    def __init__(self):
        self.secret_key = settings.SECRET_KEY
        self.algorithm = "HS256"
        self.security = HTTPBearer()

    async def authenticate_websocket(self, websocket: WebSocket, 
                                   token: Optional[str] = None) -> Optional[User]:
        """
        Authenticate WebSocket connection
        
        Args:
            websocket: WebSocket connection
            token: JWT token (from query params, headers, or initial message)
            
        Returns:
            User object if authenticated, None otherwise
        """
        try:
            # Try to get token from different sources
            if not token:
                token = await self._extract_token_from_websocket(websocket)
            
            if not token:
                logger.warning("No authentication token provided for WebSocket")
                return None
            
            # Verify JWT token
            user = await self._verify_token(token)
            if not user:
                logger.warning("Invalid token for WebSocket authentication")
                return None
            
            # Check if user is active
            if not user.is_active:
                logger.warning(f"Inactive user attempted WebSocket connection: {user.id}")
                return None
            
            logger.info(f"WebSocket authenticated for user: {user.id}")
            return user
            
        except Exception as e:
            logger.error(f"WebSocket authentication error: {e}")
            return None

    async def _extract_token_from_websocket(self, websocket: WebSocket) -> Optional[str]:
        """Extract token from WebSocket connection"""
        try:
            # Try query parameters first
            token = websocket.query_params.get("token")
            if token:
                return token
            
            # Try headers
            auth_header = websocket.headers.get("authorization")
            if auth_header and auth_header.startswith("Bearer "):
                return auth_header.split(" ")[1]
            
            # Try cookies
            token = websocket.cookies.get("access_token")
            if token:
                return token
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting token from WebSocket: {e}")
            return None

    async def _verify_token(self, token: str) -> Optional[User]:
        """Verify JWT token and get user"""
        try:
            # Decode JWT token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            user_id: str = payload.get("sub")
            
            if user_id is None:
                return None
            
            # Check token expiration
            exp = payload.get("exp")
            if exp and datetime.utcnow().timestamp() > exp:
                logger.warning(f"Expired token for user: {user_id}")
                return None
            
            # Get user from database
            async for db in get_db():
                result = await db.execute(
                    select(User).where(User.id == user_id)
                )
                user = result.scalar_one_or_none()
                
                if user:
                    # Update last seen
                    user.last_seen = datetime.utcnow()
                    await db.commit()
                
                return user
            
        except JWTError as e:
            logger.warning(f"JWT decode error: {e}")
            return None
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None

    async def authenticate_message(self, message_data: Dict[str, Any]) -> Optional[User]:
        """
        Authenticate user from message data
        Used for initial authentication message
        """
        try:
            auth_data = message_data.get("auth", {})
            token = auth_data.get("token")
            
            if not token:
                return None
            
            return await self._verify_token(token)
            
        except Exception as e:
            logger.error(f"Message authentication error: {e}")
            return None

    def generate_websocket_token(self, user_id: str, expires_delta: Optional[timedelta] = None) -> str:
        """
        Generate a JWT token specifically for WebSocket connections
        Can have different expiration than regular API tokens
        """
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=24)  # Longer expiration for WebSocket
        
        to_encode = {
            "sub": user_id,
            "exp": expire,
            "type": "websocket",
            "iat": datetime.utcnow()
        }
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    async def refresh_websocket_token(self, current_token: str) -> Optional[str]:
        """Refresh WebSocket token if it's close to expiration"""
        try:
            payload = jwt.decode(current_token, self.secret_key, algorithms=[self.algorithm])
            user_id = payload.get("sub")
            exp = payload.get("exp")
            
            if not user_id or not exp:
                return None
            
            # If token expires within 1 hour, refresh it
            current_time = datetime.utcnow().timestamp()
            if exp - current_time < 3600:  # 1 hour
                return self.generate_websocket_token(user_id)
            
            return None
            
        except JWTError:
            return None

    async def validate_room_access(self, user: User, room_id: str, 
                                 room_type: str = "private") -> bool:
        """
        Validate if user has access to a specific room
        """
        try:
            # For private rooms, users can create/join any room
            if room_type == "private":
                return True
            
            # For shared rooms, implement access control logic
            if room_type == "shared":
                # Could check database for room permissions
                # For now, allow all authenticated users
                return True
            
            # For group rooms, check membership
            if room_type == "group":
                # Implement group membership checking
                # This would query the database for group membership
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Room access validation error: {e}")
            return False

    async def log_websocket_activity(self, user_id: str, activity_type: str, 
                                   metadata: Dict[str, Any] = None):
        """Log WebSocket activity for security monitoring"""
        try:
            log_data = {
                "user_id": user_id,
                "activity_type": activity_type,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }
            
            # In a production system, this would go to a security log
            logger.info(f"WebSocket activity: {log_data}")
            
        except Exception as e:
            logger.error(f"Error logging WebSocket activity: {e}")

# Global authenticator instance
websocket_auth = WebSocketAuthenticator()
import time
from typing import Callable, Optional
from fastapi import Request, Response, HTTPException, Depends
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.core.logger import LoggerMixin
from app.database.database import async_session_factory, get_db
from app.database.models import User

class AuthMiddleware(BaseHTTPMiddleware, LoggerMixin):
    """JWT Authentication middleware"""
    
    # Endpoints that don't require authentication
    PUBLIC_ENDPOINTS = {
        "/",
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/v1/auth/login",
        "/api/v1/auth/register",
        "/api/v1/auth/refresh",
    }
    
    def __init__(self, app):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with authentication"""
        
        # Skip authentication for public endpoints
        if self._is_public_endpoint(request.url.path):
            return await call_next(request)
        
        # Extract and verify token
        token = self._extract_token(request)
        
        if not token:
            return self._create_unauthorized_response("Missing authentication token")
        
        try:
            # Verify and decode token
            payload = jwt.decode(
                token, 
                settings.SECRET_KEY, 
                algorithms=[settings.ALGORITHM]
            )
            
            # Extract user information
            user_id = payload.get("sub")
            if not user_id:
                return self._create_unauthorized_response("Invalid token: missing user ID")
            
            # Check token expiration
            exp = payload.get("exp")
            if exp and time.time() > exp:
                return self._create_unauthorized_response("Token has expired")
            
            # Load user from database
            user = await self._load_user(int(user_id))
            if not user:
                return self._create_unauthorized_response("User not found")
            
            if not user.is_active:
                return self._create_unauthorized_response("User account is disabled")
            
            # Add user info to request state
            request.state.user_id = user.id
            request.state.user = user
            request.state.token_payload = payload
            
            # Update last login time
            await self._update_last_login(user)
            
        except JWTError as e:
            self.logger.warning(f"JWT validation failed: {e}")
            return self._create_unauthorized_response("Invalid authentication token")
        
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return self._create_unauthorized_response("Authentication failed")
        
        # Process request
        response = await call_next(request)
        
        return response
    
    def _is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public (doesn't require auth)"""
        
        # Check exact matches
        if path in self.PUBLIC_ENDPOINTS:
            return True
        
        # Check patterns
        public_patterns = [
            "/static/",
            "/favicon.ico",
        ]
        
        for pattern in public_patterns:
            if path.startswith(pattern):
                return True
        
        return False
    
    def _extract_token(self, request: Request) -> Optional[str]:
        """Extract JWT token from request"""
        
        # Check Authorization header (Bearer token)
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix
        
        # Check cookie
        token_cookie = request.cookies.get("access_token")
        if token_cookie:
            return token_cookie
        
        # Check query parameter (less secure, for debugging only)
        if settings.ENVIRONMENT == "development":
            token_param = request.query_params.get("token")
            if token_param:
                return token_param
        
        return None
    
    async def _load_user(self, user_id: int) -> Optional[User]:
        """Load user from database"""
        
        try:
            async with async_session_factory() as session:
                from sqlalchemy import select
                
                stmt = select(User).where(User.id == user_id)
                result = await session.execute(stmt)
                user = result.scalar_one_or_none()
                
                return user
                
        except Exception as e:
            self.logger.error(f"Failed to load user {user_id}: {e}")
            return None
    
    async def _update_last_login(self, user: User):
        """Update user's last login timestamp"""
        
        try:
            async with async_session_factory() as session:
                from sqlalchemy import update
                
                stmt = update(User).where(User.id == user.id).values(
                    last_login=time.time()
                )
                await session.execute(stmt)
                await session.commit()
                
        except Exception as e:
            self.logger.warning(f"Failed to update last login for user {user.id}: {e}")
    
    def _create_unauthorized_response(self, message: str) -> JSONResponse:
        """Create unauthorized response"""
        
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "code": 401,
                    "message": message,
                    "type": "authentication_error"
                }
            },
            headers={
                "WWW-Authenticate": "Bearer"
            }
        )

class RequireRole:
    """Decorator to require specific user roles"""
    
    def __init__(self, *required_roles: str):
        self.required_roles = set(required_roles)
    
    def __call__(self, func):
        async def wrapper(request: Request, *args, **kwargs):
            # Check if user is authenticated
            if not hasattr(request.state, 'user'):
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required"
                )
            
            # Check user role
            user = request.state.user
            if user.role not in self.required_roles:
                raise HTTPException(
                    status_code=403,
                    detail=f"Access denied. Required roles: {', '.join(self.required_roles)}"
                )
            
            return await func(request, *args, **kwargs)
        
        return wrapper

class RequirePermission:
    """Decorator to require specific permissions"""
    
    def __init__(self, permission: str):
        self.permission = permission
    
    def __call__(self, func):
        async def wrapper(request: Request, *args, **kwargs):
            # Check if user is authenticated
            if not hasattr(request.state, 'user'):
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required"
                )
            
            # For now, admin users have all permissions
            user = request.state.user
            if user.role != "admin":
                # In a real implementation, you'd check user permissions from database
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission denied: {self.permission}"
                )
            
            return await func(request, *args, **kwargs)
        
        return wrapper

def create_access_token(data: dict, expires_delta: Optional[int] = None) -> str:
    """Create JWT access token"""
    
    to_encode = data.copy()
    
    if expires_delta:
        expire = time.time() + expires_delta
    else:
        expire = time.time() + (settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.SECRET_KEY, 
        algorithm=settings.ALGORITHM
    )
    
    return encoded_jwt

def verify_token(token: str) -> Optional[dict]:
    """Verify and decode JWT token"""
    
    try:
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=[settings.ALGORITHM]
        )
        return payload
    
    except JWTError:
        return None


async def get_current_user(request: Request, db: AsyncSession = Depends(get_db)) -> User:
    """Get current authenticated user (dependency for routes)"""
    
    # Check if user is set by middleware
    if hasattr(request.state, 'user') and request.state.user:
        return request.state.user
    
    # Extract token manually if middleware didn't process it
    token = None
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:]
    
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Authentication token missing"
        )
    
    # Verify token
    payload = verify_token(token)
    if not payload:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    
    # Get user ID from token
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Invalid token: missing user ID"
        )
    
    # Load user from database
    stmt = select(User).where(User.id == int(user_id))
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=401,
            detail="User not found"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=401,
            detail="User account is disabled"
        )
    
    return user
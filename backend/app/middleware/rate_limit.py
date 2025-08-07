import time
import asyncio
from typing import Callable, Dict, Optional
from collections import defaultdict, deque
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.config import settings
from app.core.logger import LoggerMixin

class RateLimitMiddleware(BaseHTTPMiddleware, LoggerMixin):
    """Rate limiting middleware with sliding window algorithm"""
    
    def __init__(self, app):
        super().__init__(app)
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque())
        self.blocked_until: Dict[str, float] = {}
        self.cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background task to cleanup old entries"""
        async def cleanup():
            while True:
                await asyncio.sleep(60)  # Cleanup every minute
                await self._cleanup_old_entries()
        
        self.cleanup_task = asyncio.create_task(cleanup())
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting"""
        
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Get client identifier
        client_id = await self._get_client_id(request)
        
        # Check if client is currently blocked
        if await self._is_blocked(client_id):
            return self._create_rate_limit_response()
        
        # Check rate limit
        if not await self._check_rate_limit(client_id, request):
            return self._create_rate_limit_response()
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = await self._get_remaining_requests(client_id)
        reset_time = await self._get_reset_time(client_id)
        
        response.headers["X-RateLimit-Limit"] = str(settings.RATE_LIMIT_REQUESTS)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(reset_time))
        response.headers["X-RateLimit-Window"] = str(settings.RATE_LIMIT_WINDOW)
        
        return response
    
    async def _get_client_id(self, request: Request) -> str:
        """Get unique client identifier"""
        
        # Try to get user ID from request state (set by auth middleware)
        if hasattr(request.state, 'user_id') and request.state.user_id:
            return f"user:{request.state.user_id}"
        
        # Fall back to IP address
        client_ip = request.client.host
        
        # Check for forwarded IP headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            client_ip = real_ip
        
        return f"ip:{client_ip}"
    
    async def _is_blocked(self, client_id: str) -> bool:
        """Check if client is currently blocked"""
        
        if client_id not in self.blocked_until:
            return False
        
        if time.time() < self.blocked_until[client_id]:
            return True
        
        # Unblock client
        del self.blocked_until[client_id]
        return False
    
    async def _check_rate_limit(self, client_id: str, request: Request) -> bool:
        """Check if request is within rate limit"""
        
        current_time = time.time()
        window_start = current_time - settings.RATE_LIMIT_WINDOW
        
        # Get request history for this client
        history = self.request_history[client_id]
        
        # Remove requests outside the window
        while history and history[0] < window_start:
            history.popleft()
        
        # Check if we're at the limit
        if len(history) >= settings.RATE_LIMIT_REQUESTS:
            self.logger.warning(
                f"Rate limit exceeded for client {client_id}",
                extra={
                    "client_id": client_id,
                    "request_count": len(history),
                    "limit": settings.RATE_LIMIT_REQUESTS,
                    "window": settings.RATE_LIMIT_WINDOW,
                    "endpoint": request.url.path
                }
            )
            
            # Block client for additional time if they're consistently over limit
            if len(history) > settings.RATE_LIMIT_REQUESTS * 1.5:
                self.blocked_until[client_id] = current_time + 300  # 5 minutes
            
            return False
        
        # Add current request to history
        history.append(current_time)
        
        return True
    
    async def _get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for client"""
        
        current_time = time.time()
        window_start = current_time - settings.RATE_LIMIT_WINDOW
        
        history = self.request_history[client_id]
        
        # Count requests in current window
        recent_requests = sum(1 for req_time in history if req_time >= window_start)
        
        return max(0, settings.RATE_LIMIT_REQUESTS - recent_requests)
    
    async def _get_reset_time(self, client_id: str) -> float:
        """Get time when rate limit resets"""
        
        history = self.request_history[client_id]
        
        if not history:
            return time.time()
        
        # Reset time is when the oldest request in the window expires
        return history[0] + settings.RATE_LIMIT_WINDOW
    
    def _create_rate_limit_response(self) -> JSONResponse:
        """Create rate limit exceeded response"""
        
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "code": 429,
                    "message": "Rate limit exceeded",
                    "type": "rate_limit_error",
                    "retry_after": settings.RATE_LIMIT_WINDOW
                }
            },
            headers={
                "Retry-After": str(settings.RATE_LIMIT_WINDOW),
                "X-RateLimit-Limit": str(settings.RATE_LIMIT_REQUESTS),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Window": str(settings.RATE_LIMIT_WINDOW)
            }
        )
    
    async def _cleanup_old_entries(self):
        """Clean up old request history entries"""
        
        current_time = time.time()
        cutoff_time = current_time - settings.RATE_LIMIT_WINDOW * 2  # Keep some buffer
        
        # Clean up request histories
        for client_id, history in list(self.request_history.items()):
            # Remove old entries
            while history and history[0] < cutoff_time:
                history.popleft()
            
            # Remove empty histories
            if not history:
                del self.request_history[client_id]
        
        # Clean up expired blocks
        for client_id, blocked_until in list(self.blocked_until.items()):
            if current_time >= blocked_until:
                del self.blocked_until[client_id]
        
        self.logger.debug(f"Cleaned up rate limit entries: {len(self.request_history)} active clients")
    
    def get_stats(self) -> Dict:
        """Get rate limiting statistics"""
        
        current_time = time.time()
        active_clients = len(self.request_history)
        blocked_clients = len(self.blocked_until)
        
        # Calculate request distribution
        request_counts = {}
        for client_id, history in self.request_history.items():
            recent_count = sum(1 for req_time in history if req_time >= current_time - settings.RATE_LIMIT_WINDOW)
            if recent_count > 0:
                request_counts[client_id] = recent_count
        
        return {
            "active_clients": active_clients,
            "blocked_clients": blocked_clients,
            "rate_limit": settings.RATE_LIMIT_REQUESTS,
            "window_seconds": settings.RATE_LIMIT_WINDOW,
            "request_distribution": request_counts,
            "top_clients": sorted(
                request_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        }
    
    async def reset_client_limit(self, client_id: str):
        """Reset rate limit for a specific client (admin function)"""
        
        if client_id in self.request_history:
            del self.request_history[client_id]
        
        if client_id in self.blocked_until:
            del self.blocked_until[client_id]
        
        self.logger.info(f"Reset rate limit for client: {client_id}")
    
    def __del__(self):
        """Clean up background task"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
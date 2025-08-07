import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.logger import LoggerMixin
from app.core.config import settings

class CostTrackingMiddleware(BaseHTTPMiddleware, LoggerMixin):
    """Middleware to track API costs and usage patterns"""
    
    def __init__(self, app):
        super().__init__(app)
        self.request_counts = {}
        self.cost_accumulator = 0.0
        self.start_time = time.time()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and track costs"""
        
        if not settings.COST_TRACKING_ENABLED:
            return await call_next(request)
        
        start_time = time.time()
        
        # Extract user info if available
        user_id = getattr(request.state, 'user_id', None)
        
        # Process request
        response = await call_next(request)
        
        # Calculate request time
        process_time = time.time() - start_time
        
        # Track request
        await self._track_request(request, response, process_time, user_id)
        
        # Add headers
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Cost-Tracking"] = "enabled"
        
        return response
    
    async def _track_request(
        self, 
        request: Request, 
        response: Response, 
        process_time: float,
        user_id: str = None
    ):
        """Track individual request metrics"""
        
        endpoint = f"{request.method} {request.url.path}"
        
        # Track request count
        self.request_counts[endpoint] = self.request_counts.get(endpoint, 0) + 1
        
        # Extract cost information from response if available
        cost = 0.0
        if hasattr(response, 'headers') and 'X-API-Cost' in response.headers:
            try:
                cost = float(response.headers['X-API-Cost'])
                self.cost_accumulator += cost
            except (ValueError, TypeError):
                pass
        
        # Log request metrics
        self.logger.debug(
            f"Request tracked",
            extra={
                "endpoint": endpoint,
                "user_id": user_id,
                "process_time": process_time,
                "status_code": response.status_code,
                "cost": cost,
                "request_count": self.request_counts[endpoint]
            }
        )
        
        # Check cost limits
        if cost > 0:
            await self._check_cost_limits(user_id, cost)
    
    async def _check_cost_limits(self, user_id: str, cost: float):
        """Check if user is approaching cost limits"""
        
        if self.cost_accumulator > settings.MONTHLY_COST_LIMIT * 0.8:
            self.logger.warning(
                f"Approaching monthly cost limit: ${self.cost_accumulator:.2f} / ${settings.MONTHLY_COST_LIMIT}"
            )
        
        if self.cost_accumulator >= settings.MONTHLY_COST_LIMIT:
            self.logger.error(
                f"Monthly cost limit exceeded: ${self.cost_accumulator:.2f}"
            )
    
    def get_metrics(self) -> dict:
        """Get current cost tracking metrics"""
        
        uptime = time.time() - self.start_time
        total_requests = sum(self.request_counts.values())
        
        return {
            "total_requests": total_requests,
            "total_cost": self.cost_accumulator,
            "avg_cost_per_request": self.cost_accumulator / max(total_requests, 1),
            "uptime_seconds": uptime,
            "requests_per_second": total_requests / max(uptime, 1),
            "endpoint_counts": self.request_counts,
            "cost_limit": settings.MONTHLY_COST_LIMIT,
            "cost_utilization": (self.cost_accumulator / settings.MONTHLY_COST_LIMIT) * 100
        }
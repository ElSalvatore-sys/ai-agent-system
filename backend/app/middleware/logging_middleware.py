import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

from app.core.logger import SensitiveDataFilter

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured request/response logging"""
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = structlog.get_logger("request")
        self.sensitive_filter = SensitiveDataFilter()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response details"""
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Log request
        await self._log_request(request, request_id)
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            await self._log_response(request, response, request_id, process_time)
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            await self._log_error(request, e, request_id, process_time)
            raise
    
    async def _log_request(self, request: Request, request_id: str):
        """Log incoming request details"""
        
        # Extract client info
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Get user info if available
        user_id = getattr(request.state, 'user_id', None)
        
        # Prepare log data
        log_data = {
            "event": "request_started",
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": client_ip,
            "user_agent": user_agent,
            "user_id": user_id,
            "headers": dict(request.headers)
        }
        
        # Filter sensitive data
        log_data = self.sensitive_filter._redact_sensitive_data(log_data)
        
        self.logger.info("Request started", **log_data)
    
    async def _log_response(
        self, 
        request: Request, 
        response: Response, 
        request_id: str, 
        process_time: float
    ):
        """Log response details"""
        
        log_data = {
            "event": "request_completed",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "process_time": round(process_time, 4),
            "response_headers": dict(response.headers) if hasattr(response, 'headers') else {}
        }
        
        # Add user context if available
        if hasattr(request.state, 'user_id'):
            log_data["user_id"] = request.state.user_id
        
        # Determine log level based on status code
        if response.status_code >= 500:
            self.logger.error("Request completed with server error", **log_data)
        elif response.status_code >= 400:
            self.logger.warning("Request completed with client error", **log_data)
        else:
            self.logger.info("Request completed successfully", **log_data)
    
    async def _log_error(
        self, 
        request: Request, 
        error: Exception, 
        request_id: str, 
        process_time: float
    ):
        """Log request error"""
        
        log_data = {
            "event": "request_error",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "process_time": round(process_time, 4)
        }
        
        # Add user context if available
        if hasattr(request.state, 'user_id'):
            log_data["user_id"] = request.state.user_id
        
        self.logger.error("Request failed with exception", **log_data, exc_info=True)
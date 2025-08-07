import logging
import sys
from pathlib import Path
from typing import Dict, Any
import structlog
from loguru import logger as loguru_logger

def setup_logging(level: str = "INFO") -> None:
    """Setup structured logging with both standard library and loguru"""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.dev.ConsoleRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Configure loguru for file logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Remove default handler
    loguru_logger.remove()
    
    # Add console handler
    loguru_logger.add(
        sys.stdout,
        level=level.upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Add file handlers
    loguru_logger.add(
        log_dir / "app.log",
        level="INFO",
        rotation="100 MB",
        retention="30 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )
    
    loguru_logger.add(
        log_dir / "error.log",
        level="ERROR",
        rotation="100 MB",
        retention="30 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )

class LoggerMixin:
    """Mixin to add structured logging to classes"""
    
    @property
    def logger(self):
        return structlog.get_logger(self.__class__.__name__)

def get_logger(name: str = None) -> structlog.BoundLogger:
    """Get a structured logger instance"""
    return structlog.get_logger(name or __name__)

# Custom log filter for sensitive data
class SensitiveDataFilter(logging.Filter):
    """Filter to redact sensitive information from logs"""
    
    sensitive_fields = {
        'password', 'token', 'key', 'secret', 'authorization',
        'api_key', 'access_token', 'refresh_token'
    }
    
    def filter(self, record):
        if hasattr(record, 'msg') and isinstance(record.msg, (dict, str)):
            record.msg = self._redact_sensitive_data(record.msg)
        return True
    
    def _redact_sensitive_data(self, data):
        if isinstance(data, dict):
            return {
                k: '[REDACTED]' if k.lower() in self.sensitive_fields else v
                for k, v in data.items()
            }
        elif isinstance(data, str):
            # Simple string redaction for common patterns
            import re
            patterns = [
                (r'password["\s]*[:=]["\s]*([^"\s,}]+)', r'password": "[REDACTED]"'),
                (r'token["\s]*[:=]["\s]*([^"\s,}]+)', r'token": "[REDACTED]"'),
                (r'key["\s]*[:=]["\s]*([^"\s,}]+)', r'key": "[REDACTED]"'),
            ]
            for pattern, replacement in patterns:
                data = re.sub(pattern, replacement, data, flags=re.IGNORECASE)
        return data
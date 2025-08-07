import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import create_async_engine
import os
from dotenv import load_dotenv

from app.core.config import settings
from app.database.database import init_db
from app.middleware.auth import AuthMiddleware
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.cost_tracking import CostTrackingMiddleware
from app.middleware.budget_manager import BudgetManagerMiddleware
from app.middleware.logging_middleware import LoggingMiddleware
from app.middleware.audit_logging import AuditLoggingMiddleware
from app.utils.cache import CacheManager
from app.routes import chat, admin, auth, agents, tasks, system, websocket_chat, analytics, generate, edge_agent, local_llm
from app.core.logger import setup_logging

load_dotenv()

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global cache manager
cache_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    global cache_manager
    
    logger.info("Starting AI Agent System API...")
    
    try:
        # Initialize database
        await init_db()
        logger.info("Database initialized successfully")
        
        # Initialize Redis cache
        cache_manager = CacheManager(settings.REDIS_URL)
        await cache_manager.connect()
        app.state.cache = cache_manager
        logger.info("Redis cache connected successfully")
        
        # Initialize AI models (async loading)
        from app.models.ai_orchestrator import AIOrchestrator
        orchestrator = AIOrchestrator()
        await orchestrator.initialize()
        app.state.ai_orchestrator = orchestrator
        logger.info("AI models initialized successfully")
        
        # Initialize Advanced AI Orchestrator
        from app.services.advanced_orchestrator_service import AdvancedOrchestratorService
        await AdvancedOrchestratorService.get_instance()
        logger.info("Advanced AI Orchestrator initialized successfully")

        # Initialize Self-Improvement Engine (optional – can be toggled via ENV)
        try:
            from app.services.self_improvement_engine import SelfImprovementEngine
            self_improvement_engine = SelfImprovementEngine.get_instance()
            await self_improvement_engine.start()
            app.state.self_improvement_engine = self_improvement_engine
            logger.info("Self-Improvement Engine started successfully")
        except Exception as sie_err:
            logger.error(f"Failed to start Self-Improvement Engine: {sie_err}")
        
        # Initialize local LLM management services
        try:
            from app.services.model_discovery import get_lifecycle_manager
            from app.services.container_orchestrator import get_container_orchestrator
            from app.services.redis_cache import get_cache_manager
            from app.services.api_gateway import get_api_gateway
            
            # Initialize lifecycle manager
            lifecycle_manager = await get_lifecycle_manager()
            app.state.lifecycle_manager = lifecycle_manager
            logger.info("Model lifecycle manager initialized successfully")
            
            # Initialize container orchestrator
            container_orchestrator = await get_container_orchestrator()
            app.state.container_orchestrator = container_orchestrator
            logger.info("Container orchestrator initialized successfully")
            
            # Initialize Redis cache manager
            redis_cache_manager = await get_cache_manager()
            app.state.redis_cache_manager = redis_cache_manager
            logger.info("Redis cache manager initialized successfully")
            
            # Initialize API gateway
            api_gateway = await get_api_gateway()
            app.state.api_gateway = api_gateway
            logger.info("API gateway initialized successfully")
            
        except Exception as llm_err:
            logger.error(f"Failed to initialize local LLM services: {llm_err}")
        
        # Start periodic local model discovery
        try:
            from app.services.model_discovery import schedule_periodic_discovery
            app.state.discovery_task = asyncio.create_task(schedule_periodic_discovery())
            logger.info("Local model discovery scheduler started")
        except Exception as disc_err:
            logger.error(f"Failed to start model discovery: {disc_err}")
        
        # Initialize WebSocket manager
        from app.services.websocket_manager import websocket_manager
        await websocket_manager.initialize()
        app.state.websocket_manager = websocket_manager
        logger.info("WebSocket manager initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down AI Agent System API...")
    
    # Cleanup local LLM services
    try:
        if hasattr(app.state, 'lifecycle_manager') and app.state.lifecycle_manager:
            await app.state.lifecycle_manager.shutdown()
            logger.info("Model lifecycle manager shutdown")
    except Exception as e:
        logger.error(f"Failed to shutdown lifecycle manager: {e}")
    
    try:
        if hasattr(app.state, 'container_orchestrator') and app.state.container_orchestrator:
            await app.state.container_orchestrator.shutdown()
            logger.info("Container orchestrator shutdown")
    except Exception as e:
        logger.error(f"Failed to shutdown container orchestrator: {e}")
    
    try:
        if hasattr(app.state, 'redis_cache_manager') and app.state.redis_cache_manager:
            await app.state.redis_cache_manager.shutdown()
            logger.info("Redis cache manager shutdown")
    except Exception as e:
        logger.error(f"Failed to shutdown Redis cache manager: {e}")
    
    try:
        if hasattr(app.state, 'api_gateway') and app.state.api_gateway:
            await app.state.api_gateway.shutdown()
            logger.info("API gateway shutdown")
    except Exception as e:
        logger.error(f"Failed to shutdown API gateway: {e}")
    
    # Cleanup Advanced AI Orchestrator
    try:
        from app.services.advanced_orchestrator_service import AdvancedOrchestratorService
        await AdvancedOrchestratorService.cleanup()
        logger.info("Advanced AI Orchestrator cleaned up")
    except Exception as e:
        logger.error(f"Failed to cleanup Advanced AI Orchestrator: {e}")
    
    # Cleanup Self-Improvement Engine
    try:
        if hasattr(app.state, 'self_improvement_engine') and app.state.self_improvement_engine:
            await app.state.self_improvement_engine.stop()
            logger.info("Self-Improvement Engine stopped")
    except Exception as e:
        logger.error(f"Failed to stop Self-Improvement Engine: {e}")

    # Stop discovery task
    try:
        if hasattr(app.state, 'discovery_task'):
            app.state.discovery_task.cancel()
            logger.info("Local model discovery task cancelled")
    except Exception as e:
        logger.error(f"Failed to cancel model discovery task: {e}")
    
    # Cleanup WebSocket manager
    try:
        from app.services.websocket_manager import websocket_manager
        await websocket_manager.cleanup()
        logger.info("WebSocket manager cleaned up")
    except Exception as e:
        logger.error(f"Failed to cleanup WebSocket manager: {e}")
    
    if cache_manager:
        await cache_manager.close()
        logger.info("Redis cache disconnected")

app = FastAPI(
    title="AI Agent System API",
    description="A comprehensive AI agent system with multi-model support, cost optimization, and real-time chat",
    version="1.0.0",
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None,
    lifespan=lifespan,
    openapi_url="/openapi.json" if settings.ENVIRONMENT == "development" else None,
)

# Security middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware
app.add_middleware(AuditLoggingMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(CostTrackingMiddleware)
app.add_middleware(BudgetManagerMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuthMiddleware)

# Exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "type": "http_error"
            }
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": 422,
                "message": "Validation error",
                "type": "validation_error",
                "details": exc.errors()
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "type": "server_error"
            }
        }
    )

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["authentication"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(websocket_chat.router, prefix="/api/v1/ws", tags=["websocket"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])
app.include_router(generate.router, prefix="/api/v1/generate", tags=["generation"])
app.include_router(agents.router, prefix="/api/v1/agents", tags=["agents"])
app.include_router(tasks.router, prefix="/api/v1/tasks", tags=["tasks"])
app.include_router(system.router, prefix="/api/v1/system", tags=["system"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])
app.include_router(edge_agent.router, prefix="/api/v1", tags=["edge"])
app.include_router(local_llm.router, prefix="/api/v1", tags=["local-llm"])

# Include integration management router
from app.api.endpoints.integrations import router as integrations_router
app.include_router(integrations_router, prefix="/api/v1/integrations", tags=["integrations"])

# Health check endpoints
@app.get("/", tags=["health"])
async def root():
    return {
        "message": "AI Agent System API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs" if settings.ENVIRONMENT == "development" else "disabled"
    }

@app.get("/health", tags=["health"])
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": asyncio.get_event_loop().time(),
        "services": {}
    }
    
    # Check database
    try:
        from app.database.database import get_db
        async for db in get_db():
            await db.execute("SELECT 1")
            health_status["services"]["database"] = {"status": "up"}
            break
    except Exception as e:
        health_status["services"]["database"] = {"status": "down", "error": str(e)}
        health_status["status"] = "unhealthy"
    
    # Check Redis
    try:
        if hasattr(app.state, 'cache') and app.state.cache:
            await app.state.cache.ping()
            health_status["services"]["redis"] = {"status": "up"}
        else:
            health_status["services"]["redis"] = {"status": "down", "error": "Not initialized"}
    except Exception as e:
        health_status["services"]["redis"] = {"status": "down", "error": str(e)}
        health_status["status"] = "unhealthy"
    
    # Check AI models
    try:
        if hasattr(app.state, 'ai_orchestrator') and app.state.ai_orchestrator:
            model_status = await app.state.ai_orchestrator.health_check()
            health_status["services"]["ai_models"] = model_status
        else:
            health_status["services"]["ai_models"] = {"status": "down", "error": "Not initialized"}
    except Exception as e:
        health_status["services"]["ai_models"] = {"status": "down", "error": str(e)}
        health_status["status"] = "unhealthy"
    
    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True if settings.ENVIRONMENT == "development" else False,
        log_level="info"
    )
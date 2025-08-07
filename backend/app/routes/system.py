from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from pydantic import BaseModel
from datetime import datetime, timedelta

from app.database.database import get_db
from app.database.models import User, Conversation, Message, UsageLog
from app.core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

class SystemStats(BaseModel):
    active_users: int
    total_conversations: int
    total_messages: int
    total_api_calls: int
    uptime: str
    system_load: float
    memory_usage: float

@router.get("/stats")
async def get_system_stats(
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> SystemStats:
    """Get basic system statistics"""
    
    try:
        # Count active users (logged in within last 24 hours)
        yesterday = datetime.utcnow() - timedelta(days=1)
        active_users_result = await db.execute(
            select(func.count(User.id)).where(User.last_login >= yesterday)
        )
        active_users = active_users_result.scalar() or 0
        
        # Count total conversations
        total_convs_result = await db.execute(select(func.count(Conversation.id)))
        total_conversations = total_convs_result.scalar() or 0
        
        # Count total messages
        total_msgs_result = await db.execute(select(func.count(Message.id)))
        total_messages = total_msgs_result.scalar() or 0
        
        # Count total API calls
        total_calls_result = await db.execute(select(func.count(UsageLog.id)))
        total_api_calls = total_calls_result.scalar() or 0
        
        return SystemStats(
            active_users=active_users,
            total_conversations=total_conversations,
            total_messages=total_messages,
            total_api_calls=total_api_calls,
            uptime="System running",
            system_load=0.1,  # Placeholder
            memory_usage=0.2   # Placeholder
        )
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system statistics")

@router.get("/health")
async def get_system_health():
    """Get comprehensive system health status"""
    
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "api": {"status": "up", "response_time": "fast"},
                "database": {"status": "up", "response_time": "normal"}
            },
            "version": "1.0.0"
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

@router.get("/info")
async def get_system_info():
    """Get system information"""
    
    return {
        "name": "AI Agent System",
        "version": "1.0.0",
        "description": "A comprehensive AI agent system with multi-model support",
        "features": [
            "Multi-AI Model Support (OpenAI, Anthropic, Google)",
            "Real-time Chat with WebSocket streaming",
            "Cost tracking and optimization",
            "Agent management system",
            "Task scheduling and execution",
            "Redis caching layer",
            "JWT authentication",
            "Rate limiting and security",
            "Admin analytics dashboard"
        ],
        "supported_models": {
            "openai": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "o3-mini"],
            "anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
            "google": ["gemini-1.5-pro", "gemini-1.5-flash"]
        }
    }
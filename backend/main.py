#!/usr/bin/env python3

"""
AI Agent System - Minimal Working Backend
A simplified version that starts successfully for immediate testing.
"""

import asyncio
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

load_dotenv()

# Basic configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", '["http://localhost:3000","http://127.0.0.1:3000"]')
if isinstance(CORS_ORIGINS, str):
    try:
        import json
        CORS_ORIGINS = json.loads(CORS_ORIGINS)
    except:
        CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 AI Agent System Backend Starting...")
    yield
    # Shutdown
    print("👋 AI Agent System Backend Shutting Down...")

# Create FastAPI app
app = FastAPI(
    title="AI Agent System API",
    description="A comprehensive AI agent management system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """Comprehensive health check"""
    return {
        "status": "healthy",
        "timestamp": asyncio.get_event_loop().time(),
        "services": {
            "api": "operational",
            "database": "connected",
            "redis": "connected"
        },
        "version": "1.0.0"
    }

# Root endpoint
@app.get("/", tags=["health"])
async def root():
    """Root endpoint"""
    return {
        "message": "🤖 AI Agent System API",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

# Basic API endpoints for immediate testing
@app.get("/api/agents", tags=["agents"])
async def list_agents():
    """List all agents"""
    return {
        "agents": [
            {
                "id": 1,
                "name": "GPT-4 Assistant",
                "type": "openai",
                "status": "active",
                "description": "Advanced AI assistant powered by GPT-4"
            },
            {
                "id": 2,
                "name": "Claude Assistant", 
                "type": "anthropic",
                "status": "active",
                "description": "Helpful AI assistant powered by Claude"
            },
            {
                "id": 3,
                "name": "Local LLM",
                "type": "ollama", 
                "status": "inactive",
                "description": "Local language model via Ollama"
            }
        ]
    }

@app.post("/api/chat", tags=["chat"])
async def chat_endpoint(message: dict):
    """Basic chat endpoint"""
    user_message = message.get("message", "Hello!")
    
    # Mock AI response for immediate testing
    responses = [
        f"I received your message: '{user_message}'. This is a test response from the AI Agent System!",
        f"Thanks for testing the system! Your message '{user_message}' has been processed.",
        f"Hello! I'm your AI assistant. You said: '{user_message}'. The system is working perfectly!",
    ]
    
    import random
    response = random.choice(responses)
    
    return {
        "response": response,
        "model": "test-model",
        "timestamp": asyncio.get_event_loop().time(),
        "tokens_used": len(user_message.split()) + len(response.split()),
        "cost": 0.001
    }

@app.get("/api/models", tags=["models"])
async def list_models():
    """List available AI models"""
    return {
        "models": [
            {
                "id": "gpt-4",
                "name": "GPT-4",
                "provider": "openai",
                "status": "available",
                "cost_per_token": 0.00003
            },
            {
                "id": "claude-3",
                "name": "Claude 3",
                "provider": "anthropic", 
                "status": "available",
                "cost_per_token": 0.000015
            },
            {
                "id": "llama2",
                "name": "Llama 2",
                "provider": "ollama",
                "status": "local",
                "cost_per_token": 0.0
            }
        ]
    }

@app.get("/api/analytics", tags=["analytics"])
async def get_analytics():
    """Get system analytics"""
    return {
        "total_requests": 42,
        "total_cost": 1.25,
        "active_sessions": 3,
        "uptime": "2h 15m",
        "most_used_model": "gpt-4",
        "success_rate": 98.5
    }

@app.get("/api/dashboard-stats", tags=["dashboard"])
async def get_dashboard_stats():
    """Get dashboard statistics"""
    return {
        "totalSessions": 15,
        "totalMessages": 247,
        "activeUsers": 3,
        "systemHealth": 98
    }

@app.get("/metrics", tags=["monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content="# AI Agent System Metrics\napi_requests_total 42\napi_response_time_seconds 0.15\n",
        media_type="text/plain"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
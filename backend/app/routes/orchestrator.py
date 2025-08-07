from fastapi import APIRouter, Request, Depends, HTTPException, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

from app.database.database import get_db
from app.services.advanced_orchestrator_service import (
    ChatOrchestratorService,
    BatchOrchestratorService,
    get_chat_orchestrator,
    get_batch_orchestrator,
    get_advanced_orchestrator
)
from app.models.advanced_ai_orchestrator import (
    RoutingStrategy,
    TaskComplexity,
    RoutingDecision
)


router = APIRouter()


# Pydantic models for API requests/responses
class ChatRouteRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    budget_limit: Optional[float] = Field(None, ge=0.0, le=100.0)
    strategy: RoutingStrategy = RoutingStrategy.BALANCED
    conversation_context: Optional[Dict[str, Any]] = None
    stream: bool = False


class ChatExecuteRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    budget_limit: Optional[float] = Field(None, ge=0.0, le=100.0)
    strategy: RoutingStrategy = RoutingStrategy.BALANCED
    conversation_context: Optional[Dict[str, Any]] = None
    system_prompt: Optional[str] = Field(None, max_length=5000)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(2048, ge=1, le=8192)


class BatchRequest(BaseModel):
    requests: List[Dict[str, Any]] = Field(..., min_items=1, max_items=100)
    priority: int = Field(0, ge=0, le=10)


class RoutingDecisionResponse(BaseModel):
    chosen_provider: str
    chosen_model: str
    reasoning: str
    estimated_cost: float
    estimated_time: float
    fallback_models: List[List[str]]
    confidence: float


class ChatExecuteResponse(BaseModel):
    content: str
    model_used: str
    execution_time: float
    attempts: int
    success: bool
    routing_decision: Dict[str, Any]
    cost: Optional[float] = None
    tokens_used: Optional[int] = None


class UserAnalyticsResponse(BaseModel):
    user_id: int
    metrics: Dict[str, Any]
    recommendations: List[Dict[str, str]]


class ModelPerformanceResponse(BaseModel):
    summary: Dict[str, Any]
    models: Dict[str, Any]
    recommendations: List[Dict[str, str]]


class BatchStatusResponse(BaseModel):
    batch_id: str
    status: str
    total_requests: Optional[int] = None
    successful: Optional[int] = None
    failed: Optional[int] = None
    created_at: Optional[str] = None
    priority: Optional[int] = None
    results: Optional[List[Any]] = None


@router.post("/chat/route", response_model=RoutingDecisionResponse)
async def route_chat_message(
    request_data: ChatRouteRequest,
    http_request: Request,
    chat_orchestrator: ChatOrchestratorService = Depends(get_chat_orchestrator)
):
    """
    Route a chat message to the optimal AI model without executing it
    
    This endpoint analyzes the message and returns the routing decision,
    allowing you to see which model would be selected and why.
    """
    user_id = getattr(http_request.state, 'user_id', 1)  # Default to user 1 if not authenticated
    
    try:
        decision = await chat_orchestrator.route_chat_message(
            message=request_data.message,
            user_id=user_id,
            conversation_context=request_data.conversation_context,
            budget_limit=request_data.budget_limit,
            strategy=request_data.strategy,
            stream=request_data.stream
        )
        
        return RoutingDecisionResponse(
            chosen_provider=decision.chosen_provider.value,
            chosen_model=decision.chosen_model,
            reasoning=decision.reasoning,
            estimated_cost=decision.estimated_cost,
            estimated_time=decision.estimated_time,
            fallback_models=[[m[0].value, m[1]] for m in decision.fallback_models],
            confidence=decision.confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to route message: {str(e)}")


@router.post("/chat/execute", response_model=ChatExecuteResponse)
async def execute_chat_request(
    request_data: ChatExecuteRequest,
    http_request: Request,
    chat_orchestrator: ChatOrchestratorService = Depends(get_chat_orchestrator)
):
    """
    Execute a chat request using the advanced orchestrator
    
    This endpoint routes the message to the optimal model and executes it,
    with automatic fallback handling if the primary model fails.
    """
    user_id = getattr(http_request.state, 'user_id', 1)
    
    try:
        result = await chat_orchestrator.execute_chat_request(
            message=request_data.message,
            user_id=user_id,
            conversation_context=request_data.conversation_context,
            budget_limit=request_data.budget_limit,
            strategy=request_data.strategy,
            system_prompt=request_data.system_prompt,
            temperature=request_data.temperature,
            max_tokens=request_data.max_tokens
        )
        
        response_obj = result["response"]
        
        return ChatExecuteResponse(
            content=response_obj.content,
            model_used=result["model_used"],
            execution_time=result["execution_time"],
            attempts=result["attempts"],
            success=result["success"],
            routing_decision=result["routing_decision"],
            cost=response_obj.cost,
            tokens_used=response_obj.tokens_used
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute chat request: {str(e)}")


@router.post("/chat/stream")
async def stream_chat_response(
    request_data: ChatExecuteRequest,
    http_request: Request,
    chat_orchestrator: ChatOrchestratorService = Depends(get_chat_orchestrator)
):
    """
    Stream a chat response using the advanced orchestrator
    
    This endpoint provides real-time streaming of the AI response,
    with intelligent model selection and fallback handling.
    """
    user_id = getattr(http_request.state, 'user_id', 1)
    
    from fastapi.responses import StreamingResponse
    import json
    
    async def generate_stream():
        try:
            async for chunk in chat_orchestrator.stream_chat_response(
                message=request_data.message,
                user_id=user_id,
                conversation_context=request_data.conversation_context,
                budget_limit=request_data.budget_limit,
                strategy=request_data.strategy,
                system_prompt=request_data.system_prompt,
                temperature=request_data.temperature,
                max_tokens=request_data.max_tokens
            ):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate_stream(), media_type="text/plain")


@router.get("/analytics/user", response_model=UserAnalyticsResponse)
async def get_user_analytics(
    http_request: Request,
    chat_orchestrator: ChatOrchestratorService = Depends(get_chat_orchestrator)
):
    """
    Get analytics and recommendations for the current user
    
    Returns usage patterns, preferences, and personalized recommendations
    for optimizing AI model usage.
    """
    user_id = getattr(http_request.state, 'user_id', 1)
    
    try:
        analytics = await chat_orchestrator.analyze_user_patterns(user_id)
        
        return UserAnalyticsResponse(
            user_id=analytics["user_id"],
            metrics=analytics["metrics"],
            recommendations=analytics["recommendations"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user analytics: {str(e)}")


@router.get("/analytics/models", response_model=ModelPerformanceResponse)
async def get_model_performance(
    chat_orchestrator: ChatOrchestratorService = Depends(get_chat_orchestrator)
):
    """
    Get comprehensive model performance analytics
    
    Returns performance metrics, cost analysis, and recommendations
    for all AI models in the system.
    """
    try:
        report = await chat_orchestrator.get_model_performance_report()
        
        return ModelPerformanceResponse(
            summary=report["summary"],
            models=report["models"],
            recommendations=report["recommendations"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model performance: {str(e)}")


@router.post("/batch/submit")
async def submit_batch_requests(
    request_data: BatchRequest,
    http_request: Request,
    batch_orchestrator: BatchOrchestratorService = Depends(get_batch_orchestrator)
):
    """
    Submit a batch of requests for processing
    
    Processes multiple AI requests efficiently in parallel,
    with intelligent model routing for each request.
    """
    user_id = getattr(http_request.state, 'user_id', 1)
    
    try:
        batch_id = await batch_orchestrator.submit_batch(
            requests=request_data.requests,
            user_id=user_id,
            priority=request_data.priority
        )
        
        return {"batch_id": batch_id, "status": "submitted"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit batch: {str(e)}")


@router.get("/batch/{batch_id}/status", response_model=BatchStatusResponse)
async def get_batch_status(
    batch_id: str,
    batch_orchestrator: BatchOrchestratorService = Depends(get_batch_orchestrator)
):
    """
    Get the status of a batch processing job
    
    Returns the current status, progress, and results (if completed)
    of a previously submitted batch job.
    """
    try:
        status = await batch_orchestrator.get_batch_status(batch_id)
        
        return BatchStatusResponse(**status)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get batch status: {str(e)}")


@router.get("/health")
async def orchestrator_health_check(
    orchestrator = Depends(get_advanced_orchestrator)
):
    """
    Check the health of the advanced orchestrator
    
    Returns the status of all AI model clients, cache connections,
    and overall system health.
    """
    try:
        health = await orchestrator.health_check()
        
        if health["status"] == "unhealthy":
            raise HTTPException(status_code=503, detail=health)
        
        return health
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/analytics/system")
async def get_system_analytics(
    orchestrator = Depends(get_advanced_orchestrator)
):
    """
    Get comprehensive system analytics
    
    Returns overall usage statistics, model performance,
    cost analytics, and system metrics.
    """
    try:
        analytics = await orchestrator.get_analytics()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "analytics": analytics
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system analytics: {str(e)}")


@router.get("/models/capabilities")
async def get_model_capabilities(
    orchestrator = Depends(get_advanced_orchestrator)
):
    """
    Get information about available AI models and their capabilities
    
    Returns detailed information about each model including costs,
    capabilities, performance characteristics, and availability.
    """
    try:
        capabilities = {}
        
        for provider_name, provider_models in orchestrator.model_capabilities.items():
            capabilities[provider_name] = {}
            
            for model_name, model_caps in provider_models.items():
                capabilities[provider_name][model_name] = {
                    "provider": model_caps.provider.value,
                    "model_name": model_caps.model_name,
                    "max_tokens": model_caps.max_tokens,
                    "context_window": model_caps.context_window,
                    "cost_per_1k_input": model_caps.cost_per_1k_input,
                    "cost_per_1k_output": model_caps.cost_per_1k_output,
                    "avg_response_time": model_caps.avg_response_time,
                    "success_rate": model_caps.success_rate,
                    "quality_score": model_caps.quality_score,
                    "supports_code": model_caps.supports_code,
                    "supports_reasoning": model_caps.supports_reasoning,
                    "supports_multimodal": model_caps.supports_multimodal,
                    "supports_streaming": model_caps.supports_streaming,
                    "is_local": model_caps.is_local,
                    "available": provider_name in orchestrator.model_clients
                }
        
        return {
            "capabilities": capabilities,
            "available_providers": list(orchestrator.model_clients.keys()),
            "total_models": sum(len(models) for models in capabilities.values())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model capabilities: {str(e)}")


@router.post("/preferences/update")
async def update_user_preferences(
    preferences: Dict[str, Any],
    http_request: Request,
    orchestrator = Depends(get_advanced_orchestrator)
):
    """
    Update user preferences for model selection
    
    Allows users to customize their preferences for cost sensitivity,
    speed vs quality trade-offs, and preferred models for different task types.
    """
    user_id = getattr(http_request.state, 'user_id', 1)
    
    try:
        # Get current preferences
        current_prefs = await orchestrator._get_user_preferences(user_id)
        
        # Update allowed fields
        updatable_fields = [
            'cost_sensitivity', 'speed_preference', 'quality_preference',
            'monthly_budget', 'preferred_models'
        ]
        
        for field in updatable_fields:
            if field in preferences:
                setattr(current_prefs, field, preferences[field])
        
        # Cache updated preferences
        from dataclasses import asdict
        cache_key = f"user_prefs:{user_id}"
        await orchestrator.cache.set(cache_key, asdict(current_prefs), ttl=3600)
        
        return {"message": "Preferences updated successfully", "preferences": asdict(current_prefs)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update preferences: {str(e)}")


# Demo/testing endpoints
@router.post("/demo/complexity-analysis")
async def analyze_task_complexity(
    task: str = Field(..., min_length=1, max_length=10000),
    orchestrator = Depends(get_advanced_orchestrator)
):
    """
    Demo endpoint to analyze task complexity
    
    Analyzes a given task and returns the complexity assessment,
    token estimation, and required capabilities.
    """
    try:
        analysis = await orchestrator._analyze_task_complexity(task)
        
        return {
            "task": task,
            "analysis": {
                "complexity": analysis.complexity.value,
                "estimated_tokens": analysis.estimated_tokens,
                "requires_code": analysis.requires_code,
                "requires_reasoning": analysis.requires_reasoning,
                "requires_multimodal": analysis.requires_multimodal,
                "confidence_score": analysis.confidence_score
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze task complexity: {str(e)}")


@router.get("/demo/routing-strategies")
async def get_routing_strategies():
    """
    Get information about available routing strategies
    
    Returns descriptions of all available routing strategies
    and when to use each one.
    """
    strategies = {
        RoutingStrategy.COST_OPTIMAL: {
            "name": "Cost Optimal",
            "description": "Selects the cheapest suitable model for the task",
            "use_cases": ["Budget-conscious usage", "Simple tasks", "High-volume processing"]
        },
        RoutingStrategy.PERFORMANCE_OPTIMAL: {
            "name": "Performance Optimal", 
            "description": "Selects the fastest model for the task",
            "use_cases": ["Real-time applications", "User-facing chat", "Time-sensitive tasks"]
        },
        RoutingStrategy.QUALITY_OPTIMAL: {
            "name": "Quality Optimal",
            "description": "Selects the highest quality model regardless of cost",
            "use_cases": ["Critical tasks", "Research", "Creative work"]
        },
        RoutingStrategy.BALANCED: {
            "name": "Balanced",
            "description": "Balances cost, speed, and quality based on user preferences",
            "use_cases": ["General usage", "Mixed workloads", "Default choice"]
        },
        RoutingStrategy.USER_PREFERENCE: {
            "name": "User Preference",
            "description": "Uses models based on user's historical preferences",
            "use_cases": ["Personalized experience", "Learned behavior", "Consistency"]
        }
    }
    
    return {
        "strategies": {strategy.value: info for strategy, info in strategies.items()},
        "default": RoutingStrategy.BALANCED.value
    }
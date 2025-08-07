"""FastAPI Routes for Local LLM Management.

Provides REST API endpoints for model lifecycle management, container orchestration,
background task management, and monitoring capabilities.
"""
from __future__ import annotations

from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field

from app.middleware.auth import get_current_user, RequireRole
from app.database.models import User, ModelProvider
from app.services.model_discovery import get_lifecycle_manager
from app.services.container_orchestrator import get_container_orchestrator
from app.services.celery_app import get_task_manager
from app.services.redis_cache import get_cache_manager
from app.services.api_gateway import get_api_gateway, GatewayRequest
from app.core.logger import get_logger
from app.security.model_verifier import verify_model_signature


logger = get_logger(__name__)
router = APIRouter(prefix="/local-llm", tags=["Local LLM Management"])


# Request/Response Models
class ModelPullRequest(BaseModel):
    provider: ModelProvider
    model_name: str
    priority: int = Field(5, ge=1, le=10)


class ModelLoadRequest(BaseModel):
    provider: ModelProvider
    model_name: str
    resource_requirements: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(5, ge=1, le=10)


class ModelScaleRequest(BaseModel):
    provider: ModelProvider
    model_name: str
    desired_instances: int = Field(ge=0, le=10)
    priority: int = Field(4, ge=1, le=10)


class ContainerCreateRequest(BaseModel):
    provider: ModelProvider
    model_name: str
    resource_requirements: Optional[Dict[str, Any]] = None


class UnifiedModelRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)
    system_prompt: Optional[str] = Field(None, max_length=5000)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(2048, ge=1, le=8192)
    stream: bool = False
    model_preference: Optional[str] = None
    provider_preference: Optional[ModelProvider] = None
    use_cache: bool = True
    timeout: float = Field(30.0, ge=1.0, le=300.0)
    metadata: Optional[Dict[str, Any]] = None


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Any] = None
    traceback: Optional[str] = None
    info: Optional[Any] = None


class ModelStatusResponse(BaseModel):
    name: str
    provider: str
    status: str
    health_score: float
    memory_usage_mb: float
    gpu_usage_percent: float
    last_request_time: Optional[datetime]
    error_count: int
    response_time_avg: float
    container_id: Optional[str] = None


class ContainerStatusResponse(BaseModel):
    container_id: str
    name: str
    model_name: str
    provider: str
    status: str
    health_status: str
    created_at: datetime
    started_at: Optional[datetime]
    cpu_usage: float
    memory_usage_mb: float
    gpu_usage: float
    network_io: Dict[str, int]
    disk_io: Dict[str, int]
    restart_count: int
    last_error: Optional[str]


class ResourceMetricsResponse(BaseModel):
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    gpu_count: int
    gpu_memory_total_mb: float
    gpu_memory_used_mb: float
    disk_usage_percent: float


class CacheStatsResponse(BaseModel):
    total_keys: int
    total_memory_mb: float
    hit_rate: float
    miss_rate: float
    evictions: int
    expired_keys: int
    connections: int


class GatewayStatsResponse(BaseModel):
    total_requests: int
    successful_requests: int
    failed_requests: int
    cached_responses: int
    avg_response_time: float
    total_tokens: int
    total_cost: float
    active_endpoints: int
    cache_hit_rate: float
    error_rate: float


# Model Management Endpoints
@router.post("/models/pull")
@RequireRole("admin")
async def pull_model(
    request: ModelPullRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """Pull/download a model"""
    try:
        task_manager = get_task_manager()
        task_id = await task_manager.pull_model_async(
            request.provider,
            request.model_name,
            request.priority
        )
        
        return {
            "task_id": task_id,
            "message": f"Model pull initiated for {request.provider.value}:{request.model_name}"
        }
        
    except Exception as e:
        logger.error(f"Model pull request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/load")
@RequireRole("admin")
async def load_model(
    request: ModelLoadRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """Load a model into memory/container"""
    try:
        task_manager = get_task_manager()
        task_id = await task_manager.load_model_async(
            request.provider,
            request.model_name,
            request.resource_requirements,
            request.priority
        )
        
        return {
            "task_id": task_id,
            "message": f"Model load initiated for {request.provider.value}:{request.model_name}"
        }
        
    except Exception as e:
        logger.error(f"Model load request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/unload")
@RequireRole("admin")
async def unload_model(
    provider: ModelProvider,
    model_name: str,
    priority: int = Query(3, ge=1, le=10),
    current_user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """Unload a model from memory/container"""
    try:
        task_manager = get_task_manager()
        task_id = await task_manager.unload_model_async(
            provider,
            model_name,
            priority
        )
        
        return {
            "task_id": task_id,
            "message": f"Model unload initiated for {provider.value}:{model_name}"
        }
        
    except Exception as e:
        logger.error(f"Model unload request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/scale")
@RequireRole("admin")
async def scale_model(
    request: ModelScaleRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """Scale a model to desired number of instances"""
    try:
        task_manager = get_task_manager()
        task_id = await task_manager.scale_model_async(
            request.provider,
            request.model_name,
            request.desired_instances,
            request.priority
        )
        
        return {
            "task_id": task_id,
            "message": f"Model scaling initiated for {request.provider.value}:{request.model_name}"
        }
        
    except Exception as e:
        logger.error(f"Model scaling request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/status", response_model=List[ModelStatusResponse])
async def get_model_statuses(
    provider: Optional[ModelProvider] = None,
    current_user: User = Depends(get_current_user)
):
    """Get status of all models or models from specific provider"""
    try:
        lifecycle_manager = await get_lifecycle_manager()
        all_statuses = await lifecycle_manager.get_all_model_statuses()
        
        # Filter by provider if specified
        if provider:
            all_statuses = [s for s in all_statuses if s.provider == provider]
        
        return [
            ModelStatusResponse(
                name=status.name,
                provider=status.provider.value,
                status=status.status,
                health_score=status.health_score,
                memory_usage_mb=status.memory_usage_mb,
                gpu_usage_percent=status.gpu_usage_percent,
                last_request_time=status.last_request_time,
                error_count=status.error_count,
                response_time_avg=status.response_time_avg,
                container_id=status.container_id
            )
            for status in all_statuses
        ]
        
    except Exception as e:
        logger.error(f"Failed to get model statuses: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/health-check")
async def health_check_model(
    provider: ModelProvider,
    model_name: str,
    priority: int = Query(2, ge=1, le=10),
    current_user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """Initiate health check for a specific model"""
    try:
        task_manager = get_task_manager()
        task_id = await task_manager.health_check_async(
            provider,
            model_name,
            priority
        )
        
        return {
            "task_id": task_id,
            "message": f"Health check initiated for {provider.value}:{model_name}"
        }
        
    except Exception as e:
        logger.error(f"Health check request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Container Management Endpoints
@router.post("/containers/create")
@RequireRole("admin")
async def create_container(
    request: ContainerCreateRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """Create a new container for a model"""
    try:
        orchestrator = await get_container_orchestrator()
        container_id = await orchestrator.create_model_container(
            request.provider,
            request.model_name,
            request.resource_requirements
        )
        
        return {
            "container_id": container_id,
            "message": f"Container created for {request.provider.value}:{request.model_name}"
        }
        
    except Exception as e:
        logger.error(f"Container creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/containers/{container_id}")
@RequireRole("admin")
async def remove_container(
    container_id: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """Remove a container"""
    try:
        orchestrator = await get_container_orchestrator()
        success = await orchestrator.remove_model_container(container_id)
        
        if success:
            return {"message": f"Container {container_id} removed successfully"}
        else:
            raise HTTPException(status_code=404, detail="Container not found")
            
    except Exception as e:
        logger.error(f"Container removal failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/containers", response_model=List[ContainerStatusResponse])
async def get_containers(
    provider: Optional[ModelProvider] = None,
    model_name: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Get status of all containers"""
    try:
        orchestrator = await get_container_orchestrator()
        
        if provider and model_name:
            containers = await orchestrator.get_model_containers(provider, model_name)
        else:
            containers = await orchestrator.get_all_containers()
        
        return [
            ContainerStatusResponse(
                container_id=container.container_id,
                name=container.name,
                model_name=container.model_name,
                provider=container.provider.value,
                status=container.status,
                health_status=container.health_status,
                created_at=container.created_at,
                started_at=container.started_at,
                cpu_usage=container.cpu_usage,
                memory_usage_mb=container.memory_usage_mb,
                gpu_usage=container.gpu_usage,
                network_io=container.network_io,
                disk_io=container.disk_io,
                restart_count=container.restart_count,
                last_error=container.last_error
            )
            for container in containers
        ]
        
    except Exception as e:
        logger.error(f"Failed to get container statuses: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/containers/{container_id}", response_model=ContainerStatusResponse)
async def get_container_status(
    container_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get status of a specific container"""
    try:
        orchestrator = await get_container_orchestrator()
        container = await orchestrator.get_container_status(container_id)
        
        if not container:
            raise HTTPException(status_code=404, detail="Container not found")
        
        return ContainerStatusResponse(
            container_id=container.container_id,
            name=container.name,
            model_name=container.model_name,
            provider=container.provider.value,
            status=container.status,
            health_status=container.health_status,
            created_at=container.created_at,
            started_at=container.started_at,
            cpu_usage=container.cpu_usage,
            memory_usage_mb=container.memory_usage_mb,
            gpu_usage=container.gpu_usage,
            network_io=container.network_io,
            disk_io=container.disk_io,
            restart_count=container.restart_count,
            last_error=container.last_error
        )
        
    except Exception as e:
        logger.error(f"Failed to get container status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Task Management Endpoints
@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get status of a background task"""
    try:
        task_manager = get_task_manager()
        task_status = await task_manager.get_task_status(task_id)
        
        return TaskStatusResponse(**task_status)
        
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/tasks/{task_id}")
async def cancel_task(
    task_id: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """Cancel a background task"""
    try:
        task_manager = get_task_manager()
        success = await task_manager.cancel_task(task_id)
        
        if success:
            return {"message": f"Task {task_id} cancelled successfully"}
        else:
            return {"message": f"Failed to cancel task {task_id}"}
            
    except Exception as e:
        logger.error(f"Failed to cancel task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/active")
async def get_active_tasks(
    current_user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """Get list of active background tasks"""
    try:
        task_manager = get_task_manager()
        active_tasks = await task_manager.get_active_tasks()
        
        return active_tasks
        
    except Exception as e:
        logger.error(f"Failed to get active tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/stats")
@RequireRole("admin")
async def get_task_stats(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get worker and task statistics"""
    try:
        task_manager = get_task_manager()
        stats = await task_manager.get_worker_stats()
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get task stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Monitoring and Metrics Endpoints
@router.get("/metrics/resources", response_model=ResourceMetricsResponse)
async def get_resource_metrics(
    current_user: User = Depends(get_current_user)
):
    """Get current system resource metrics"""
    try:
        lifecycle_manager = await get_lifecycle_manager()
        metrics = await lifecycle_manager.get_resource_metrics()
        
        return ResourceMetricsResponse(
            cpu_percent=metrics.cpu_percent,
            memory_percent=metrics.memory_percent,
            memory_available_mb=metrics.memory_available_mb,
            gpu_count=metrics.gpu_count,
            gpu_memory_total_mb=metrics.gpu_memory_total_mb,
            gpu_memory_used_mb=metrics.gpu_memory_used_mb,
            disk_usage_percent=metrics.disk_usage_percent
        )
        
    except Exception as e:
        logger.error(f"Failed to get resource metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/gpu")
async def get_gpu_metrics(
    current_user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """Get GPU resource metrics"""
    try:
        orchestrator = await get_container_orchestrator()
        gpu_resources = await orchestrator.get_gpu_resources()
        
        return [
            {
                "device_id": gpu.device_id,
                "name": gpu.name,
                "memory_total_mb": gpu.memory_total_mb,
                "memory_used_mb": gpu.memory_used_mb,
                "memory_available_mb": gpu.memory_available_mb,
                "utilization_percent": gpu.utilization_percent,
                "temperature": gpu.temperature,
                "power_usage": gpu.power_usage,
                "allocated_to": gpu.allocated_to
            }
            for gpu in gpu_resources
        ]
        
    except Exception as e:
        logger.error(f"Failed to get GPU metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/cache", response_model=CacheStatsResponse)
async def get_cache_stats(
    current_user: User = Depends(get_current_user)
):
    """Get cache statistics"""
    try:
        cache_manager = await get_cache_manager()
        stats = await cache_manager.get_cache_stats()
        
        return CacheStatsResponse(
            total_keys=stats.total_keys,
            total_memory_mb=stats.total_memory_mb,
            hit_rate=stats.hit_rate,
            miss_rate=stats.miss_rate,
            evictions=stats.evictions,
            expired_keys=stats.expired_keys,
            connections=stats.connections
        )
        
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/gateway", response_model=GatewayStatsResponse)
async def get_gateway_stats(
    current_user: User = Depends(get_current_user)
):
    """Get API gateway statistics"""
    try:
        gateway = await get_api_gateway()
        stats = await gateway.get_gateway_stats()
        
        return GatewayStatsResponse(
            total_requests=stats.total_requests,
            successful_requests=stats.successful_requests,
            failed_requests=stats.failed_requests,
            cached_responses=stats.cached_responses,
            avg_response_time=stats.avg_response_time,
            total_tokens=stats.total_tokens,
            total_cost=stats.total_cost,
            active_endpoints=stats.active_endpoints,
            cache_hit_rate=stats.cache_hit_rate,
            error_rate=stats.error_rate
        )
        
    except Exception as e:
        logger.error(f"Failed to get gateway stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Unified Model Request Endpoint
@router.post("/generate")
async def generate_unified(
    request: UnifiedModelRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Process a unified model request through the API gateway"""
    try:
        gateway = await get_api_gateway()
        
        gateway_request = GatewayRequest(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=request.stream,
            model_preference=request.model_preference,
            provider_preference=request.provider_preference,
            user_id=current_user.id,
            use_cache=request.use_cache,
            timeout=request.timeout,
            metadata=request.metadata
        )
        
        response = await gateway.process_request(gateway_request)
        
        return {
            "content": response.content,
            "model_used": response.model_used,
            "provider": response.provider.value,
            "endpoint_id": response.endpoint_id,
            "tokens_used": response.tokens_used,
            "cost": response.cost,
            "response_time": response.response_time,
            "cached": response.cached,
            "attempts": response.attempts,
            "fallback_used": response.fallback_used,
            "metadata": response.metadata
        }
        
    except Exception as e:
        logger.error(f"Unified generation request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Cache Management Endpoints
@router.delete("/cache/clear")
@RequireRole("admin")
async def clear_cache(
    category: Optional[str] = None,
    current_user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """Clear cache entries"""
    try:
        cache_manager = await get_cache_manager()
        
        if category:
            deleted = await cache_manager.clear_category(category)
            return {"message": f"Cleared {deleted} entries from {category} cache"}
        else:
            # Clear all cache (implement if needed)
            return {"message": "Cache clear not implemented for all categories"}
            
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/flush-expired")
@RequireRole("admin")
async def flush_expired_cache(
    current_user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """Flush expired cache entries"""
    try:
        cache_manager = await get_cache_manager()
        expired_count = await cache_manager.flush_expired_keys()
        
        return {"message": f"Flushed {expired_count} expired cache entries"}
        
    except Exception as e:
        logger.error(f"Cache flush failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# Model Verification Endpoint
# -----------------------------------------------------------------------------
class ModelVerifyRequest(BaseModel):
    model_path: str
    signature_path: str
    public_key_path: str


@router.post("/models/verify")
@RequireRole("admin")
async def verify_model_signature_endpoint(
    request: ModelVerifyRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """Verify the detached Ed25519 signature of a local model file."""
    from pathlib import Path
    try:
        ok = verify_model_signature(
            Path(request.model_path),
            Path(request.signature_path),
            Path(request.public_key_path),
        )
        if ok:
            return {"status": "valid", "message": "Model signature verified"}
        raise HTTPException(status_code=400, detail="Invalid model signature")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signature verification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
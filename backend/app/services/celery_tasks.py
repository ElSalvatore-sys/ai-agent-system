"""Celery Tasks for Background Model Management Operations.

Contains all the background tasks for model lifecycle management including
pulling, loading, scaling, monitoring, and cleanup operations.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from celery import current_task
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.celery_app import celery_app
from app.database.models import ModelProvider
from app.database.database import get_db
from app.core.logger import get_logger


logger = get_logger(__name__)


def update_task_progress(current: int, total: int, message: str = ""):
    """Update task progress"""
    if current_task:
        current_task.update_state(
            state="PROGRESS",
            meta={
                "current": current,
                "total": total,
                "message": message,
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def pull_model(self, provider_str: str, model_name: str) -> Dict[str, Any]:
    """Pull/download a model"""
    try:
        provider = ModelProvider(provider_str)
        logger.info(f"Starting model pull: {provider.value}:{model_name}")
        
        update_task_progress(0, 100, "Initializing model pull")
        
        # Import here to avoid circular imports
        from app.services.model_discovery import get_lifecycle_manager
        
        async def _pull_model():
            manager = await get_lifecycle_manager()
            
            update_task_progress(25, 100, "Connecting to model service")
            result = await manager.pull_model(provider, model_name)
            
            update_task_progress(100, 100, "Model pull completed")
            return result
        
        # Run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(_pull_model())
            logger.info(f"Model pull completed: {provider.value}:{model_name}")
            return result
        finally:
            loop.close()
            
    except Exception as exc:
        logger.error(f"Model pull failed: {provider_str}:{model_name} - {exc}")
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying model pull: {provider_str}:{model_name} (attempt {self.request.retries + 1})")
            raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))
        
        # Final failure
        return {
            "status": "error",
            "message": f"Failed to pull model after {self.max_retries} attempts: {str(exc)}"
        }


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def load_model(self, provider_str: str, model_name: str, resource_requirements: Dict[str, Any]) -> Dict[str, Any]:
    """Load a model into memory/container"""
    try:
        provider = ModelProvider(provider_str)
        logger.info(f"Starting model load: {provider.value}:{model_name}")
        
        update_task_progress(0, 100, "Initializing model load")
        
        from app.services.container_orchestrator import get_container_orchestrator
        
        async def _load_model():
            orchestrator = await get_container_orchestrator()
            
            update_task_progress(25, 100, "Creating container")
            container_id = await orchestrator.create_model_container(
                provider, model_name, resource_requirements
            )
            
            update_task_progress(75, 100, "Starting container")
            # Container startup is handled in create_model_container
            
            update_task_progress(100, 100, "Model loaded successfully")
            return {
                "status": "success",
                "container_id": container_id,
                "message": f"Model {provider.value}:{model_name} loaded successfully"
            }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(_load_model())
            logger.info(f"Model load completed: {provider.value}:{model_name}")
            return result
        finally:
            loop.close()
            
    except Exception as exc:
        logger.error(f"Model load failed: {provider_str}:{model_name} - {exc}")
        
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying model load: {provider_str}:{model_name} (attempt {self.request.retries + 1})")
            raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))
        
        return {
            "status": "error",
            "message": f"Failed to load model after {self.max_retries} attempts: {str(exc)}"
        }


@celery_app.task(bind=True, max_retries=2, default_retry_delay=30)
def unload_model(self, provider_str: str, model_name: str) -> Dict[str, Any]:
    """Unload a model from memory/container"""
    try:
        provider = ModelProvider(provider_str)
        logger.info(f"Starting model unload: {provider.value}:{model_name}")
        
        update_task_progress(0, 100, "Finding model containers")
        
        from app.services.container_orchestrator import get_container_orchestrator
        
        async def _unload_model():
            orchestrator = await get_container_orchestrator()
            
            # Find containers for this model
            containers = await orchestrator.get_model_containers(provider, model_name)
            
            update_task_progress(25, 100, f"Found {len(containers)} containers to remove")
            
            removed_count = 0
            for container in containers:
                await orchestrator.remove_model_container(container.container_id)
                removed_count += 1
                
                progress = 25 + (removed_count / len(containers)) * 75
                update_task_progress(int(progress), 100, f"Removed {removed_count}/{len(containers)} containers")
            
            update_task_progress(100, 100, "Model unloaded successfully")
            return {
                "status": "success",
                "containers_removed": removed_count,
                "message": f"Model {provider.value}:{model_name} unloaded successfully"
            }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(_unload_model())
            logger.info(f"Model unload completed: {provider.value}:{model_name}")
            return result
        finally:
            loop.close()
            
    except Exception as exc:
        logger.error(f"Model unload failed: {provider_str}:{model_name} - {exc}")
        
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=30)
        
        return {
            "status": "error",
            "message": f"Failed to unload model: {str(exc)}"
        }


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def scale_model(self, provider_str: str, model_name: str, desired_instances: int) -> Dict[str, Any]:
    """Scale a model to desired number of instances"""
    try:
        provider = ModelProvider(provider_str)
        logger.info(f"Starting model scaling: {provider.value}:{model_name} to {desired_instances} instances")
        
        update_task_progress(0, 100, "Initializing scaling operation")
        
        from app.services.container_orchestrator import get_container_orchestrator
        
        async def _scale_model():
            orchestrator = await get_container_orchestrator()
            
            update_task_progress(25, 100, "Analyzing current instances")
            current_containers = await orchestrator.get_model_containers(provider, model_name)
            current_instances = len(current_containers)
            
            update_task_progress(50, 100, f"Current instances: {current_instances}, Target: {desired_instances}")
            
            # Perform scaling
            result_containers = await orchestrator.scale_model(provider, model_name, desired_instances)
            
            update_task_progress(100, 100, "Scaling completed")
            return {
                "status": "success",
                "previous_instances": current_instances,
                "current_instances": len(result_containers),
                "container_ids": [c.container_id if hasattr(c, 'container_id') else str(c) for c in result_containers],
                "message": f"Model {provider.value}:{model_name} scaled to {len(result_containers)} instances"
            }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(_scale_model())
            logger.info(f"Model scaling completed: {provider.value}:{model_name}")
            return result
        finally:
            loop.close()
            
    except Exception as exc:
        logger.error(f"Model scaling failed: {provider_str}:{model_name} - {exc}")
        
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=30 * (2 ** self.request.retries))
        
        return {
            "status": "error",
            "message": f"Failed to scale model: {str(exc)}"
        }


@celery_app.task(bind=True)
def health_check(self, provider_str: str, model_name: str) -> Dict[str, Any]:
    """Perform health check on a model"""
    try:
        provider = ModelProvider(provider_str)
        
        from app.services.model_discovery import get_lifecycle_manager
        
        async def _health_check():
            manager = await get_lifecycle_manager()
            health_score = await manager._check_model_health(provider, model_name)
            
            status = "healthy" if health_score > 0.7 else "unhealthy" if health_score > 0 else "offline"
            
            return {
                "status": "success",
                "health_status": status,
                "health_score": health_score,
                "timestamp": datetime.utcnow().isoformat(),
                "message": f"Health check completed for {provider.value}:{model_name}"
            }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(_health_check())
            return result
        finally:
            loop.close()
            
    except Exception as exc:
        logger.error(f"Health check failed: {provider_str}:{model_name} - {exc}")
        return {
            "status": "error",
            "health_status": "unknown",
            "health_score": 0.0,
            "message": f"Health check failed: {str(exc)}"
        }


@celery_app.task
def periodic_health_check() -> Dict[str, Any]:
    """Periodic health check for all models"""
    try:
        from app.services.model_discovery import get_lifecycle_manager
        
        async def _periodic_health_check():
            manager = await get_lifecycle_manager()
            statuses = await manager.get_all_model_statuses()
            
            health_results = []
            for status in statuses:
                health_score = await manager._check_model_health(status.provider, status.name)
                status.health_score = health_score
                
                health_results.append({
                    "model": f"{status.provider.value}:{status.name}",
                    "health_score": health_score,
                    "status": status.status
                })
            
            return {
                "status": "success",
                "checked_models": len(health_results),
                "results": health_results,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(_periodic_health_check())
            logger.info(f"Periodic health check completed for {result['checked_models']} models")
            return result
        finally:
            loop.close()
            
    except Exception as exc:
        logger.error(f"Periodic health check failed: {exc}")
        return {
            "status": "error",
            "message": f"Periodic health check failed: {str(exc)}"
        }


@celery_app.task
def periodic_resource_monitoring() -> Dict[str, Any]:
    """Periodic resource monitoring"""
    try:
        from app.services.model_discovery import get_lifecycle_manager
        from app.services.container_orchestrator import get_container_orchestrator
        
        async def _resource_monitoring():
            manager = await get_lifecycle_manager()
            orchestrator = await get_container_orchestrator()
            
            # Get system resource metrics
            system_metrics = await manager.get_resource_metrics()
            
            # Get container resource usage
            container_usage = await orchestrator.get_resource_usage()
            
            # Get GPU resources
            gpu_resources = await orchestrator.get_gpu_resources()
            
            return {
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "system_metrics": {
                    "cpu_percent": system_metrics.cpu_percent,
                    "memory_percent": system_metrics.memory_percent,
                    "memory_available_mb": system_metrics.memory_available_mb,
                    "disk_usage_percent": system_metrics.disk_usage_percent,
                },
                "container_usage": container_usage,
                "gpu_resources": [
                    {
                        "device_id": gpu.device_id,
                        "name": gpu.name,
                        "memory_used_mb": gpu.memory_used_mb,
                        "memory_total_mb": gpu.memory_total_mb,
                        "utilization_percent": gpu.utilization_percent,
                        "temperature": gpu.temperature,
                        "allocated_to": gpu.allocated_to
                    }
                    for gpu in gpu_resources
                ]
            }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(_resource_monitoring())
            return result
        finally:
            loop.close()
            
    except Exception as exc:
        logger.error(f"Resource monitoring failed: {exc}")
        return {
            "status": "error",
            "message": f"Resource monitoring failed: {str(exc)}"
        }


@celery_app.task
def periodic_container_cleanup() -> Dict[str, Any]:
    """Periodic cleanup of stopped/failed containers"""
    try:
        from app.services.container_orchestrator import get_container_orchestrator
        
        async def _container_cleanup():
            orchestrator = await get_container_orchestrator()
            
            # Get all containers
            all_containers = await orchestrator.get_all_containers()
            
            # Find containers that need cleanup
            cleanup_candidates = [
                container for container in all_containers
                if container.status in ["exited", "dead", "removing"]
                or container.health_status == "unhealthy"
            ]
            
            cleaned_up = 0
            for container in cleanup_candidates:
                try:
                    success = await orchestrator.remove_model_container(container.container_id)
                    if success:
                        cleaned_up += 1
                except Exception as e:
                    logger.error(f"Failed to cleanup container {container.container_id}: {e}")
            
            return {
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "total_containers": len(all_containers),
                "cleanup_candidates": len(cleanup_candidates),
                "cleaned_up": cleaned_up
            }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(_container_cleanup())
            logger.info(f"Container cleanup completed: {result['cleaned_up']} containers cleaned up")
            return result
        finally:
            loop.close()
            
    except Exception as exc:
        logger.error(f"Container cleanup failed: {exc}")
        return {
            "status": "error",
            "message": f"Container cleanup failed: {str(exc)}"
        }


@celery_app.task
def periodic_model_discovery() -> Dict[str, Any]:
    """Periodic model discovery"""
    try:
        from app.services.model_discovery import run_discovery_once
        
        async def _model_discovery():
            await run_discovery_once()
            return {
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Model discovery completed"
            }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(_model_discovery())
            logger.info("Periodic model discovery completed")
            return result
        finally:
            loop.close()
            
    except Exception as exc:
        logger.error(f"Model discovery failed: {exc}")
        return {
            "status": "error",
            "message": f"Model discovery failed: {str(exc)}"
        }


@celery_app.task(bind=True)
def batch_model_operations(self, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Execute multiple model operations in batch"""
    try:
        update_task_progress(0, len(operations), "Starting batch operations")
        
        results = []
        for i, operation in enumerate(operations):
            try:
                op_type = operation["type"]
                provider_str = operation["provider"]
                model_name = operation["model_name"]
                
                if op_type == "pull":
                    result = pull_model(provider_str, model_name)
                elif op_type == "load":
                    resource_req = operation.get("resource_requirements", {})
                    result = load_model(provider_str, model_name, resource_req)
                elif op_type == "unload":
                    result = unload_model(provider_str, model_name)
                elif op_type == "scale":
                    instances = operation["desired_instances"]
                    result = scale_model(provider_str, model_name, instances)
                else:
                    result = {"status": "error", "message": f"Unknown operation type: {op_type}"}
                
                results.append({
                    "operation": operation,
                    "result": result
                })
                
                update_task_progress(i + 1, len(operations), f"Completed {i + 1}/{len(operations)} operations")
                
            except Exception as e:
                results.append({
                    "operation": operation,
                    "result": {"status": "error", "message": str(e)}
                })
        
        successful = len([r for r in results if r["result"]["status"] == "success"])
        
        return {
            "status": "success",
            "total_operations": len(operations),
            "successful": successful,
            "failed": len(operations) - successful,
            "results": results
        }
        
    except Exception as exc:
        logger.error(f"Batch operations failed: {exc}")
        return {
            "status": "error",
            "message": f"Batch operations failed: {str(exc)}"
        }
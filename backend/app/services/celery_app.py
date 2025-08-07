"""Celery Application for Background Model Management Tasks.

Provides distributed task queue for model lifecycle operations including
pulling, loading, unloading, and scaling operations.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from celery import Celery, Task
from celery.signals import worker_ready, worker_shutdown
from kombu import Queue

from app.core.config import settings
from app.database.models import ModelProvider


logger = logging.getLogger(__name__)


# Celery configuration
celery_app = Celery(
    "llm_background_tasks",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=[
        "app.services.celery_tasks"
    ]
)

# Celery settings
celery_app.conf.update(
    # Task routing
    task_routes={
        "app.services.celery_tasks.pull_model": {"queue": "model_management"},
        "app.services.celery_tasks.load_model": {"queue": "model_management"},
        "app.services.celery_tasks.unload_model": {"queue": "model_management"},
        "app.services.celery_tasks.scale_model": {"queue": "scaling"},
        "app.services.celery_tasks.health_check": {"queue": "monitoring"},
        "app.services.celery_tasks.resource_monitoring": {"queue": "monitoring"},
        "app.services.celery_tasks.container_cleanup": {"queue": "cleanup"},
    },
    
    # Queue configuration
    task_default_queue="default",
    task_queues=(
        Queue("default", routing_key="default"),
        Queue("model_management", routing_key="model_management"),
        Queue("scaling", routing_key="scaling"),
        Queue("monitoring", routing_key="monitoring"),
        Queue("cleanup", routing_key="cleanup"),
    ),
    
    # Task execution settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task retry settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,
    
    # Worker settings
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    worker_disable_rate_limits=False,
    
    # Result backend settings
    result_expires=3600,  # 1 hour
    result_persistent=True,
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Beat schedule for periodic tasks
    beat_schedule={
        "health-check-models": {
            "task": "app.services.celery_tasks.periodic_health_check",
            "schedule": 300.0,  # Every 5 minutes
        },
        "resource-monitoring": {
            "task": "app.services.celery_tasks.periodic_resource_monitoring",
            "schedule": 30.0,  # Every 30 seconds
        },
        "container-cleanup": {
            "task": "app.services.celery_tasks.periodic_container_cleanup",
            "schedule": 3600.0,  # Every hour
        },
        "model-discovery": {
            "task": "app.services.celery_tasks.periodic_model_discovery",
            "schedule": 900.0,  # Every 15 minutes
        },
    },
)


class CallbackTask(Task):
    """Base task class with callback support"""
    
    def on_success(self, retval, task_id, args, kwargs):
        """Called when task succeeds"""
        logger.info(f"Task {self.name} ({task_id}) succeeded: {retval}")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails"""
        logger.error(f"Task {self.name} ({task_id}) failed: {exc}")
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is retried"""
        logger.warning(f"Task {self.name} ({task_id}) retrying: {exc}")


# Set the base task class
celery_app.Task = CallbackTask


@worker_ready.connect
def at_start(sender, **kwargs):
    """Called when Celery worker starts"""
    logger.info("Celery worker started and ready")


@worker_shutdown.connect
def at_shutdown(sender, **kwargs):
    """Called when Celery worker shuts down"""
    logger.info("Celery worker shutting down")


class TaskManager:
    """High-level interface for managing background tasks"""
    
    def __init__(self):
        self.app = celery_app
    
    async def pull_model_async(
        self, 
        provider: ModelProvider, 
        model_name: str,
        priority: int = 5
    ) -> str:
        """Queue a model pull task"""
        from app.services.celery_tasks import pull_model
        
        task = pull_model.apply_async(
            args=[provider.value, model_name],
            priority=priority,
            queue="model_management"
        )
        
        logger.info(f"Queued model pull task: {task.id} for {provider.value}:{model_name}")
        return task.id
    
    async def load_model_async(
        self, 
        provider: ModelProvider, 
        model_name: str,
        resource_requirements: Optional[Dict[str, Any]] = None,
        priority: int = 5
    ) -> str:
        """Queue a model load task"""
        from app.services.celery_tasks import load_model
        
        task = load_model.apply_async(
            args=[provider.value, model_name, resource_requirements or {}],
            priority=priority,
            queue="model_management"
        )
        
        logger.info(f"Queued model load task: {task.id} for {provider.value}:{model_name}")
        return task.id
    
    async def unload_model_async(
        self, 
        provider: ModelProvider, 
        model_name: str,
        priority: int = 3
    ) -> str:
        """Queue a model unload task"""
        from app.services.celery_tasks import unload_model
        
        task = unload_model.apply_async(
            args=[provider.value, model_name],
            priority=priority,
            queue="model_management"
        )
        
        logger.info(f"Queued model unload task: {task.id} for {provider.value}:{model_name}")
        return task.id
    
    async def scale_model_async(
        self, 
        provider: ModelProvider, 
        model_name: str,
        desired_instances: int,
        priority: int = 4
    ) -> str:
        """Queue a model scaling task"""
        from app.services.celery_tasks import scale_model
        
        task = scale_model.apply_async(
            args=[provider.value, model_name, desired_instances],
            priority=priority,
            queue="scaling"
        )
        
        logger.info(
            f"Queued model scaling task: {task.id} for {provider.value}:{model_name} "
            f"to {desired_instances} instances"
        )
        return task.id
    
    async def health_check_async(
        self, 
        provider: ModelProvider, 
        model_name: str,
        priority: int = 2
    ) -> str:
        """Queue a health check task"""
        from app.services.celery_tasks import health_check
        
        task = health_check.apply_async(
            args=[provider.value, model_name],
            priority=priority,
            queue="monitoring"
        )
        
        return task.id
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a task"""
        task_result = self.app.AsyncResult(task_id)
        
        return {
            "task_id": task_id,
            "status": task_result.status,
            "result": task_result.result if task_result.ready() else None,
            "traceback": task_result.traceback if task_result.failed() else None,
            "info": task_result.info,
        }
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        try:
            self.app.control.revoke(task_id, terminate=True)
            logger.info(f"Cancelled task: {task_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return False
    
    async def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get list of active tasks"""
        inspect = self.app.control.inspect()
        active_tasks = inspect.active()
        
        if not active_tasks:
            return []
        
        tasks = []
        for worker, worker_tasks in active_tasks.items():
            for task in worker_tasks:
                tasks.append({
                    "task_id": task["id"],
                    "name": task["name"],
                    "args": task["args"],
                    "kwargs": task["kwargs"],
                    "worker": worker,
                    "time_start": task.get("time_start"),
                })
        
        return tasks
    
    async def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        inspect = self.app.control.inspect()
        
        stats = inspect.stats()
        registered = inspect.registered()
        active = inspect.active()
        scheduled = inspect.scheduled()
        reserved = inspect.reserved()
        
        return {
            "workers": list(stats.keys()) if stats else [],
            "stats": stats or {},
            "registered_tasks": registered or {},
            "active_tasks": active or {},
            "scheduled_tasks": scheduled or {},
            "reserved_tasks": reserved or {},
        }
    
    async def purge_queue(self, queue_name: str) -> int:
        """Purge all tasks from a queue"""
        try:
            purged = self.app.control.purge()
            logger.info(f"Purged {purged} tasks from queue {queue_name}")
            return purged
        except Exception as e:
            logger.error(f"Failed to purge queue {queue_name}: {e}")
            return 0
    
    def get_task_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get task execution history"""
        # This would typically query a database or monitoring system
        # For now, return empty list as placeholder
        return []


# Global instance
task_manager = TaskManager()


def get_task_manager() -> TaskManager:
    """Get the global task manager instance"""
    return task_manager
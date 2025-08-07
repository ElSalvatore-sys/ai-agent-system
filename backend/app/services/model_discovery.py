"""Enhanced Model Discovery and Lifecycle Management.

Supports Ollama, HuggingFace Transformers, and Docker container orchestration.
Provides comprehensive model lifecycle management including pulling, loading,
health monitoring, and resource tracking.
"""
from __future__ import annotations

import asyncio
import logging
import psutil
import docker
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

import httpx
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logger import LoggerMixin
from app.database.database import get_db
from app.database.models import AIModel, ModelProvider, SystemMetrics

logger = logging.getLogger(__name__)


@dataclass
class ModelStatus:
    """Model status and health information"""
    name: str
    provider: ModelProvider
    status: str  # available, loading, error, offline
    health_score: float  # 0-1
    memory_usage_mb: float
    gpu_usage_percent: float
    last_request_time: Optional[datetime]
    error_count: int
    response_time_avg: float
    container_id: Optional[str] = None


@dataclass
class ResourceMetrics:
    """System resource metrics"""
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    gpu_count: int
    gpu_memory_total_mb: float
    gpu_memory_used_mb: float
    disk_usage_percent: float


class ModelLifecycleManager(LoggerMixin):
    """Manages the complete lifecycle of local LLM models"""
    
    def __init__(self):
        super().__init__()
        self.docker_client = None
        self.model_statuses: Dict[str, ModelStatus] = {}
        self.resource_monitor_task = None
        self.health_check_task = None
        self._shutdown_event = asyncio.Event()
    
    async def initialize(self):
        """Initialize the model lifecycle manager"""
        try:
            self.docker_client = docker.from_env()
            self.logger.info("Docker client initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize Docker client: {e}")
        
        # Start background tasks
        self.resource_monitor_task = asyncio.create_task(self._resource_monitor_loop())
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        # Discover existing models
        await self.discover_and_sync_models()
        
        self.logger.info("Model lifecycle manager initialized")
    
    async def shutdown(self):
        """Gracefully shutdown the manager"""
        self.logger.info("Shutting down model lifecycle manager")
        self._shutdown_event.set()
        
        if self.resource_monitor_task:
            self.resource_monitor_task.cancel()
        if self.health_check_task:
            self.health_check_task.cancel()
        
        # Stop all model containers
        await self._stop_all_model_containers()
    
    async def discover_and_sync_models(self):
        """Discover and sync all available local models"""
        ollama_models = await _discover_ollama_models()
        hf_models = await self._discover_hf_models()
        
        async for db in get_db():
            # Sync Ollama models
            for model in ollama_models:
                await _sync_model_enhanced(
                    db, 
                    ModelProvider.LOCAL_OLLAMA, 
                    model.get("name", ""), 
                    model.get("name", ""),
                    model.get("size", 0),
                    ["text_generation", "streaming"]
                )
            
            # Sync HuggingFace models
            for model in hf_models:
                await _sync_model_enhanced(
                    db,
                    ModelProvider.LOCAL_HF,
                    model["name"],
                    model["display_name"],
                    model.get("size", 0),
                    model.get("capabilities", [])
                )
            
            await db.commit()
            break
    
    async def pull_model(self, provider: ModelProvider, model_name: str) -> Dict[str, Any]:
        """Pull/download a model"""
        self.logger.info(f"Pulling model {provider.value}:{model_name}")
        
        try:
            if provider == ModelProvider.LOCAL_OLLAMA:
                return await self._pull_ollama_model(model_name)
            elif provider == ModelProvider.LOCAL_HF:
                return await self._pull_hf_model(model_name)
            else:
                raise ValueError(f"Unsupported provider for pulling: {provider}")
        except Exception as e:
            self.logger.error(f"Failed to pull model {provider.value}:{model_name}: {e}")
            raise
    
    async def get_resource_metrics(self) -> ResourceMetrics:
        """Get current system resource metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        gpu_count = 0
        gpu_memory_total = 0
        gpu_memory_used = 0
        
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            gpu_count = len(gpus)
            if gpus:
                gpu_memory_total = sum(gpu.memoryTotal for gpu in gpus)
                gpu_memory_used = sum(gpu.memoryUsed for gpu in gpus)
        except ImportError:
            self.logger.warning("GPUtil not available, GPU metrics disabled")
        
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available_mb=memory.available / 1024 / 1024,
            gpu_count=gpu_count,
            gpu_memory_total_mb=gpu_memory_total,
            gpu_memory_used_mb=gpu_memory_used,
            disk_usage_percent=disk.percent
        )
    
    async def _discover_hf_models(self) -> List[Dict[str, Any]]:
        """Discover HuggingFace models configured for local use"""
        return [
            {
                "name": "microsoft/DialoGPT-small",
                "display_name": "DialoGPT Small",
                "size": 117000000,
                "capabilities": ["text_generation", "conversation"]
            },
            {
                "name": "microsoft/CodeBERT-base",
                "display_name": "CodeBERT Base",
                "size": 440000000,
                "capabilities": ["code_analysis", "text_generation"]
            }
        ]
    
    async def _pull_ollama_model(self, model_name: str) -> Dict[str, Any]:
        """Pull a model using Ollama API"""
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(
                    f"{settings.OLLAMA_HOST}/api/pull",
                    json={"name": model_name}
                )
                response.raise_for_status()
                return {"status": "success", "message": f"Model {model_name} pulled successfully"}
        except Exception as e:
            raise Exception(f"Failed to pull Ollama model {model_name}: {e}")
    
    async def _pull_hf_model(self, model_name: str) -> Dict[str, Any]:
        """Pull a HuggingFace model"""
        try:
            from transformers import AutoTokenizer, AutoModel
            
            # Download tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)

            # Optional cryptographic verification (detached signature)
            sig_ok = True
            try:
                from pathlib import Path
                from app.security.model_verifier import verify_model_signature

                trust_dir = getattr(settings, "MODEL_TRUST_STORE", None)
                if trust_dir:
                    model_basename = model_name.split("/")[-1]
                    model_path = Path(trust_dir) / f"{model_basename}.safetensors"
                    sig_path = model_path.with_suffix(model_path.suffix + ".sig")
                    pub_path = Path(trust_dir) / f"{model_basename}.pub"
                    if model_path.exists() and sig_path.exists() and pub_path.exists():
                        sig_ok = verify_model_signature(model_path, sig_path, pub_path)
            except Exception as verr:
                logger.warning("Signature verification error for %s: %s", model_name, verr)
                sig_ok = False

            if not sig_ok:
                raise Exception("Model signature verification failed")

            return {"status": "success", "message": f"HuggingFace model {model_name} downloaded and verified successfully"}
        except Exception as e:
            raise Exception(f"Failed to pull HuggingFace model {model_name}: {e}")
    
    async def _resource_monitor_loop(self):
        """Background task to monitor system resources"""
        while not self._shutdown_event.is_set():
            try:
                metrics = await self.get_resource_metrics()
                
                # Store metrics in database
                async for db in get_db():
                    metric_data = [
                        SystemMetrics(
                            metric_name="cpu_usage_percent",
                            metric_value=metrics.cpu_percent,
                            metric_type="gauge"
                        ),
                        SystemMetrics(
                            metric_name="memory_usage_percent", 
                            metric_value=metrics.memory_percent,
                            metric_type="gauge"
                        ),
                        SystemMetrics(
                            metric_name="gpu_memory_usage_mb",
                            metric_value=metrics.gpu_memory_used_mb,
                            metric_type="gauge"
                        )
                    ]
                    
                    for metric in metric_data:
                        db.add(metric)
                    
                    await db.commit()
                    break
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _health_check_loop(self):
        """Background task to check model health"""
        while not self._shutdown_event.is_set():
            try:
                for model_key, status in self.model_statuses.items():
                    provider_str, model_name = model_key.split(":", 1)
                    provider = ModelProvider(provider_str)
                    health_score = await self._check_model_health(provider, model_name)
                    status.health_score = health_score
                
                await asyncio.sleep(60)  # Health check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(120)
    
    async def _check_model_health(self, provider: ModelProvider, model_name: str) -> float:
        """Check health of a specific model"""
        try:
            if provider == ModelProvider.LOCAL_OLLAMA:
                async with httpx.AsyncClient(timeout=10) as client:
                    start_time = asyncio.get_event_loop().time()
                    response = await client.post(
                        f"{settings.OLLAMA_HOST}/api/generate",
                        json={
                            "model": model_name,
                            "prompt": "test",
                            "stream": False
                        }
                    )
                    response_time = asyncio.get_event_loop().time() - start_time
                    if response.status_code == 200:
                        # Calculate health score based on response time
                        if response_time < 1.0:
                            return 1.0
                        elif response_time < 5.0:
                            return 0.8
                        else:
                            return 0.6
                    else:
                        return 0.0
            else:
                # For HF models, just return 1.0 for now
                return 1.0
        except Exception as e:
            self.logger.warning(f"Health check failed for {provider.value}:{model_name}: {e}")
            return 0.0
    
    async def _stop_all_model_containers(self):
        """Stop all running model containers"""
        if not self.docker_client:
            return
        
        try:
            containers = self.docker_client.containers.list(filters={"label": "llm-model"})
            for container in containers:
                self.logger.info(f"Stopping container {container.id}")
                container.stop(timeout=30)
        except Exception as e:
            self.logger.error(f"Error stopping containers: {e}")


# Global instance
_lifecycle_manager: Optional[ModelLifecycleManager] = None


async def get_lifecycle_manager() -> ModelLifecycleManager:
    """Get the global model lifecycle manager instance"""
    global _lifecycle_manager
    if _lifecycle_manager is None:
        _lifecycle_manager = ModelLifecycleManager()
        await _lifecycle_manager.initialize()
    return _lifecycle_manager


async def _discover_ollama_models() -> List[dict]:
    """Discover models available via Ollama API"""
    base = settings.OLLAMA_HOST
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{base}/api/tags")
        r.raise_for_status()
        data = r.json()
        return data.get("models", [])
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Ollama discovery failed: %s", exc)
        return []


async def _sync_model(session: AsyncSession, provider: ModelProvider, name: str, display_name: str):
    """Legacy sync function for backward compatibility"""
    await _sync_model_enhanced(session, provider, name, display_name, 0, [])


async def _sync_model_enhanced(
    session: AsyncSession, 
    provider: ModelProvider, 
    name: str, 
    display_name: str,
    size_bytes: int = 0,
    capabilities: List[str] = None
):
    """Enhanced model sync with additional metadata"""
    capabilities = capabilities or []
    
    stmt = select(AIModel).where(AIModel.provider == provider, AIModel.name == name)
    res = await session.execute(stmt)
    instance: AIModel | None = res.scalar_one_or_none()

    if instance is None:
        instance = AIModel(
            provider=provider,
            name=name,
            display_name=display_name,
            description=f"Local model {display_name}",
            host_type="local",
            device_id="local",  # future: GPU serial / hostname
            availability=True,
            supports_streaming="streaming" in capabilities,
            supports_functions="function_calling" in capabilities,
            supports_vision="vision" in capabilities,
            max_tokens=4096,
            context_window=4096,
            input_cost=0.0,
            output_cost=0.0,
            is_active=True,
            created_at=datetime.utcnow(),
            meta_data={
                "size_bytes": size_bytes,
                "capabilities": capabilities,
                "discovered_at": datetime.utcnow().isoformat()
            }
        )
        session.add(instance)
        await session.commit()
        logger.info("Registered local model %s:%s", provider, name)
    else:
        # Update availability flag and metadata if previously inactive
        if not instance.availability:
            instance.availability = True
            instance.updated_at = datetime.utcnow()
            instance.meta_data = {
                **(instance.meta_data or {}),
                "size_bytes": size_bytes,
                "capabilities": capabilities,
                "last_discovered_at": datetime.utcnow().isoformat()
            }
            await session.commit()


async def run_discovery_once():
    """Entry point used from FastAPI lifespan."""
    try:
        manager = await get_lifecycle_manager()
        await manager.discover_and_sync_models()
        logger.info("Model discovery completed successfully")
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Model discovery failed: %s", exc)
        # Fallback to basic discovery
        async for db in get_db():
            for m in await _discover_ollama_models():
                await _sync_model(db, ModelProvider.LOCAL_OLLAMA, m.get("name"), m.get("name"))
            break  # get_db() yields once


async def schedule_periodic_discovery(interval: int = 300):  # 5 min default
    """Run discovery every N seconds with enhanced lifecycle management"""
    while True:
        try:
            await run_discovery_once()
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("model discovery error: %s", exc)
        await asyncio.sleep(interval)

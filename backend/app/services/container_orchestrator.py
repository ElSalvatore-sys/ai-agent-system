"""Container Orchestration Service for Local LLM Management.

Provides Docker container management with GPU allocation, resource constraints,
and automatic scaling for local LLM models.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

import docker
import docker.errors
from docker.models.containers import Container
from docker.types import Mount, DeviceRequest

from app.core.config import settings
from app.core.logger import LoggerMixin
from app.database.models import ModelProvider


logger = logging.getLogger(__name__)


@dataclass
class ContainerConfig:
    """Configuration for a model container"""
    image: str
    model_name: str
    provider: ModelProvider
    environment: Dict[str, str]
    ports: Dict[str, int]
    mounts: List[Mount]
    memory_limit: str
    cpu_limit: float
    gpu_device_ids: List[str]
    health_check: Dict[str, Any]
    restart_policy: Dict[str, str]
    labels: Dict[str, str]


@dataclass
class ContainerStatus:
    """Container status information"""
    container_id: str
    name: str
    model_name: str
    provider: ModelProvider
    status: str  # running, stopped, error, starting
    health_status: str  # healthy, unhealthy, starting
    created_at: datetime
    started_at: Optional[datetime]
    cpu_usage: float
    memory_usage_mb: float
    gpu_usage: float
    network_io: Dict[str, int]
    disk_io: Dict[str, int]
    restart_count: int
    last_error: Optional[str]


@dataclass
class GPUResource:
    """GPU resource information"""
    device_id: str
    name: str
    memory_total_mb: float
    memory_used_mb: float
    memory_available_mb: float
    utilization_percent: float
    temperature: float
    power_usage: float
    allocated_to: Optional[str] = None  # container_id


class ContainerOrchestrator(LoggerMixin):
    """Manages Docker containers for local LLM models"""
    
    def __init__(self):
        super().__init__()
        self.docker_client = None
        self.containers: Dict[str, ContainerStatus] = {}
        self.gpu_resources: Dict[str, GPUResource] = {}
        self.container_configs: Dict[str, ContainerConfig] = {}
        self._monitor_task = None
        self._scaling_task = None
        self._shutdown_event = asyncio.Event()
        
        # Configuration
        self.max_containers_per_model = 3
        self.min_memory_mb = 512
        self.max_memory_mb = 8192
        self.network_name = "llm-network"
        self.base_data_dir = Path("/var/lib/llm-models")
    
    async def initialize(self):
        """Initialize the container orchestrator"""
        try:
            self.docker_client = docker.from_env()
            
            # Create network if it doesn't exist
            await self._ensure_network()
            
            # Discover GPU resources
            await self._discover_gpu_resources()
            
            # Load existing containers
            await self._load_existing_containers()
            
            # Start background tasks
            self._monitor_task = asyncio.create_task(self._monitoring_loop())
            self._scaling_task = asyncio.create_task(self._scaling_loop())
            
            self.logger.info("Container orchestrator initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize container orchestrator: {e}")
            raise
    
    async def shutdown(self):
        """Gracefully shutdown the orchestrator"""
        self.logger.info("Shutting down container orchestrator")
        self._shutdown_event.set()
        
        if self._monitor_task:
            self._monitor_task.cancel()
        if self._scaling_task:
            self._scaling_task.cancel()
        
        # Stop all managed containers
        await self._stop_all_containers()
    
    async def create_model_container(
        self, 
        provider: ModelProvider, 
        model_name: str,
        resource_requirements: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new container for a model"""
        self.logger.info(f"Creating container for {provider.value}:{model_name}")
        
        try:
            # Generate container configuration
            config = await self._generate_container_config(
                provider, model_name, resource_requirements
            )
            
            # Allocate GPU resources if needed
            gpu_devices = await self._allocate_gpu_resources(config.gpu_device_ids)
            
            # Create the container
            container = await self._create_container(config, gpu_devices)
            
            # Start the container
            await self._start_container(container.id)
            
            # Store configuration and status
            self.container_configs[container.id] = config
            self.containers[container.id] = ContainerStatus(
                container_id=container.id,
                name=container.name,
                model_name=model_name,
                provider=provider,
                status="starting",
                health_status="starting",
                created_at=datetime.utcnow(),
                started_at=datetime.utcnow(),
                cpu_usage=0.0,
                memory_usage_mb=0.0,
                gpu_usage=0.0,
                network_io={"rx_bytes": 0, "tx_bytes": 0},
                disk_io={"read_bytes": 0, "write_bytes": 0},
                restart_count=0,
                last_error=None
            )
            
            self.logger.info(f"Container {container.id} created for {provider.value}:{model_name}")
            return container.id
            
        except Exception as e:
            self.logger.error(f"Failed to create container for {provider.value}:{model_name}: {e}")
            raise
    
    async def stop_model_container(self, container_id: str) -> bool:
        """Stop a model container"""
        try:
            container = self.docker_client.containers.get(container_id)
            container.stop(timeout=30)
            
            # Free GPU resources
            if container_id in self.container_configs:
                config = self.container_configs[container_id]
                await self._free_gpu_resources(config.gpu_device_ids)
            
            # Update status
            if container_id in self.containers:
                self.containers[container_id].status = "stopped"
            
            self.logger.info(f"Container {container_id} stopped")
            return True
            
        except docker.errors.NotFound:
            self.logger.warning(f"Container {container_id} not found")
            return False
        except Exception as e:
            self.logger.error(f"Failed to stop container {container_id}: {e}")
            return False
    
    async def remove_model_container(self, container_id: str) -> bool:
        """Remove a model container"""
        try:
            # Stop first if running
            await self.stop_model_container(container_id)
            
            # Remove container
            container = self.docker_client.containers.get(container_id)
            container.remove(force=True)
            
            # Clean up tracking
            self.containers.pop(container_id, None)
            self.container_configs.pop(container_id, None)
            
            self.logger.info(f"Container {container_id} removed")
            return True
            
        except docker.errors.NotFound:
            self.logger.warning(f"Container {container_id} not found")
            return False
        except Exception as e:
            self.logger.error(f"Failed to remove container {container_id}: {e}")
            return False
    
    async def scale_model(
        self, 
        provider: ModelProvider, 
        model_name: str, 
        desired_instances: int
    ) -> List[str]:
        """Scale a model to desired number of instances"""
        model_key = f"{provider.value}:{model_name}"
        
        # Find existing containers for this model
        existing_containers = [
            container_id for container_id, status in self.containers.items()
            if status.provider == provider and status.model_name == model_name
        ]
        
        current_instances = len(existing_containers)
        
        self.logger.info(
            f"Scaling {model_key} from {current_instances} to {desired_instances} instances"
        )
        
        if desired_instances > current_instances:
            # Scale up
            new_containers = []
            for _ in range(desired_instances - current_instances):
                try:
                    container_id = await self.create_model_container(provider, model_name)
                    new_containers.append(container_id)
                except Exception as e:
                    self.logger.error(f"Failed to scale up {model_key}: {e}")
                    break
            return existing_containers + new_containers
            
        elif desired_instances < current_instances:
            # Scale down
            containers_to_remove = existing_containers[desired_instances:]
            for container_id in containers_to_remove:
                await self.remove_model_container(container_id)
            return existing_containers[:desired_instances]
        
        return existing_containers
    
    async def get_container_status(self, container_id: str) -> Optional[ContainerStatus]:
        """Get status of a specific container"""
        return self.containers.get(container_id)
    
    async def get_model_containers(
        self, 
        provider: ModelProvider, 
        model_name: str
    ) -> List[ContainerStatus]:
        """Get all containers for a specific model"""
        return [
            status for status in self.containers.values()
            if status.provider == provider and status.model_name == model_name
        ]
    
    async def get_all_containers(self) -> List[ContainerStatus]:
        """Get status of all managed containers"""
        return list(self.containers.values())
    
    async def get_gpu_resources(self) -> List[GPUResource]:
        """Get current GPU resource status"""
        await self._update_gpu_usage()
        return list(self.gpu_resources.values())
    
    async def get_resource_usage(self) -> Dict[str, Any]:
        """Get overall resource usage statistics"""
        total_containers = len(self.containers)
        running_containers = len([c for c in self.containers.values() if c.status == "running"])
        
        total_memory = sum(c.memory_usage_mb for c in self.containers.values())
        total_cpu = sum(c.cpu_usage for c in self.containers.values())
        
        gpu_usage = await self.get_gpu_resources()
        total_gpu_memory = sum(gpu.memory_used_mb for gpu in gpu_usage)
        
        return {
            "total_containers": total_containers,
            "running_containers": running_containers,
            "total_memory_mb": total_memory,
            "total_cpu_percent": total_cpu,
            "total_gpu_memory_mb": total_gpu_memory,
            "gpu_devices": len(self.gpu_resources),
            "network_name": self.network_name,
            "data_directory": str(self.base_data_dir)
        }
    
    async def _generate_container_config(
        self, 
        provider: ModelProvider, 
        model_name: str,
        resource_requirements: Optional[Dict[str, Any]] = None
    ) -> ContainerConfig:
        """Generate container configuration for a model"""
        resource_requirements = resource_requirements or {}
        
        # Base configuration
        container_name = f"llm-{provider.value}-{model_name}-{uuid.uuid4().hex[:8]}"
        
        # Environment variables
        environment = {
            "MODEL_NAME": model_name,
            "PROVIDER": provider.value,
            "PYTHONUNBUFFERED": "1"
        }
        
        # Port mappings
        ports = {"8080/tcp": None}  # Auto-assign host port
        
        # Volume mounts
        model_data_dir = self.base_data_dir / provider.value / model_name
        model_data_dir.mkdir(parents=True, exist_ok=True)
        
        mounts = [
            Mount(
                target="/app/models",
                source=str(model_data_dir),
                type="bind"
            )
        ]
        
        # Resource limits
        memory_limit = resource_requirements.get("memory_mb", 2048)
        cpu_limit = resource_requirements.get("cpu_cores", 1.0)
        gpu_device_ids = resource_requirements.get("gpu_device_ids", [])
        
        # Provider-specific configuration
        if provider == ModelProvider.LOCAL_OLLAMA:
            image = "ollama/ollama:latest"
            environment.update({
                "OLLAMA_HOST": "0.0.0.0",
                "OLLAMA_PORT": "8080"
            })
        elif provider == ModelProvider.LOCAL_HF:
            image = "huggingface/transformers-pytorch-gpu:latest"
            environment.update({
                "HF_HOME": "/app/models",
                "TRANSFORMERS_CACHE": "/app/models"
            })
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Health check
        health_check = {
            "test": ["CMD", "curl", "-f", "http://localhost:8080/health"],
            "interval": 30000000000,  # 30 seconds in nanoseconds
            "timeout": 10000000000,   # 10 seconds
            "retries": 3,
            "start_period": 60000000000  # 60 seconds
        }
        
        # Restart policy
        restart_policy = {"Name": "unless-stopped"}
        
        # Labels
        labels = {
            "llm-model": "true",
            "provider": provider.value,
            "model": model_name,
            "managed-by": "llm-orchestrator"
        }
        
        return ContainerConfig(
            image=image,
            model_name=model_name,
            provider=provider,
            environment=environment,
            ports=ports,
            mounts=mounts,
            memory_limit=f"{memory_limit}m",
            cpu_limit=cpu_limit,
            gpu_device_ids=gpu_device_ids,
            health_check=health_check,
            restart_policy=restart_policy,
            labels=labels
        )
    
    async def _create_container(
        self, 
        config: ContainerConfig, 
        gpu_devices: List[DeviceRequest]
    ) -> Container:
        """Create a Docker container with the given configuration"""
        container_name = f"llm-{config.provider.value}-{config.model_name}-{uuid.uuid4().hex[:8]}"
        
        create_kwargs = {
            "image": config.image,
            "name": container_name,
            "environment": config.environment,
            "ports": config.ports,
            "mounts": config.mounts,
            "mem_limit": config.memory_limit,
            "cpu_period": 100000,
            "cpu_quota": int(config.cpu_limit * 100000),
            "restart_policy": config.restart_policy,
            "labels": config.labels,
            "network": self.network_name,
            "detach": True,
            "healthcheck": config.health_check,
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges"],
            "read_only": True
        }
        
        # Add GPU devices if available
        if gpu_devices:
            create_kwargs["device_requests"] = gpu_devices
        
        return self.docker_client.containers.create(**create_kwargs)
    
    async def _start_container(self, container_id: str):
        """Start a container"""
        container = self.docker_client.containers.get(container_id)
        container.start()
        
        # Wait for container to be ready
        await self._wait_for_container_health(container_id)
    
    async def _wait_for_container_health(self, container_id: str, timeout: int = 300):
        """Wait for a container to become healthy"""
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).seconds < timeout:
            try:
                container = self.docker_client.containers.get(container_id)
                health = container.attrs.get("State", {}).get("Health", {})
                status = health.get("Status", "starting")
                
                if status == "healthy":
                    self.logger.info(f"Container {container_id} is healthy")
                    return
                elif status == "unhealthy":
                    raise Exception(f"Container {container_id} is unhealthy")
                
                await asyncio.sleep(5)
                
            except docker.errors.NotFound:
                raise Exception(f"Container {container_id} not found")
        
        raise Exception(f"Container {container_id} health check timeout")
    
    async def _discover_gpu_resources(self):
        """Discover available GPU resources"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            
            for i, gpu in enumerate(gpus):
                self.gpu_resources[str(i)] = GPUResource(
                    device_id=str(i),
                    name=gpu.name,
                    memory_total_mb=gpu.memoryTotal,
                    memory_used_mb=gpu.memoryUsed,
                    memory_available_mb=gpu.memoryFree,
                    utilization_percent=gpu.load * 100,
                    temperature=gpu.temperature,
                    power_usage=0.0  # GPUtil doesn't provide power info
                )
            
            self.logger.info(f"Discovered {len(self.gpu_resources)} GPU devices")
            
        except ImportError:
            self.logger.warning("GPUtil not available, GPU management disabled")
        except Exception as e:
            self.logger.error(f"Failed to discover GPU resources: {e}")
    
    async def _allocate_gpu_resources(self, requested_devices: List[str]) -> List[DeviceRequest]:
        """Allocate GPU resources for a container"""
        if not requested_devices or not self.gpu_resources:
            return []
        
        device_requests = []
        
        for device_id in requested_devices:
            if device_id in self.gpu_resources:
                gpu = self.gpu_resources[device_id]
                if gpu.allocated_to is None:
                    device_requests.append(
                        DeviceRequest(device_ids=[device_id], capabilities=[["gpu"]])
                    )
                    # Mark as allocated (will be updated with container_id later)
                    gpu.allocated_to = "pending"
                else:
                    self.logger.warning(f"GPU {device_id} already allocated to {gpu.allocated_to}")
        
        return device_requests
    
    async def _free_gpu_resources(self, device_ids: List[str]):
        """Free GPU resources from a container"""
        for device_id in device_ids:
            if device_id in self.gpu_resources:
                self.gpu_resources[device_id].allocated_to = None
    
    async def _update_gpu_usage(self):
        """Update GPU usage statistics"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            
            for i, gpu in enumerate(gpus):
                device_id = str(i)
                if device_id in self.gpu_resources:
                    resource = self.gpu_resources[device_id]
                    resource.memory_used_mb = gpu.memoryUsed
                    resource.memory_available_mb = gpu.memoryFree
                    resource.utilization_percent = gpu.load * 100
                    resource.temperature = gpu.temperature
                    
        except Exception as e:
            self.logger.error(f"Failed to update GPU usage: {e}")
    
    async def _ensure_network(self):
        """Ensure the LLM network exists"""
        try:
            self.docker_client.networks.get(self.network_name)
            self.logger.info(f"Network {self.network_name} already exists")
        except docker.errors.NotFound:
            self.docker_client.networks.create(
                self.network_name,
                driver="bridge",
                labels={"managed-by": "llm-orchestrator"}
            )
            self.logger.info(f"Created network {self.network_name}")
    
    async def _load_existing_containers(self):
        """Load existing managed containers"""
        try:
            containers = self.docker_client.containers.list(
                all=True,
                filters={"label": "llm-model=true"}
            )
            
            for container in containers:
                # Extract metadata from labels
                labels = container.labels
                provider = ModelProvider(labels.get("provider", "local_ollama"))
                model_name = labels.get("model", "unknown")
                
                # Create status object
                self.containers[container.id] = ContainerStatus(
                    container_id=container.id,
                    name=container.name,
                    model_name=model_name,
                    provider=provider,
                    status=container.status,
                    health_status="unknown",
                    created_at=datetime.fromisoformat(container.attrs["Created"].split(".")[0]),
                    started_at=None,
                    cpu_usage=0.0,
                    memory_usage_mb=0.0,
                    gpu_usage=0.0,
                    network_io={"rx_bytes": 0, "tx_bytes": 0},
                    disk_io={"read_bytes": 0, "write_bytes": 0},
                    restart_count=container.attrs["RestartCount"],
                    last_error=None
                )
            
            self.logger.info(f"Loaded {len(self.containers)} existing containers")
            
        except Exception as e:
            self.logger.error(f"Failed to load existing containers: {e}")
    
    async def _monitoring_loop(self):
        """Background task to monitor container status and resource usage"""
        while not self._shutdown_event.is_set():
            try:
                await self._update_container_stats()
                await self._update_gpu_usage()
                await asyncio.sleep(30)  # Update every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)
    
    async def _scaling_loop(self):
        """Background task for automatic scaling decisions"""
        while not self._shutdown_event.is_set():
            try:
                await self._auto_scale_models()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Scaling loop error: {e}")
                await asyncio.sleep(120)
    
    async def _update_container_stats(self):
        """Update container statistics"""
        for container_id, status in self.containers.items():
            try:
                container = self.docker_client.containers.get(container_id)
                
                # Update basic status
                status.status = container.status
                
                # Get health status
                health = container.attrs.get("State", {}).get("Health", {})
                status.health_status = health.get("Status", "unknown")
                
                # Get resource usage stats
                stats = container.stats(stream=False)
                
                # CPU usage
                cpu_stats = stats["cpu_stats"]
                prev_cpu_stats = stats["precpu_stats"]
                
                cpu_delta = cpu_stats["cpu_usage"]["total_usage"] - prev_cpu_stats["cpu_usage"]["total_usage"]
                system_delta = cpu_stats["system_cpu_usage"] - prev_cpu_stats["system_cpu_usage"]
                
                if system_delta > 0:
                    cpu_count = len(cpu_stats["cpu_usage"]["percpu_usage"])
                    status.cpu_usage = (cpu_delta / system_delta) * cpu_count * 100.0
                
                # Memory usage
                memory_stats = stats["memory_stats"]
                if "usage" in memory_stats:
                    status.memory_usage_mb = memory_stats["usage"] / 1024 / 1024
                
                # Network I/O
                networks = stats.get("networks", {})
                total_rx = sum(net["rx_bytes"] for net in networks.values())
                total_tx = sum(net["tx_bytes"] for net in networks.values())
                status.network_io = {"rx_bytes": total_rx, "tx_bytes": total_tx}
                
                # Disk I/O
                blkio_stats = stats.get("blkio_stats", {})
                io_service_bytes = blkio_stats.get("io_service_bytes_recursive", [])
                
                read_bytes = sum(
                    entry["value"] for entry in io_service_bytes 
                    if entry["op"] == "Read"
                )
                write_bytes = sum(
                    entry["value"] for entry in io_service_bytes 
                    if entry["op"] == "Write"
                )
                status.disk_io = {"read_bytes": read_bytes, "write_bytes": write_bytes}
                
            except docker.errors.NotFound:
                # Container was removed
                self.containers.pop(container_id, None)
            except Exception as e:
                self.logger.error(f"Failed to update stats for container {container_id}: {e}")
    
    async def _auto_scale_models(self):
        """Implement automatic scaling logic based on usage patterns"""
        # This is a placeholder for more sophisticated auto-scaling logic
        # You could implement scaling based on:
        # - Request queue length
        # - Response time metrics
        # - CPU/GPU utilization
        # - Time-based patterns
        pass
    
    async def _stop_all_containers(self):
        """Stop all managed containers"""
        for container_id in list(self.containers.keys()):
            await self.stop_model_container(container_id)


# Global instance
_orchestrator: Optional[ContainerOrchestrator] = None


async def get_container_orchestrator() -> ContainerOrchestrator:
    """Get the global container orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ContainerOrchestrator()
        await _orchestrator.initialize()
    return _orchestrator
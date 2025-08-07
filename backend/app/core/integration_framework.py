"""
Core Integration Framework for LLM Platform
Provides extensible plugin architecture and integration management
"""

import asyncio
import importlib
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Type, Union, Callable
from uuid import uuid4
import logging

from pydantic import BaseModel, validator
from app.core.logger import LoggerMixin


class IntegrationStatus(str, Enum):
    """Integration status enumeration"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DISABLED = "disabled"


class IntegrationType(str, Enum):
    """Integration type enumeration"""
    MODEL_PLATFORM = "model_platform"
    DEV_TOOL = "dev_tool"
    MONITORING = "monitoring"
    EXTERNAL_SERVICE = "external_service"
    API_COMPATIBILITY = "api_compatibility"
    PIPELINE = "pipeline"


@dataclass
class IntegrationConfig:
    """Configuration for integrations"""
    name: str
    type: IntegrationType
    enabled: bool = True
    priority: int = 100
    dependencies: List[str] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)
    health_check_interval: int = 60  # seconds
    retry_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_retries": 3,
        "backoff_factor": 2,
        "max_backoff": 300
    })


class IntegrationMetrics(BaseModel):
    """Metrics for integration monitoring"""
    integration_name: str
    status: IntegrationStatus
    last_health_check: datetime
    health_check_count: int = 0
    error_count: int = 0
    success_count: int = 0
    avg_response_time: float = 0.0
    uptime_percentage: float = 100.0
    last_error: Optional[str] = None


class IntegrationEvent(BaseModel):
    """Event emitted by integrations"""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    integration_name: str
    event_type: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseIntegration(ABC, LoggerMixin):
    """Base class for all integrations"""
    
    def __init__(self, config: IntegrationConfig):
        super().__init__()
        self.config = config
        self.status = IntegrationStatus.INITIALIZING
        self.metrics = IntegrationMetrics(
            integration_name=config.name,
            status=self.status,
            last_health_check=datetime.utcnow()
        )
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._event_handlers: List[Callable[[IntegrationEvent], None]] = []
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the integration"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Perform health check"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the integration gracefully"""
        pass
    
    async def start(self) -> bool:
        """Start the integration"""
        try:
            self.logger.info(f"Starting integration: {self.config.name}")
            success = await self.initialize()
            
            if success:
                self.status = IntegrationStatus.ACTIVE
                self._start_health_monitoring()
                self.logger.info(f"Integration {self.config.name} started successfully")
            else:
                self.status = IntegrationStatus.ERROR
                self.logger.error(f"Failed to start integration: {self.config.name}")
            
            return success
            
        except Exception as e:
            self.status = IntegrationStatus.ERROR
            self.metrics.error_count += 1
            self.metrics.last_error = str(e)
            self.logger.error(f"Error starting integration {self.config.name}: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop the integration"""
        self.logger.info(f"Stopping integration: {self.config.name}")
        self._shutdown_event.set()
        
        if self._health_check_task:
            self._health_check_task.cancel()
        
        await self.shutdown()
        self.status = IntegrationStatus.INACTIVE
        self.logger.info(f"Integration {self.config.name} stopped")
    
    def _start_health_monitoring(self) -> None:
        """Start health check monitoring"""
        if self.config.health_check_interval > 0:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def _health_check_loop(self) -> None:
        """Health check monitoring loop"""
        while not self._shutdown_event.is_set():
            try:
                start_time = datetime.utcnow()
                is_healthy = await self.health_check()
                response_time = (datetime.utcnow() - start_time).total_seconds()
                
                self.metrics.last_health_check = datetime.utcnow()
                self.metrics.health_check_count += 1
                
                # Update response time (rolling average)
                if self.metrics.avg_response_time == 0:
                    self.metrics.avg_response_time = response_time
                else:
                    self.metrics.avg_response_time = (
                        self.metrics.avg_response_time * 0.9 + response_time * 0.1
                    )
                
                if is_healthy:
                    if self.status == IntegrationStatus.ERROR:
                        self.status = IntegrationStatus.ACTIVE
                        self._emit_event("health_recovered", {"response_time": response_time})
                    self.metrics.success_count += 1
                else:
                    self.status = IntegrationStatus.ERROR
                    self.metrics.error_count += 1
                    self._emit_event("health_check_failed", {"response_time": response_time})
                
                # Calculate uptime percentage
                total_checks = self.metrics.health_check_count
                successful_checks = self.metrics.success_count
                self.metrics.uptime_percentage = (successful_checks / total_checks) * 100 if total_checks > 0 else 0
                
            except Exception as e:
                self.logger.error(f"Health check error for {self.config.name}: {e}")
                self.metrics.error_count += 1
                self.metrics.last_error = str(e)
                self.status = IntegrationStatus.ERROR
            
            await asyncio.sleep(self.config.health_check_interval)
    
    def _emit_event(self, event_type: str, data: Dict[str, Any] = None) -> None:
        """Emit an integration event"""
        event = IntegrationEvent(
            integration_name=self.config.name,
            event_type=event_type,
            data=data or {},
            metadata={"integration_type": self.config.type.value}
        )
        
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception as e:
                self.logger.error(f"Error in event handler: {e}")
    
    def add_event_handler(self, handler: Callable[[IntegrationEvent], None]) -> None:
        """Add an event handler"""
        self._event_handlers.append(handler)
    
    def remove_event_handler(self, handler: Callable[[IntegrationEvent], None]) -> None:
        """Remove an event handler"""
        if handler in self._event_handlers:
            self._event_handlers.remove(handler)


class IntegrationRegistry:
    """Registry for managing integrations"""
    
    def __init__(self):
        self.integrations: Dict[str, BaseIntegration] = {}
        self.configs: Dict[str, IntegrationConfig] = {}
        self._startup_order: List[str] = []
        self._dependency_graph: Dict[str, List[str]] = {}
        self.logger = logging.getLogger(__name__)
    
    def register(self, integration: BaseIntegration) -> None:
        """Register an integration"""
        name = integration.config.name
        self.integrations[name] = integration
        self.configs[name] = integration.config
        self._update_dependency_graph()
        self.logger.info(f"Registered integration: {name}")
    
    def unregister(self, name: str) -> None:
        """Unregister an integration"""
        if name in self.integrations:
            del self.integrations[name]
            del self.configs[name]
            self._update_dependency_graph()
            self.logger.info(f"Unregistered integration: {name}")
    
    def get_integration(self, name: str) -> Optional[BaseIntegration]:
        """Get an integration by name"""
        return self.integrations.get(name)
    
    def get_integrations_by_type(self, integration_type: IntegrationType) -> List[BaseIntegration]:
        """Get all integrations of a specific type"""
        return [
            integration for integration in self.integrations.values()
            if integration.config.type == integration_type
        ]
    
    def get_active_integrations(self) -> List[BaseIntegration]:
        """Get all active integrations"""
        return [
            integration for integration in self.integrations.values()
            if integration.status == IntegrationStatus.ACTIVE
        ]
    
    def _update_dependency_graph(self) -> None:
        """Update the dependency graph for startup ordering"""
        self._dependency_graph = {}
        for name, config in self.configs.items():
            self._dependency_graph[name] = config.dependencies.copy()
        
        # Calculate startup order using topological sort
        self._startup_order = self._topological_sort()
    
    def _topological_sort(self) -> List[str]:
        """Perform topological sort for dependency resolution"""
        in_degree = {name: 0 for name in self.configs.keys()}
        
        # Calculate in-degrees
        for name, deps in self._dependency_graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] += 1
        
        # Sort by priority and in-degree
        queue = []
        for name in sorted(self.configs.keys(), key=lambda x: (in_degree[x], -self.configs[x].priority)):
            if in_degree[name] == 0:
                queue.append(name)
        
        result = []
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            for neighbor in self._dependency_graph.get(current, []):
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
        
        return result
    
    async def start_all(self) -> Dict[str, bool]:
        """Start all registered integrations in dependency order"""
        results = {}
        
        for name in self._startup_order:
            if name in self.integrations and self.configs[name].enabled:
                integration = self.integrations[name]
                success = await integration.start()
                results[name] = success
                
                if not success:
                    self.logger.error(f"Failed to start integration {name}, continuing with others")
            else:
                self.logger.info(f"Skipping disabled integration: {name}")
                results[name] = False
        
        return results
    
    async def stop_all(self) -> None:
        """Stop all integrations in reverse dependency order"""
        for name in reversed(self._startup_order):
            if name in self.integrations:
                integration = self.integrations[name]
                if integration.status == IntegrationStatus.ACTIVE:
                    await integration.stop()
    
    def get_metrics(self) -> Dict[str, IntegrationMetrics]:
        """Get metrics for all integrations"""
        return {
            name: integration.metrics 
            for name, integration in self.integrations.items()
        }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary"""
        total = len(self.integrations)
        if total == 0:
            return {"status": "no_integrations", "healthy": 0, "total": 0}
        
        healthy = sum(1 for i in self.integrations.values() if i.status == IntegrationStatus.ACTIVE)
        error = sum(1 for i in self.integrations.values() if i.status == IntegrationStatus.ERROR)
        
        overall_status = "healthy" if healthy == total else "degraded" if healthy > 0 else "unhealthy"
        
        return {
            "status": overall_status,
            "healthy": healthy,
            "error": error,
            "total": total,
            "uptime_percentage": (healthy / total) * 100 if total > 0 else 0
        }


class IntegrationManager(LoggerMixin):
    """Main integration manager"""
    
    def __init__(self):
        super().__init__()
        self.registry = IntegrationRegistry()
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the integration manager"""
        self.logger.info("Initializing Integration Manager")
        
        # Auto-discover and register integrations
        await self._discover_integrations()
        
        self._initialized = True
        self.logger.info("Integration Manager initialized")
    
    async def _discover_integrations(self) -> None:
        """Auto-discover integrations from the integrations directory"""
        integrations_dir = Path(__file__).parent.parent / "integrations"
        
        if not integrations_dir.exists():
            self.logger.warning("Integrations directory not found")
            return
        
        for module_file in integrations_dir.glob("*.py"):
            if module_file.name.startswith("_"):
                continue
            
            try:
                module_name = f"app.integrations.{module_file.stem}"
                module = importlib.import_module(module_name)
                
                # Look for integration classes
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseIntegration) and 
                        obj != BaseIntegration):
                        
                        # Look for default configuration
                        config_name = f"DEFAULT_{name.upper()}_CONFIG"
                        if hasattr(module, config_name):
                            config = getattr(module, config_name)
                            integration = obj(config)
                            self.registry.register(integration)
                            self.logger.info(f"Auto-discovered integration: {obj.__name__}")
            
            except Exception as e:
                self.logger.error(f"Failed to load integration from {module_file}: {e}")
    
    async def start_integrations(self) -> Dict[str, bool]:
        """Start all integrations"""
        if not self._initialized:
            await self.initialize()
        
        return await self.registry.start_all()
    
    async def stop_integrations(self) -> None:
        """Stop all integrations"""
        await self.registry.stop_all()
    
    def get_integration(self, name: str) -> Optional[BaseIntegration]:
        """Get an integration by name"""
        return self.registry.get_integration(name)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        return self.registry.get_health_summary()
    
    def get_metrics(self) -> Dict[str, IntegrationMetrics]:
        """Get all integration metrics"""
        return self.registry.get_metrics()


# Global integration manager instance
integration_manager = IntegrationManager()


async def get_integration_manager() -> IntegrationManager:
    """Get the global integration manager"""
    if not integration_manager._initialized:
        await integration_manager.initialize()
    return integration_manager
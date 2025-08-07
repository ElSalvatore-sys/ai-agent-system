"""
Integration Management API Endpoints
Provides REST API for managing and monitoring integrations
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import logging

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

from app.core.integration_framework import (
    get_integration_manager, IntegrationManager, IntegrationStatus, IntegrationType
)
from app.middleware.auth import get_current_user, RequireRole
from app.database.models import User

logger = logging.getLogger(__name__)
router = APIRouter()


class IntegrationStatusResponse(BaseModel):
    """Response model for integration status"""
    name: str
    type: str
    status: str
    enabled: bool
    priority: int
    last_health_check: datetime
    health_check_count: int
    error_count: int
    success_count: int
    uptime_percentage: float
    avg_response_time: float
    last_error: Optional[str] = None


class IntegrationHealthResponse(BaseModel):
    """Response model for overall integration health"""
    status: str
    healthy: int
    error: int
    total: int
    uptime_percentage: float
    timestamp: datetime


class IntegrationMetricsResponse(BaseModel):
    """Response model for integration metrics"""
    integration_name: str
    metrics: Dict[str, Any]
    timestamp: datetime


class IntegrationConfigRequest(BaseModel):
    """Request model for updating integration configuration"""
    enabled: Optional[bool] = None
    settings: Optional[Dict[str, Any]] = None
    health_check_interval: Optional[int] = None


@router.get("/", response_model=List[IntegrationStatusResponse])
async def get_all_integrations(
    current_user: User = Depends(get_current_user),
    integration_manager: IntegrationManager = Depends(get_integration_manager)
):
    """Get status of all integrations"""
    try:
        metrics = integration_manager.get_metrics()
        
        response = []
        for name, metric in metrics.items():
            response.append(IntegrationStatusResponse(
                name=metric.integration_name,
                type=integration_manager.registry.configs[name].type.value,
                status=metric.status.value,
                enabled=integration_manager.registry.configs[name].enabled,
                priority=integration_manager.registry.configs[name].priority,
                last_health_check=metric.last_health_check,
                health_check_count=metric.health_check_count,
                error_count=metric.error_count,
                success_count=metric.success_count,
                uptime_percentage=metric.uptime_percentage,
                avg_response_time=metric.avg_response_time,
                last_error=metric.last_error
            ))
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get integrations status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=IntegrationHealthResponse)
async def get_integrations_health(
    integration_manager: IntegrationManager = Depends(get_integration_manager)
):
    """Get overall health status of all integrations"""
    try:
        health_summary = integration_manager.get_health_status()
        
        return IntegrationHealthResponse(
            status=health_summary["status"],
            healthy=health_summary["healthy"],
            error=health_summary["error"],
            total=health_summary["total"],
            uptime_percentage=health_summary["uptime_percentage"],
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to get integration health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{integration_name}/status", response_model=IntegrationStatusResponse)
async def get_integration_status(
    integration_name: str,
    current_user: User = Depends(get_current_user),
    integration_manager: IntegrationManager = Depends(get_integration_manager)
):
    """Get status of a specific integration"""
    try:
        integration = integration_manager.get_integration(integration_name)
        if not integration:
            raise HTTPException(status_code=404, detail=f"Integration '{integration_name}' not found")
        
        metrics = integration_manager.get_metrics()
        if integration_name not in metrics:
            raise HTTPException(status_code=404, detail=f"No metrics found for integration '{integration_name}'")
        
        metric = metrics[integration_name]
        config = integration_manager.registry.configs[integration_name]
        
        return IntegrationStatusResponse(
            name=metric.integration_name,
            type=config.type.value,
            status=metric.status.value,
            enabled=config.enabled,
            priority=config.priority,
            last_health_check=metric.last_health_check,
            health_check_count=metric.health_check_count,
            error_count=metric.error_count,
            success_count=metric.success_count,
            uptime_percentage=metric.uptime_percentage,
            avg_response_time=metric.avg_response_time,
            last_error=metric.last_error
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get integration status for {integration_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{integration_name}/metrics", response_model=IntegrationMetricsResponse)
async def get_integration_metrics(
    integration_name: str,
    current_user: User = Depends(get_current_user),
    integration_manager: IntegrationManager = Depends(get_integration_manager)
):
    """Get detailed metrics for a specific integration"""
    try:
        integration = integration_manager.get_integration(integration_name)
        if not integration:
            raise HTTPException(status_code=404, detail=f"Integration '{integration_name}' not found")
        
        # Get integration-specific metrics
        metrics = {}
        
        # Add common metrics
        integration_metrics = integration_manager.get_metrics()
        if integration_name in integration_metrics:
            metric = integration_metrics[integration_name]
            metrics.update({
                "status": metric.status.value,
                "health_check_count": metric.health_check_count,
                "error_count": metric.error_count,
                "success_count": metric.success_count,
                "uptime_percentage": metric.uptime_percentage,
                "avg_response_time": metric.avg_response_time,
                "last_health_check": metric.last_health_check.isoformat()
            })
        
        # Add integration-specific metrics if available
        if hasattr(integration, 'get_custom_metrics'):
            custom_metrics = await integration.get_custom_metrics()
            metrics.update(custom_metrics)
        
        return IntegrationMetricsResponse(
            integration_name=integration_name,
            metrics=metrics,
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metrics for integration {integration_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{integration_name}/restart")
@RequireRole("admin")
async def restart_integration(
    integration_name: str,
    current_user: User = Depends(get_current_user),
    integration_manager: IntegrationManager = Depends(get_integration_manager)
):
    """Restart a specific integration"""
    try:
        integration = integration_manager.get_integration(integration_name)
        if not integration:
            raise HTTPException(status_code=404, detail=f"Integration '{integration_name}' not found")
        
        # Stop integration
        await integration.stop()
        logger.info(f"Stopped integration: {integration_name}")
        
        # Start integration
        success = await integration.start()
        if success:
            logger.info(f"Restarted integration: {integration_name}")
            return {"message": f"Integration '{integration_name}' restarted successfully"}
        else:
            logger.error(f"Failed to restart integration: {integration_name}")
            raise HTTPException(status_code=500, detail=f"Failed to restart integration '{integration_name}'")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restart integration {integration_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{integration_name}/config")
@RequireRole("admin")
async def update_integration_config(
    integration_name: str,
    config_update: IntegrationConfigRequest,
    current_user: User = Depends(get_current_user),
    integration_manager: IntegrationManager = Depends(get_integration_manager)
):
    """Update configuration for a specific integration"""
    try:
        integration = integration_manager.get_integration(integration_name)
        if not integration:
            raise HTTPException(status_code=404, detail=f"Integration '{integration_name}' not found")
        
        config = integration_manager.registry.configs[integration_name]
        
        # Update configuration
        updated = False
        if config_update.enabled is not None:
            config.enabled = config_update.enabled
            updated = True
        
        if config_update.settings is not None:
            config.settings.update(config_update.settings)
            updated = True
        
        if config_update.health_check_interval is not None:
            config.health_check_interval = config_update.health_check_interval
            updated = True
        
        if updated:
            # Restart integration to apply new configuration
            await integration.stop()
            success = await integration.start()
            
            if success:
                logger.info(f"Updated configuration for integration: {integration_name}")
                return {"message": f"Configuration updated for integration '{integration_name}'"}
            else:
                logger.error(f"Failed to restart integration after config update: {integration_name}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Configuration updated but failed to restart integration '{integration_name}'"
                )
        else:
            return {"message": "No configuration changes provided"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update config for integration {integration_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/types/{integration_type}")
async def get_integrations_by_type(
    integration_type: str,
    current_user: User = Depends(get_current_user),
    integration_manager: IntegrationManager = Depends(get_integration_manager)
):
    """Get all integrations of a specific type"""
    try:
        # Validate integration type
        try:
            type_enum = IntegrationType(integration_type)
        except ValueError:
            valid_types = [t.value for t in IntegrationType]
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid integration type. Valid types: {valid_types}"
            )
        
        integrations = integration_manager.registry.get_integrations_by_type(type_enum)
        metrics = integration_manager.get_metrics()
        
        response = []
        for integration in integrations:
            name = integration.config.name
            if name in metrics:
                metric = metrics[name]
                response.append(IntegrationStatusResponse(
                    name=metric.integration_name,
                    type=integration.config.type.value,
                    status=metric.status.value,
                    enabled=integration.config.enabled,
                    priority=integration.config.priority,
                    last_health_check=metric.last_health_check,
                    health_check_count=metric.health_check_count,
                    error_count=metric.error_count,
                    success_count=metric.success_count,
                    uptime_percentage=metric.uptime_percentage,
                    avg_response_time=metric.avg_response_time,
                    last_error=metric.last_error
                ))
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get integrations by type {integration_type}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/restart-all")
@RequireRole("admin")
async def restart_all_integrations(
    current_user: User = Depends(get_current_user),
    integration_manager: IntegrationManager = Depends(get_integration_manager)
):
    """Restart all integrations"""
    try:
        # Stop all integrations
        await integration_manager.stop_integrations()
        logger.info("Stopped all integrations")
        
        # Start all integrations
        results = await integration_manager.start_integrations()
        
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        logger.info(f"Restarted {successful}/{total} integrations successfully")
        
        return {
            "message": f"Restarted {successful}/{total} integrations",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Failed to restart all integrations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events")
async def get_integration_events(
    integration_name: Optional[str] = Query(None, description="Filter by integration name"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of events to return"),
    current_user: User = Depends(get_current_user)
):
    """Get recent integration events"""
    try:
        # This would typically query an event store or log aggregation system
        # For now, return a placeholder response
        events = [
            {
                "event_id": "evt_123",
                "integration_name": "ollama",
                "event_type": "health_recovered",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {"response_time": 0.5},
                "metadata": {"integration_type": "model_platform"}
            }
        ]
        
        # Apply filters
        if integration_name:
            events = [e for e in events if e["integration_name"] == integration_name]
        
        if event_type:
            events = [e for e in events if e["event_type"] == event_type]
        
        # Apply limit
        events = events[:limit]
        
        return {
            "events": events,
            "total": len(events),
            "filters": {
                "integration_name": integration_name,
                "event_type": event_type,
                "limit": limit
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get integration events: {e}")
        raise HTTPException(status_code=500, detail=str(e))
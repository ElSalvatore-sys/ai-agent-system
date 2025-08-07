from fastapi import APIRouter
from app.services.system_service import SystemService

router = APIRouter()
system_service = SystemService()

@router.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    return await system_service.get_stats()

@router.get("/health")
async def get_system_health():
    """Get system health status"""
    return await system_service.get_health()

@router.post("/restart")
async def restart_system():
    """Restart the system"""
    return await system_service.restart()

@router.get("/logs")
async def get_system_logs(limit: int = 100):
    """Get system logs"""
    return await system_service.get_logs(limit)
from fastapi import APIRouter
from app.api.endpoints import agents, tasks, system
from app.routes import orchestrator

router = APIRouter()

router.include_router(agents.router, prefix="/agents", tags=["agents"])
router.include_router(tasks.router, prefix="/tasks", tags=["tasks"])
router.include_router(system.router, prefix="/system", tags=["system"])
router.include_router(orchestrator.router, prefix="/orchestrator", tags=["orchestrator"])
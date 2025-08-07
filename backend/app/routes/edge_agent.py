"""Endpoints for edge agent registration and heartbeat"""
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, Request, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.database.database import get_db
from app.database.models import AIModel, ModelProvider
from app.core.logger import LoggerMixin

router = APIRouter(prefix="/edge", tags=["edge"])


class EdgeModelDescriptor(BaseModel):
    provider: ModelProvider
    name: str
    display_name: Optional[str] = None
    capabilities: Optional[List[str]] = []
    context_window: Optional[int] = 4096


class RegisterRequest(BaseModel):
    device_id: str
    models: List[EdgeModelDescriptor]


class HeartbeatRequest(BaseModel):
    device_id: str
    available: bool = True
    model_availability: Optional[Dict[str, bool]] = None  # key = provider:name


class EdgeAgentRoutes(LoggerMixin):
    @staticmethod
    async def _upsert_model(db: AsyncSession, device_id: str, desc: EdgeModelDescriptor):
        provider = desc.provider
        name = desc.name
        stmt = (
            AIModel.__table__.select()
            .where(AIModel.provider == provider, AIModel.name == name)
        )
        res = await db.execute(stmt)
        m: AIModel | None = res.scalar_one_or_none()
        if m is None:
            m = AIModel(
                provider=provider,
                name=name,
                display_name=desc.display_name or name,
                host_type="edge",
                device_id=device_id,
                availability=True,
                supports_streaming=False,
                input_cost=0.0,
                output_cost=0.0,
                created_at=datetime.utcnow(),
            )
            db.add(m)
        else:
            m.device_id = device_id
            m.availability = True
            m.updated_at = datetime.utcnow()
        await db.commit()


@router.post("/register")
async def register_edge_agent(payload: RegisterRequest, db: AsyncSession = Depends(get_db)):
    if not payload.models:
        raise HTTPException(status_code=400, detail="No models provided")
    for m in payload.models:
        await EdgeAgentRoutes._upsert_model(db, payload.device_id, m)
    return {"status": "registered", "models": len(payload.models)}


@router.post("/heartbeat")
async def edge_heartbeat(payload: HeartbeatRequest, db: AsyncSession = Depends(get_db)):
    stmt = AIModel.__table__.update().where(AIModel.device_id == payload.device_id)
    stmt = stmt.values(availability=payload.available, updated_at=datetime.utcnow())
    await db.execute(stmt)
    await db.commit()
    return {"status": "ok"}

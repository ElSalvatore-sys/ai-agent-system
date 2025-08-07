from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, and_, func
from pydantic import BaseModel, Field
from datetime import datetime

from app.database.database import get_db
from app.database.models import Agent, Task, User, TaskStatus
from app.core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

# Pydantic models
class AgentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)
    system_prompt: Optional[str] = Field(None, max_length=5000)
    model_provider: str = Field("openai")
    model_name: str = Field("gpt-4")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(2048, ge=1, le=8192)
    is_public: bool = Field(False)
    capabilities: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

class AgentUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)
    system_prompt: Optional[str] = Field(None, max_length=5000)
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=8192)
    is_active: Optional[bool] = None
    is_public: Optional[bool] = None
    capabilities: Optional[List[str]] = None
    tags: Optional[List[str]] = None

class AgentResponse(BaseModel):
    id: int
    uuid: str
    name: str
    description: Optional[str]
    system_prompt: Optional[str]
    model_provider: str
    model_name: str
    temperature: float
    max_tokens: int
    is_active: bool
    is_public: bool
    capabilities: List[str]
    tags: List[str]
    created_at: datetime
    updated_at: Optional[datetime]
    task_count: int = 0
    active_task_count: int = 0

@router.get("/", response_model=List[AgentResponse])
async def get_agents(
    request: Request,
    db: AsyncSession = Depends(get_db),
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0),
    search: Optional[str] = None,
    is_public: Optional[bool] = None,
    is_active: Optional[bool] = None
):
    """Get agents with filtering options"""
    
    if not hasattr(request.state, 'user'):
        raise HTTPException(status_code=401, detail="Authentication required")
    
    user = request.state.user
    
    try:
        # Build query
        stmt = select(
            Agent,
            func.count(func.distinct(Task.id)).label('task_count'),
            func.count(func.distinct(
                func.case((Task.status.in_([TaskStatus.PENDING, TaskStatus.RUNNING]), Task.id))
            )).label('active_task_count')
        ).outerjoin(Task)
        
        # Filter by user's agents or public agents
        stmt = stmt.where(Agent.is_public == True)
        
        # Apply filters
        if search:
            stmt = stmt.where(
                func.or_(
                    Agent.name.ilike(f"%{search}%"),
                    Agent.description.ilike(f"%{search}%")
                )
            )
        
        if is_public is not None:
            stmt = stmt.where(Agent.is_public == is_public)
        
        if is_active is not None:
            stmt = stmt.where(Agent.is_active == is_active)
        
        stmt = stmt.group_by(Agent.id).order_by(desc(Agent.updated_at)).limit(limit).offset(offset)
        
        result = await db.execute(stmt)
        agents_data = result.all()
        
        return [
            AgentResponse(
                id=agent.id,
                uuid=agent.uuid,
                name=agent.name,
                description=agent.description,
                system_prompt=agent.system_prompt,
                model_provider=agent.model_provider.value,
                model_name=agent.model_name,
                temperature=agent.temperature,
                max_tokens=agent.max_tokens,
                is_active=agent.is_active,
                is_public=agent.is_public,
                capabilities=agent.capabilities or [],
                tags=agent.tags or [],
                created_at=agent.created_at,
                updated_at=agent.updated_at,
                task_count=task_count or 0,
                active_task_count=active_task_count or 0
            )
            for agent, task_count, active_task_count in agents_data
        ]
        
    except Exception as e:
        logger.error(f"Failed to get agents: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve agents")

@router.post("/", response_model=AgentResponse)
async def create_agent(
    agent_data: AgentCreate,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """Create a new agent"""
    
    if not hasattr(request.state, 'user'):
        raise HTTPException(status_code=401, detail="Authentication required")
    
    user = request.state.user
    
    try:
        from app.database.models import ModelProvider
        
        # Validate model provider
        try:
            model_provider = ModelProvider(agent_data.model_provider)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid model provider")
        
        # Create agent
        agent = Agent(
            name=agent_data.name,
            description=agent_data.description,
            system_prompt=agent_data.system_prompt,
            model_provider=model_provider,
            model_name=agent_data.model_name,
            temperature=agent_data.temperature,
            max_tokens=agent_data.max_tokens,
            is_public=agent_data.is_public,
            capabilities=agent_data.capabilities,
            tags=agent_data.tags,
            is_active=True
        )
        
        db.add(agent)
        await db.commit()
        await db.refresh(agent)
        
        logger.info(f"Agent created: {agent.name} by user {user.username}")
        
        return AgentResponse(
            id=agent.id,
            uuid=agent.uuid,
            name=agent.name,
            description=agent.description,
            system_prompt=agent.system_prompt,
            model_provider=agent.model_provider.value,
            model_name=agent.model_name,
            temperature=agent.temperature,
            max_tokens=agent.max_tokens,
            is_active=agent.is_active,
            is_public=agent.is_public,
            capabilities=agent.capabilities or [],
            tags=agent.tags or [],
            created_at=agent.created_at,
            updated_at=agent.updated_at,
            task_count=0,
            active_task_count=0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to create agent: {e}")
        raise HTTPException(status_code=500, detail="Failed to create agent")
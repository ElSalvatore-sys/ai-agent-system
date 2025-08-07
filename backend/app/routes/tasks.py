from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Request, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, and_, func
from pydantic import BaseModel, Field
from datetime import datetime

from app.database.database import get_db
from app.database.models import Task, Agent, User, TaskStatus
from app.core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

# Pydantic models
class TaskCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    agent_id: Optional[int] = None
    priority: int = Field(0, ge=0, le=10)
    input_data: Dict[str, Any] = Field(default_factory=dict)
    estimated_duration: Optional[int] = Field(None, ge=1)  # seconds

class TaskResponse(BaseModel):
    id: int
    uuid: str
    title: str
    description: Optional[str]
    status: str
    priority: int
    agent_id: Optional[int]
    agent_name: Optional[str]
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    error_message: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    estimated_duration: Optional[int]
    actual_duration: Optional[int]

@router.get("/", response_model=List[TaskResponse])
async def get_tasks(
    request: Request,
    db: AsyncSession = Depends(get_db),
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0),
    status: Optional[TaskStatus] = None,
    agent_id: Optional[int] = None
):
    """Get tasks with filtering options"""
    
    if not hasattr(request.state, 'user'):
        raise HTTPException(status_code=401, detail="Authentication required")
    
    user = request.state.user
    
    try:
        # Build query with join to get agent name
        stmt = select(Task, Agent.name.label('agent_name')).outerjoin(Agent, Task.agent_id == Agent.id)
        
        # Filter by user's tasks
        stmt = stmt.where(Task.user_id == user.id)
        
        # Apply filters
        if status:
            stmt = stmt.where(Task.status == status)
        
        if agent_id:
            stmt = stmt.where(Task.agent_id == agent_id)
        
        stmt = stmt.order_by(desc(Task.created_at)).limit(limit).offset(offset)
        
        result = await db.execute(stmt)
        tasks_data = result.all()
        
        return [
            TaskResponse(
                id=task.id,
                uuid=task.uuid,
                title=task.title,
                description=task.description,
                status=task.status.value,
                priority=task.priority,
                agent_id=task.agent_id,
                agent_name=agent_name,
                input_data=task.input_data or {},
                output_data=task.output_data or {},
                error_message=task.error_message,
                created_at=task.created_at,
                updated_at=task.updated_at,
                started_at=task.started_at,
                completed_at=task.completed_at,
                estimated_duration=task.estimated_duration,
                actual_duration=task.actual_duration
            )
            for task, agent_name in tasks_data
        ]
        
    except Exception as e:
        logger.error(f"Failed to get tasks: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve tasks")

@router.post("/", response_model=TaskResponse)
async def create_task(
    task_data: TaskCreate,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """Create a new task"""
    
    if not hasattr(request.state, 'user'):
        raise HTTPException(status_code=401, detail="Authentication required")
    
    user = request.state.user
    
    try:
        # Create task
        task = Task(
            title=task_data.title,
            description=task_data.description,
            agent_id=task_data.agent_id,
            user_id=user.id,
            priority=task_data.priority,
            input_data=task_data.input_data,
            estimated_duration=task_data.estimated_duration,
            status=TaskStatus.PENDING
        )
        
        db.add(task)
        await db.commit()
        await db.refresh(task)
        
        logger.info(f"Task created: {task.title} by user {user.username}")
        
        return TaskResponse(
            id=task.id,
            uuid=task.uuid,
            title=task.title,
            description=task.description,
            status=task.status.value,
            priority=task.priority,
            agent_id=task.agent_id,
            agent_name=None,
            input_data=task.input_data or {},
            output_data=task.output_data or {},
            error_message=task.error_message,
            created_at=task.created_at,
            updated_at=task.updated_at,
            started_at=task.started_at,
            completed_at=task.completed_at,
            estimated_duration=task.estimated_duration,
            actual_duration=task.actual_duration
        )
        
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to create task: {e}")
        raise HTTPException(status_code=500, detail="Failed to create task")
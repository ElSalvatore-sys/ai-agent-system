from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from enum import Enum

class AgentStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class AgentType(str, Enum):
    PROCESSING = "processing"
    ANALYSIS = "analysis"
    AUTOMATION = "automation"
    MONITORING = "monitoring"

class AgentBase(BaseModel):
    name: str
    description: Optional[str] = None
    type: AgentType
    config: Optional[dict] = None

class AgentCreate(AgentBase):
    pass

class AgentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    type: Optional[AgentType] = None
    status: Optional[AgentStatus] = None
    config: Optional[dict] = None

class Agent(AgentBase):
    id: int
    status: AgentStatus = AgentStatus.INACTIVE
    created_at: datetime
    updated_at: Optional[datetime] = None
    last_active: Optional[datetime] = None

    class Config:
        from_attributes = True
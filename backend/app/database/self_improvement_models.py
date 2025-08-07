from __future__ import annotations

"""Database models that power the autonomous self-improvement engine.

Adding them in a separate module avoids making the already very large
`database/models.py` file even bigger while still sharing the global
SQLAlchemy `Base` instance so that Alembic can discover the tables
automatically.

NOTE: Remember to include this module in Alembic's `target_metadata`
configuration (env.py) so the new tables are picked up by `alembic
revision --autogenerate`.
"""

import enum
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import Column, DateTime, Enum as SQLEnum, Integer, String, Text, JSON
from sqlalchemy.sql import func

from app.database.database import Base


class ImprovementTaskStatus(str, enum.Enum):
    """Lifecycle stages of an autonomous improvement task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ImprovementTaskType(str, enum.Enum):
    """High-level categories of self-improvement actions.

    Mirror the capability groups described in design docs so that the
    planner can easily decide which executor to invoke.
    """

    ANALYSIS = "analysis"  # Generate new insights from telemetry
    CODE_OPTIMISATION = "code_optimisation"
    MODEL_ROUTING_UPDATE = "model_routing_update"
    KNOWLEDGE_BASE_UPDATE = "knowledge_base_update"
    ARCHITECTURE_EVOLUTION = "architecture_evolution"


class ImprovementTask(Base):
    __tablename__ = "improvement_tasks"

    id = Column(Integer, primary_key=True, index=True)
    task_type = Column(SQLEnum(ImprovementTaskType), nullable=False)
    status = Column(SQLEnum(ImprovementTaskStatus), default=ImprovementTaskStatus.PENDING, nullable=False)

    # Arbitrary JSON payload – schema depends on task_type
    payload = Column(JSON, default=dict)

    # Metrics snapshot before & after task execution (optional)
    metrics_before = Column(JSON, default=dict)
    metrics_after = Column(JSON, default=dict)

    error_message = Column(Text)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ImprovementTask id={self.id} type={self.task_type} status={self.status}>"


class PromptPattern(Base):
    """Frequent prompt structure mined from successful interactions."""

    __tablename__ = "prompt_patterns"

    id = Column(Integer, primary_key=True, index=True)

    pattern = Column(String(512), nullable=False, unique=True)
    support = Column(Integer, default=0)  # Number of occurrences
    confidence = Column(Integer, default=0)  # Success rate percentage
    success_delta = Column(Integer, default=0)  # vs global baseline (percentage points)

    last_seen = Column(DateTime(timezone=True), server_default=func.now())

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self) -> str:  # pragma: no cover
        return f"<PromptPattern pattern='{self.pattern[:30]}...' support={self.support}>"
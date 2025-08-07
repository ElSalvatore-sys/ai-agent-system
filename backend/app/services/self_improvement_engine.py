from __future__ import annotations

"""Autonomous Self-Improvement Engine
-------------------------------------

This module hosts the *minimum viable* skeleton for the self-improving
capabilities described in the architectural blueprint.  It purposefully
keeps the initial implementation lightweight while providing clear hook
points (TODO markers) where future logic can be plugged in.

The engine spins up three cooperative background loops:

• *Analyzer* – crunches raw telemetry and stores high-level findings.
• *Planner*  – converts findings into actionable ``ImprovementTask``s.
• *Executor* – executes one task at a time, applies changes, validates
  and records the outcome.

Each loop is intentionally throttled (``asyncio.sleep``) so that the
extra load on your production instance is negligible until the advanced
analytics are added.

Add ``await SelfImprovementEngine.get_instance().start()`` in the
application lifespan to activate the loops.
"""

import asyncio
import logging
from contextlib import suppress
from datetime import datetime
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from app.core.logger import LoggerMixin, get_logger
from app.database.database import get_db
from app.database.self_improvement_models import (
    ImprovementTask, ImprovementTaskStatus, ImprovementTaskType,
)


class SelfImprovementEngine(LoggerMixin):
    """Singleton orchestrating analysis, planning and execution."""

    _instance: Optional["SelfImprovementEngine"] = None

    ANALYZER_PERIOD = 60         # seconds – run every minute
    PLANNER_PERIOD = 30          # seconds
    EXECUTOR_BACKOFF = 5         # seconds – idle wait when no tasks

    def __init__(self) -> None:
        self._running = False
        self._analyzer_task: Optional[asyncio.Task] = None
        self._planner_task: Optional[asyncio.Task] = None
        self._executor_task: Optional[asyncio.Task] = None
        # Lazily initialised DB session generator
        self._db_gen = get_db()
        self.logger = get_logger(self.__class__.__name__)

    # ---------------------------------------------------------------------
    # Lifecycle helpers
    # ---------------------------------------------------------------------
    @classmethod
    def get_instance(cls) -> "SelfImprovementEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def start(self) -> None:
        if self._running:
            return
        self.logger.info("Starting Self-Improvement Engine…")
        self._running = True
        self._analyzer_task = asyncio.create_task(self._run_analyzer_loop(), name="SI-Analyzer")
        self._planner_task = asyncio.create_task(self._run_planner_loop(), name="SI-Planner")
        self._executor_task = asyncio.create_task(self._run_executor_loop(), name="SI-Executor")

    async def stop(self) -> None:
        if not self._running:
            return
        self.logger.info("Stopping Self-Improvement Engine…")
        self._running = False
        for task in (self._analyzer_task, self._planner_task, self._executor_task):
            if task:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    # ------------------------------------------------------------------
    # Analyzer – turn raw telemetry into findings (TODO)
    # ------------------------------------------------------------------
    async def _run_analyzer_loop(self) -> None:
        while self._running:
            try:
                await self._analyze_metrics()
            except Exception as exc:  # pragma: no cover – best-effort resilience
                self.logger.error(f"Analyzer error: {exc}", exc_info=True)
            await asyncio.sleep(self.ANALYZER_PERIOD)

    async def _analyze_metrics(self) -> None:
        """Placeholder – aggregate metrics & persist findings.

        In the full implementation this function will:
        • Query AIRequest, UsageLog, CostTracking tables.
        • Compute failure probability, cost efficiency etc.
        • Write results to PerformanceMetrics or a dedicated findings table.
        """
        self.logger.debug("Analyzing telemetry… (stub)")
        # TODO: real metrics crunching

    # ------------------------------------------------------------------
    # Planner – convert findings → ImprovementTask records
    # ------------------------------------------------------------------
    async def _run_planner_loop(self) -> None:
        while self._running:
            try:
                await self._plan_tasks()
            except Exception as exc:
                self.logger.error(f"Planner error: {exc}", exc_info=True)
            await asyncio.sleep(self.PLANNER_PERIOD)

    async def _plan_tasks(self) -> None:
        """Dummy planner inserts a heartbeat task every hour (demo purpose)."""
        async for db in self._db_gen:
            # Insert demo task at :00 of every hour if none exists
            now = datetime.utcnow()
            if now.minute == 0 and now.second < self.PLANNER_PERIOD:
                exists = await db.scalar(select(ImprovementTask.id).where(
                    ImprovementTask.task_type == ImprovementTaskType.ANALYSIS,
                    ImprovementTask.status == ImprovementTaskStatus.PENDING,
                ))
                if not exists:
                    db.add(ImprovementTask(task_type=ImprovementTaskType.ANALYSIS))
                    await db.commit()
                    self.logger.info("Queued hourly ANALYSIS task")
            break  # Exit generator context

    # ------------------------------------------------------------------
    # Executor – pick next task, run & record outcome
    # ------------------------------------------------------------------
    async def _run_executor_loop(self) -> None:
        while self._running:
            try:
                worked = await self._execute_next_task()
                if not worked:
                    await asyncio.sleep(self.EXECUTOR_BACKOFF)
            except Exception as exc:
                self.logger.error(f"Executor error: {exc}", exc_info=True)
                await asyncio.sleep(self.EXECUTOR_BACKOFF)

    async def _execute_next_task(self) -> bool:
        """Fetch oldest pending task and process it.

        Returns True if a task was processed, False if queue empty.
        """
        async for db in self._db_gen:
            task: ImprovementTask | None = await db.scalar(
                select(ImprovementTask).where(
                    ImprovementTask.status == ImprovementTaskStatus.PENDING
                ).order_by(ImprovementTask.created_at.asc())
            )
            if not task:
                return False

            self.logger.info("Executing task %s", task)
            task.status = ImprovementTaskStatus.IN_PROGRESS
            task.started_at = datetime.utcnow()
            await db.commit()
            await db.refresh(task)

            try:
                # Dispatch based on task_type (stub)
                if task.task_type == ImprovementTaskType.ANALYSIS:
                    await self._exec_analysis_task(task)
                elif task.task_type == ImprovementTaskType.CODE_OPTIMISATION:
                    await self._exec_code_opt_task(task)
                # TODO: other task types
                task.status = ImprovementTaskStatus.COMPLETED
            except Exception as exec_err:
                task.status = ImprovementTaskStatus.FAILED
                task.error_message = str(exec_err)
                self.logger.error("Task %s failed: %s", task.id, exec_err, exc_info=True)
            finally:
                task.completed_at = datetime.utcnow()
                await db.commit()
                return True  # processed exactly one task

    # ------------------------------------------------------------------
    # Executors for different task categories (stubs)
    # ------------------------------------------------------------------
    async def _exec_analysis_task(self, task: ImprovementTask) -> None:  # noqa: D401
        """Analyse stored telemetry and write findings (stub)."""
        await asyncio.sleep(0.1)  # simulate work

    async def _exec_code_opt_task(self, task: ImprovementTask) -> None:  # noqa: D401
        """Optimise code automatically (stub)."""
        await asyncio.sleep(0.1)

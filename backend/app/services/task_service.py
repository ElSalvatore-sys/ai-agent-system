from typing import List, Optional
from datetime import datetime
from app.models.task import Task, TaskCreate, TaskUpdate, TaskStatus, TaskPriority

class TaskService:
    def __init__(self):
        self._tasks = [
            Task(
                id=1,
                title="Process user data",
                description="Process incoming user registration data",
                agent_id=1,
                priority=TaskPriority.HIGH,
                status=TaskStatus.COMPLETED,
                created_at=datetime.now(),
                completed_at=datetime.now()
            ),
            Task(
                id=2,
                title="Analyze sentiment",
                description="Analyze sentiment of user feedback",
                agent_id=2,
                priority=TaskPriority.MEDIUM,
                status=TaskStatus.RUNNING,
                created_at=datetime.now(),
                started_at=datetime.now()
            ),
            Task(
                id=3,
                title="Schedule backup",
                description="Schedule daily database backup",
                agent_id=3,
                priority=TaskPriority.LOW,
                status=TaskStatus.PENDING,
                created_at=datetime.now()
            )
        ]
        self._next_id = 4

    async def get_all_tasks(self) -> List[Task]:
        return self._tasks

    async def get_task(self, task_id: int) -> Optional[Task]:
        return next((task for task in self._tasks if task.id == task_id), None)

    async def create_task(self, task_create: TaskCreate) -> Task:
        task = Task(
            id=self._next_id,
            title=task_create.title,
            description=task_create.description,
            agent_id=task_create.agent_id,
            priority=task_create.priority,
            params=task_create.params,
            status=TaskStatus.PENDING,
            created_at=datetime.now()
        )
        self._tasks.append(task)
        self._next_id += 1
        return task

    async def update_task(self, task_id: int, task_update: TaskUpdate) -> Optional[Task]:
        task = await self.get_task(task_id)
        if not task:
            return None
        
        update_data = task_update.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(task, field, value)
        
        task.updated_at = datetime.now()
        
        if task_update.status == TaskStatus.RUNNING and not task.started_at:
            task.started_at = datetime.now()
        elif task_update.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            task.completed_at = datetime.now()
            
        return task

    async def delete_task(self, task_id: int) -> bool:
        task = await self.get_task(task_id)
        if not task:
            return False
        
        self._tasks.remove(task)
        return True
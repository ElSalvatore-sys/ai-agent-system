from fastapi import APIRouter, HTTPException
from typing import List
from app.models.task import Task, TaskCreate, TaskUpdate
from app.services.task_service import TaskService

router = APIRouter()
task_service = TaskService()

@router.get("/", response_model=List[Task])
async def get_tasks():
    """Get all tasks"""
    return await task_service.get_all_tasks()

@router.get("/{task_id}", response_model=Task)
async def get_task(task_id: int):
    """Get a specific task by ID"""
    task = await task_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@router.post("/", response_model=Task)
async def create_task(task: TaskCreate):
    """Create a new task"""
    return await task_service.create_task(task)

@router.put("/{task_id}", response_model=Task)
async def update_task(task_id: int, task_update: TaskUpdate):
    """Update an existing task"""
    task = await task_service.update_task(task_id, task_update)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@router.delete("/{task_id}")
async def delete_task(task_id: int):
    """Delete a task"""
    success = await task_service.delete_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"message": "Task deleted successfully"}
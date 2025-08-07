from fastapi import APIRouter, HTTPException
from typing import List
from app.models.agent import Agent, AgentCreate, AgentUpdate
from app.services.agent_service import AgentService

router = APIRouter()
agent_service = AgentService()

@router.get("/", response_model=List[Agent])
async def get_agents():
    """Get all agents"""
    return await agent_service.get_all_agents()

@router.get("/{agent_id}", response_model=Agent)
async def get_agent(agent_id: int):
    """Get a specific agent by ID"""
    agent = await agent_service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@router.post("/", response_model=Agent)
async def create_agent(agent: AgentCreate):
    """Create a new agent"""
    return await agent_service.create_agent(agent)

@router.put("/{agent_id}", response_model=Agent)
async def update_agent(agent_id: int, agent_update: AgentUpdate):
    """Update an existing agent"""
    agent = await agent_service.update_agent(agent_id, agent_update)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@router.delete("/{agent_id}")
async def delete_agent(agent_id: int):
    """Delete an agent"""
    success = await agent_service.delete_agent(agent_id)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"message": "Agent deleted successfully"}

@router.post("/{agent_id}/start")
async def start_agent(agent_id: int):
    """Start an agent"""
    success = await agent_service.start_agent(agent_id)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"message": "Agent started successfully"}

@router.post("/{agent_id}/stop")
async def stop_agent(agent_id: int):
    """Stop an agent"""
    success = await agent_service.stop_agent(agent_id)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"message": "Agent stopped successfully"}
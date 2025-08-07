from typing import List, Optional
from datetime import datetime
from app.models.agent import Agent, AgentCreate, AgentUpdate, AgentStatus

class AgentService:
    def __init__(self):
        self._agents = [
            Agent(
                id=1,
                name="Data Processor",
                description="Processes incoming data streams",
                type="processing",
                status=AgentStatus.ACTIVE,
                created_at=datetime.now(),
                last_active=datetime.now()
            ),
            Agent(
                id=2,
                name="Content Analyzer",
                description="Analyzes content for insights",
                type="analysis",
                status=AgentStatus.INACTIVE,
                created_at=datetime.now(),
                last_active=datetime.now()
            ),
            Agent(
                id=3,
                name="Task Scheduler",
                description="Schedules and manages tasks",
                type="automation",
                status=AgentStatus.ACTIVE,
                created_at=datetime.now(),
                last_active=datetime.now()
            )
        ]
        self._next_id = 4

    async def get_all_agents(self) -> List[Agent]:
        return self._agents

    async def get_agent(self, agent_id: int) -> Optional[Agent]:
        return next((agent for agent in self._agents if agent.id == agent_id), None)

    async def create_agent(self, agent_create: AgentCreate) -> Agent:
        agent = Agent(
            id=self._next_id,
            name=agent_create.name,
            description=agent_create.description,
            type=agent_create.type,
            config=agent_create.config,
            status=AgentStatus.INACTIVE,
            created_at=datetime.now()
        )
        self._agents.append(agent)
        self._next_id += 1
        return agent

    async def update_agent(self, agent_id: int, agent_update: AgentUpdate) -> Optional[Agent]:
        agent = await self.get_agent(agent_id)
        if not agent:
            return None
        
        update_data = agent_update.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(agent, field, value)
        
        agent.updated_at = datetime.now()
        return agent

    async def delete_agent(self, agent_id: int) -> bool:
        agent = await self.get_agent(agent_id)
        if not agent:
            return False
        
        self._agents.remove(agent)
        return True

    async def start_agent(self, agent_id: int) -> bool:
        agent = await self.get_agent(agent_id)
        if not agent:
            return False
        
        agent.status = AgentStatus.ACTIVE
        agent.last_active = datetime.now()
        return True

    async def stop_agent(self, agent_id: int) -> bool:
        agent = await self.get_agent(agent_id)
        if not agent:
            return False
        
        agent.status = AgentStatus.INACTIVE
        return True
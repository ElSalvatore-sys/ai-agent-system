from datetime import datetime
from typing import Dict, Any, List

class SystemService:
    def __init__(self):
        self._logs = [
            {"timestamp": datetime.now(), "level": "INFO", "message": "System started"},
            {"timestamp": datetime.now(), "level": "INFO", "message": "Agent service initialized"},
            {"timestamp": datetime.now(), "level": "INFO", "message": "Task service initialized"},
        ]

    async def get_stats(self) -> Dict[str, Any]:
        return {
            "active_agents": 2,
            "total_agents": 3,
            "pending_tasks": 1,
            "completed_tasks": 1,
            "running_tasks": 1,
            "total_tasks": 3,
            "uptime": "2h 34m",
            "system_load": 0.65,
            "memory_usage": 0.42,
            "disk_usage": 0.23
        }

    async def get_health(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "timestamp": datetime.now(),
            "services": {
                "api": {"status": "up", "response_time": "12ms"},
                "database": {"status": "up", "response_time": "3ms"},
                "redis": {"status": "up", "response_time": "1ms"},
                "agents": {"status": "up", "active_count": 2}
            }
        }

    async def restart(self) -> Dict[str, str]:
        self._logs.append({
            "timestamp": datetime.now(),
            "level": "WARNING",
            "message": "System restart initiated"
        })
        return {"message": "System restart initiated"}

    async def get_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self._logs[-limit:]
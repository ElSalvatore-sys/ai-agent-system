@echo off
echo Stopping LLM Optimization System
echo =================================

echo Stopping all services...
docker-compose -f docker-compose.local.yml -f docker-compose.gpu.yml down

echo Cleaning up unused containers and networks...
docker system prune -f --filter "until=24h"

echo.
echo All services stopped successfully!
echo Data volumes are preserved for next startup.
echo.
echo To completely remove all data, run:
echo docker-compose -f docker-compose.local.yml down -v
echo.
pause
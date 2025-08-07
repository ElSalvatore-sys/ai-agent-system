@echo off
echo Starting LLM Optimization System (Local Development)
echo ================================================

echo Checking if Docker is running...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

echo Pulling latest images...
docker-compose -f docker-compose.local.yml pull

echo Starting services...
docker-compose -f docker-compose.local.yml up -d

echo Waiting for services to be ready...
timeout /t 30 /nobreak >nul

echo Checking service health...
docker-compose -f docker-compose.local.yml ps

echo.
echo Services are starting up! Access points:
echo ==========================================
echo Frontend:        http://localhost:7000
echo Backend API:     http://localhost:8000
echo API Docs:        http://localhost:8000/docs
echo Monitoring:      http://localhost:9090
echo Task Monitor:    http://localhost:5555 (admin/admin123)
echo.
echo Database:        localhost:5432 (postgres/postgres123)
echo Redis:           localhost:6379
echo Ollama:          localhost:11434
echo.

echo Downloading initial models (this may take a while)...
timeout /t 10 /nobreak >nul
docker exec llm-ollama ollama pull llama2:7b
docker exec llm-ollama ollama pull codellama:7b

echo.
echo Setup complete! Press any key to view logs...
pause >nul

docker-compose -f docker-compose.local.yml logs -f
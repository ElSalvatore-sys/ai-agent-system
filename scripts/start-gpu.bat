@echo off
echo Starting LLM Optimization System with GPU Support
echo ================================================

echo Checking if Docker is running...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

echo Checking NVIDIA GPU support...
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo Warning: NVIDIA GPU not detected or drivers not installed.
    echo Falling back to CPU-only mode...
    call start-local.bat
    exit /b 0
)

echo NVIDIA GPU detected! Starting with GPU acceleration...
docker-compose -f docker-compose.local.yml -f docker-compose.gpu.yml pull

echo Starting GPU-enabled services...
docker-compose -f docker-compose.local.yml -f docker-compose.gpu.yml up -d

echo Waiting for services to be ready...
timeout /t 30 /nobreak >nul

echo Checking service health...
docker-compose -f docker-compose.local.yml -f docker-compose.gpu.yml ps

echo.
echo GPU-enabled services are starting up! Access points:
echo ===================================================
echo Frontend:        http://localhost:7000
echo Backend API:     http://localhost:8000
echo API Docs:        http://localhost:8000/docs
echo Monitoring:      http://localhost:9090
echo Task Monitor:    http://localhost:5555 (admin/admin123)
echo.

echo Downloading GPU-optimized models...
timeout /t 10 /nobreak >nul
docker exec llm-ollama ollama pull llama2:7b
docker exec llm-ollama ollama pull codellama:7b
docker exec llm-ollama ollama pull mistral:7b

echo.
echo GPU setup complete! Press any key to view logs...
pause >nul

docker-compose -f docker-compose.local.yml -f docker-compose.gpu.yml logs -f
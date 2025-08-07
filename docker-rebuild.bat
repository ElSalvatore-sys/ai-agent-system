@echo off
echo ========================================
echo    AI Agent System - Docker Rebuild
echo ========================================
echo.

echo 🔄 Stopping existing containers...
docker-compose down

echo 🧹 Cleaning up Docker cache...
docker system prune -f

echo 🔨 Rebuilding frontend container with all dependencies...
docker-compose build --no-cache frontend

echo 🔨 Rebuilding backend container...
docker-compose build --no-cache backend

echo 🚀 Starting all services...
docker-compose up -d

echo.
echo ⏳ Waiting for services to start...
timeout /t 30 /nobreak > nul

echo.
echo ========================================
echo    System Status Check
echo ========================================

echo Checking Backend...
powershell -Command "try { $response = Invoke-WebRequest -Uri 'http://localhost:8000/health' -UseBasicParsing -TimeoutSec 5; if ($response.StatusCode -eq 200) { Write-Host '✅ Backend: RUNNING' -ForegroundColor Green } else { Write-Host '❌ Backend: ERROR' -ForegroundColor Red } } catch { Write-Host '❌ Backend: NOT RESPONDING' -ForegroundColor Red }"

echo Checking Frontend...
powershell -Command "try { $response = Invoke-WebRequest -Uri 'http://localhost:8080' -UseBasicParsing -TimeoutSec 5; if ($response.StatusCode -eq 200) { Write-Host '✅ Frontend: RUNNING' -ForegroundColor Green } else { Write-Host '❌ Frontend: ERROR' -ForegroundColor Red } } catch { Write-Host '❌ Frontend: NOT RESPONDING' -ForegroundColor Red }"

echo.
echo ========================================
echo    Access URLs
echo ========================================
echo 🌐 Frontend Application: http://localhost:8080
echo 📊 Backend API: http://localhost:8000
echo 📚 API Documentation: http://localhost:8000/docs
echo 📈 Prometheus: http://localhost:9090
echo 📊 Grafana: http://localhost:3001
echo.

echo ========================================
echo    Advanced Features Available
echo ========================================
echo 🎯 Enhanced Dashboard: http://localhost:8080/
echo 💬 Multi-Model Chat: http://localhost:8080/enhanced-chat
echo 📈 Monitoring: http://localhost:8080/monitoring
echo 🎨 Component Showcase: http://localhost:8080/showcase
echo ⚙️ Settings: http://localhost:8080/settings
echo.

echo 🚀 Your advanced AI Agent System is ready!
echo.
echo 💡 To view logs:
echo    docker-compose logs -f frontend
echo    docker-compose logs -f backend
echo.
echo 💡 To stop:
echo    docker-compose down
echo.
pause 
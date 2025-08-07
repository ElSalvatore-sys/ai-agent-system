@echo off
echo ========================================
echo    AI Agent System - Advanced Edition
echo ========================================
echo.
echo Starting comprehensive AI Agent System...
echo This includes:
echo - Advanced Dashboard with cost tracking
echo - Multi-model AI chat interface
echo - Real-time monitoring and analytics
echo - Local model management
echo - Performance optimization
echo - WebSocket real-time communication
echo.

echo [1/4] Starting Backend (FastAPI) on http://localhost:8000
start "AI Agent Backend" powershell -Command "cd backend; python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

echo [2/4] Waiting for backend to initialize...
timeout /t 5 /nobreak > nul

echo [3/4] Starting Frontend (React) on http://localhost:8080
start "AI Agent Frontend" powershell -Command "cd frontend; npm run dev"

echo [4/4] Waiting for frontend to initialize...
timeout /t 8 /nobreak > nul

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
echo 🔍 API Explorer: http://localhost:8000/redoc
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

echo Press any key to open the main application...
pause > nul

echo Opening AI Agent System...
start http://localhost:8080

echo.
echo ========================================
echo    System Started Successfully!
echo ========================================
echo.
echo 💡 Tips:
echo - Use Ctrl+C in the terminal windows to stop services
echo - Check the browser console for any JavaScript errors
echo - Monitor the terminal windows for backend logs
echo - The system includes comprehensive error handling
echo.
echo 🚀 Your advanced AI Agent System is ready for use! 
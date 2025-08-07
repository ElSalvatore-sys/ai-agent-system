@echo off
echo Starting AI Agent System...
echo.

echo Starting Backend (FastAPI) on http://localhost:8000
start "Backend" powershell -Command "cd backend; python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

echo Starting Frontend (React) on http://localhost:8080
start "Frontend" powershell -Command "cd frontend; npm run dev"

echo.
echo Services are starting...
echo Backend: http://localhost:8000
echo Frontend: http://localhost:8080
echo API Docs: http://localhost:8000/docs
echo.
echo Press any key to open the frontend in your browser...
pause > nul
start http://localhost:8080 
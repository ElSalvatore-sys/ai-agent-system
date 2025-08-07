@echo off
echo Installing AI Agent System dependencies...

echo Installing frontend dependencies...
cd frontend
npm install
cd ..

echo Installing backend dependencies...
cd backend
pip install -r requirements.txt
cd ..

echo Dependencies installed successfully!
echo Run 'scripts\dev.bat' to start the development environment.
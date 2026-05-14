@echo off
echo ======================================================
echo  ScolioVis - Installation and Startup
echo ======================================================

echo [1/3] Installing Backend dependencies...
cd backend
if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate
pip install -r requirements.txt
cd ..

echo [2/3] Installing Frontend dependencies...
cd frontend
call npm install
cd ..

echo [3/3] Starting Application...
start "ScolioVis Backend" cmd /k "cd backend && venv\Scripts\activate && python -m uvicorn main:app --port 8000 --reload"
start "ScolioVis Frontend" cmd /k "cd frontend && npm run dev"

echo ======================================================
echo  Installation Complete! 
echo  Backend: http://localhost:8000
echo  Frontend: http://localhost:3000
echo ======================================================
pause

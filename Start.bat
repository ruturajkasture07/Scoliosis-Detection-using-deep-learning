@echo off
echo ======================================================
echo  ScolioVis - Direct Startup
echo ======================================================

echo Starting Backend API...
start "ScolioVis Backend" cmd /k "cd backend && venv\Scripts\activate && python -m uvicorn main:app --port 8000 --reload"

echo Starting Frontend UI...
start "ScolioVis Frontend" cmd /k "cd frontend && npm run dev"

echo ======================================================
echo  Launch triggered! 
echo  Please wait a few seconds for the servers to boot.
echo ======================================================
timeout /t 5

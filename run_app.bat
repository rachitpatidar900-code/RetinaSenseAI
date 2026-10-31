@echo off
cd /d "%~dp0"
if not exist venv\Scripts\activate.bat (
  echo First-time setup. This takes several minutes...
  py -3.12 -m venv venv
  call venv\Scripts\activate
  pip install -r requirements.txt
) else (
  call venv\Scripts\activate
)
start "" cmd /c "timeout /t 5 >nul & start http://localhost:8000"
uvicorn main:app --port 8000
pause
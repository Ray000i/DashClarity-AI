@echo off
REM =====================================================================
REM DashClarity AI - Web Application Startup Script
REM Activates project virtual environment and starts Flask development server
REM =====================================================================

cd /d "%~dp0"

IF NOT EXIST ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\activate.bat
    echo Please create the venv before running this script.
    pause
    exit /b 1
)

echo [INFO] Activating virtual environment (.venv)...
call .venv\Scripts\activate.bat

echo [INFO] Starting DashClarity AI Web Server on http://127.0.0.1:5000 ...
python app.py

pause

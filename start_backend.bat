@echo off
echo ========================================
echo Starting Backend Service
echo ========================================
cd /d "%~dp0"
call conda activate Whisper
echo.
echo Backend starting...
python backend/app.py
pause

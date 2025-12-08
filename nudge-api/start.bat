@echo off
title Nudge Coach API
cd /d "%~dp0"

echo ================================================
echo           NUDGE COACH API LAUNCHER
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

echo Starting Nudge Coach API...
echo.

REM Run the startup script
python start.py %*

pause


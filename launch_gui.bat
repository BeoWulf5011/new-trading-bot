@echo off
REM CryptoBot Matrix Edition Launcher for Windows
REM Automatically starts the Matrix-themed web GUI

echo =========================================
echo   CryptoBot Matrix Edition
echo   Version 2.0.0
echo =========================================
echo.
echo Starting Matrix-themed Web GUI...
echo.

REM Default port
set PORT=8080

REM Check if custom port provided
if not "%1"=="" set PORT=%1

REM Launch the GUI
cd /d "%~dp0"
python gui_web_matrix.py --port %PORT%

echo.
echo GUI server stopped.
pause

@echo off
rem Starts the detour desktop app. Double-click it or run it from any folder.
rem Extra arguments go to the app, for example: run_app.bat --browser
cd /d "%~dp0"

set "VENV_PYTHON=%USERPROFILE%\detour-venv\Scripts\python.exe"
if not exist "%VENV_PYTHON%" (
    echo No se encontro el entorno de Python en %USERPROFILE%\detour-venv
    echo Crealo con:  python -m venv "%USERPROFILE%\detour-venv"
    echo y luego:     "%VENV_PYTHON%" -m pip install -r requirements\evaluation.txt -r requirements\app.txt pywebview
    pause
    exit /b 1
)

"%VENV_PYTHON%" scripts\run_app.py %*
if errorlevel 1 (
    echo.
    echo La app termino con un error. Revisa logs\detour.log
    pause
)

@echo off
TITLE Aviogel Launcher

:: 1. TENTATIVO DI LIBERARE LA PORTA 8000
echo Sto controllando la porta 8000...
FOR /F "tokens=5" %%T IN ('netstat -a -n -o ^| findstr ":8000" ^| findstr "LISTENING"') DO (
    echo Kill processo PID %%T sulla porta 8000
    TASKKILL /F /PID %%T
)

if exist .venv\Scripts\activate (
    call .venv\Scripts\activate
) else (
    echo ERRORE: Virtual environment non trovato in .venv\Scripts\
    pause
    exit /b
)

cd backend

echo Avvio Django Server...
start "Django Backend" python manage.py runserver 0.0.0.0:8000

echo Attendo 5 secondi per l'avvio del backend...
timeout /t 5 /nobreak >nul


cd ..\aviogelFrontend

echo Avvio Ionic...

ionic serve

pause
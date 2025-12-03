#!/bin/bash

sudo fuser -k 8000/tcp || true &

source .venv/bin/activate

echo Avvio Django Server...
cd backend

# sudo -E nohup /home/aviogel/Documenti/aviogelSW/backend/piEnv/bin/python manage.py runserver 0.0.0.0:8000 >> "$LOGFILE" 2>&1
sudo -E nohup python manage.py runserver 0.0.0.0:8000 &

sleep 5

echo Avvio Ionic ...
cd ../aviogelFrontend

ionic serve

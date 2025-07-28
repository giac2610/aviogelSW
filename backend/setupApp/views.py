import json
import os
from django.conf import settings
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import SettingsSerializer
from aviogelMotor.views import load_system_config
# Percorso del file settings.json
SETTINGS_FILE = os.path.join(settings.BASE_DIR, 'config', 'setup.json')

def read_settings():
    """Legge il file settings.json e restituisce i dati"""
    try:
        with open(SETTINGS_FILE, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

@api_view(['GET'])
def get_settings(request):
    """Restituisce le impostazioni di sistema"""
    return Response(read_settings())

def write_settings(data):
    """Scrive i dati nel file settings.json"""
    with open(SETTINGS_FILE, 'w') as file:
        json.dump(data, file, indent=4)

@api_view(['POST'])
def update_settings(request):
    """Aggiorna il file settings.json con nuovi dati"""
    settings_data = read_settings()
    serializer = SettingsSerializer(data=request.data, partial=True)
    if serializer.is_valid():
        settings_data.update(serializer.validated_data)  # Aggiorna solo i campi forniti
        write_settings(settings_data)
        load_system_config()
        return Response({"success": True, "settings": settings_data})
    
    return Response(serializer.errors, status=400)

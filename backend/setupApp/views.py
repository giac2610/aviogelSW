import json
import os
from django.conf import settings
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .serializers import SettingsSerializer
from aviogelMotor.views import load_system_config
import socket


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

@api_view(['GET'])
def get_ip_address(request):
    """
    Restituisce l'indirizzo IP locale della macchina.
    """
    try:
        # Crea un socket per trovare l'IP locale.
        # Connettendosi a un server esterno (senza inviare dati),
        # il sistema operativo espone l'IP dell'interfaccia di rete primaria.
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)) 
        ip_address = s.getsockname()[0]
        s.close()
        return Response({"ip_address": ip_address})
    except Exception:
        # Metodo alternativo nel caso in cui il primo fallisca
        try:
            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)
            return Response({"ip_address": ip_address})
        except Exception as e:
            # Se entrambi i metodi falliscono, restituisce un errore
            error_msg = f"Impossibile determinare l'indirizzo IP: {e}"
            return Response({"error": error_msg}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

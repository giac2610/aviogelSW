import json
import os
import time
import threading
import sys
from django.conf import settings
from unittest.mock import MagicMock
from .serializers import SettingsSerializer

# SU MAC: Simula il modulo pigpio per evitare errori
if sys.platform == "darwin":
    sys.modules["pigpio"] = MagicMock()
import pigpio  # type: ignore

from django.http import JsonResponse
from rest_framework.decorators import api_view

SETTINGS_FILE = os.path.join(settings.BASE_DIR, 'config', 'setup.json')

# ------------------------------------------------------------------------------
# Caricamento configurazione
# ------------------------------------------------------------------------------
def load_motor_config():
    with open("config/setup.json", "r") as file:
        return json.load(file)

def reload_motor_config():
    global config, motor_configs
    config = load_motor_config()
    motor_configs = config.get("motors", {})


config = load_motor_config()
motor_configs = config.get("motors", {})

# ------------------------------------------------------------------------------
# Mappatura motori e inizializzazione pigpio
# ------------------------------------------------------------------------------
MOTORS = {
    "syringe": {"STEP": 12, "DIR": 5, "EN": 7},
    "conveyor": {"STEP": 18, "DIR": 27, "EN": 8},
    "extruder": {"STEP": 13, "DIR": 6, "EN": 3},
}

SERVOS = {
    "extruder": {"signal":9},
    "syringe": {"signal":11},
}    

pi = pigpio.pi()
if not pi.connected:
    raise Exception("Non connesso a pigpio")

for motor in MOTORS.values():
    pi.set_mode(motor["STEP"], pigpio.OUTPUT)
    pi.set_mode(motor["DIR"], pigpio.OUTPUT)    
    pi.set_mode(motor["EN"], pigpio.OUTPUT)    

running_flags = {"extruder": False, "conveyor": False, "syringe": False}

# ------------------------------------------------------------------------------
# Funzione di conversione passi e frequenza
# ------------------------------------------------------------------------------
def compute_motor_params(motor_id):
    motor_conf = motor_configs.get(motor_id, {})
    stepOneRev = motor_conf.get("stepOneRev", 200.0)
    microstep = motor_conf.get("microstep", 8)
    pitch = motor_conf.get("pitch", 5)
    maxSpeed = motor_conf.get("maxSpeed", 250.0)
    steps_per_mm = (stepOneRev * microstep) / pitch
    freq = maxSpeed * steps_per_mm
    return steps_per_mm, max(1, freq)

def write_settings(data):
    """Scrive i dati nel file settings.json"""
    with open(SETTINGS_FILE, 'w') as file:
        json.dump(data, file, indent=4)
# ------------------------------------------------------------------------------
# API: Aggiornamento configurazione
# ------------------------------------------------------------------------------
@api_view(['POST'])
def update_config():
    global config, motor_configs
    try:
        config = load_motor_config()
        motor_configs = config.get("motors", {})
        return JsonResponse({}, status=204)  # No Content
    except Exception as e:
        return JsonResponse({"error": "Errore caricamento config", "detail": str(e)}, status=500)

# ------------------------------------------------------------------------------
# API: Movimento motore
# ------------------------------------------------------------------------------
@api_view(['POST'])
def move_motor(request):
    """
    Muove uno o più motori ai target specificati.
    Il body della richiesta deve contenere i target per ogni motore.
    """
    global running_flags
    try:
        data = json.loads(request.body)
        targets = data.get("targets", {})  # Dizionario con i target per ogni motore
        print("Targets:", targets)
    except Exception as e:
        return JsonResponse({"error": "Dati non validi", "detail": str(e)}, status=400)

    threads = []

    # Gestione dell'EN per syringe e altri motori
    if "syringe" in targets:
        # Disabilita conveyor ed extruder se syringe è nei target
        for motor_id in ["conveyor", "extruder"]:
            motor = MOTORS.get(motor_id)
            if motor:
                pi.write(motor["EN"], 1)  # Disabilita conveyor ed extruder
                # print(f"Disabilitato {motor_id}")
        # Abilita syringe
        syringe_motor = MOTORS.get("syringe")
        if syringe_motor:
            pi.write(syringe_motor["EN"], 0)  # Abilita syringe
            # print(f"Disabilitato {motor_id}")
    elif any(motor_id in targets for motor_id in ["conveyor", "extruder"]):
        # Disabilita syringe se conveyor o extruder sono nei target
        syringe_motor = MOTORS.get("syringe")
        if syringe_motor:
            pi.write(syringe_motor["EN"], 1)  # Disabilita syringe
            # print(f"Disabilitato {motor_id}")
        # Abilita conveyor ed extruder
        for motor_id in ["conveyor", "extruder"]:
            motor = MOTORS.get(motor_id)
            if motor:
                pi.write(motor["EN"], 0)  # Abilita conveyor ed extruder
                # print(f"Disabilitato {motor_id}")

    for motor_id, target in targets.items():
        if motor_id not in MOTORS:
            return JsonResponse({"error": f"Motore non valido: {motor_id}"}, status=400)

        try:
            distance = float(target)
        except ValueError:
            return JsonResponse({"error": f"Target non valido per il motore {motor_id}"}, status=400)

        motor = MOTORS[motor_id]
        direction = 1 if distance >= 0 else 0
        pi.write(motor["DIR"], direction)

        steps_per_mm, freq = compute_motor_params(motor_id)
        steps = abs(distance) * steps_per_mm
        total_time = steps / freq

        running_flags[motor_id] = True

        def motor_thread(motor_id, motor, freq, total_time):
            try:
                pi.hardware_PWM(motor["STEP"], int(freq), 500000)
                start_time = time.time()
                while running_flags[motor_id] and (time.time() - start_time) < total_time:
                    time.sleep(0.01)
            finally:
                pi.hardware_PWM(motor["STEP"], 0, 0)
                running_flags[motor_id] = False

        thread = threading.Thread(target=motor_thread, args=(motor_id, motor, freq, total_time), daemon=True)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return JsonResponse({"status": "Movimento avviato", "targets": targets})
# ------------------------------------------------------------------------------
# API: Stop motori
# ------------------------------------------------------------------------------
@api_view(['POST'])
def stop_motor(body):
    global running_flags
    for key in running_flags:
        running_flags[key] = False
    for motor in MOTORS.values():
        pi.hardware_PWM(motor["STEP"], 0, 0)
    return JsonResponse({"status": "Motori fermati"})

@api_view(['POST'])
def save_motor_config(request):
    global config, motor_configs
    try:
        # Carica la configurazione esistente
        config = load_motor_config()
        settings_data = config.get("motors", {})  # Ottieni i dati dei motori esistenti
    except Exception as e:
        return JsonResponse({"error": "Errore caricamento config", "detail": str(e)}, status=500)

    # Serializza i dati ricevuti
    serializer = SettingsSerializer(data=request.data, partial=True)
    if serializer.is_valid():
        # Aggiorna solo i campi forniti all'interno di "motors"
        settings_data.update(serializer.validated_data)
        config["motors"] = settings_data  # Aggiorna la chiave "motors" nel file di configurazione

        # Rimuovi eventuali duplicazioni di "motors" all'interno di "motors"
        if "motors" in config["motors"]:
            del config["motors"]["motors"]
            
        if "camera" in config["motors"]:
            del config["motors"]["camera"]

        # Scrivi i dati aggiornati nel file di configurazione
        write_settings(config)
        reload_motor_config()
        # Restituisci la configurazione aggiornata
        return JsonResponse({"success": True, "settings": config})
    
    # Restituisci errori di validazione
    return JsonResponse(serializer.errors, status=400)
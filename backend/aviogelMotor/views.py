import json
import os
import time
import threading
import sys
from django.conf import settings
from unittest.mock import MagicMock
from .serializers import SettingsSerializer
from django.views.decorators.csrf import csrf_exempt

# SU MAC: Simula il modulo pigpio per evitare errori
if sys.platform == "darwin":
    sys.modules["pigpio"] = MagicMock()
import pigpio  # type: ignore

from django.http import JsonResponse
from rest_framework.decorators import api_view

SETTINGS_FILE = os.path.join(settings.BASE_DIR, 'config', 'setup.json')

# Percorsi dei file di configurazione
CONFIG_DIR = os.path.join(os.path.dirname(__file__), '../config')
SETTINGS_FILE = os.path.join(CONFIG_DIR, 'setup.json')
EXAMPLE_JSON_PATH = os.path.join(CONFIG_DIR, 'setup.example.json')

# Verifica ed eventualmente rigenera setup.json
if not os.path.exists(SETTINGS_FILE):
    if not os.path.exists(EXAMPLE_JSON_PATH):
        raise FileNotFoundError(f"File di esempio mancante: {EXAMPLE_JSON_PATH}")
    from shutil import copyfile
    copyfile(EXAMPLE_JSON_PATH, SETTINGS_FILE)
    print(f"[INFO] File di configurazione creato da setup.example.json")
# ------------------------------------------------------------------------------
# Caricamento configurazione
# ------------------------------------------------------------------------------
def load_motor_config():
    with open(SETTINGS_FILE, "r") as file:
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
    "extruder": {"STEP": 13, "DIR": 6, "EN": 3}, # X
    "syringe": {"STEP": 18, "DIR": 27, "EN": 8}, # Z
    "conveyor": {"STEP": 12, "DIR": 5, "EN": 7}, # Y
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

# Variabile globale per memorizzare le velocità correnti
current_speeds = {"extruder": 0, "conveyor": 0, "syringe": 0}

# ------------------------------------------------------------------------------
# Funzione di conversione passi e frequenza
# ------------------------------------------------------------------------------
def compute_motor_params(motor_id):
    """
    Calcola i parametri del motore una sola volta per ridurre i calcoli ripetuti.
    """
    motor_conf = motor_configs.get(motor_id, {})
    stepOneRev = motor_conf.get("stepOneRev", 200.0)
    microstep = motor_conf.get("microstep", 8)
    pitch = motor_conf.get("pitch", 5)
    maxSpeed = motor_conf.get("maxSpeed", 250.0)
    acceleration = motor_conf.get("acceleration", 800.0)

    steps_per_mm = (stepOneRev * microstep) / pitch
    max_freq = maxSpeed * steps_per_mm
    accel_steps = int((max_freq ** 2) / (2 * acceleration * steps_per_mm))

    return {
        "steps_per_mm": steps_per_mm,
        "max_freq": max(1, max_freq),
        "accel_steps": accel_steps,
        "decel_steps": accel_steps,  # Usato lo stesso valore di accel_steps
        "acceleration": acceleration,
    }

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
def generate_waveform(motor_targets):
    """
    Genera una waveform DMA multicanale per muovere i motori sincronizzati,
    includendo accelerazione e decelerazione.
    """
    wave = []
    max_total_steps = 0
    pulse_plan = {}

    # Prepara i parametri
    for motor_id, distance in motor_targets.items():
        params = compute_motor_params(motor_id)
        steps = int(abs(distance) * params["steps_per_mm"])
        max_freq = params["max_freq"]
        accel_steps = params["accel_steps"]
        decel_steps = params["decel_steps"]

        direction = 1 if distance >= 0 else 0
        motor = MOTORS[motor_id]
        pi.write(motor["DIR"], direction)
        pi.write(motor["EN"], 0)

        pulse_plan[motor_id] = {
            "pin": motor["STEP"],
            "steps": steps,
            "max_freq": max_freq,
            "accel_steps": accel_steps,
            "decel_steps": decel_steps,
            "next_step": 0
        }

        max_total_steps = max(max_total_steps, steps)

    # Reset wave
    pi.wave_clear()

    # Timeline simulata con accelerazione e decelerazione
    t = 0
    while any(p["next_step"] < p["steps"] for p in pulse_plan.values()):
        on_pulses = []
        off_pulses = []

        for motor_id, plan in pulse_plan.items():
            if plan["next_step"] < plan["steps"]:
                # Calcola la frequenza in base alla fase del movimento
                if plan["next_step"] < plan["accel_steps"]:
                    # Accelerazione
                    freq = (plan["next_step"] / plan["accel_steps"]) * plan["max_freq"]
                elif plan["next_step"] > plan["steps"] - plan["decel_steps"]:
                    # Decelerazione
                    remaining_steps = plan["steps"] - plan["next_step"]
                    freq = (remaining_steps / plan["decel_steps"]) * plan["max_freq"]
                else:
                    # Velocità costante
                    freq = plan["max_freq"]

                freq = max(1, min(freq, plan["max_freq"]))  # Limita la frequenza
                delay_us = int(1_000_000 / freq)  # Calcola il ritardo in microsecondi

                # Genera impulsi
                on_pulses.append(pigpio.pulse(1 << plan["pin"], 0, 5))
                off_pulses.append(pigpio.pulse(0, 1 << plan["pin"], delay_us - 5))
                plan["next_step"] += 1

        wave.extend(on_pulses)
        wave.extend(off_pulses)
        t += min(p["next_step"] for p in pulse_plan.values())

    pi.wave_add_generic(wave)
    wave_id = pi.wave_create()
    return wave_id


@api_view(['POST'])
def move_motor(request):
    """
    Nuova versione: muove i motori generando una waveform DMA multicanale.
    """
    try:
        reload_motor_config()
        data = json.loads(request.body)
        targets = data.get("targets", {})
        if not targets:
            return JsonResponse({"error": "Nessun target fornito"}, status=400)

        for motor_id in targets:
            if motor_id not in MOTORS:
                return JsonResponse({"error": f"Motore non valido: {motor_id}"}, status=400)

        # Gestione del pin EN per evitare interferenze tra i motori
        if "syringe" in targets:
            for motor_id in ["conveyor", "extruder"]:
                pi.write(MOTORS[motor_id]["EN"], 1)  # Disabilita conveyor ed extruder
            pi.write(MOTORS["syringe"]["EN"], 0)  # Abilita syringe
        elif any(motor_id in targets for motor_id in ["conveyor", "extruder"]):
            pi.write(MOTORS["syringe"]["EN"], 1)  # Disabilita syringe
            for motor_id in ["conveyor", "extruder"]:
                if motor_id in targets:
                    pi.write(MOTORS[motor_id]["EN"], 0)  # Abilita conveyor o extruder

        # Ferma eventuali movimenti precedenti
        pi.wave_tx_stop()

        wave_id = generate_waveform(targets)
        if wave_id >= 0:
            pi.wave_send_once(wave_id)
            while pi.wave_tx_busy():
                time.sleep(0.01)
            pi.wave_delete(wave_id)

            # Mantieni EN attivo per i motori in movimento
            for motor_id in targets:
                pi.write(MOTORS[motor_id]["EN"], 0)

            return JsonResponse({"status": "Movimento completato"})
        else:
            return JsonResponse({"error": "Errore creazione waveform"}, status=500)

    except Exception as e:
        return JsonResponse({"error": "Errore interno", "detail": str(e)}, status=500)

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

@csrf_exempt
@api_view(['GET'])
def get_motor_speeds(request):
    """
    Restituisce le velocità correnti dei motori.
    """
    global current_speeds
    # Assicurarsi che le chiavi siano sempre presenti
    response = {
        "syringe": current_speeds.get("syringe", 0),
        "extruder": current_speeds.get("extruder", 0),
        "conveyor": current_speeds.get("conveyor", 0),
    }
    return JsonResponse(response)
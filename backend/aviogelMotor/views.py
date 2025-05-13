import json
import os
import time
import threading
import sys
from django.conf import settings
from unittest.mock import MagicMock
from .serializers import SettingsSerializer
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from rest_framework.decorators import api_view

# SU MAC: Simula il modulo pigpio per evitare errori
if sys.platform == "darwin":
    sys.modules["pigpio"] = MagicMock()
import pigpio  # type: ignore

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
    "extruder": {"STEP": 13, "DIR": 6, "EN": 3},  # X
    "syringe": {"STEP": 18, "DIR": 27, "EN": 8},  # Z
    "conveyor": {"STEP": 12, "DIR": 5, "EN": 7},  # Y
}

SERVOS = {
    "extruder": {"signal": 9},
    "syringe": {"signal": 11},
}

# Inizializzazione pigpio
pi = pigpio.pi()
if not pi.connected:
    raise Exception("Non connesso a pigpio")

for motor in MOTORS.values():
    pi.set_mode(motor["STEP"], pigpio.OUTPUT)
    pi.set_mode(motor["DIR"], pigpio.OUTPUT)
    pi.set_mode(motor["EN"], pigpio.OUTPUT)

running_flags = {motor: False for motor in MOTORS.keys()}
current_speeds = {motor: 0 for motor in MOTORS.keys()}

# ------------------------------------------------------------------------------
# Funzione di conversione passi e frequenza
# ------------------------------------------------------------------------------
def compute_motor_params(motor_id):
    """
    Calcola i parametri del motore una sola volta per ridurre i calcoli ripetuti.
    """
    motor_conf = motor_configs.get(motor_id, {})
    step_one_rev = motor_conf.get("stepOneRev", 200.0)
    microstep = motor_conf.get("microstep", 8)
    pitch = motor_conf.get("pitch", 5)
    max_speed = motor_conf.get("maxSpeed", 250.0)
    acceleration = motor_conf.get("acceleration", 800.0)

    steps_per_mm = (step_one_rev * microstep) / pitch
    max_freq = max_speed * steps_per_mm
    accel_steps = int((max_freq ** 2) / (2 * acceleration * steps_per_mm))

    return {
        "steps_per_mm": steps_per_mm,
        "max_freq": max(1, max_freq),
        "accel_steps": accel_steps,
        "decel_steps": accel_steps,
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
        reload_motor_config()
        return JsonResponse({"log": "Configurazione aggiornata", "status": "success"}, status=204)
    except Exception as e:
        return JsonResponse({"log": "Errore aggiornamento configurazione", "error": str(e)}, status=500)

# ------------------------------------------------------------------------------
# API: Movimento motore
# ------------------------------------------------------------------------------
def generate_waveform(motor_targets):
    """
    Genera una waveform DMA multicanale per muovere i motori sincronizzati,
    includendo accelerazione e decelerazione. Utilizza una wave chain per
    gestire un numero elevato di impulsi.
    """
    wave = []
    pulse_plan = {}
    MAX_PULSES_PER_WAVE = 10000

    # Prepara i parametri
    for motor_id, distance in motor_targets.items():
        params = compute_motor_params(motor_id)
        steps = int(abs(distance) * params["steps_per_mm"])
        direction = 1 if distance >= 0 else 0
        motor = MOTORS[motor_id]

        pi.write(motor["DIR"], direction)
        pi.write(motor["EN"], 0)

        pulse_plan[motor_id] = {
            "pin": motor["STEP"],
            "steps": steps,
            "max_freq": params["max_freq"],
            "accel_steps": params["accel_steps"],
            "decel_steps": params["decel_steps"],
            "next_step": 0,
        }

    # Reset wave
    pi.wave_clear()
    wave_ids = []

    while any(plan["next_step"] < plan["steps"] for plan in pulse_plan.values()):
        on_pulses, off_pulses = [], []

        for motor_id, plan in pulse_plan.items():
            if plan["next_step"] < plan["steps"]:
                freq = compute_frequency(plan)
                delay_us = int(1_000_000 / freq)

                on_pulses.append(pigpio.pulse(1 << plan["pin"], 0, 5))
                off_pulses.append(pigpio.pulse(0, 1 << plan["pin"], delay_us - 5))
                plan["next_step"] += 1

        wave.extend(on_pulses + off_pulses)

        if len(wave) >= MAX_PULSES_PER_WAVE:
            wave_ids.append(create_wave(wave[:MAX_PULSES_PER_WAVE]))
            wave = wave[MAX_PULSES_PER_WAVE:]

    if wave:
        wave_ids.append(create_wave(wave))

    # Correzione: aggiunta di parentesi quadre attorno alla comprensione della lista
    pi.wave_chain([255, 0] + [wave_id for wave_id in wave_ids])
    return wave_ids

def compute_frequency(plan):
    """Calcola la frequenza in base alla fase del movimento."""
    if plan["next_step"] < plan["accel_steps"]:
        return (plan["next_step"] / plan["accel_steps"]) * plan["max_freq"]
    elif plan["next_step"] > plan["steps"] - plan["decel_steps"]:
        remaining_steps = plan["steps"] - plan["next_step"]
        return (remaining_steps / plan["decel_steps"]) * plan["max_freq"]
    return plan["max_freq"]

def create_wave(pulses):
    """Crea una waveform e restituisce il suo ID."""
    pi.wave_add_generic(pulses)
    return pi.wave_create()

@api_view(['POST'])
def move_motor(request):
    """
    Muove i motori generando una waveform DMA multicanale.
    """
    global pi
    try:
        reload_motor_config()
        data = json.loads(request.body)
        targets = data.get("targets", {})
        if not targets:
            return JsonResponse({"log": "Nessun target fornito", "error": "Input non valido"}, status=400)

        validate_targets(targets)
        manage_motor_pins(targets)

        pi.wave_tx_stop()
        ensure_pigpio_connection()

        wave_ids = generate_waveform(targets)
        if wave_ids:
            threading.Thread(target=execute_waveform, args=(wave_ids,), daemon=True).start()
            return JsonResponse({"log": "Movimento avviato", "status": "success"})
        return JsonResponse({"log": "Errore creazione waveform", "error": "Waveform non valida"}, status=500)

    except Exception as e:
        return handle_exception(e)

def validate_targets(targets):
    """Valida i target forniti."""
    for motor_id in targets:
        if motor_id not in MOTORS:
            raise ValueError(f"Motore non valido: {motor_id}")

def manage_motor_pins(targets):
    """Gestisce i pin EN per evitare interferenze tra i motori."""
    if "syringe" in targets:
        for motor_id in ["conveyor", "extruder"]:
            pi.write(MOTORS[motor_id]["EN"], 1)
        pi.write(MOTORS["syringe"]["EN"], 0)
    elif any(motor_id in targets for motor_id in ["conveyor", "extruder"]):
        pi.write(MOTORS["syringe"]["EN"], 1)
        for motor_id in ["conveyor", "extruder"]:
            if motor_id in targets:
                pi.write(MOTORS[motor_id]["EN"], 0)

def ensure_pigpio_connection():
    """Verifica e ristabilisce la connessione a pigpio."""
    if not pi.connected:
        pi.stop()
        time.sleep(1)
        pi = pigpio.pi()
        if not pi.connected:
            raise Exception("Impossibile riconnettersi a pigpio")

def execute_waveform(wave_ids):
    """Esegue la waveform in un thread separato."""
    try:
        for wave_id in wave_ids:
            pi.wave_send_once(wave_id)
            while pi.wave_tx_busy():
                time.sleep(0.01)
            pi.wave_delete(wave_id)
    except Exception as e:
        print(f"[ERROR] Errore durante l'esecuzione della waveform: {e}")

def handle_exception(e):
    """Gestisce le eccezioni e restituisce un JsonResponse."""
    import traceback
    error_details = traceback.format_exc()
    print(f"[ERROR] Movimento motore fallito: {error_details}")
    return JsonResponse({"log": "Errore interno durante il movimento", "error": str(e)}, status=500)

# ------------------------------------------------------------------------------
# API: Stop motori
# ------------------------------------------------------------------------------
@api_view(['POST'])
def stop_motor(body):
    global running_flags
    try:
        for key in running_flags:
            running_flags[key] = False
        for motor in MOTORS.values():
            pi.hardware_PWM(motor["STEP"], 0, 0)
        return JsonResponse({"log": "Motori fermati con successo", "status": "success"})
    except Exception as e:
        return JsonResponse({"log": "Errore durante lo stop dei motori", "error": str(e)}, status=500)

# ------------------------------------------------------------------------------
# API: Salvataggio configurazione
# ------------------------------------------------------------------------------
@api_view(['POST'])
def save_motor_config(request):
    global config, motor_configs
    try:
        config = load_motor_config()
        settings_data = config.get("motors", {})
    except Exception as e:
        return JsonResponse({"log": "Errore durante il salvataggio della configurazione", "error": str(e)}, status=500)

    serializer = SettingsSerializer(data=request.data, partial=True)
    if serializer.is_valid():
        settings_data.update(serializer.validated_data)
        config["motors"] = settings_data
        config["motors"].pop("motors", None)
        config["motors"].pop("camera", None)

        write_settings(config)
        reload_motor_config()
        return JsonResponse({"log": "Configurazione salvata con successo", "success": True, "settings": config})

    return JsonResponse(serializer.errors, status=400)

# ------------------------------------------------------------------------------
# API: Velocità motori
# ------------------------------------------------------------------------------
@csrf_exempt
@api_view(['GET'])
def get_motor_speeds(request):
    global current_speeds
    try:
        response = {motor: current_speeds.get(motor, 0) for motor in MOTORS.keys()}
        return JsonResponse({"log": "Velocità motori recuperate", "speeds": response})
    except Exception as e:
        return JsonResponse({"log": "Errore durante il recupero delle velocità", "error": str(e)}, status=500)
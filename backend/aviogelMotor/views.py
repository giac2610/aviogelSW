import json
import time
import threading
import sys
from unittest.mock import MagicMock

# SU MAC: Simula il modulo pigpio per evitare errori
if sys.platform == "darwin":
    sys.modules["pigpio"] = MagicMock()
import pigpio  # type: ignore

from django.http import JsonResponse
from rest_framework.decorators import api_view

# ------------------------------------------------------------------------------
# Caricamento configurazione
# ------------------------------------------------------------------------------
def load_motor_config():
    with open("config/setup.json", "r") as file:
        return json.load(file)

config = load_motor_config()
motor_configs = config.get("motors", {})

# ------------------------------------------------------------------------------
# Mappatura motori e inizializzazione pigpio
# ------------------------------------------------------------------------------
MOTORS = {
    "extruder": {"STEP": 12, "DIR": 27},
    "conveyor": {"STEP": 13, "DIR": 23},
    "syringe": {"STEP": 18, "DIR": 24},
}

pi = pigpio.pi()
if not pi.connected:
    raise Exception("Non connesso a pigpio")

for motor in MOTORS.values():
    pi.set_mode(motor["STEP"], pigpio.OUTPUT)
    pi.set_mode(motor["DIR"], pigpio.OUTPUT)

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

# ------------------------------------------------------------------------------
# API: Aggiornamento configurazione
# ------------------------------------------------------------------------------
@api_view(['POST'])
def update_config(request):
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
    global running_flags
    try:
        data = json.loads(request.body)
        motor_id = data.get("motor")
        distance = float(data.get("distance", 0))
    except Exception as e:
        return JsonResponse({"error": "Dati non validi", "detail": str(e)}, status=400)
    
    if motor_id not in MOTORS:
        return JsonResponse({"error": "Motore non valido"}, status=400)
    
    motor = MOTORS[motor_id]
    direction = 1 if distance >= 0 else 0
    pi.write(motor["DIR"], direction)
    
    steps_per_mm, freq = compute_motor_params(motor_id)
    steps = abs(distance) * steps_per_mm
    total_time = steps / freq

    running_flags[motor_id] = True
    
    def motor_thread():
        pi.hardware_PWM(motor["STEP"], int(freq), 500000)
        start_time = time.time()
        while running_flags[motor_id] and (time.time() - start_time) < total_time:
            time.sleep(0.01)
        pi.hardware_PWM(motor["STEP"], 0, 0)
        running_flags[motor_id] = False

    threading.Thread(target=motor_thread, daemon=True).start()
    return JsonResponse({"status": "Movimento avviato", "motor": motor_id})

# ------------------------------------------------------------------------------
# API: Stop motori
# ------------------------------------------------------------------------------
@api_view(['POST'])
def stop_motor(request):
    global running_flags
    for key in running_flags:
        running_flags[key] = False
    for motor in MOTORS.values():
        pi.hardware_PWM(motor["STEP"], 0, 0)
    return JsonResponse({"status": "Motori fermati"})

# -*- coding: utf-8 -*-

import json
import os
import time
import threading
import sys
import logging
import queue
from shutil import copyfile
from itertools import groupby

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view

# --- Blocco di Simulazione ---
IS_RPI = sys.platform == "linux" or sys.platform == "linux2"
if not IS_RPI:
    from unittest.mock import MagicMock
    sys.modules["pigpio"] = MagicMock()
    logging.warning(f"Piattaforma non-Raspberry Pi rilevata ({sys.platform}). 'pigpio' simulato.")
import pigpio  # type: ignore

# --- Configurazione Percorsi, Logging, etc. ---
try:
    CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CURRENT_SCRIPT_DIR = os.getcwd()
CONFIG_DIR = os.path.join(CURRENT_SCRIPT_DIR, '../config')
SETTINGS_FILE = os.path.join(CONFIG_DIR, 'setup.json')
EXAMPLE_JSON_PATH = os.path.join(CONFIG_DIR, 'setup.example.json')
LOG_FILE = os.path.join(CURRENT_SCRIPT_DIR, 'motorLog.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s:%(funcName)s] - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

def initialize_config_file():
    if not os.path.exists(SETTINGS_FILE):
        logging.warning(f"'{SETTINGS_FILE}' non trovato.")
        if not os.path.exists(EXAMPLE_JSON_PATH):
            raise FileNotFoundError(f"'{EXAMPLE_JSON_PATH}' mancante.")
        os.makedirs(CONFIG_DIR, exist_ok=True)
        copyfile(EXAMPLE_JSON_PATH, SETTINGS_FILE)
        logging.info(f"Config creata da esempio in '{SETTINGS_FILE}'")
initialize_config_file()

class SettingsSerializer:
    def __init__(self, data=None, partial=False): self.initial_data = data; self._validated_data = data or {}; self.errors = {}
    def is_valid(self, raise_exception=False): return True
    @property
    def validated_data(self): return self._validated_data

pi = None
def get_pigpio_instance():
    global pi
    if pi and pi.connected: return pi
    logging.warning("pigpio non connesso. Tentativo di (ri)connessione...")
    if pi:
        try: pi.stop()
        except: pass
    try:
        if IS_RPI:
            pi = pigpio.pi()
            if pi.connected: logging.info("Connessione a pigpio stabilita."); _initialize_gpio_pins(); return pi
        else:
            pi = pigpio.pi(); logging.info("Utilizzo di istanza pigpio simulata."); return pi
    except Exception as e:
        logging.error(f"Fallimento init pigpio: {e}"); pi = None; return None

MOTORS = {"extruder": {"STEP": 13, "DIR": 6, "EN": 3}, "syringe": {"STEP": 18, "DIR": 27, "EN": 8}, "conveyor": {"STEP": 12, "DIR": 5, "EN": 7}}
current_speeds = {motor: 0 for motor in MOTORS.keys()}

def _initialize_gpio_pins():
    pi_instance = get_pigpio_instance()
    if not pi_instance or not getattr(pi_instance, 'connected', False): return
    for motor_name, pins in MOTORS.items():
        try:
            pi_instance.set_mode(pins["STEP"], pigpio.OUTPUT); pi_instance.set_mode(pins["DIR"], pigpio.OUTPUT)
            pi_instance.set_mode(pins["EN"], pigpio.OUTPUT); pi_instance.write(pins["EN"], 1)
        except Exception as e: logging.error(f"Errore config pin per '{motor_name}': {e}")

motor_configs = {}
def load_motor_config():
    global motor_configs
    try:
        with open(SETTINGS_FILE, "r") as f:
            full_config = json.load(f)
            motor_configs = full_config.get("motors", {})
            logging.debug("Configurazione motori caricata/ricaricata.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Errore caricamento '{SETTINGS_FILE}': {e}."); motor_configs = {}
    return motor_configs

def write_settings(data):
    with open(SETTINGS_FILE, 'w') as f: json.dump(data, f, indent=4)
    logging.info(f"Configurazione salvata in '{SETTINGS_FILE}'.")

load_motor_config(); get_pigpio_instance()

motor_command_queue = queue.Queue()

# ==============================================================================
# Motor Worker CORRETTO con gestione finally per task_done()
# ==============================================================================
def motor_worker():
    logging.info("Motor worker avviato.")
    while True:
        targets = motor_command_queue.get()
        try:
            logging.info(f"Worker: nuovo comando {targets}")
            execute_movement(targets)
            logging.info(f"Worker: comando {targets} completato.")
        except Exception as e:
            # L'errore viene comunque loggato
            logging.error(f"Errore durante l'esecuzione del comando {targets} nel motor_worker: {e}", exc_info=True)
        finally:
            # Assicuriamoci che la coda venga notificata in ogni caso, anche in caso di errore
            motor_command_queue.task_done()

threading.Thread(target=motor_worker, daemon=True, name="MotorWorker").start()

# ==============================================================================
# Core Logic: Generazione ed Esecuzione Waveform
# ==============================================================================
MAX_PULSES_PER_WAVE = 4096

def compute_motor_params(motor_id):
    conf = motor_configs.get(motor_id, {}); steps_per_rev = float(conf.get("stepOneRev", 200.0)); microsteps = int(conf.get("microstep", 8))
    pitch = float(conf.get("pitch", 5.0)); max_speed_mms = float(conf.get("maxSpeed", 250.0)); acceleration_mms2 = float(conf.get("acceleration", 800.0))
    if pitch == 0: raise ValueError(f"'pitch' per '{motor_id}' non può essere zero.")
    steps_per_mm = (steps_per_rev * microsteps) / pitch; max_freq_hz = max_speed_mms * steps_per_mm
    accel_steps_s2 = acceleration_mms2 * steps_per_mm
    accel_steps = int((max_freq_hz ** 2) / (2 * accel_steps_s2)) if accel_steps_s2 > 0 else 0
    return {"steps_per_mm": steps_per_mm, "max_freq": max_freq_hz, "accel_steps": max(1, accel_steps)}

def compute_step_frequency(current_step, total_steps, accel_steps, max_freq):
    decel_start_step = total_steps - accel_steps
    if current_step < accel_steps: return max(1.0, (current_step + 1) / accel_steps * max_freq)
    elif current_step >= decel_start_step:
        steps_into_decel = current_step - decel_start_step; remaining_steps = accel_steps - steps_into_decel
        return max(1.0, remaining_steps / accel_steps * max_freq)
    else: return max_freq

def _generate_motor_event_timeline(motor_id, distance_mm):
    params = compute_motor_params(motor_id); step_pin = MOTORS[motor_id]["STEP"]
    total_steps = int(abs(distance_mm) * params["steps_per_mm"])
    if total_steps == 0: return []
    accel_steps = min(params["accel_steps"], total_steps // 2); timeline = []; current_time_us = 0.0
    for step in range(total_steps):
        freq = compute_step_frequency(step, total_steps, accel_steps, params["max_freq"])
        half_period_us = 500_000.0 / freq
        timeline.append({'time_us': current_time_us, 'pin': step_pin, 'state': 1})
        current_time_us += half_period_us
        timeline.append({'time_us': current_time_us, 'pin': step_pin, 'state': 0})
        current_time_us += half_period_us
    return timeline

def stop_all_motors():
    pi_instance = get_pigpio_instance()
    if not pi_instance or not pi_instance.connected: return
    try:
        pi_instance.wave_tx_stop()
        pi_instance.wave_clear()
        for pins in MOTORS.values():
            pi_instance.write(pins["EN"], 1)
        logging.info("Tutti i motori sono stati fermati e disabilitati.")
    except Exception as e:
        logging.error(f"Errore durante lo stop dei motori: {e}")

# ==============================================================================
# Funzione execute_movement con pulizia (finally) CORRETTA
# ==============================================================================
def execute_movement(targets):
    pi_instance = get_pigpio_instance()
    if not pi_instance or not pi_instance.connected:
        raise ConnectionError("pigpio non connesso.")

    master_timeline = []
    active_motor_pins_en_mask = 0
    for motor_id, distance_mm in targets.items():
        if motor_id not in MOTORS or distance_mm == 0:
            continue
        pi_instance.write(MOTORS[motor_id]["DIR"], 1 if distance_mm >= 0 else 0)
        active_motor_pins_en_mask |= (1 << MOTORS[motor_id]["EN"])
        logging.info(f"Generazione timeline per {motor_id} ({distance_mm}mm)...")
        master_timeline.extend(_generate_motor_event_timeline(motor_id, distance_mm))
    
    if not master_timeline:
        logging.warning("Nessun movimento richiesto."); return

    logging.info(f"Timeline master creata con {len(master_timeline)} eventi. Ordinamento...")
    master_timeline.sort(key=lambda e: e['time_us'])
    logging.info("Ordinamento completato. Inizio esecuzione a flusso con `wave_chain`...")

    try:
        # Abilita i motori necessari impostando i loro pin EN a 0 (LOW)
        for motor_id, pins in MOTORS.items():
            if (1 << pins["EN"]) & active_motor_pins_en_mask:
                 pi_instance.write(pins["EN"], 0)
        
        time.sleep(0.01)

        wave_pulses = []
        wave_chain_data = []

        last_time_us = 0
        for time_us, events_at_time in groupby(master_timeline, key=lambda e: e['time_us']):
            delay_us = time_us - last_time_us
            if delay_us > 0:
                wave_pulses.append(pigpio.pulse(0, 0, int(round(delay_us))))
            
            on_mask = 0
            off_mask = 0
            for event in events_at_time:
                if event['state'] == 1:
                    on_mask |= (1 << event['pin'])
                else:
                    off_mask |= (1 << event['pin'])
            wave_pulses.append(pigpio.pulse(on_mask, off_mask, 0))
            last_time_us = time_us

            if len(wave_pulses) >= MAX_PULSES_PER_WAVE:
                pi_instance.wave_add_generic(wave_pulses)
                wave_id = pi_instance.wave_create()
                wave_chain_data.extend([255, 0, wave_id])
                wave_pulses = []

        if wave_pulses:
            pi_instance.wave_add_generic(wave_pulses)
            wave_id = pi_instance.wave_create()
            wave_chain_data.extend([255, 0, wave_id])

        wave_chain_data.extend([255, 2, 0, 0])

        if wave_chain_data:
            logging.info(f"Invio della catena di {len(wave_chain_data) // 3} waveform...")
            pi_instance.wave_chain(wave_chain_data)

            while pi_instance.wave_tx_busy():
                time.sleep(0.05)
            
            logging.info("Esecuzione catena waveform completata.")

    finally:
        # Sequenza di pulizia robusta
        try:
            # 1. Ferma qualsiasi trasmissione di waveform in corso
            pi_instance.wave_tx_stop()
        except Exception as e:
            logging.error(f"Errore durante pi.wave_tx_stop(): {e}")

        try:
            # 2. Pulisci tutte le waveform dalla memoria di pigpio
            pi_instance.wave_clear()
        except Exception as e:
            logging.error(f"Errore durante pi.wave_clear(): {e}")
            
        # 3. Disabilita tutti i motori (impostando EN a 1 HIGH)
        for pins in MOTORS.values():
            try:
                if pi_instance.connected:
                    pi_instance.write(pins["EN"], 1)
            except Exception as e:
                logging.error(f"Errore disabilitazione pin EN {pins['EN']}: {e}")
        
        logging.info("Pulizia post-movimento completata (stop, clear, motori disabilitati).")


def handle_exception(e):
    import traceback; error_details = traceback.format_exc(); logging.error(f"Errore interno: {error_details}")
    error_type = type(e).__name__; error_message = str(e)
    return JsonResponse({"log": f"Errore interno: {error_type}", "error": error_message}, status=500)

# ==============================================================================
# API VIEWS (Originali)
# ==============================================================================
@api_view(['POST'])
def update_config_view(request):
    try: load_motor_config(); return JsonResponse({"log": "Configurazione aggiornata", "status": "success"}, status=200)
    except Exception as e: logging.error(f"Errore aggiornamento config: {str(e)}"); return JsonResponse({"log": "Errore aggiornamento configurazione", "error": str(e)}, status=500)

@api_view(['POST'])
def move_motor_view(request):
    try:
        data = json.loads(request.body); targets = data.get("targets")
        if not targets: return JsonResponse({"log": "Nessun target fornito", "error": "Input non valido"}, status=400)
        motor_command_queue.put(targets)
        return JsonResponse({"log": "Movimento messo in coda", "status": "queued"})
    except Exception as e: return handle_exception(e)

@api_view(['POST'])
def stop_motor_view(request):
    try:
        logging.info("Richiesta di STOP motori ricevuta.")
        while not motor_command_queue.empty():
            try: motor_command_queue.get_nowait()
            except queue.Empty: continue
        stop_all_motors()
        # Svuotare la coda e chiamare task_done per ogni elemento rimosso
        # per sbloccare qualsiasi `queue.join()` in attesa.
        while not motor_command_queue.empty():
            try:
                motor_command_queue.get_nowait()
                motor_command_queue.task_done()
            except queue.Empty:
                continue
        return JsonResponse({"log": "Motori fermati e coda pulita con successo", "status": "success"})
    except Exception as e: return handle_exception(e)

@api_view(['POST'])
def save_motor_config_view(request):
    try:
        try:
            with open(SETTINGS_FILE, 'r') as f: full_config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError): full_config = {}
        if 'motors' not in full_config: full_config['motors'] = {}
        new_data = json.loads(request.body)
        serializer = SettingsSerializer(data=new_data)
        if serializer.is_valid():
            full_config['motors'].update(serializer.validated_data)
            write_settings(full_config) 
            load_motor_config()
            return JsonResponse({"log": "Configurazione motori salvata", "success": True, "settings": full_config})
        else: return JsonResponse({"log": "Errore di validazione", "errors": serializer.errors}, status=400)
    except Exception as e: return handle_exception(e)

@csrf_exempt
@api_view(['GET'])
def get_motor_speeds_view(request):
    global current_speeds
    try:
        speeds_snapshot = {motor: current_speeds.get(motor, 0) for motor in MOTORS.keys()}
        return JsonResponse({"log": "Velocità motori (stato placeholder)", "speeds": speeds_snapshot})
    except Exception as e: return handle_exception(e)

@api_view(['POST'])
def start_simulation_view(request):
    try:
        simulation_steps = []; extruder_direction = 1
        for _ in range(5):
            for _ in range(3): simulation_steps.append({"syringe": 5}); simulation_steps.append({"extruder": 50 * extruder_direction})
            extruder_direction *= -1; simulation_steps.append({"conveyor": 50})
        logging.info("Avvio simulazione predefinita...")
        for i, step in enumerate(simulation_steps):
            logging.info(f"Accodamento passo simulazione {i+1}: {step}"); motor_command_queue.put(step)
        
        motor_command_queue.join()
        logging.info("Simulazione completata.")
        return JsonResponse({"log": "Simulazione completata con successo", "status": "success"})
    except Exception as e: return handle_exception(e)

@api_view(['POST'])
def execute_route_view(request):
    try:
        data = json.loads(request.body); route = data.get("route", [])
        if not isinstance(route, list) or not route: return JsonResponse({"log": "Percorso non valido", "error": "Input non valido"}, status=400)
        logging.info(f"Inizio esecuzione rotta con {len(route)} passi.")
        for idx, step in enumerate(route):
            if idx == 0 and all((isinstance(v, (int, float)) and v == 0) for v in step.values()):
                logging.info(f"Salto primo movimento nullo: {step}"); continue
            step_to_execute = dict(step)
            logging.info(f"Accodamento passo rotta {idx+1}: {step_to_execute}")
            motor_command_queue.put(step_to_execute)

        motor_command_queue.join()
        logging.info("Rotta completata.")
        return JsonResponse({"log": "Rotta eseguita con successo", "status": "success"})
    except Exception as e:
        return handle_exception(e)
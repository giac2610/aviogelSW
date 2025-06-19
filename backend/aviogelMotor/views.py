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

# --- Blocco di Simulazione per Ambienti non-Raspberry Pi (es. macOS, Windows) ---
IS_RPI = sys.platform == "linux" or sys.platform == "linux2"
if not IS_RPI:
    from unittest.mock import MagicMock
    sys.modules["pigpio"] = MagicMock()
    logging.warning(f"Piattaforma non-Raspberry Pi rilevata ({sys.platform}). Il modulo 'pigpio' è stato simulato.")
import pigpio  # type: ignore

# --- Configurazione dei Percorsi e dei File ---
try:
    CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CURRENT_SCRIPT_DIR = os.getcwd()

CONFIG_DIR = os.path.join(CURRENT_SCRIPT_DIR, '../config')
SETTINGS_FILE = os.path.join(CONFIG_DIR, 'setup.json')
EXAMPLE_JSON_PATH = os.path.join(CONFIG_DIR, 'setup.example.json')
LOG_FILE = os.path.join(CURRENT_SCRIPT_DIR, 'motorLog.log')

# --- Inizializzazione e Configurazione del Logging ---
logging.basicConfig(
    level=logging.INFO, # Impostato a INFO per ridurre il verbosismo, DEBUG per maggiori dettagli
    format='%(asctime)s - %(levelname)s - [%(threadName)s:%(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# --- Gestione File di Configurazione ---
def initialize_config_file():
    if not os.path.exists(SETTINGS_FILE):
        # ... (codice identico a prima)
        logging.warning(f"File di configurazione '{SETTINGS_FILE}' non trovato.")
        if not os.path.exists(EXAMPLE_JSON_PATH):
            error_msg = f"File di esempio '{EXAMPLE_JSON_PATH}' mancante. Impossibile creare la configurazione."
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            os.makedirs(CONFIG_DIR, exist_ok=True)
            copyfile(EXAMPLE_JSON_PATH, SETTINGS_FILE)
            logging.info(f"File di configurazione creato da esempio in '{SETTINGS_FILE}'")
        except Exception as e:
            logging.critical(f"Impossibile creare il file di configurazione: {e}")
            raise

initialize_config_file()

# --- Placeholder per il Serializer di Django ---
class SettingsSerializer:
    # ... (codice identico a prima)
    def __init__(self, data=None, partial=False): self.initial_data = data; self._validated_data = data or {}; self.errors = {}
    def is_valid(self, raise_exception=False): return True
    @property
    def validated_data(self): return self._validated_data

# --- Gestione della Connessione a pigpio ---
pi = None
def get_pigpio_instance():
    # ... (codice identico a prima)
    global pi
    if pi and pi.connected: return pi
    logging.warning("Connessione a pigpio non attiva. Tentativo di (ri)connessione...")
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
        logging.error(f"Fallimento critico durante l'inizializzazione di pigpio: {e}"); pi = None; return None

# --- Definizione Motori e Pin ---
MOTORS = {
    "extruder": {"STEP": 13, "DIR": 6, "EN": 3},
    "syringe": {"STEP": 18, "DIR": 27, "EN": 8},
    "conveyor": {"STEP": 12, "DIR": 5, "EN": 7},
}
current_speeds = {motor: 0 for motor in MOTORS.keys()}

def _initialize_gpio_pins():
    # ... (codice identico a prima)
    pi_instance = get_pigpio_instance()
    if not pi_instance or not getattr(pi_instance, 'connected', False): return
    for motor_name, pins in MOTORS.items():
        try:
            pi_instance.set_mode(pins["STEP"], pigpio.OUTPUT)
            pi_instance.set_mode(pins["DIR"], pigpio.OUTPUT)
            pi_instance.set_mode(pins["EN"], pigpio.OUTPUT)
            pi_instance.write(pins["EN"], 1)
        except Exception as e:
            logging.error(f"Errore config pin per '{motor_name}': {e}")


# --- Caricamento e Gestione Configurazione Motori ---
motor_configs = {}
def load_motor_config():
    # ... (codice identico a prima)
    global motor_configs
    try:
        with open(SETTINGS_FILE, "r") as f:
            full_config = json.load(f); motor_configs = full_config.get("motors", {}); logging.info("Configurazione motori caricata/ricaricata.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Errore caricamento '{SETTINGS_FILE}': {e}."); motor_configs = {}
    return motor_configs

def write_settings(data):
    # ... (codice identico a prima)
    try:
        with open(SETTINGS_FILE, 'w') as f: json.dump(data, f, indent=4); logging.info(f"Configurazione salvata in '{SETTINGS_FILE}'.")
    except Exception as e:
        logging.error(f"Impossibile salvare config: {e}"); raise

# Inizializzazione all'avvio
load_motor_config()
get_pigpio_instance()

# --- Logica del Worker per Movimenti Asincroni ---
motor_command_queue = queue.Queue()
def motor_worker():
    # ... (codice identico a prima)
    logging.info("Motor worker avviato.")
    while True:
        try:
            targets = motor_command_queue.get(); logging.info(f"Worker: nuovo comando {targets}"); execute_movement(targets)
            logging.info(f"Worker: comando {targets} completato."); motor_command_queue.task_done()
        except Exception as e:
            logging.error(f"Errore nel motor_worker: {e}", exc_info=True)
threading.Thread(target=motor_worker, daemon=True, name="MotorWorker").start()

# ==============================================================================
# Core Logic: Generazione ed Esecuzione Waveform (NUOVA LOGICA EVENT-BASED)
# ==============================================================================
MAX_PULSES_PER_WAVE = 8192 # Limite più generoso
MAX_WAVE_CHAIN_SIZE = 250

def compute_motor_params(motor_id):
    # ... (codice identico a prima)
    conf = motor_configs.get(motor_id, {})
    steps_per_rev = float(conf.get("stepOneRev", 200.0)); microsteps = int(conf.get("microstep", 8))
    pitch = float(conf.get("pitch", 5.0)); max_speed_mms = float(conf.get("maxSpeed", 250.0))
    acceleration_mms2 = float(conf.get("acceleration", 800.0))
    if pitch == 0: raise ValueError(f"'pitch' per '{motor_id}' non può essere zero.")
    steps_per_mm = (steps_per_rev * microsteps) / pitch; max_freq_hz = max_speed_mms * steps_per_mm
    accel_steps_s2 = acceleration_mms2 * steps_per_mm
    accel_steps = int((max_freq_hz ** 2) / (2 * accel_steps_s2)) if accel_steps_s2 > 0 else 0
    return {"steps_per_mm": steps_per_mm, "max_freq": max_freq_hz, "accel_steps": max(1, accel_steps)}

def compute_step_frequency(current_step, total_steps, accel_steps, max_freq):
    # ... (codice identico a prima)
    decel_start_step = total_steps - accel_steps
    if current_step < accel_steps: return max(1.0, (current_step + 1) / accel_steps * max_freq)
    elif current_step >= decel_start_step:
        steps_into_decel = current_step - decel_start_step; remaining_steps = accel_steps - steps_into_decel
        return max(1.0, remaining_steps / accel_steps * max_freq)
    else: return max_freq

def _generate_motor_event_timeline(motor_id, distance_mm):
    """
    [NUOVA FUNZIONE HELPER]
    Calcola la timeline di eventi (timestamp, pin, stato) per un singolo motore.
    """
    params = compute_motor_params(motor_id)
    step_pin = MOTORS[motor_id]["STEP"]
    total_steps = int(abs(distance_mm) * params["steps_per_mm"])
    if total_steps == 0:
        return []

    accel_steps = min(params["accel_steps"], total_steps // 2)
    
    timeline = []
    current_time_us = 0.0

    for step in range(total_steps):
        freq = compute_step_frequency(step, total_steps, accel_steps, params["max_freq"])
        half_period_us = 500_000.0 / freq

        # Evento di salita (ON)
        timeline.append({'time_us': current_time_us, 'pin': step_pin, 'state': 1})
        current_time_us += half_period_us
        # Evento di discesa (OFF)
        timeline.append({'time_us': current_time_us, 'pin': step_pin, 'state': 0})
        current_time_us += half_period_us

    return timeline

def generate_movement_waveforms(targets):
    """
    [LOGICA COMPLETAMENTE RISCRITTA]
    Genera una waveform event-based fondendo le timeline di eventi di tutti i motori.
    """
    pi_instance = get_pigpio_instance()
    if not pi_instance or not pi_instance.connected:
        raise ConnectionError("pigpio non connesso.")

    master_timeline = []
    active_motor_pins_en = 0

    for motor_id, distance_mm in targets.items():
        if motor_id not in MOTORS or distance_mm == 0:
            continue
        
        pi_instance.write(MOTORS[motor_id]["DIR"], 1 if distance_mm >= 0 else 0)
        active_motor_pins_en |= (1 << MOTORS[motor_id]["EN"])
        
        logging.info(f"Generazione timeline per {motor_id} ({distance_mm}mm)...")
        motor_timeline = _generate_motor_event_timeline(motor_id, distance_mm)
        master_timeline.extend(motor_timeline)

    if not master_timeline:
        return []

    logging.info(f"Timeline master creata con {len(master_timeline)} eventi. Ordinamento...")
    master_timeline.sort(key=lambda e: e['time_us'])
    logging.info("Ordinamento completato. Generazione waveform da eventi...")

    pi_instance.wave_clear()
    
    # 1. Waveform per abilitare i motori
    pi_instance.wave_add_generic([pigpio.pulse(0, active_motor_pins_en, 100)])
    enable_wave_id = pi_instance.wave_create()

    # 2. Waveform di movimento principale
    movement_wave_ids = []
    wave_pulses = []
    last_time_us = 0
    
    # Raggruppa eventi che accadono allo stesso istante per creare un singolo impulso
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
        
        wave_pulses.append(pigpio.pulse(on_mask, off_mask, 0)) # Applica i cambiamenti istantaneamente
        last_time_us = time_us

        if len(wave_pulses) >= MAX_PULSES_PER_WAVE:
            pi_instance.wave_add_generic(wave_pulses)
            movement_wave_ids.append(pi_instance.wave_create())
            wave_pulses = []
    
    # Aggiungi l'ultimo blocco di impulsi se presente
    if wave_pulses:
        pi_instance.wave_add_generic(wave_pulses)
        movement_wave_ids.append(pi_instance.wave_create())

    # 3. Waveform per disabilitare i motori
    pi_instance.wave_add_generic([pigpio.pulse(active_motor_pins_en, 0, 100)])
    disable_wave_id = pi_instance.wave_create()
    
    final_chain = [enable_wave_id] + movement_wave_ids + [disable_wave_id]
    logging.info(f"Generazione waveform completata. Catena di {len(final_chain)} wave ID creata.")
    return final_chain


# --- Funzioni di Esecuzione e Gestione (Identiche a prima) ---
def execute_wave_chain(wave_ids):
    # ... (codice identico a prima)
    pi_instance = get_pigpio_instance()
    if not pi_instance or not pi_instance.connected: raise ConnectionError("pigpio non connesso.")
    if not wave_ids: return
    offset = 0
    while offset < len(wave_ids):
        chunk = wave_ids[offset : offset + MAX_WAVE_CHAIN_SIZE]; offset += len(chunk)
        pi_instance.wave_chain(chunk)
        while pi_instance.wave_tx_busy(): time.sleep(0.01)

def stop_all_motors():
    # ... (codice identico a prima)
    pi_instance = get_pigpio_instance()
    if not pi_instance or not pi_instance.connected: return
    try:
        pi_instance.wave_tx_stop()
        for pins in MOTORS.values(): pi_instance.write(pins["EN"], 1)
        logging.info("Tutti i motori sono stati fermati e disabilitati.")
    except Exception as e:
        logging.error(f"Errore durante lo stop dei motori: {e}")

def execute_movement(targets):
    # ... (codice identico a prima)
    try:
        if not isinstance(targets, dict): raise ValueError("I target devono essere un dizionario.")
        for motor_id, distance in targets.items():
            if motor_id not in MOTORS: raise ValueError(f"Motore non valido: {motor_id}")
            if not isinstance(distance, (int, float)): raise ValueError(f"La distanza per {motor_id} deve essere numerica.")
        wave_ids = generate_movement_waveforms(targets)
        if wave_ids: execute_wave_chain(wave_ids)
    except Exception as e:
        logging.error(f"Errore critico durante esecuzione movimento per {targets}: {e}", exc_info=True)
        stop_all_motors(); raise

def handle_exception(e):
    # ... (codice identico a prima)
    import traceback; error_details = traceback.format_exc(); logging.error(f"Errore interno: {error_details}")
    error_type = type(e).__name__; error_message = str(e)
    return JsonResponse({"log": f"Errore interno: {error_type}", "error": error_message}, status=500)

# ==============================================================================
# API VIEWS (Interfaccia Originale Mantenuta, NESSUNA MODIFICA QUI)
# ==============================================================================

@api_view(['POST'])
def update_config_view(request):
    try: load_motor_config(); return JsonResponse({"log": "Configurazione aggiornata", "status": "success"}, status=200)
    except Exception as e:
        logging.error(f"Errore aggiornamento configurazione: {str(e)}"); return JsonResponse({"log": "Errore aggiornamento configurazione", "error": str(e)}, status=500)

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
            motor_command_queue.task_done()
        stop_all_motors()
        return JsonResponse({"log": "Motori fermati con successo", "status": "success"})
    except Exception as e: return handle_exception(e)

@api_view(['POST'])
def save_motor_config_view(request):
    try:
        with open(SETTINGS_FILE, 'r') as f:
            try: full_config = json.load(f)
            except json.JSONDecodeError: full_config = {}
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
            logging.info(f"Esecuzione passo simulazione {i+1}: {step}"); execute_movement(step)
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
            step_to_execute = dict(step); step_to_execute["syringe"] = -10
            logging.info(f"Esecuzione passo rotta {idx+1}: {step_to_execute}"); execute_movement(step_to_execute)
        logging.info("Rotta completata.")
        return JsonResponse({"log": "Rotta eseguita con successo", "status": "success"})
    except Exception as e: return handle_exception(e)
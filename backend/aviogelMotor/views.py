# -*- coding: utf-8 -*-

# Requisito: numpy. Eseguire 'pip install numpy'
import numpy as np

import json
import os
import time
import threading
import sys
import logging
import queue
import math
from shutil import copyfile
from dataclasses import dataclass

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view

# --- Blocco di Simulazione ---
IS_RPI = sys.platform == "linux" or sys.platform == "linux2"
if not IS_RPI:
    from unittest.mock import MagicMock
    sys.modules["pigpio"] = MagicMock()
    logging.warning(f"Piattaforma non-Raspberry Pi rilevata ({sys.platform}). 'pigpio' simulato.")
import pigpio #type: ignore

# --- Configurazione e Logging ---
try:
    CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CURRENT_SCRIPT_DIR = os.getcwd()
CONFIG_DIR = os.path.join(CURRENT_SCRIPT_DIR, '../config')
SETTINGS_FILE = os.path.join(CONFIG_DIR, 'setup.json')
LOG_FILE = os.path.join(CURRENT_SCRIPT_DIR, 'motorLog.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s:%(funcName)s] - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

# --- MAPPATURA HARDWARE FONDAMENTALE ---
MOTORS = {
    "extruder": {"STEP": 13, "DIR": 6, "EN": 3},
    "syringe": {"STEP": 18, "DIR": 27, "EN": 8},
    "conveyor": {"STEP": 12, "DIR": 5, "EN": 7}
}

# --- PARAMETRO PER ACCELERAZIONE LINEARE ---
# Definisce su quanti passi avviene la rampa di accelerazione/decelerazione.
LINEAR_ACCEL_STEPS = 400

# ==============================================================================
# ARCHITETTURA REFACTORING: Classi per la Gestione del Movimento
# ==============================================================================

@dataclass
class MotorConfig:
    """Struttura dati per contenere la configurazione completa di un singolo motore."""
    name: str
    step_pin: int
    dir_pin: int
    en_pin: int
    steps_per_mm: float
    max_freq_hz: float

class MotionPlanner:
    """Il cervello del sistema. Pianifica movimenti coordinati."""
    def __init__(self, motor_configs: dict[str, MotorConfig]):
        self.motor_configs = motor_configs
        logging.info(f"MotionPlanner inizializzato con motori: {list(motor_configs.keys())}")

    def _generate_trapezoidal_profile(self, total_steps: int, max_freq_hz: float) -> list[float]:
        """
        Genera i timestamp per ogni passo usando un profilo Trapezoidale (accelerazione lineare).
        Ottimizzato con NumPy.
        """
        if total_steps == 0:
            return []

        accel_steps = min(LINEAR_ACCEL_STEPS, total_steps // 2)
        decel_start_step = total_steps - accel_steps

        i = np.arange(total_steps, dtype=np.float64)
        freq = np.full(total_steps, max_freq_hz, dtype=np.float64)

        # Fase di accelerazione lineare
        if accel_steps > 0:
            accel_mask = i < accel_steps
            freq[accel_mask] = max_freq_hz * (i[accel_mask] + 1) / accel_steps

        # Fase di decelerazione lineare
        if decel_start_step < total_steps:
            decel_mask = i >= decel_start_step
            steps_into_decel = i[decel_mask] - decel_start_step
            remaining_steps = accel_steps - steps_into_decel
            freq[decel_mask] = max_freq_hz * remaining_steps / accel_steps
        
        np.maximum(freq, 1.0, out=freq) # Assicura che la frequenza non sia mai zero

        periods_us = 1_000_000.0 / freq
        timestamps_us = np.cumsum(periods_us)
        
        return timestamps_us.tolist()

    def plan_move(self, targets: dict[str, float]) -> tuple[list, set, dict]:
        """Pianifica un movimento coordinato, con logica di generazione impulsi corretta per prevenire la perdita di passi."""
        if not targets:
            return [], set(), {}

        move_data = {}
        for motor_id, distance in targets.items():
            if distance == 0 or motor_id not in self.motor_configs:
                continue
            config = self.motor_configs[motor_id]
            move_data[motor_id] = {
                "steps": int(abs(distance) * config.steps_per_mm),
                "dir": 1 if distance >= 0 else 0,
                "config": config
            }
        
        if not move_data:
            return [], set(), {}

        master_id = max(move_data, key=lambda k: move_data[k]["steps"])
        master_steps = move_data[master_id]["steps"]
        if master_steps == 0:
            return [], set(), {}

        logging.info(f"Pianificazione movimento. Asse Master: '{master_id}' ({master_steps} passi).")
        # --- MODIFICA: Chiamata alla nuova funzione di profilo lineare ---
        master_profile_ts = self._generate_trapezoidal_profile(master_steps, move_data[master_id]["config"].max_freq_hz)

        dir_on_mask = 0
        dir_off_mask = 0
        for motor_id, move in move_data.items():
            if move["dir"] == 1:
                dir_on_mask |= (1 << move["config"].dir_pin)
            else:
                dir_off_mask |= (1 << move["config"].dir_pin)
        
        setup_pulse = pigpio.pulse(dir_on_mask, dir_off_mask, 20)
        final_pulses = [setup_pulse]
        
        bresenham_errors = {mid: -master_steps / 2 for mid in move_data if mid != master_id}
        last_time_us = 0.0
        for i in range(master_steps):
            on_mask = 1 << move_data[master_id]["config"].step_pin
            for slave_id in bresenham_errors:
                bresenham_errors[slave_id] += move_data[slave_id]["steps"]
                if bresenham_errors[slave_id] > 0:
                    on_mask |= 1 << move_data[slave_id]["config"].step_pin
                    bresenham_errors[slave_id] -= master_steps
            
            current_time_us = master_profile_ts[i]
            total_period_us = current_time_us - last_time_us

            if total_period_us >= 2:
                on_width_us = 2
                off_width_us = int(round(total_period_us - on_width_us))
                final_pulses.append(pigpio.pulse(on_mask, 0, on_width_us))
                final_pulses.append(pigpio.pulse(0, on_mask, off_width_us))
            elif total_period_us > 0:
                final_pulses.append(pigpio.pulse(on_mask, 0, 1))
                final_pulses.append(pigpio.pulse(0, on_mask, 0))

            last_time_us = current_time_us

        active_motors = {m["config"].name for m in move_data.values()}
        directions = {mid: m["dir"] for mid, m in move_data.items()}
        return final_pulses, active_motors, directions

class MotorController:
    """Gestisce l'interazione diretta con pigpio e l'hardware."""
    def __init__(self, motor_configs: dict[str, MotorConfig]):
        self.pi = self._get_pigpio_instance()
        self.motor_configs = motor_configs
        self._initialize_gpio_pins()
        self.MAX_PULSES_PER_WAVE = 4096

    def _get_pigpio_instance(self):
        logging.info("Tentativo di connessione a pigpio...")
        try:
            if IS_RPI:
                pi = pigpio.pi()
                if pi.connected:
                    logging.info("Connessione a pigpio stabilita.")
                    return pi
            else:
                logging.info("Utilizzo di istanza pigpio simulata.")
                return pigpio.pi()
        except Exception as e:
            logging.error(f"Fallimento init pigpio: {e}")
        return None

    def _initialize_gpio_pins(self):
        if not self.pi or not self.pi.connected: return
        for config in self.motor_configs.values():
            self.pi.set_mode(config.step_pin, pigpio.OUTPUT)
            self.pi.set_mode(config.dir_pin, pigpio.OUTPUT)
            self.pi.set_mode(config.en_pin, pigpio.OUTPUT)
            self.pi.write(config.en_pin, 1)

    def execute_timeline(self, timeline: list, active_motors: set, directions: dict):
        if not self.pi or not self.pi.connected:
            raise ConnectionError("Esecuzione fallita: pigpio non connesso.")
        if not timeline:
            logging.warning("Timeline vuota, nessun movimento da eseguire.")
            return

        created_wave_ids = []
        try:
            for motor_name in active_motors:
                config = self.motor_configs[motor_name]
                self.pi.write(config.en_pin, 0)
            time.sleep(0.01)

            wave_chain_data = []
            for i in range(0, len(timeline), self.MAX_PULSES_PER_WAVE):
                chunk = timeline[i:i + self.MAX_PULSES_PER_WAVE]
                self.pi.wave_add_generic(chunk)
                wave_id = self.pi.wave_create()
                if wave_id < 0:
                    raise pigpio.error(f"Errore creazione waveform (codice {wave_id})")
                created_wave_ids.append(wave_id)
                wave_chain_data.extend([255, 0, wave_id])
            
            wave_chain_data.extend([255, 2, 0, 0])

            logging.info(f"Esecuzione catena di {len(created_wave_ids)} waveforms...")
            self.pi.wave_chain(wave_chain_data)
            while self.pi.wave_tx_busy():
                time.sleep(0.05)
            logging.info("Esecuzione catena waveform completata.")

        finally:
            if self.pi and self.pi.connected:
                self.pi.wave_tx_stop()
                for wave_id in created_wave_ids:
                    try: self.pi.wave_delete(wave_id)
                    except pigpio.error: pass
                for config in self.motor_configs.values():
                    self.pi.write(config.en_pin, 1)
                logging.info(f"Pulizia post-movimento completata (cancellate {len(created_wave_ids)} waveforms).")

# ==============================================================================
# Inizializzazione del Sistema
# ==============================================================================
def load_system_config() -> dict[str, MotorConfig]:
    """Carica la configurazione dal file JSON e la trasforma in oggetti MotorConfig."""
    configs = {}
    try:
        with open(SETTINGS_FILE, "r") as f:
            full_config = json.load(f).get("motors", {})
            for name, params in full_config.items():
                if name not in MOTORS: continue
                
                steps_per_rev = float(params.get("stepOneRev", 200.0))
                microsteps = int(params.get("microstep", 8))
                pitch = float(params.get("pitch", 5.0))
                if pitch == 0: raise ValueError(f"'pitch' per '{name}' non può essere zero.")
                max_speed_mms = float(params.get("maxSpeed", 250.0))
                
                steps_per_mm = (steps_per_rev * microsteps) / pitch
                max_freq_hz = max_speed_mms * steps_per_mm

                configs[name] = MotorConfig(
                    name=name,
                    step_pin=MOTORS[name]["STEP"],
                    dir_pin=MOTORS[name]["DIR"],
                    en_pin=MOTORS[name]["EN"],
                    steps_per_mm=steps_per_mm,
                    max_freq_hz=max_freq_hz
                )
            logging.info("Configurazione motori caricata e parsata.")
    except Exception as e:
        logging.error(f"Errore caricamento configurazione: {e}")
    return configs

# Istanze globali dei controller di sistema
MOTOR_CONFIGS = load_system_config()
MOTION_PLANNER = MotionPlanner(MOTOR_CONFIGS)
MOTOR_CONTROLLER = MotorController(MOTOR_CONFIGS)

motor_command_queue = queue.Queue()
SYSTEM_CONFIG_LOCK = threading.Lock()

def motor_worker():
    """Worker thread che orchestra la pianificazione e l'esecuzione dei movimenti."""
    logging.info("Motor worker avviato con architettura professionale.")
    while True:
        targets = motor_command_queue.get()
        try:
            with SYSTEM_CONFIG_LOCK:
                logging.info(f"Worker: ricevuto comando {targets}")
                timeline, active_motors, directions = MOTION_PLANNER.plan_move(targets)
                MOTOR_CONTROLLER.execute_timeline(timeline, active_motors, directions)
            logging.info(f"Worker: comando {targets} completato con successo.")
        except Exception as e:
            logging.error(f"Errore critico nel motor_worker su comando {targets}: {e}", exc_info=True)
        finally:
            motor_command_queue.task_done()
            time.sleep(0.1)

def handle_exception(e):
    import traceback
    error_details = traceback.format_exc()
    logging.error(f"Errore interno API: {error_details}")
    return JsonResponse({"log": f"Errore interno: {type(e).__name__}", "error": str(e)}, status=500)

# ==============================================================================
# AVVIO DEL THREAD WORKER
# ==============================================================================
if os.environ.get('RUN_MAIN') == 'true':
    logging.info("Processo principale di Django rilevato. Avvio del MotorWorker...")
    threading.Thread(target=motor_worker, daemon=True, name="MotorWorker").start()
else:
    logging.info("Processo di reload rilevato. Il MotorWorker non verrà avviato qui.")

# ==============================================================================
# API VIEWS (Interfaccia esterna preservata)
# ==============================================================================

@api_view(['POST'])
def move_motor_view(request):
    try:
        data = json.loads(request.body)
        targets = data.get("targets")
        if not targets or not isinstance(targets, dict):
            return JsonResponse({"log": "Input non valido", "error": "targets deve essere un dizionario"}, status=400)
        motor_command_queue.put(targets)
        return JsonResponse({"log": "Movimento messo in coda", "status": "queued"})
    except Exception as e:
        return handle_exception(e)

@api_view(['POST'])

def execute_route_view(request):

    try:

        data = json.loads(request.body)
        route = data.get("route", [])
        if not isinstance(route, list):
            return JsonResponse({"log": "Percorso non valido", "error": "Input non valido"}, status=400)

        logging.info(f"Accodamento rotta con {len(route)} passi.")

        for step in route:
            if isinstance(step, dict):  
                motor_command_queue.put(step)
        return JsonResponse({"log": f"Rotta con {len(route)} passi accodata con successo.", "status": "queued"})
    except Exception as e:
        return handle_exception(e)
    
@api_view(['POST'])
def stop_motor_view(request):
    try:
        logging.info("Richiesta di STOP motori ricevuta.")
        while not motor_command_queue.empty():
            try:
                motor_command_queue.get_nowait()
                motor_command_queue.task_done()
            except queue.Empty:
                continue
        
        if MOTOR_CONTROLLER.pi and MOTOR_CONTROLLER.pi.connected:
            MOTOR_CONTROLLER.pi.wave_tx_stop()

        logging.info("Movimento fermato e coda di comandi pulita.")
        return JsonResponse({"log": "Comando di stop inviato.", "status": "success"})
    except Exception as e:
        return handle_exception(e)

@api_view(['POST'])
def update_config_view(request):
    """Ricarica a caldo la configurazione dei motori dal file JSON."""
    global MOTOR_CONFIGS, MOTION_PLANNER, MOTOR_CONTROLLER
    
    logging.warning("Inizio procedura di Hot-Reload della configurazione di sistema.")
    with SYSTEM_CONFIG_LOCK:
        try:
            if MOTOR_CONTROLLER.pi and MOTOR_CONTROLLER.pi.connected:
                MOTOR_CONTROLLER.pi.wave_tx_stop()
            logging.info("Movimento corrente interrotto per aggiornamento configurazione.")

            new_configs = load_system_config()
            if not new_configs:
                raise ValueError("Caricamento nuova configurazione fallito o file vuoto.")

            MOTOR_CONFIGS = new_configs
            MOTOR_CONTROLLER = MotorController(MOTOR_CONFIGS)
            MOTION_PLANNER = MotionPlanner(MOTOR_CONFIGS)
            
            logging.info("Hot-Reload completato. Il sistema ora usa la nuova configurazione.")
            return JsonResponse({"log": "Configurazione aggiornata e ricaricata con successo", "status": "success"})
        except Exception as e:
            logging.error(f"Fallimento durante la procedura di Hot-Reload: {e}", exc_info=True)
            return JsonResponse({"log": "Errore critico durante l'hot-reload.", "error": str(e)}, status=500)

@api_view(['POST'])
def start_simulation_view(request):
    """Esegue una sequenza di movimenti predefinita e bloccante."""
    try:
        simulation_steps = []
        extruder_direction = 1
        for _ in range(5):
            for _ in range(3):
                simulation_steps.append({"syringe": 5})
                simulation_steps.append({"extruder": 50 * extruder_direction})
            extruder_direction *= -1
            simulation_steps.append({"conveyor": 50})
        
        logging.info("Avvio simulazione predefinita...")
        for i, step in enumerate(simulation_steps):
            logging.info(f"Accodamento passo simulazione {i+1}: {step}")
            motor_command_queue.put(step)
        
        motor_command_queue.join()
        
        logging.info("Simulazione completata.")
        return JsonResponse({"log": "Simulazione completata con successo", "status": "success"})
    except Exception as e:
        return handle_exception(e)

@api_view(['POST'])
def save_motor_config_view(request):
    try:
        with open(SETTINGS_FILE, 'r') as f:
            full_config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        full_config = {}
    
    new_data = json.loads(request.body)
    full_config.setdefault('motors', {}).update(new_data)

    with open(SETTINGS_FILE, 'w') as f:
        json.dump(full_config, f, indent=4)
    
    msg = "Configurazione motori salvata. Chiamare /motors/update_config/ per applicare le modifiche a caldo."
    logging.info(msg)
    return JsonResponse({"log": msg, "success": True})

@csrf_exempt
@api_view(['GET'])
def get_motor_speeds_view(request):
    return JsonResponse({"log": "API di velocità non implementata nella nuova architettura", "speeds": {}})
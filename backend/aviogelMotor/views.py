# -*- coding: utf-8 -*-

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

# --- MAPPATURA HARDWARE FONDAMENTALE (Correzione) ---
# Questa mappatura è critica e definisce i collegamenti fisici.
MOTORS = {
    "extruder": {"STEP": 13, "DIR": 6, "EN": 3},
    "syringe": {"STEP": 18, "DIR": 27, "EN": 8},
    "conveyor": {"STEP": 12, "DIR": 5, "EN": 7}
}

# --- Parametri di Movimento Professionale ---
# Numero di passi per completare la fase di "jerk control".
# Un valore più alto produce una curva più dolce ma un'accelerazione più lenta.
S_CURVE_ACCEL_STEPS = 400

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
    """Il cervello del sistema. Pianifica movimenti coordinati con profilo S-Curve."""
    def __init__(self, motor_configs: dict[str, MotorConfig]):
        self.motor_configs = motor_configs
        logging.info(f"MotionPlanner inizializzato con motori: {list(motor_configs.keys())}")

    def _generate_s_curve_profile(self, total_steps: int, max_freq_hz: float) -> list[float]:
        """Genera i timestamp per ogni passo usando un profilo S-Curve."""
        if total_steps == 0:
            return []
            
        timestamps_us = [0.0] * total_steps
        current_time_us = 0.0
        accel_steps = min(S_CURVE_ACCEL_STEPS, total_steps // 2)
        decel_start_step = total_steps - accel_steps

        for i in range(total_steps):
            freq = 1.0
            if i < accel_steps:
                completion = i / accel_steps
                factor = 0.5 * (1 - math.cos(completion * math.pi))
                freq = max(1.0, max_freq_hz * factor)
            elif i >= decel_start_step:
                steps_into_decel = i - decel_start_step
                completion = steps_into_decel / accel_steps
                factor = 0.5 * (1 - math.cos((1 - completion) * math.pi))
                freq = max(1.0, max_freq_hz * factor)
            else:
                freq = max_freq_hz

            period_us = 1_000_000.0 / freq
            current_time_us += period_us
            timestamps_us[i] = current_time_us
            
        return timestamps_us

    def plan_move(self, targets: dict[str, float]) -> tuple[list, set, dict]:
        """Pianifica un movimento coordinato (interpolazione lineare)."""
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
        master_profile_ts = self._generate_s_curve_profile(master_steps, move_data[master_id]["config"].max_freq_hz)

        final_pulses = []
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
            pulse_width_us = int(round(total_period_us / 2))
            
            if pulse_width_us > 0:
                final_pulses.append(pigpio.pulse(on_mask, 0, pulse_width_us))
                final_pulses.append(pigpio.pulse(0, on_mask, pulse_width_us))
            
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
        return None

    def _initialize_gpio_pins(self):
        if not self.pi or not self.pi.connected: return
        for config in self.motor_configs.values():
            self.pi.set_mode(config.step_pin, pigpio.OUTPUT)
            self.pi.set_mode(config.dir_pin, pigpio.OUTPUT)
            self.pi.set_mode(config.en_pin, pigpio.OUTPUT)
            self.pi.write(config.en_pin, 1) # Disabilita di default

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
                self.pi.write(config.dir_pin, directions[motor_name])
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

def motor_worker():
    """Worker thread che orchestra la pianificazione e l'esecuzione dei movimenti."""
    logging.info("Motor worker avviato con architettura professionale.")
    while True:
        targets = motor_command_queue.get()
        try:
            logging.info(f"Worker: ricevuto comando {targets}")
            timeline, active_motors, directions = MOTION_PLANNER.plan_move(targets)
            MOTOR_CONTROLLER.execute_timeline(timeline, active_motors, directions)
            logging.info(f"Worker: comando {targets} completato con successo.")
        except Exception as e:
            logging.error(f"Errore critico nel motor_worker su comando {targets}: {e}", exc_info=True)
        finally:
            motor_command_queue.task_done()

threading.Thread(target=motor_worker, daemon=True, name="MotorWorker").start()

def handle_exception(e):
    import traceback
    error_details = traceback.format_exc()
    logging.error(f"Errore interno API: {error_details}")
    return JsonResponse({"log": f"Errore interno: {type(e).__name__}", "error": str(e)}, status=500)

# ==============================================================================
# API VIEWS (Interfaccia esterna, logica di business delegata)
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
    logging.info(f"Configurazione salvata. RIAVVIARE IL SERVIZIO per applicare le modifiche.")
    
    return JsonResponse({"log": "Configurazione motori salvata. È richiesto un riavvio per applicare.", "success": True})

@csrf_exempt
@api_view(['GET'])
def get_motor_speeds_view(request):
    return JsonResponse({"log": "API di velocità non implementata nella nuova architettura", "speeds": {}})
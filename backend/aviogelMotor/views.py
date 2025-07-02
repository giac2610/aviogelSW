# -*- coding: utf-8 -*-

# ==============================================================================
# ARCHITETTURA STREAMING con Accelerazione per Motore (Versione Corretta e Completa)
# Questa versione corregge i problemi di esecuzione a scatti, blocco,
# e controllo della direzione.
# ==============================================================================

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

SWITCHES = {
    "extruder": {"Start": 23, "End": 24},
    "syringe": {"Start": 19, "End": 26},
}

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
    acceleration_mmss: float

class MotionPlanner:
    def __init__(self, motor_configs: dict[str, MotorConfig]):
        self.motor_configs = motor_configs
        logging.info(f"MotionPlanner inizializzato con motori: {list(motor_configs.keys())}")

    def _generate_trapezoidal_profile(self, total_steps: int, max_freq_hz: float, accel_mmss: float, steps_per_mm: float) -> list[float]:
        if total_steps == 0:
            return []

        v_max = max_freq_hz / steps_per_mm
        if accel_mmss <= 0: 
            accel_mmss = 1.0 # Prevenzione divisione per zero
        t_accel = v_max / accel_mmss
        s_accel = 0.5 * accel_mmss * t_accel**2
        steps_accel = int(s_accel * steps_per_mm)
        steps_accel = min(steps_accel, total_steps // 2)

        if steps_accel == 0 and total_steps > 0:
             steps_accel = 1
        if steps_accel * 2 > total_steps:
            steps_accel = total_steps // 2
            v_peak = (accel_mmss * (steps_accel / steps_per_mm))**0.5
            v_max = v_peak
            t_accel = v_max / accel_mmss if accel_mmss > 0 else 0

        i = np.arange(total_steps, dtype=np.float64)
        freq = np.full(total_steps, v_max * steps_per_mm, dtype=np.float64)

        if steps_accel > 0:
            accel_mask = i < steps_accel
            t_vals = np.sqrt(2 * (i[accel_mask] + 1) / (accel_mmss * steps_per_mm))
            freq[accel_mask] = accel_mmss * t_vals * steps_per_mm

            decel_start_step = total_steps - steps_accel
            decel_mask = i >= decel_start_step
            steps_into_decel = i[decel_mask] - decel_start_step
            t_vals_decel = np.sqrt(2 * (steps_accel - steps_into_decel) / (accel_mmss * steps_per_mm))
            freq[decel_mask] = accel_mmss * t_vals_decel * steps_per_mm

        np.maximum(freq, 1.0, out=freq)
        periods_us = 1_000_000.0 / freq
        return np.cumsum(periods_us).tolist()

    def plan_move_streamed(self, targets: dict[str, float], switch_states: dict, pi=None, chunk_size: int = 250, max_chunks: int = 10):
        if not targets:
            return (None for _ in range(0)), set(), {}
    
        # --- AGGIORNAMENTO STATO FINECORSA ---
        if pi is not None:
            for motor, pins in SWITCHES.items():
                for name, pin in pins.items():
                    switch_id = f"{motor}_{name.lower()}"
                    switch_states[switch_id] = pi.read(pin) == 1
    
        move_data = {}
        for motor_id, distance in targets.items():
            if distance == 0 or motor_id not in self.motor_configs:
                continue
            
            direction = 1 if distance >= 0 else 0
            logging.info(f"Calcolo movimento per '{motor_id}': distanza {distance} mm, direzione {'positiva' if direction == 1 else 'negativa'}.")
            # --- TEST TEMPORANEO: La logica dei finecorsa è disabilitata ---
            if direction == 1 and switch_states.get(f"{motor_id}_start"):
                logging.info(f"Sblocco logico: il movimento positivo per '{motor_id}' resetta lo stato del finecorsa START.")
                switch_states[f"{motor_id}_start"] = False
    
            if direction == 0 and switch_states.get(f"{motor_id}_end"):
                logging.info(f"Sblocco logico: il movimento negativo per '{motor_id}' resetta lo stato del finecorsa END.")
                switch_states[f"{motor_id}_end"] = False
    
            if direction == 0 and switch_states.get(f"{motor_id}_start"):
                logging.warning(f"Movimento per '{motor_id}' bloccato: si sta tentando di superare il finecorsa START attivo.")
                continue
            if direction == 1 and switch_states.get(f"{motor_id}_end"):
                logging.warning(f"Movimento per '{motor_id}' bloccato: si sta tentando di superare il finecorsa END attivo.")
                continue
            
            config = self.motor_configs[motor_id]
            move_data[motor_id] = {
                "steps": int(abs(distance) * config.steps_per_mm), "dir": direction, "config": config
            }
    
        if not move_data:
            return (None for _ in range(0)), set(), {}
    
        master_id = max(move_data, key=lambda k: move_data[k]["steps"])
        master_data = move_data[master_id]
        master_steps = master_data["steps"]
    
        if master_steps == 0:
            return (None for _ in range(0)), set(), {}
    
        accel_for_this_move = master_data["config"].acceleration_mmss
        logging.info(f"Streaming pianificato. Master: '{master_id}' ({master_steps} passi). Uso accelerazione di {accel_for_this_move} mm/s^2.")
    
        master_profile_ts = self._generate_trapezoidal_profile(
            master_steps,
            master_data["config"].max_freq_hz,
            accel_for_this_move,
            master_data["config"].steps_per_mm
        )
    
        active_motors = {m["config"].name for m in move_data.values()}
        directions_to_set = {mid: move['dir'] for mid, move in move_data.items()}
    
        # --- Suddivisione in macro-movimenti ---
        steps_per_macro = int((chunk_size * max_chunks) // 2)  # //2 perché ogni step sono 2 pulse
        total_macro = (master_steps + steps_per_macro - 1) // steps_per_macro
    
        def make_pulse_generator_macro(start_step, end_step, macro_idx, total_macro):
            def pulse_generator():
                bresenham_errors = {mid: -master_steps / 2 for mid in move_data if mid != master_id}
                last_time_us, pulse_chunk = 0.0, []
                chunk_count = 0
                total_pulse = 0
                for i in range(start_step, end_step):
                    on_mask = 1 << master_data["config"].step_pin
                    for slave_id in bresenham_errors:
                        bresenham_errors[slave_id] += move_data[slave_id]["steps"]
                        if bresenham_errors[slave_id] > 0:
                            on_mask |= 1 << move_data[slave_id]["config"].step_pin
                            bresenham_errors[slave_id] -= master_steps
                    current_time_us = master_profile_ts[i]
                    total_period_us = current_time_us - last_time_us
                    if total_period_us >= 2:
                        pulse_chunk.extend([pigpio.pulse(on_mask, 0, 2), pigpio.pulse(0, on_mask, int(round(total_period_us - 2)))])
                    elif total_period_us > 0:
                        pulse_chunk.extend([pigpio.pulse(on_mask, 0, 1), pigpio.pulse(0, on_mask, 0)])
                    last_time_us = current_time_us
                    if len(pulse_chunk) >= chunk_size:
                        chunk_count += 1
                        total_pulse += len(pulse_chunk)
                        logging.info(f"[PulseGen] Macro {macro_idx+1}/{total_macro} - Yield chunk {chunk_count}: {len(pulse_chunk)} pulse ({len(pulse_chunk)//2} passi)")
                        yield pulse_chunk
                        pulse_chunk = []
                if pulse_chunk:
                    chunk_count += 1
                    total_pulse += len(pulse_chunk)
                    logging.info(f"[PulseGen] Macro {macro_idx+1}/{total_macro} - Yield chunk {chunk_count}: {len(pulse_chunk)} pulse ({len(pulse_chunk)//2} passi)")
                    yield pulse_chunk
                logging.info(f"[PulseGen] Macro {macro_idx+1}/{total_macro} - Totale chunk: {chunk_count}, pulse: {total_pulse}, passi: {total_pulse//2}")
            return pulse_generator
    
        macro_generators = []
        for macro_idx in range(total_macro):
            start_step = macro_idx * steps_per_macro
            end_step = min((macro_idx + 1) * steps_per_macro, master_steps)
            macro_generators.append(make_pulse_generator_macro(start_step, end_step, macro_idx, total_macro))
    
        # Se serve solo un macro-movimento, restituisci il generatore singolo per retrocompatibilità
        if len(macro_generators) == 1:
            return macro_generators[0](), active_motors, directions_to_set
        else:
            return macro_generators, active_motors, directions_to_set
class MotorController:
    def __init__(self, motor_configs: dict[str, MotorConfig]):
        self.pi = self._get_pigpio_instance()
        self.motor_configs = motor_configs
        self.switch_states = {}
        self._callbacks = []
        self._pin_to_switch_map = {}
        self.last_move_interrupted = False
        self._initialize_gpio_pins()

    def _get_pigpio_instance(self):
        logging.info("Tentativo di connessione a pigpio...")
        try:
            if IS_RPI:
                pi = pigpio.pi()
                if not pi.connected: raise ConnectionError("Connessione a pigpiod fallita.")
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
            self.pi.set_pull_up_down(config.dir_pin, pigpio.PUD_DOWN)
            self.pi.set_mode(config.dir_pin, pigpio.OUTPUT)
            self.pi.set_mode(config.en_pin, pigpio.OUTPUT)
            self.pi.write(config.en_pin, 1)
            self.pi.write(config.dir_pin, 0)

        for motor, pins in SWITCHES.items():
            for name, pin in pins.items():
                switch_id = f"{motor}_{name.lower()}"
                self.pi.set_mode(pin, pigpio.INPUT)
                self.pi.set_pull_up_down(pin, pigpio.PUD_DOWN)
                self.switch_states[switch_id] = self.pi.read(pin) == 1 
                self._pin_to_switch_map[pin] = switch_id
                cb = self.pi.callback(pin, pigpio.RISING_EDGE, self._switch_callback)
                self._callbacks.append(cb)
        logging.info(f"Finecorsa inizializzati in modalità PULL-DOWN. Stato attuale: {self.switch_states}")

    def _switch_callback(self, gpio, level, tick):
        switch_id = self._pin_to_switch_map.get(gpio)
        if switch_id and not self.switch_states.get(switch_id, True):
            self.switch_states[switch_id] = True
            self.last_move_interrupted = True
            logging.warning(f"!!! FINE CORSA ATTIVATO: {switch_id.upper()} (GPIO: {gpio}) !!! ARRESTO IMMEDIATO.")
            if self.pi.wave_tx_busy():
                self.pi.wave_tx_stop()
            motor, state = switch_id.split('_')
            other_state = 'end' if state == 'start' else 'start'
            other_switch_id = f"{motor}_{other_state}"
            if other_switch_id in self.switch_states:
                self.switch_states[other_switch_id] = False

    def execute_streamed_move(self, pulse_generator: object, active_motors: set, directions: dict):
        if not self.pi or not self.pi.connected:
            raise ConnectionError("Esecuzione fallita: pigpio non connesso.")
        
        self.last_move_interrupted = False

        # Imposta e mantieni lo stato dei pin di direzione
        for motor_id, direction in directions.items():
            dir_pin = self.motor_configs[motor_id].dir_pin
            self.pi.write(dir_pin, direction)
            logging.info(f"Impostata direzione per '{motor_id}' a {direction}")

        # Attiva i motori (pin EN)
        for motor_name in active_motors:
            self.pi.write(self.motor_configs[motor_name].en_pin, 0)
        time.sleep(0.01)

        wave_ids = []
        try:
            for chunk in pulse_generator:
                if not chunk: 
                    continue
                logging.debug(f"Chunk con {len(chunk)} pulses")
                self.pi.wave_add_generic(chunk)
                wave_id = self.pi.wave_create()

                if wave_id >= 0:
                    wave_ids.append(wave_id)
                else:
                    logging.error(f"Errore critico (codice {wave_id}) durante creazione waveform. Movimento annullato.")
                    return

            if not wave_ids:
                logging.warning("Nessuna waveform valida generata, nessun movimento eseguito.")
                return

            logging.info(f"Invio catena a pigpio con {len(wave_ids)} waves.")
            self.pi.wave_chain(wave_ids)

            while self.pi.wave_tx_busy():
                if self.last_move_interrupted:
                    logging.warning("Interruzione da finecorsa rilevata. Arresto della catena.")
                    self.pi.wave_tx_stop()
                    break
                time.sleep(0.05)
            
            if self.last_move_interrupted:
                logging.warning("MOVIMENTO INTERROTTO da un finecorsa.")
            else:
                logging.info("Movimento completato normalmente.")

        finally:
            if self.pi and self.pi.connected:
                if self.pi.wave_tx_busy():
                    self.pi.wave_tx_stop()
                self.pi.wave_clear()
                for config in self.motor_configs.values():
                    self.pi.write(config.en_pin, 1)
                    self.pi.write(config.dir_pin, 0)
                    logging.info(f"DIR pin: {config.dir_pin} | Valore impostato: 0")
                logging.info(f"Pulizia post-movimento completata (wave_clear eseguito).")

    def execute_homing_sequence(self, motor_name: str):
        if motor_name not in SWITCHES:
            logging.error(f"Impossibile eseguire homing: '{motor_name}' non ha finecorsa.")
            return
        logging.info(f"Avvio sequenza di Homing per '{motor_name}'...")
        switch_pin = SWITCHES[motor_name]['Start']
        switch_id = f"{motor_name}_start"
        config = self.motor_configs[motor_name]
        for cb in self._callbacks:
            if cb.gpio() == switch_pin:
                cb.cancel()
                break
        homing_hit = threading.Event()
        def homing_callback(gpio, level, tick):
            if level == 1:
                self.pi.wave_tx_stop()
                self.switch_states[switch_id] = True
                self.switch_states[f"{motor_name}_end"] = False
                homing_hit.set()
                logging.info(f"Homing: finecorsa {switch_id.upper()} raggiunto.")
        cb_homing = self.pi.callback(switch_pin, pigpio.RISING_EDGE, homing_callback)
        self.pi.write(config.en_pin, 0)
        self.pi.write(config.dir_pin, 0)
        self.pi.wave_clear()
        period_us = int(1_000_000 / 1000)
        pulse = [pigpio.pulse(1 << config.step_pin, 0, period_us // 2), pigpio.pulse(0, 1 << config.step_pin, period_us // 2)]
        self.pi.wave_add_generic(pulse)
        wave_id = self.pi.wave_create()
        self.pi.wave_send_repeat(wave_id)
        hit = homing_hit.wait(timeout=30)
        self.pi.wave_tx_stop()
        self.pi.wave_delete(wave_id)
        cb_homing.cancel()
        self._callbacks = [cb for cb in self._callbacks if cb.gpio() != switch_pin]
        new_cb = self.pi.callback(switch_pin, pigpio.RISING_EDGE, self._switch_callback)
        self._callbacks.append(new_cb)
        self.pi.write(config.en_pin, 1)
        if not hit: logging.error(f"Homing fallito per '{motor_name}': timeout.")
        else: logging.info(f"Homing per '{motor_name}' completato.")

def load_system_config() -> dict[str, MotorConfig]:
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
                acceleration_mmss = float(params.get("acceleration", 100.0))
                steps_per_mm = (steps_per_rev * microsteps) / pitch
                max_freq_hz = max_speed_mms * steps_per_mm

                configs[name] = MotorConfig(
                    name=name,
                    step_pin=MOTORS[name]["STEP"],
                    dir_pin=MOTORS[name]["DIR"],
                    en_pin=MOTORS[name]["EN"],
                    steps_per_mm=steps_per_mm,
                    max_freq_hz=max_freq_hz,
                    acceleration_mmss=acceleration_mmss
                )
            logging.info("Configurazione motori caricata (acceleration interpretata come mm/s^2).")
    except Exception as e:
        logging.error(f"Errore caricamento configurazione: {e}")
    return configs

# --- Blocco di Inizializzazione e Worker ---

MOTOR_CONFIGS = load_system_config()
MOTION_PLANNER = MotionPlanner(MOTOR_CONFIGS)
MOTOR_CONTROLLER = MotorController(MOTOR_CONFIGS)
motor_command_queue = queue.Queue()
SYSTEM_CONFIG_LOCK = threading.Lock()
            
def motor_worker():
    logging.info("Motor worker avviato con architettura a streaming e gestione finecorsa.")
    while True:
        task = motor_command_queue.get()
        try:
            command = task.get("command", "move")
            if command == "move":
                targets = task.get("targets", {})
                logging.info(f"Worker: ricevuto comando MOVE {targets}")
                with SYSTEM_CONFIG_LOCK:
                    current_switch_states = MOTOR_CONTROLLER.switch_states.copy()
                    # Passa anche l'istanza pi per aggiornare i finecorsa reali
                    pulse_generator, active_motors, directions = MOTION_PLANNER.plan_move_streamed(
                        targets, current_switch_states, pi=MOTOR_CONTROLLER.pi
                    )
                MOTOR_CONTROLLER.execute_streamed_move(pulse_generator, active_motors, directions)

            elif command == "home":
                motor_to_home = task.get("motor")
                logging.info(f"Worker: ricevuto comando HOME per '{motor_to_home}'")
                if motor_to_home: MOTOR_CONTROLLER.execute_homing_sequence(motor_to_home)
            
            logging.info(f"Worker: task {task} completato con successo.")
        except Exception as e:
            logging.error(f"Errore critico nel motor_worker su task {task}: {e}", exc_info=True)
        finally:
            motor_command_queue.task_done()
            time.sleep(0.1)

def handle_exception(e):
    import traceback
    error_details = traceback.format_exc()
    logging.error(f"Errore interno API: {error_details}")
    return JsonResponse({"log": f"Errore interno: {type(e).__name__}", "error": str(e)}, status=500)

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
        motor_command_queue.put({"command": "move", "targets": targets})
        return JsonResponse({"log": "Movimento messo in coda", "status": "queued"})
    except Exception as e: return handle_exception(e)

@api_view(['POST'])
def home_motor_view(request):
    try:
        data = json.loads(request.body)
        motor_to_home = data.get("motor")
        if not motor_to_home or not motor_to_home not in MOTORS:
            return JsonResponse({"log": "Input non valido", "error": f"Specificare un motore valido: {list(MOTORS.keys())}"}, status=400)
        logging.info(f"Richiesta API di Homing per il motore: {motor_to_home}")
        motor_command_queue.put({"command": "home", "motor": motor_to_home})
        return JsonResponse({"log": f"Comando di Homing per '{motor_to_home}' messo in coda.", "status": "queued"})
    except Exception as e: return handle_exception(e)

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
                motor_command_queue.put({"command": "move", "targets": step})
        return JsonResponse({"log": f"Rotta con {len(route)} passi accodata.", "status": "queued"})
    except Exception as e: return handle_exception(e)

@api_view(['POST'])
def stop_motor_view(request):
    try:
        logging.info("Richiesta di STOP motori ricevuta.")
        while not motor_command_queue.empty():
            try: motor_command_queue.get_nowait(); motor_command_queue.task_done()
            except queue.Empty: continue
        if MOTOR_CONTROLLER.pi and MOTOR_CONTROLLER.pi.connected:
            MOTOR_CONTROLLER.pi.wave_tx_stop()
        logging.info("Movimento fermato e coda di comandi pulita.")
        return JsonResponse({"log": "Comando di stop inviato.", "status": "success"})
    except Exception as e: return handle_exception(e)

@api_view(['POST'])
def update_config_view(request):
    global MOTOR_CONFIGS, MOTION_PLANNER, MOTOR_CONTROLLER
    logging.warning("Inizio procedura di Hot-Reload della configurazione di sistema.")
    with SYSTEM_CONFIG_LOCK:
        try:
            if MOTOR_CONTROLLER.pi and MOTOR_CONTROLLER.pi.connected and MOTOR_CONTROLLER.pi.wave_tx_busy():
                MOTOR_CONTROLLER.pi.wave_tx_stop()
            logging.info("Movimento corrente interrotto per aggiornamento configurazione.")
            
            new_configs = load_system_config()
            if not new_configs: 
                raise ValueError("Caricamento nuova configurazione fallito o file vuoto.")
            
            MOTOR_CONFIGS = new_configs
            MOTION_PLANNER = MotionPlanner(new_configs)
            if MOTOR_CONTROLLER and MOTOR_CONTROLLER.pi and MOTOR_CONTROLLER.pi.connected:
                for cb in MOTOR_CONTROLLER._callbacks:
                    cb.cancel()
            MOTOR_CONTROLLER = MotorController(new_configs)
            
            logging.info("Hot-Reload completato. Il sistema ora usa la nuova configurazione.")
            return JsonResponse({"log": "Configurazione aggiornata e ricaricata con successo.", "status": "success"})
        except Exception as e:
            logging.error(f"Fallimento durante la procedura di Hot-Reload: {e}", exc_info=True)
            return JsonResponse({"log": "Errore critico durante l'hot-reload.", "error": str(e)}, status=500)

@api_view(['POST'])
def start_simulation_view(request):
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
            motor_command_queue.put({"command": "move", "targets": step})
        return JsonResponse({"log": "Simulazione accodata. Controllare i log per il completamento.", "status": "queued"})
    except Exception as e: return handle_exception(e)

@api_view(['POST'])
def save_motor_config_view(request):
    try:
        with open(SETTINGS_FILE, 'r') as f: 
            full_config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError): full_config = {}
    new_data = json.loads(request.body)
    full_config.setdefault('motors', {}).update(new_data)
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(full_config, f, indent=4)
    msg = "Configurazione motori salvata. Chiamare /motors/update_config/ per applicare le modifiche."
    logging.info(msg)
    return JsonResponse({"log": msg, "success": True})

@csrf_exempt
@api_view(['GET'])
def get_motor_speeds_view(request):
    return JsonResponse({"log": "API di velocità non implementata nella nuova architettura", "speeds": {}})

@api_view(['GET'])
def get_motor_max_speeds_view(request):
    try:
        speeds = {}
        for name, config in MOTOR_CONFIGS.items():
            # Calcola la velocità massima in mm/s
            speed_mm_s = config.max_freq_hz / config.steps_per_mm
            speeds[name] = round(speed_mm_s, 3)
        return JsonResponse({"status": "success", "speeds": speeds})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)
    
@api_view(['GET'])
def get_motor_status_view(request):
    try:
        with SYSTEM_CONFIG_LOCK:
            status_data = {
                "switch_states": MOTOR_CONTROLLER.switch_states,
                "last_move_interrupted": MOTOR_CONTROLLER.last_move_interrupted,
            }
        return JsonResponse(status_data)
    except Exception as e:
        return handle_exception(e)
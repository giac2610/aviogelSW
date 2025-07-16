# -*- coding: utf-8 -*-

# ==============================================================================
# ARCHITETTURA STREAMING con Accelerazione per Motore (Versione Finale Stabile)
# Questa versione implementa la pianificazione a macro-movimenti e
# un'esecuzione "a tappe" continua per la massima stabilità e fluidità.
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
    level=logging.warning,
    format='%(asctime)s - %(levelname)s - [%(threadName)s:%(funcName)s] - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler()]
)

# --- MAPPATURA HARDWARE FONDAMENTALE ---
MOTORS = {
    "extruder": {"STEP": 13, "DIR": 6, "EN": 3},
    "syringe": {"STEP": 18, "DIR": 27, "EN": 8},
    "conveyor": {"STEP": 12, "DIR": 5, "EN": 7}
}

SWITCHES = {
    "extruder": {"Start": 23, "End": 24},
    # "syringe": {"Start": 19, "End": 26},
    "syringe": {"Start": 26, "End": 19},
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
    homeDir: bool
    
class MotionPlanner:
    def __init__(self, motor_configs: dict[str, MotorConfig]):
        self.motor_configs = motor_configs
        logging.warning(f"MotionPlanner inizializzato con motori: {list(motor_configs.keys())}")

    def _generate_trapezoidal_profile(self, total_steps: int, max_freq_hz: float, accel_mmss: float, steps_per_mm: float) -> list[float]:
        # Questa funzione è corretta
        if total_steps == 0: return []
        v_max = max_freq_hz / steps_per_mm
        if accel_mmss <= 0: accel_mmss = 1.0
        t_accel = v_max / accel_mmss
        s_accel = 0.5 * accel_mmss * t_accel**2
        steps_accel = int(s_accel * steps_per_mm)
        steps_accel = min(steps_accel, total_steps // 2)
        if steps_accel == 0 and total_steps > 0: steps_accel = 1
        if steps_accel * 2 > total_steps:
            steps_accel = total_steps // 2
            v_peak = math.sqrt(2 * accel_mmss * (steps_accel / steps_per_mm))
            v_max = v_peak
        i = np.arange(1, total_steps + 1, dtype=np.float64)
        freq = np.full(total_steps, v_max * steps_per_mm, dtype=np.float64)
        if steps_accel > 0:
            accel_mask = i <= steps_accel
            v_i_accel = np.sqrt(2 * accel_mmss * (i[accel_mask] / steps_per_mm))
            freq[accel_mask] = v_i_accel * steps_per_mm
            decel_start_step = total_steps - steps_accel
            decel_mask = i > decel_start_step
            steps_from_end = total_steps - i[decel_mask] + 1
            v_i_decel = np.sqrt(2 * accel_mmss * (steps_from_end / steps_per_mm))
            freq[decel_mask] = v_i_decel * steps_per_mm
        np.maximum(freq, 1.0, out=freq)
        periods_us = 1_000_000.0 / freq
        return periods_us.tolist()

    def plan_move_streamed(self, targets: dict[str, float], switch_states: dict, pi=None):
        if not targets: return None, set(), {}, {}
        if pi:
            for motor, pins in SWITCHES.items():
                for name, pin in pins.items():
                    switch_id = f"{motor}_{name.lower()}"
                    is_pressed = pi.read(pin) == 1
                    switch_states[switch_id] = is_pressed

        move_data = {}
        for motor_id, distance in targets.items():
            if distance == 0 or motor_id not in self.motor_configs: continue
            direction = 1 if distance >= 0 else 0
            if motor_id != "conveyor":
                if (direction == 0 and switch_states.get(f"{motor_id}_start")) or \
                   (direction == 1 and switch_states.get(f"{motor_id}_end")):
                    logging.warning(f"Movimento per '{motor_id}' bloccato da finecorsa attivo.")
                    continue
            config = self.motor_configs[motor_id]
            move_data[motor_id] = {"steps": int(abs(distance) * config.steps_per_mm), "dir": direction, "config": config}

        if not move_data: return None, set(), {}, {}

        master_id = max(move_data, key=lambda k: move_data[k]["steps"])
        master_data = move_data[master_id]

        # --- MODIFICA CHIAVE ---
        # Salviamo i passi totali del master PRIMA di troncarli. Questo è il valore corretto per l'algoritmo.
        master_steps_total = master_data["steps"]

        # Ora possiamo tranquillamente troncare il numero di passi per il blocco corrente.
        master_steps_chunk = master_steps_total

        MAX_STEPS_PER_COMMAND = 5500
        if master_steps_chunk > MAX_STEPS_PER_COMMAND:
            logging.warning(f"MOVIMENTO TROPPO LUNGO ({master_steps_chunk} passi)! Troncato al limite di sicurezza di {MAX_STEPS_PER_COMMAND} passi per questo blocco.")
            master_steps_chunk = MAX_STEPS_PER_COMMAND

        if master_steps_chunk == 0: return None, set(), {}, {}

        periods_list = self._generate_trapezoidal_profile(
            master_steps_chunk, master_data["config"].max_freq_hz, master_data["config"].acceleration_mmss, master_data["config"].steps_per_mm
        )
        active_motors = {m["config"].name for m in move_data.values()}
        directions_to_set = {mid: move['dir'] for mid, move in move_data.items()}

        def single_pulse_generator():
            # --- MODIFICA CHIAVE ---
            # L'algoritmo di Bresenham ora usa master_steps_total per il calcolo della proporzione.
            bresenham_errors = {mid: -master_steps_total / 2 for mid in move_data if mid != master_id}

            # Il ciclo for, invece, itera solo per la lunghezza del blocco corrente.
            for i in range(master_steps_chunk):
                on_mask = 1 << master_data["config"].step_pin
                for slave_id in bresenham_errors:
                    # L'incremento usa ancora i passi totali dello slave (corretto).
                    bresenham_errors[slave_id] += move_data[slave_id]["steps"]
                    if bresenham_errors[slave_id] > 0:
                        on_mask |= 1 << move_data[slave_id]["config"].step_pin
                        # Il decremento usa i passi totali del master (ora è corretto!).
                        bresenham_errors[slave_id] -= master_steps_total

                total_period_us = periods_list[i]
                pulse_width_us = max(1, int(total_period_us / 2))
                yield pigpio.pulse(on_mask, 0, pulse_width_us)
                yield pigpio.pulse(0, on_mask, max(1, int(total_period_us - pulse_width_us)))

        # Calcoliamo i passi effettivi per ogni motore in questo blocco
        steps_in_chunk = {master_id: master_steps_chunk}
        if master_steps_total > 0: # Evita divisione per zero se il movimento è nullo
            for slave_id, data in move_data.items():
                if slave_id != master_id:
                    # La logica di Bresenham distribuisce i passi in modo proporzionale
                    steps_in_chunk[slave_id] = int((data["steps"] / master_steps_total) * master_steps_chunk)

        logging.warning(f"Pianificato blocco per '{master_id}' ({master_steps_chunk} passi). Motori attivi: {list(active_motors)}")
        return single_pulse_generator(), active_motors, directions_to_set, steps_in_chunk
class MotorController:
    def __init__(self, motor_configs: dict[str, MotorConfig]):
        self.pi = self._get_pigpio_instance()
        self.motor_configs = motor_configs
        self.switch_states = {}
        self.motor_error_state = {motor: False for motor in motor_configs.keys()}
        self._callbacks = {}
        self._pin_to_switch_map = {}
        self.last_move_interrupted = False
        self._initialize_gpio_pins()

    def _get_pigpio_instance(self):
        logging.warning("Tentativo di connessione a pigpio...")
        try:
            if IS_RPI:
                pi = pigpio.pi()
                if not pi.connected: raise ConnectionError("Connessione a pigpiod fallita.")
                logging.warning("Connessione a pigpio stabilita.")
                return pi
            else:
                logging.warning("Utilizzo di istanza pigpio simulata.")
                return pigpio.pi()
        except Exception as e:
            logging.warning(f"Fallimento init pigpio: {e}")
            return None

    def _initialize_gpio_pins(self):
        if not self.pi or not self.pi.connected: return
        for config in self.motor_configs.values():
            self.pi.set_mode(config.step_pin, pigpio.OUTPUT)
            self.pi.set_pull_up_down(config.dir_pin, pigpio.PUD_DOWN)
            self.pi.set_mode(config.dir_pin, pigpio.OUTPUT)
            self.pi.set_mode(config.en_pin, pigpio.OUTPUT)
            self.pi.write(config.en_pin, 0)
            self.pi.write(config.dir_pin, 0)

        for motor, pins in SWITCHES.items():
            for name, pin in pins.items():
                switch_id = f"{motor}_{name.lower()}"
                self.pi.set_mode(pin, pigpio.INPUT)
                self.pi.set_pull_up_down(pin, pigpio.PUD_DOWN)
                is_pressed = self.pi.read(pin) == 1
                self.switch_states[switch_id] = is_pressed
                self._pin_to_switch_map[pin] = switch_id
                cb = self.pi.callback(pin, pigpio.EITHER_EDGE, self._switch_callback)
                self._callbacks[pin] = cb
        logging.warning(f"Finecorsa inizializzati. Stato attuale: {self.switch_states}")

    def _switch_callback(self, gpio, level, tick):
        switch_id = self._pin_to_switch_map.get(gpio)
        if switch_id:
            is_pressed = (level == 1)
            self.switch_states[switch_id] = is_pressed
            if is_pressed:
                self.last_move_interrupted = True
                logging.warning(f"!!! FINE CORSA ATTIVATO: {switch_id.upper()} (GPIO: {gpio}) !!!")
                if self.pi.wave_tx_busy(): 
                    self.pi.wave_tx_stop()

    def _prepare_waves_from_generator(self, pulse_generator: object) -> list[int]:
        if not self.pi or not self.pi.connected:
            raise ConnectionError("Preparazione fallita: pigpio non connesso.")
        
        PULSE_BLOCK_THRESHOLD = 3800
        wave_ids = []
        all_pulses = list(pulse_generator)
        if not all_pulses: return []

        for i in range(0, len(all_pulses), PULSE_BLOCK_THRESHOLD):
            pulse_block = all_pulses[i:i + PULSE_BLOCK_THRESHOLD]
            self.pi.wave_add_generic(pulse_block)
            wave_id = self.pi.wave_create()
            if wave_id >= 0:
                wave_ids.append(wave_id)
            else:
                for wid in wave_ids: self.pi.wave_delete(wid)
                raise pigpio.error(f'Creazione onda fallita con codice {wave_id}')
        return wave_ids

    def execute_homing_sequence(self, motor_name: str):
        if motor_name not in SWITCHES:
            logging.warning(f"Impossibile eseguire homing: '{motor_name}' non ha finecorsa.")
            return

        self.motor_error_state[motor_name] = False

        logging.warning(f"Avvio sequenza di Homing per '{motor_name}'...")
        start_switch_pin = SWITCHES[motor_name]['Start']
        end_switch_pin = SWITCHES[motor_name]['End']
        switch_id = f"{motor_name}_start"
        end_switch_id = f"{motor_name}_end"
        config = self.motor_configs[motor_name]
        
        logging.warning("Homing: Fase 1 - Ricerca veloce del finecorsa...")

        homing_hit = threading.Event()
        end_switch_hit = threading.Event()

        # Callback per il finecorsa di START
        def homing_callback(gpio, level, tick):
            if level == 1:
                self.pi.wave_tx_stop()
                homing_hit.set()

        # Callback per il finecorsa di END (emergenza) ##
        def end_switch_callback(gpio, level, tick):
            if level == 1:
                self.pi.wave_tx_stop()
                end_switch_hit.set()

        # Cancella eventuali callback esistenti sui pin che useremo
        for pin in [start_switch_pin, end_switch_pin]:
            existing_cb = self._callbacks.pop(pin, None)
            if existing_cb: existing_cb.cancel()

        # Imposta i nuovi callback temporanei per l'homing
        cb_homing = self.pi.callback(start_switch_pin, pigpio.RISING_EDGE, homing_callback)
        cb_end = self.pi.callback(end_switch_pin, pigpio.RISING_EDGE, end_switch_callback)

        self.pi.write(config.en_pin, 0)
        self.pi.write(config.dir_pin, config.homeDir)
        self.pi.wave_clear()

        if(MOTOR_CONTROLLER.switch_states.get(f"{motor_name}_start")):
            logging.warning(f"Fase 1 Homing per '{motor_name}' già completato: finecorsa di START attivo.")
            logging.warning("Homing: Fase 2 - Back-off lento per rilascio sensore...")
            backoff_done = threading.Event()
            def backoff_callback(gpio, level, tick):
                if level == 0:
                    self.pi.wave_tx_stop()
                    backoff_done.set()
    
            cb_backoff = self.pi.callback(start_switch_pin, pigpio.FALLING_EDGE, backoff_callback)
            self.pi.write(config.dir_pin, not config.homeDir) 
    
            period_us_slow = int(1_000_000 / 200) # 200 Hz
            pulse_slow = [pigpio.pulse(1 << config.step_pin, 0, period_us_slow // 2), pigpio.pulse(0, 1 << config.step_pin, period_us_slow // 2)]
            self.pi.wave_add_generic(pulse_slow)
            wave_id_slow = self.pi.wave_create()
            self.pi.wave_send_repeat(wave_id_slow)
    
            backed_off = backoff_done.wait(timeout=10)
            self.pi.wave_tx_stop()
            self.pi.wave_delete(wave_id_slow)
            cb_backoff.cancel()
    
            self._callbacks[start_switch_pin] = self.pi.callback(start_switch_pin, pigpio.EITHER_EDGE, self._switch_callback)
            self._callbacks[end_switch_pin] = self.pi.callback(end_switch_pin, pigpio.EITHER_EDGE, self._switch_callback)
            self.pi.write(config.en_pin, 1)
    
            if not backed_off:
                self.switch_states[switch_id] = True
                logging.warning(f"Homing Fase 2 (Back-off) fallita per '{motor_name}': timeout.")
            else:
                self.switch_states[switch_id] = False
                logging.warning(f"Homing per '{motor_name}' completato con successo. Posizione zero definita.")
    
            self.switch_states[end_switch_id] = False
        else:
            period_us = int(1_000_000 / 2000) # 2000 Hz
            pulse = [pigpio.pulse(1 << config.step_pin, 0, period_us // 2), pigpio.pulse(0, 1 << config.step_pin, period_us // 2)]
            self.pi.wave_add_generic(pulse)
            wave_id = self.pi.wave_create()
            self.pi.wave_send_repeat(wave_id)
            
            start_time = time.time()
            timeout = 10
            while not (homing_hit.is_set() or end_switch_hit.is_set()):
                if time.time() - start_time > timeout:
                    self.pi.wave_tx_stop() # Ferma il motore in caso di timeout
                    break
                time.sleep(0.01)

            self.pi.wave_delete(wave_id)
            cb_homing.cancel()
            cb_end.cancel()

        if end_switch_hit.is_set():
            self.pi.write(config.en_pin, 1)
            self.motor_error_state[motor_name] = True
            self._callbacks[start_switch_pin] = self.pi.callback(start_switch_pin, pigpio.EITHER_EDGE, self._switch_callback)
            self._callbacks[end_switch_pin] = self.pi.callback(end_switch_pin, pigpio.EITHER_EDGE, self._switch_callback)
            logging.warning(f"Homing per '{motor_name}' fallito: Finecorsa di EMERGENZA (END) attivato!")
            logging.warning(f"Il motore '{motor_name}' è bloccato. Eseguire un nuovo homing per sbloccarlo.")
            return

        if not homing_hit.is_set():
            self.pi.write(config.en_pin, 1)
            self._callbacks[start_switch_pin] = self.pi.callback(start_switch_pin, pigpio.EITHER_EDGE, self._switch_callback)
            self._callbacks[end_switch_pin] = self.pi.callback(end_switch_pin, pigpio.EITHER_EDGE, self._switch_callback)
            logging.warning(f"Homing Fase 1 fallita per '{motor_name}': timeout.")
 
            return

        logging.warning("Homing: Finecorsa toccato.")
        time.sleep(0.1)

        # --- FASE 2: Back-off lento dal finecorsa ---
        logging.warning("Homing: Fase 2 - Back-off lento per rilascio sensore...")
        backoff_done = threading.Event()
        def backoff_callback(gpio, level, tick):
            if level == 0:
                self.pi.wave_tx_stop()
                backoff_done.set()

        cb_backoff = self.pi.callback(start_switch_pin, pigpio.FALLING_EDGE, backoff_callback)
        self.pi.write(config.dir_pin, not config.homeDir) 

        period_us_slow = int(1_000_000 / 200) # 200 Hz
        pulse_slow = [pigpio.pulse(1 << config.step_pin, 0, period_us_slow // 2), pigpio.pulse(0, 1 << config.step_pin, period_us_slow // 2)]
        self.pi.wave_add_generic(pulse_slow)
        wave_id_slow = self.pi.wave_create()
        self.pi.wave_send_repeat(wave_id_slow)

        backed_off = backoff_done.wait(timeout=10)
        self.pi.wave_tx_stop()
        self.pi.wave_delete(wave_id_slow)
        cb_backoff.cancel()

        self._callbacks[start_switch_pin] = self.pi.callback(start_switch_pin, pigpio.EITHER_EDGE, self._switch_callback)
        self._callbacks[end_switch_pin] = self.pi.callback(end_switch_pin, pigpio.EITHER_EDGE, self._switch_callback)
        self.pi.write(config.en_pin, 1)

        if not backed_off:
            self.switch_states[switch_id] = True
            logging.warning(f"Homing Fase 2 (Back-off) fallita per '{motor_name}': timeout.")
        else:
            self.switch_states[switch_id] = False
            logging.warning(f"Homing per '{motor_name}' completato con successo. Posizione zero definita.")

        self.switch_states[end_switch_id] = False
    
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
                homeDir = bool(params.get("homeDir", True))
                configs[name] = MotorConfig(
                    name=name,
                    step_pin=MOTORS[name]["STEP"],
                    dir_pin=MOTORS[name]["DIR"],
                    en_pin=MOTORS[name]["EN"],
                    steps_per_mm=steps_per_mm,
                    max_freq_hz=max_freq_hz,
                    acceleration_mmss=acceleration_mmss,
                    homeDir=homeDir
                )
            logging.warning("Configurazione motori caricata.")
    except Exception as e:
        logging.warning(f"Errore caricamento configurazione: {e}")
    return configs

# --- Blocco di Inizializzazione e Worker ---
MOTOR_CONFIGS = load_system_config()
MOTION_PLANNER = MotionPlanner(MOTOR_CONFIGS)
MOTOR_CONTROLLER = MotorController(MOTOR_CONFIGS)
motor_command_queue = queue.Queue()
SYSTEM_CONFIG_LOCK = threading.Lock()
            
def motor_worker():
    logging.warning("Motor worker avviato con architettura a cicli continui.")
    while True:
        task = motor_command_queue.get()
        try:
            command = task.get("command", "move")
            
            if command == "move":
                remaining_targets = task.get("targets", {}).copy()
                
                # Cicla finché c'è ancora distanza significativa da percorrere
                while any(abs(dist) > 0.01 for dist in remaining_targets.values()):
                    if MOTOR_CONTROLLER.last_move_interrupted:
                        # Trova quali motori sono bloccati da finecorsa
                        blocked = []
                        for motor_id in list(remaining_targets.keys()):
                            if motor_id != "conveyor":
                                # Recupera la distanza e la configurazione del motore
                                distance = remaining_targets[motor_id]
                                # config = MOTOR_CONFIGS[motor_id] 
                                config = MOTOR_CONTROLLER.motor_configs.get(motor_id, None)
                                # Calcola la direzione (0 o 1) del movimento residuo
                                direction_for_pin = 1 if distance >= 0 else 0
                                # Recupera la direzione "verso start" (0 o 1) dalla configurazione
                                config_dir_to_start = config.get("homeDir", 1)
                                # Controlla se il movimento è verso un finecorsa attivo
                                is_moving_towards_start = (direction_for_pin == config_dir_to_start)
                                is_moving_towards_end = (direction_for_pin != config_dir_to_start)
                                # final check for end switches
                                is_blocked_at_start = is_moving_towards_start and MOTOR_CONTROLLER.switch_states.get(f"{motor_id}_start")
                                is_blocked_at_end = is_moving_towards_end and MOTOR_CONTROLLER.switch_states.get(f"{motor_id}_end")
                                if is_blocked_at_start or is_blocked_at_end:
                                    blocked.append(motor_id)
                        for motor_id in blocked:
                            logging.warning(f"Motore '{motor_id}' bloccato da finecorsa: rimosso dai target.")
                            del remaining_targets[motor_id]
                        MOTOR_CONTROLLER.last_move_interrupted = False
                        if not remaining_targets:
                            logging.warning("Tutti i motori bloccati da finecorsa. Uscita dal ciclo.")
                            break
                        # Rimuovi eventuali motori ancora bloccati prima di accodare il residuo
                        for motor_id in list(remaining_targets.keys()):
                            if motor_id != "conveyor":
                                if (remaining_targets[motor_id] < 0 and MOTOR_CONTROLLER.switch_states.get(f"{motor_id}_end")) or \
                                   (remaining_targets[motor_id] > 0 and MOTOR_CONTROLLER.switch_states.get(f"{motor_id}_start")):
                                    logging.warning(f"Motore '{motor_id}' ancora bloccato da finecorsa: non verrà accodato nel residuo.")
                                    del remaining_targets[motor_id]
                        # Accoda solo se rimangono target validi
                        if any(abs(dist) > 0.001 for dist in remaining_targets.values()):
                            logging.warning(f"Accodo movimento residuo per motori non bloccati: {remaining_targets}")
                            motor_command_queue.put({"command": "move", "targets": remaining_targets.copy()})
                        break
                    with SYSTEM_CONFIG_LOCK:
                        current_switch_states = MOTOR_CONTROLLER.switch_states.copy()
                        pulse_generator, active_motors, directions, steps_executed = MOTION_PLANNER.plan_move_streamed(
                            remaining_targets, current_switch_states, pi=MOTOR_CONTROLLER.pi
                        )

                    if not active_motors or not pulse_generator:
                        logging.warning("Nessun movimento pianificato per i target rimanenti. Uscita dal ciclo.")
                        break

                    all_wave_ids = []
                    try:
                        all_wave_ids = MOTOR_CONTROLLER._prepare_waves_from_generator(pulse_generator)
                        if not all_wave_ids:
                            continue
                        
                        # Esecuzione del blocco
                        for motor_id, direction in directions.items():
                            MOTOR_CONTROLLER.pi.write(MOTOR_CONTROLLER.motor_configs[motor_id].dir_pin, direction)
                        for motor_name in active_motors:
                            MOTOR_CONTROLLER.pi.write(MOTOR_CONTROLLER.motor_configs[motor_name].en_pin, 0)
                        
                        MOTOR_CONTROLLER.pi.wave_chain(all_wave_ids)
                        
                        while MOTOR_CONTROLLER.pi.wave_tx_busy():
                            if MOTOR_CONTROLLER.last_move_interrupted:
                                break
                            time.sleep(0.05)
                        
                    finally:
                        # Pulizia dopo ogni blocco
                        if MOTOR_CONTROLLER.pi and MOTOR_CONTROLLER.pi.connected:
                            if MOTOR_CONTROLLER.pi.wave_tx_busy(): MOTOR_CONTROLLER.pi.wave_tx_stop()
                            MOTOR_CONTROLLER.pi.wave_clear()
                            for config in MOTOR_CONTROLLER.motor_configs.values():
                                try: MOTOR_CONTROLLER.pi.write(config.en_pin, 0)
                                except Exception: pass
                    
                    # Aggiorna le distanze rimanenti
                    for motor_id, steps in steps_executed.items():
                        if motor_id in remaining_targets:
                            config = MOTOR_CONFIGS[motor_id]
                            distance_moved_mm = steps / config.steps_per_mm
                            logging.warning(f"Blocco da compiere. Distanza: {distance_moved_mm}")
                            if remaining_targets[motor_id] > 0:
                                remaining_targets[motor_id] = max(0, remaining_targets[motor_id] - distance_moved_mm)
                            else:
                                remaining_targets[motor_id] = min(0, remaining_targets[motor_id] + distance_moved_mm)
                    
                    logging.warning(f"Blocco completato. Distanze rimanenti: {remaining_targets}")

            elif command == "home":
                motor_to_home = task.get("motor")
                if motor_to_home:
                    MOTOR_CONTROLLER.execute_homing_sequence(motor_to_home)
            
            logging.warning(f"Worker: task '{task.get('command')}' completato.")

        except Exception as e:
            logging.warning(f"Errore critico nel motor_worker su task {task}: {e}", exc_info=True)
        finally:
            motor_command_queue.task_done()
            time.sleep(0.1)
                 
def handle_exception(e):
    import traceback
    error_details = traceback.format_exc()
    logging.warning(f"Errore interno API: {error_details}")
    return JsonResponse({"log": f"Errore interno: {type(e).__name__}", "error": str(e)}, status=500)

if os.environ.get('RUN_MAIN') == 'true':
    logging.warning("Processo principale di Django rilevato. Avvio del MotorWorker...")
    threading.Thread(target=motor_worker, daemon=True, name="MotorWorker").start()
else:
    logging.warning("Processo di reload rilevato. Il MotorWorker non verrà avviato qui.")

# ==============================================================================
# API VIEWS (Interfaccia esterna preservata)
# ==============================================================================
# Le API Views rimangono invariate. Sono state omesse per brevità ma sono identiche
# a quelle nel tuo file.
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
        if not motor_to_home or motor_to_home not in MOTORS:
            return JsonResponse({"log": "Input non valido", "error": f"Specificare un motore valido: {list(MOTORS.keys())}"}, status=400)
        logging.warning(f"Richiesta API di Homing per il motore: {motor_to_home}")
        motor_command_queue.put({"command": "home", "motor": motor_to_home})
        return JsonResponse({"log": f"Comando di Homing per '{motor_to_home}' messo in coda.", "status": "queued"})
    except Exception as e: return handle_exception(e)

@api_view(['POST'])
def execute_route_view(request):
    try:
        motor_command_queue.put({"command": "home", "motor": "extruder"})
        data = json.loads(request.body)
        route = data.get("route", [])
        logging.warning("ricevuta questa  rotta: " + str(route))
        if not isinstance(route, list):
            return JsonResponse({"log": "Percorso non valido", "error": "Input non valido"}, status=400)
        logging.warning(f"Accodamento rotta con {len(route)} passi.")
        for step in route:
            if isinstance(step, dict): 
                motor_command_queue.put({"command": "move", "targets": step})
        motor_command_queue.put({"command": "home", "motor": "extruder"})
        return JsonResponse({"log": f"Rotta con {len(route)} passi accodata.", "status": "queued"})
    except Exception as e: return handle_exception(e)

@api_view(['POST'])
def stop_motor_view(request):
    try:
        logging.warning("Richiesta di STOP motori ricevuta.")
        while not motor_command_queue.empty():
            try: motor_command_queue.get_nowait(); motor_command_queue.task_done()
            except queue.Empty: continue
        if MOTOR_CONTROLLER.pi and MOTOR_CONTROLLER.pi.connected:
            MOTOR_CONTROLLER.pi.wave_tx_stop()
        logging.warning("Movimento fermato e coda di comandi pulita.")
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
                MOTOR_CONTROLLER.pi.wave_clear()
            logging.warning("Movimento corrente interrotto per aggiornamento configurazione.")
            
            if MOTOR_CONTROLLER and MOTOR_CONTROLLER.pi and MOTOR_CONTROLLER.pi.connected:
                for pin, cb in MOTOR_CONTROLLER._callbacks.items():
                    cb.cancel()
            
            new_configs = load_system_config()
            if not new_configs: 
                raise ValueError("Caricamento nuova configurazione fallito o file vuoto.")
            
            MOTOR_CONFIGS = new_configs
            MOTION_PLANNER = MotionPlanner(new_configs)
            MOTOR_CONTROLLER = MotorController(new_configs)
            
            logging.warning("Hot-Reload completato. Il sistema ora usa la nuova configurazione.")
            return JsonResponse({"log": "Configurazione aggiornata e ricaricata con successo.", "status": "success"})
        except Exception as e:
            logging.warning(f"Fallimento durante la procedura di Hot-Reload: {e}", exc_info=True)
            MOTOR_CONFIGS = load_system_config()
            MOTION_PLANNER = MotionPlanner(MOTOR_CONFIGS)
            MOTOR_CONTROLLER = MotorController(MOTOR_CONFIGS)
            return JsonResponse({"log": "Errore critico durante l'hot-reload. Ricaricata config precedente.", "error": str(e)}, status=500)

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
        logging.warning("Avvio simulazione predefinita...")
        for i, step in enumerate(simulation_steps):
            logging.warning(f"Accodamento passo simulazione {i+1}: {step}")
            motor_command_queue.put({"command": "move", "targets": step})
        return JsonResponse({"log": "Simulazione accodata.", "status": "queued"})
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
    logging.warning(msg)
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
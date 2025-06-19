import json
import os
import time
import threading
import sys
import logging
import queue
from django.conf import settings
from unittest.mock import MagicMock
# Assicurati che il percorso a serializers sia corretto per il tuo progetto Django
# from .serializers import SettingsSerializer # Se è nella stessa app
# Oppure: from nome_app.serializers import SettingsSerializer
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from rest_framework.decorators import api_view

# SU MAC: Simula il modulo pigpio per evitare errori
if sys.platform == "darwin":
    sys.modules["pigpio"] = MagicMock()
import pigpio  # type: ignore

# Percorsi dei file di configurazione
# Utilizza __file__ per percorsi relativi allo script corrente
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../config')
SETTINGS_FILE = os.path.join(CONFIG_DIR, 'setup.json')
EXAMPLE_JSON_PATH = os.path.join(CONFIG_DIR, 'setup.example.json')

# Verifica ed eventualmente rigenera setup.json
if not os.path.exists(SETTINGS_FILE):
    if not os.path.exists(EXAMPLE_JSON_PATH):
        # Potrebbe essere preferibile loggare un errore o creare un file di default qui
        # piuttosto che sollevare un'eccezione che blocca l'avvio.
        print(f"[WARN] File di esempio mancante: {EXAMPLE_JSON_PATH}. Verrà creato un file di settings vuoto o di default se possibile.")
        # Per ora, manteniamo l'eccezione se il file di esempio è cruciale per la creazione.
        raise FileNotFoundError(f"File di esempio mancante: {EXAMPLE_JSON_PATH}")
    from shutil import copyfile
    # Assicurati che la directory CONFIG_DIR esista
    os.makedirs(CONFIG_DIR, exist_ok=True)
    copyfile(EXAMPLE_JSON_PATH, SETTINGS_FILE)
    print(f"[INFO] File di configurazione creato da setup.example.json in {SETTINGS_FILE}")

motor_command_queue = queue.Queue()
motor_worker_started = False

    

# Configurazione del logging
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, 'motorLog.log')
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s' # Aggiunto funcName per contesto
)

# Istanza di Serializer (se SettingsSerializer è definito)
# Questo va importato correttamente. Se non è definito, commentalo o definiscilo.
# from .serializers import SettingsSerializer # Esempio
class SettingsSerializer: # Placeholder se non disponibile
    def __init__(self, data=None, partial=False): self.data=data; self.validated_data=data or {}; self.errors="No real serializer"
    def is_valid(self): return True


# Funzione per loggare le risposte JSON
def log_json_response(response):
    try:
        logging.debug(f"JSON Response: {response.content.decode('utf-8')}")
    except Exception as e:
        logging.error(f"Error decoding JSON response for logging: {e}")

# Funzione per loggare gli errori
def log_error(error_message):
    logging.error(error_message)

def motor_worker():
    while True:
        wave_ids_dict = motor_command_queue.get()
        try:
            start_motor_movement_independent(wave_ids_dict)
        except Exception as e:
            log_error(f"Errore nel thread motor_worker durante l'esecuzione del movimento: {e}")
        finally:
            motor_command_queue.task_done()
            
if not motor_worker_started:
    threading.Thread(target=motor_worker, daemon=True).start()
    motor_worker_started = True
# ------------------------------------------------------------------------------
# Caricamento configurazione
# ------------------------------------------------------------------------------
def load_motor_config():
    try:
        with open(SETTINGS_FILE, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        logging.error(f"File di configurazione {SETTINGS_FILE} non trovato. Verrà usata una configurazione vuota/default.")
        return {"motors": {}} # Ritorna una config di default per evitare crash
    except json.JSONDecodeError:
        logging.error(f"Errore nel decodificare {SETTINGS_FILE}. Verrà usata una configurazione vuota/default.")
        return {"motors": {}}


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
pi = None
try:
    pi = pigpio.pi()
    if not pi.connected:
        logging.error("Non connesso a pigpio daemon. Assicurati che pigpiod sia in esecuzione.")
        # Non sollevare eccezione qui, altrimenti il modulo non può essere importato senza pigpio
        # Le funzioni che usano 'pi' dovranno gestire 'pi is None' o 'pi.connected is False'
except NameError: # pigpio non è definito (es. su MAC senza MagicMock completo per pi())
    logging.warning("pigpio.pi() non disponibile (probabilmente non su Raspberry Pi o pigpio non installato).")
except Exception as e: # Altre eccezioni da pigpio.pi()
    logging.error(f"Errore durante l'inizializzazione di pigpio: {e}")


if pi and pi.connected:
    for motor_name, motor_pins in MOTORS.items():
        try:
            pi.set_mode(motor_pins["STEP"], pigpio.OUTPUT)
            pi.set_mode(motor_pins["DIR"], pigpio.OUTPUT)
            pi.set_mode(motor_pins["EN"], pigpio.OUTPUT)
            logging.info(f"Perni configurati per il motore: {motor_name}")
        except Exception as e:
            logging.error(f"Errore nella configurazione dei perni per {motor_name}: {e}")
else:
    logging.warning("Connessione a pigpio non riuscita. Il controllo motore non funzionerà.")

running_flags = {motor: False for motor in MOTORS.keys()}
current_speeds = {motor: 0 for motor in MOTORS.keys()}

# ------------------------------------------------------------------------------
# Funzione di conversione passi e frequenza
# ------------------------------------------------------------------------------
def compute_motor_params(motor_id):
    motor_conf = motor_configs.get(motor_id, {})
    step_one_rev = float(motor_conf.get("stepOneRev", 200.0))
    microstep = int(motor_conf.get("microstep", 8))
    pitch = float(motor_conf.get("pitch", 5.0)) # mm per rivoluzione
    max_speed = float(motor_conf.get("maxSpeed", 250.0)) # mm/s
    acceleration = float(motor_conf.get("acceleration", 800.0)) # mm/s^2

    logging.debug(f"Parametri motore {motor_id}: stepOneRev={step_one_rev}, microstep={microstep}, "
                  f"pitch={pitch}, maxSpeed={max_speed}, acceleration={acceleration}")

    if pitch == 0:
        logging.error(f"Il 'pitch' per il motore {motor_id} non può essere zero.")
        raise ValueError(f"Pitch nullo per il motore {motor_id}")

    steps_per_rev_effettivi = step_one_rev * microstep
    steps_per_mm = steps_per_rev_effettivi / pitch
    
    # Frequenza massima in passi/secondo
    max_freq_steps_sec = max_speed * steps_per_mm
    
    # Steps necessari per accelerare da 0 a max_freq_steps_sec
    # v^2 = u^2 + 2as => s = v^2 / (2a)  (u=0)
    # s in mm, v in mm/s, a in mm/s^2
    # steps_for_accel_dist = (max_speed**2 / (2 * acceleration)) * steps_per_mm
    # Frequenza in Hz (impulsi/sec). accel in impulsi/s^2
    # accel_steps_per_sec2 = acceleration * steps_per_mm
    # accel_steps = int((max_freq_steps_sec ** 2) / max(1, (2 * accel_steps_per_sec2)))
    
    # Calcolo semplificato per accel_steps basato sul tempo per raggiungere max_speed alla data accelerazione
    # t = max_speed / acceleration (tempo per raggiungere maxSpeed in secondi)
    # freq_ramp_time_s = max_freq_steps_sec / (acceleration * steps_per_mm) # Tempo per raggiungere max_freq
    # accel_steps = int(0.5 * max_freq_steps_sec * freq_ramp_time_s) # Steps durante accelerazione triangolare

    # Utilizziamo la formula originale, ma assicuriamoci che l'accelerazione in steps/s^2 sia usata
    acceleration_steps_s2 = acceleration * steps_per_mm
    if acceleration_steps_s2 == 0:
         accel_steps = steps_per_mm * 1 # Default a 1mm di accelerazione se accelerazione è zero
         logging.warning(f"L'accelerazione per il motore {motor_id} è zero. accel_steps impostato su {accel_steps}")
    else:
        accel_steps = int((max_freq_steps_sec ** 2) / (2 * acceleration_steps_s2))


    if accel_steps == 0:
        accel_steps = 1 # Deve essere almeno 1 per evitare divisioni per zero in compute_frequency

    logging.debug(f"Esito calcoli per motore {motor_id}: steps_per_mm={steps_per_mm}, max_freq={max_freq_steps_sec}, "
                  f"accel_steps={accel_steps}, decel_steps={accel_steps}")

    return {
        "steps_per_mm": steps_per_mm,
        "max_freq": max_freq_steps_sec, # Ora è in steps/sec
        "accel_steps": accel_steps,
        "decel_steps": accel_steps, # Assumendo decelerazione simmetrica
    }

def write_settings(data):
    """Scrive i dati nel file settings.json"""
    try:
        with open(SETTINGS_FILE, 'w') as file:
            json.dump(data, file, indent=4)
        logging.info(f"Configurazione salvata in {SETTINGS_FILE}")
    except Exception as e:
        log_error(f"Errore durante la scrittura di settings.json: {e}")

# ------------------------------------------------------------------------------
# API: Aggiornamento configurazione
# ------------------------------------------------------------------------------
@api_view(['POST'])
def update_config_view(request): # Rinominata per evitare conflitto con variabile 'config'
    global config, motor_configs
    try:
        reload_motor_config()
        response = JsonResponse({"log": "Configurazione aggiornata", "status": "success"}, status=200) # 204 significa No Content
        log_json_response(response)
        return response
    except Exception as e:
        log_error(f"Errore aggiornamento configurazione: {str(e)}")
        response = JsonResponse({"log": "Errore aggiornamento configurazione", "error": str(e)}, status=500)
        log_json_response(response)
        return response

# ------------------------------------------------------------------------------
# API: Movimento motore
# ------------------------------------------------------------------------------
MAX_PULSES_PER_WAVE = 5000  # Impulsi per singola waveform pigpio (un impulso è ON o OFF)
MAX_WAVE_CHAIN_SIZE = 250   # Numero massimo di ID di waveform in una singola chiamata a pi.wave_chain()

def generate_waveform(motor_targets):
    """
    Genera una lista di ID di waveform DMA multicanale per muovere i motori.
    NON esegue il movimento.
    """
    global pi
    if not pi or not pi.connected:
        log_error("generate_waveform: pigpio non connesso.")
        raise ConnectionError("pigpio non connesso.")

    wave_accumulator = [] # Accumula impulsi (on/off) per la waveform corrente
    generated_wave_ids = [] # Lista di ID delle waveform create

    # Prepara i piani di impulsi per ogni motore
    pulse_plans = {}
    for motor_id, distance in motor_targets.items():
        if motor_id not in MOTORS:
            log_error(f"Motore {motor_id} non trovato in MOTORS.")
            raise ValueError(f"Motore non valido: {motor_id}")
        
        logging.debug(f"Target ricevuto per motore {motor_id}: distanza={distance}")
        params = compute_motor_params(motor_id)
        steps_to_make = int(abs(distance) * params["steps_per_mm"])
        direction = 1 if distance >= 0 else 0
        motor_pins = MOTORS[motor_id]

        pi.write(motor_pins["DIR"], direction)
        pi.write(motor_pins["EN"], 0) # Abilita motore

        pulse_plans[motor_id] = {
            "pin": motor_pins["STEP"],
            "steps_total": steps_to_make,
            "max_freq": params["max_freq"],
            "accel_steps": min(params["accel_steps"], steps_to_make // 2 if steps_to_make > 1 else 0), # Evita accel_steps > steps_total/2
            "decel_steps": min(params["decel_steps"], steps_to_make // 2 if steps_to_make > 1 else 0),
            "steps_done": 0,
        }
        # Correzione per accel/decel steps se steps_total è molto piccolo
        if steps_to_make <= 1:
            pulse_plans[motor_id]["accel_steps"] = 0
            pulse_plans[motor_id]["decel_steps"] = 0

        logging.debug(f"Piano di impulsi per motore {motor_id}: {pulse_plans[motor_id]}")

    # Pulisce le waveform definite precedentemente in pigpio
    pi.wave_clear() 
    logging.debug("Waveform precedenti cancellate da pigpio.")

    # Genera gli impulsi e le waveform
    active_motors = True
    while active_motors:
        on_pulses_this_tick = []
        off_pulses_this_tick = []
        active_motors = False # Resetta flag, verrà impostato a True se qualche motore ha ancora passi da fare

        for motor_id, plan in pulse_plans.items():
            if plan["steps_done"] < plan["steps_total"]:
                active_motors = True # Almeno un motore è ancora in movimento
                current_step_in_movement = plan["steps_done"]
                
                # Passa il piano intero a compute_frequency
                freq = compute_frequency(plan, current_step_in_movement) 
                
                if freq <= 0:
                    logging.error(f"Frequenza non valida ({freq} Hz) per motore {motor_id} allo step {current_step_in_movement}.")
                    # Potrebbe essere meglio gestire questo errore in modo più robusto, es. usando una freq minima
                    raise ValueError(f"Frequenza non valida per motore {motor_id}: {freq}")
                
                # delay_us è la durata totale dell'impulso (ON+OFF), quindi 1/freq
                # L'impulso ON è breve (es. 5us), il resto è OFF.
                pulse_duration_us = int(1000000 / freq) # Durata totale di un ciclo di step in microsecondi
                on_time_us = 5 # Breve impulso ON

                if pulse_duration_us <= on_time_us: # Frequenza troppo alta, l'impulso ON consumerebbe tutto il tempo
                    off_time_us = 1 # Minimo off time
                    on_time_us = max(1, pulse_duration_us - off_time_us) # Aggiusta on_time se necessario
                    logging.warning(f"Alta frequenza per {motor_id}: pulse_duration_us ({pulse_duration_us}) <= on_time_us. Aggiustato on_time a {on_time_us}.")
                else:
                    off_time_us = pulse_duration_us - on_time_us

                on_pulses_this_tick.append(pigpio.pulse(1 << plan["pin"], 0, on_time_us))
                off_pulses_this_tick.append(pigpio.pulse(0, 1 << plan["pin"], off_time_us))
                plan["steps_done"] += 1
        
        if on_pulses_this_tick: # Se sono stati generati impulsi in questo tick
            wave_accumulator.extend(on_pulses_this_tick)
            wave_accumulator.extend(off_pulses_this_tick)

        # Se l'accumulatore ha abbastanza impulsi per una waveform o se tutti i motori hanno finito
        if len(wave_accumulator) >= MAX_PULSES_PER_WAVE or (not active_motors and wave_accumulator):
            created_wave_id = create_wave(wave_accumulator) # Crea la waveform con gli impulsi accumulati
            if created_wave_id is not None: # create_wave potrebbe restituire None in caso di errore grave
                 generated_wave_ids.append(created_wave_id)
            wave_accumulator = [] # Resetta l'accumulatore

            # Non c'è più il break per MAX_WAVE_CHAIN_SIZE qui, si accumulano tutti gli ID.
            # La gestione di MAX_WAVE_CHAIN_SIZE è demandata a execute_wave_chain.

    if not generated_wave_ids and any(p["steps_total"] > 0 for p in pulse_plans.values()):
        logging.warning("Nessun wave_id generato nonostante ci fossero motori con target di step > 0.")
        # Questo potrebbe indicare un problema logico o che MAX_PULSES_PER_WAVE è troppo grande
        # rispetto al numero totale di impulsi per movimenti molto brevi.

    logging.info(f"Generati {len(generated_wave_ids)} ID di waveform.")
    return generated_wave_ids

def generate_waveform_independent(motor_targets):
    """
    Genera una waveform separata per ogni motore, ognuna con la propria velocità.
    Ritorna un dizionario {motor_id: wave_id}.
    """
    global pi
    if not pi or not pi.connected:
        log_error("generate_waveform_independent: pigpio non connesso.")
        raise ConnectionError("pigpio non connesso.")

    wave_ids = {}
    for motor_id, distance in motor_targets.items():
        if motor_id not in MOTORS:
            log_error(f"Motore {motor_id} non trovato in MOTORS.")
            raise ValueError(f"Motore non valido: {motor_id}")

        params = compute_motor_params(motor_id)
        steps_to_make = int(abs(distance) * params["steps_per_mm"])
        direction = 1 if distance >= 0 else 0
        motor_pins = MOTORS[motor_id]

        pi.write(motor_pins["DIR"], direction)
        pi.write(motor_pins["EN"], 0)  # Abilita motore

        pulses = []
        for step in range(steps_to_make):
            freq = compute_frequency({
                "steps_total": steps_to_make,
                "accel_steps": min(params["accel_steps"], steps_to_make // 2 if steps_to_make > 1 else 0),
                "decel_steps": min(params["decel_steps"], steps_to_make // 2 if steps_to_make > 1 else 0),
                "max_freq": params["max_freq"]
            }, step)
            pulse_duration_us = int(1000000 / freq)
            on_time_us = 5
            off_time_us = max(1, pulse_duration_us - on_time_us)
            pulses.append(pigpio.pulse(1 << motor_pins["STEP"], 0, on_time_us))
            pulses.append(pigpio.pulse(0, 1 << motor_pins["STEP"], off_time_us))

        pi.wave_clear()
        wave_id = create_wave(pulses)
        if wave_id is not None:
            wave_ids[motor_id] = wave_id
        else:
            log_error(f"Waveform non creata per motore {motor_id}")

    return wave_ids

def start_motor_movement(wave_ids_to_execute):
    """
    Avvia il movimento dei motori eseguendo la wave_chain generata.
    Questa funzione è pensata per essere eseguita in un thread.
    """
    if not wave_ids_to_execute:
        logging.warning("start_motor_movement: Nessuna waveform ID fornita per l'esecuzione.")
        return # Non sollevare eccezioni in un thread senza un gestore specifico se possibile

    try:
        logging.debug(f"start_motor_movement: Esecuzione della wave_chain con {len(wave_ids_to_execute)} wave_ids.")
        execute_wave_chain(wave_ids_to_execute)
        logging.info("Movimento completato (tutti i chunk della wave_chain inviati).")
    except Exception as e:
        # Logga l'eccezione che potrebbe verificarsi in execute_wave_chain
        log_error(f"Errore in start_motor_movement (thread): {e}")
        # Qui potresti voler segnalare lo stato di errore globalmente se necessario
    finally:
        # Qui si potrebbero disabilitare i motori (pi.write(motor["EN"], 1)) se desiderato dopo ogni movimento
        # Ma dipende dalla logica dell'applicazione (se i motori devono rimanere "holding")
        pass

def start_motor_movement_independent(wave_ids_dict):
    """
    Avvia tutte le waveform dei motori in parallelo.
    """
    global pi
    if not pi or not pi.connected:
        log_error("start_motor_movement_independent: pigpio non connesso.")
        return

    try:
        # Avvia tutte le waveform contemporaneamente
        for motor_id, wave_id in wave_ids_dict.items():
            pin = MOTORS[motor_id]["STEP"]
            pi.wave_send_once(wave_id)  # Avvia la waveform su tutti i pin

        # Attendi che tutti i motori abbiano finito
        while any(pi.wave_tx_busy() for _ in wave_ids_dict):
            time.sleep(0.001)
        logging.info("Movimento parallelo completato.")
    except Exception as e:
        log_error(f"Errore in start_motor_movement_independent: {e}")
        
@api_view(['POST'])
def move_motor_view(request):
    global pi
    if not pi or not pi.connected:
        log_error("move_motor_view: pigpio non connesso.")
        return JsonResponse({"log": "Errore: pigpio non connesso", "error": "Pigpio connection issue"}, status=503)

    try:
        reload_motor_config()
        data = json.loads(request.body)
        logging.debug(f"Richiesta ricevuta: {data}")
        targets = data.get("targets", {})

        if not targets:
            response = JsonResponse({"log": "Nessun target fornito", "error": "Input non valido"}, status=400)
            log_json_response(response)
            return response

        validate_targets(targets)
        manage_motor_pins(targets)
        ensure_pigpio_connection()

        wave_ids_dict = generate_waveform_independent(targets)
        if wave_ids_dict:
            motor_command_queue.put(wave_ids_dict)
            response = JsonResponse({"log": "Movimento messo in coda", "status": "queued", "wave_ids_count": len(wave_ids_dict)})
            log_json_response(response)
            return response
        else:
            response = JsonResponse({"log": "Nessuna waveform generata, nessun movimento avviato.", "error": "Waveform non creata o vuota"}, status=500)
            log_json_response(response)
            return response

    except ValueError as ve:
        log_error(f"Errore di valore durante il movimento del motore: {str(ve)}")
        return JsonResponse({"log": "Errore nei dati forniti", "error": str(ve)}, status=400)
    except ConnectionError as ce:
        return JsonResponse({"log": "Errore di connessione pigpio", "error": str(ce)}, status=503)
    except Exception as e:
        log_error(f"Errore generico durante il movimento del motore: {str(e)}")
        return handle_exception(e)
    
def execute_wave_chain(all_wave_ids):
    """
    Esegue una lista di ID di waveform (all_wave_ids) inviandoli a pigpio
    in chunk che rispettano MAX_WAVE_CHAIN_SIZE.
    """
    global pi
    if not pi or not pi.connected:
        log_error("execute_wave_chain: pigpio non connesso.")
        raise ConnectionError("pigpio non connesso.")

    ensure_pigpio_connection() # Assicura che la connessione sia attiva
    
    try:
        offset = 0
        chain_number = 0
        while offset < len(all_wave_ids):
            chain_number += 1
            # Estrai il chunk di wave_ids per la catena corrente
            current_chunk_wave_ids = all_wave_ids[offset : offset + MAX_WAVE_CHAIN_SIZE]
            offset += len(current_chunk_wave_ids)

            if not current_chunk_wave_ids: # Non dovrebbe succedere se all_wave_ids non era vuota
                logging.warning("execute_wave_chain: Tentativo di eseguire un chunk vuoto.")
                continue

            # La chain per pigpio è semplicemente la lista degli ID di waveform per questo chunk
            # Non è necessario anteporre [255,0] per una singola esecuzione sequenziale.
            # pi.wave_chain() accetta una lista di wave ID.
            logging.debug(f"Esecuzione chunk {chain_number} della catena con {len(current_chunk_wave_ids)} wave IDs: {current_chunk_wave_ids}")
            
            pi.wave_chain(current_chunk_wave_ids)
            
            # Attendi che questo chunk della catena sia stato trasmesso
            # Questo rende la funzione bloccante per la durata del chunk.
            # Il chiamante (start_motor_movement) è in un thread, quindi l'API non si blocca.
            while pi.wave_tx_busy():
                time.sleep(0.001) # Sleep breve, ma attenzione a non sovraccaricare la CPU con polling troppo aggressivo
            
            logging.debug(f"Chunk {chain_number} completato.")

        logging.info(f"Tutti i {chain_number} chunk della wave_chain sono stati eseguiti.")

    except Exception as e:
        log_error(f"Errore critico durante l'esecuzione della wave_chain: {e}")
        # Potrebbe essere utile tentare di fermare i motori qui o segnalare un errore grave
        pi.wave_tx_stop() # Tenta di fermare la trasmissione in caso di errore
        raise # Rilancia l'eccezione per essere gestita dal chiamante (start_motor_movement)

def compute_frequency(plan, current_step_in_movement):
    """Calcola la frequenza in base alla fase del movimento (accelerazione, velocità costante, decelerazione)."""
    total_steps = plan["steps_total"]
    accel_steps = plan["accel_steps"]
    decel_steps = plan["decel_steps"] # Dovrebbe essere calcolato come steps_total - inizio_decelerazione
    max_freq = plan["max_freq"]

    # Calcola il punto in cui inizia la decelerazione
    decel_starts_at_step = total_steps - decel_steps

    if accel_steps == 0 and decel_steps == 0: # Nessuna accelerazione/decelerazione (es. movimento molto breve)
        return max_freq
    
    # Fase di Accelerazione
    if current_step_in_movement < accel_steps:
        if accel_steps == 0: return max_freq # Evita divisione per zero se accel_steps è 0 ma siamo qui
        # Interpolazione lineare della frequenza da ~0 a max_freq
        # Inizia da una frequenza minima > 0 per evitare problemi
        min_freq_during_accel = max_freq * 0.01 # Ad esempio 1% della max_freq come base
        if min_freq_during_accel == 0: min_freq_during_accel = 1.0 # Assoluto minimo
        
        # La frequenza dovrebbe aumentare linearmente o quadraticamente (per accelerazione costante)
        # Per accelerazione costante, la velocità (e quindi la frequenza) aumenta linearmente col tempo.
        # Se assumiamo che ogni step prenda un tempo inversamente proporzionale alla frequenza istantanea,
        # il calcolo diventa più complesso.
        # Una rampa di frequenza lineare rispetto al numero di step:
        # Freq = (current_step / accel_steps) * max_freq
        # Rendiamola più robusta per current_step = 0
        calculated_freq = ((current_step_in_movement + 1) / accel_steps) * max_freq
        return max(min_freq_during_accel, calculated_freq)

    # Fase di Decelerazione
    elif current_step_in_movement >= decel_starts_at_step:
        if decel_steps == 0: return max_freq # Evita divisione per zero
        steps_into_decel = current_step_in_movement - decel_starts_at_step
        remaining_decel_steps = decel_steps - steps_into_decel
        
        min_freq_during_decel = max_freq * 0.01
        if min_freq_during_decel == 0: min_freq_during_decel = 1.0

        calculated_freq = (remaining_decel_steps / decel_steps) * max_freq
        return max(min_freq_during_decel, calculated_freq)
        
    # Fase di Velocità Costante
    else:
        return max_freq

def create_wave(pulses):
    """Crea una waveform da una lista di impulsi e restituisce il suo ID."""
    global pi
    if not pi or not pi.connected:
        log_error("create_wave: pigpio non connesso.")
        return None # Non si può creare una wave senza pigpio

    ensure_pigpio_connection()
    if not pulses:
        logging.warning("create_wave: Tentativo di creare una waveform da una lista di impulsi vuota.")
        return None
    try:
        pi.wave_add_generic(pulses)
        wave_id = pi.wave_create()
        logging.debug(f"Waveform creata con ID: {wave_id} da {len(pulses)} impulsi.")
        return wave_id
    except (ConnectionResetError, BrokenPipeError, pigpio.error) as e: # Aggiunto pigpio.error per errori specifici della libreria
        log_error(f"Errore durante la creazione della waveform: {e}. Tentativo di riconnessione...")
        ensure_pigpio_connection()
        if pi.connected:
            try:
                # Ritenta l'operazione dopo la riconnessione
                pi.wave_add_generic(pulses)
                wave_id = pi.wave_create()
                logging.info(f"Waveform creata con ID: {wave_id} dopo riconnessione.")
                return wave_id
            except Exception as e2:
                log_error(f"Errore nel ritentare la creazione della waveform: {e2}")
                return None
        else:
            log_error("Impossibile creare waveform, riconnessione a pigpio fallita.")
            return None
    except Exception as e_generic: # Altri errori generici
        log_error(f"Errore generico in create_wave: {e_generic}")
        return None

def validate_targets(targets):
    """Valida i target forniti."""
    if not isinstance(targets, dict):
        raise ValueError("I target devono essere un dizionario.")
    for motor_id, distance in targets.items():
        if motor_id not in MOTORS:
            raise ValueError(f"Motore non valido: {motor_id}")
        if not isinstance(distance, (int, float)):
            raise ValueError(f"La distanza per il motore {motor_id} deve essere un numero.")

def manage_motor_pins(targets):
    """Gestisce i pin EN per evitare interferenze tra i motori."""
    global pi
    if not pi or not pi.connected:
        log_error("manage_motor_pins: pigpio non connesso.")
        return

    # Logica di esempio: disabilita tutti i motori non nel target, abilita quelli nel target
    # Questa logica potrebbe necessitare di affinamento basata sui requisiti specifici.
    # Ad esempio, se syringe si muove, gli altri sono disabilitati.
    # Se extruder o conveyor si muovono, syringe è disabilitata.
    # Se si muovono extruder E conveyor insieme, syringe è disabilitata.

    moving_syringe = "syringe" in targets
    moving_others = any(m in targets for m in ["extruder", "conveyor"])

    if moving_syringe:
        # Se la siringa si muove, disabilita extruder e conveyor
        if "extruder" in MOTORS: pi.write(MOTORS["extruder"]["EN"], 1) # Disable
        if "conveyor" in MOTORS: pi.write(MOTORS["conveyor"]["EN"], 1) # Disable
        pi.write(MOTORS["syringe"]["EN"], 0) # Enable syringe
        logging.debug("Abilitata syringe, disabilitati extruder e conveyor.")
    elif moving_others:
        # Se extruder o conveyor (o entrambi) si muovono, disabilita syringe
        pi.write(MOTORS["syringe"]["EN"], 1) # Disable syringe
        if "extruder" in targets:
            pi.write(MOTORS["extruder"]["EN"], 0) # Enable extruder
            logging.debug("Abilitato extruder.")
        else:
            if "extruder" in MOTORS: pi.write(MOTORS["extruder"]["EN"], 1) # Disable extruder se non in target

        if "conveyor" in targets:
            pi.write(MOTORS["conveyor"]["EN"], 0) # Enable conveyor
            logging.debug("Abilitato conveyor.")
        else:
            if "conveyor" in MOTORS: pi.write(MOTORS["conveyor"]["EN"], 1) # Disable conveyor se non in target
    else:
        # Nessun motore specificato nei target per questa logica,
        # potrebbero essere tutti disabilitati o mantenere lo stato precedente.
        # Per sicurezza, disabilitiamoli se non sono nei target (dipende dal comportamento desiderato)
        # for motor_id_loop in MOTORS:
        #    if motor_id_loop not in targets:
        #        pi.write(MOTORS[motor_id_loop]["EN"], 1) # Disable
        logging.debug("Nessun target specifico per la logica EN, stato pin EN non modificato attivamente qui.")


def ensure_pigpio_connection():
    """Verifica e ristabilisce la connessione a pigpio."""
    global pi
    if pi and pi.connected:
        return True

    logging.warning("Connessione a pigpio persa o non stabilita. Tentativo di (ri)connessione...")
    if pi:
        try:
            pi.stop() # Ferma la sessione pigpio esistente se presente
        except Exception as e_stop:
            logging.error(f"Errore durante pi.stop(): {e_stop}")
    
    time.sleep(0.5) # Breve pausa prima di ritentare

    try:
        new_pi_instance = pigpio.pi()
        if new_pi_instance.connected:
            pi = new_pi_instance # Assegna la nuova istanza connessa
            logging.info("Riconnessione a pigpio riuscita.")
            # Necessario riconfigurare i modi dei pin dopo una nuova istanza di pi
            for motor_pins_dict in MOTORS.values():
                pi.set_mode(motor_pins_dict["STEP"], pigpio.OUTPUT)
                pi.set_mode(motor_pins_dict["DIR"], pigpio.OUTPUT)
                pi.set_mode(motor_pins_dict["EN"], pigpio.OUTPUT)
            return True
        else:
            logging.error("Impossibile riconnettersi a pigpio. La nuova istanza non è connessa.")
            # pi rimane la vecchia istanza non connessa o None
            return False
    except Exception as e_conn:
        logging.error(f"Eccezione durante il tentativo di riconnessione a pigpio: {e_conn}")
        # pi rimane la vecchia istanza non connessa o None
        return False


def handle_exception(e):
    """Gestisce le eccezioni e restituisce un JsonResponse."""
    import traceback
    error_details = traceback.format_exc()
    log_error(f"Errore interno: {error_details}")
    # Estrai un messaggio più conciso dall'eccezione per il JSON response
    error_type = type(e).__name__
    error_message = str(e)
    response = JsonResponse({"log": f"Errore interno: {error_type}", "error": error_message}, status=500)
    log_json_response(response)
    return response

# ------------------------------------------------------------------------------
# API: Stop motori
# ------------------------------------------------------------------------------
@api_view(['POST'])
def stop_motor_view(request): # Rinominata per coerenza e per evitare conflitti
    global running_flags, pi
    if not pi or not pi.connected:
        log_error("stop_motor_view: pigpio non connesso.")
        return JsonResponse({"log": "Errore: pigpio non connesso", "error": "Pigpio connection issue"}, status=503)

    try:
        logging.info("Richiesta di stop motori ricevuta.")
        pi.wave_tx_stop() # Ferma qualsiasi trasmissione di waveform DMA
        # Potrebbe essere necessario cancellare le waveform correnti se si vuole un arresto più pulito
        # pi.wave_clear() # Attenzione: questo cancella TUTTE le definizioni di waveform

        for key in running_flags:
            running_flags[key] = False
        
        # Disabilita i motori o imposta PWM a zero
        for motor_id, motor_pins in MOTORS.items():
            # Opzione 1: Disabilita i motori (potrebbero perdere la posizione se non hanno freno)
            pi.write(motor_pins["EN"], 1) 
            # Opzione 2: Se usi PWM hardware per lo step (non comune con waveform DMA per lo step)
            # pi.hardware_PWM(motor_pins["STEP"], 0, 0) # Freq 0, DutyCycle 0
            logging.debug(f"Motore {motor_id} fermato e disabilitato (EN=1).")

        response = JsonResponse({"log": "Motori fermati con successo", "status": "success"})
        log_json_response(response)
        return response
    except Exception as e:
        log_error(f"Errore durante lo stop dei motori: {str(e)}")
        response = JsonResponse({"log": "Errore durante lo stop dei motori", "error": str(e)}, status=500)
        log_json_response(response)
        return response

# ------------------------------------------------------------------------------
# API: Salvataggio configurazione
# ------------------------------------------------------------------------------
@api_view(['POST'])
def save_motor_config_view(request): # Rinominata
    global config, motor_configs
    current_config = {}
    try:
        # Carica la configurazione esistente per aggiornarla
        current_config = load_motor_config() 
    except Exception as e:
        # Se il caricamento fallisce, current_config potrebbe essere vuota o default
        log_error(f"Errore durante il caricamento della configurazione per il salvataggio: {str(e)}")
        # Non necessariamente un errore fatale per il salvataggio, potremmo sovrascrivere con nuovi dati.
        # Ma è bene saperlo.

    # Utilizza i dati dalla richiesta per aggiornare la sezione 'motors'
    # Assumiamo che request.data contenga la configurazione SOLO per la chiave 'motors'
    # o che SettingsSerializer gestisca la struttura completa.
    # Se request.data è { "motors": { "extruder": {...} } }, allora va bene.
    # Se request.data è solo { "extruder": {...} }, SettingsSerializer dovrebbe saperlo.
    
    # Per sicurezza, estraiamo la parte 'motors' dalla request.data se presente,
    # altrimenti usiamo request.data direttamente se SettingsSerializer lo gestisce.
    data_to_serialize = request.data.get("motors", request.data)

    serializer = SettingsSerializer(data=data_to_serialize, partial=True) # Partial=True permette aggiornamenti parziali

    if serializer.is_valid():
        # Aggiorna la sezione 'motors' della configurazione caricata
        if "motors" not in current_config:
            current_config["motors"] = {}
        current_config["motors"].update(serializer.validated_data)
        
        # Rimuovi chiavi obsolete se necessario (logica originale)
        # current_config["motors"].pop("motors", None) # Probabilmente un errore di battitura, rimuove la chiave stessa
        # current_config["motors"].pop("camera", None) # Rimuove 'camera' da DENTRO 'motors'

        write_settings(current_config) # Scrive l'intera struttura 'config' aggiornata
        reload_motor_config() # Ricarica la configurazione globale
        
        response = JsonResponse({"log": "Configurazione motori salvata con successo", "success": True, "settings": current_config})
        log_json_response(response)
        return response
    else:
        log_error(f"Errore di validazione durante salvataggio configurazione: {serializer.errors}")
        response = JsonResponse({"log": "Errore di validazione", "errors": serializer.errors}, status=400)
        log_json_response(response)
        return response

# ------------------------------------------------------------------------------
# API: Velocità motori (Placeholder - la velocità attuale non è tracciata in modo granulare)
# ------------------------------------------------------------------------------
@csrf_exempt # Considera se csrf_exempt è sicuro per questa GET request
@api_view(['GET'])
def get_motor_speeds_view(request): # Rinominata
    global current_speeds 
    # current_speeds non viene aggiornato dinamicamente con la frequenza attuale nel codice fornito.
    # Questa funzione restituirà i valori iniziali (0) o l'ultimo valore impostato manualmente (che non avviene).
    # Per velocità reali, bisognerebbe calcolarle durante `compute_frequency` e aggiornare uno stato globale.
    try:
        # Simula un recupero, ma i valori non sono aggiornati in tempo reale
        # con la logica attuale delle waveform.
        # La "velocità" qui è più uno placeholder.
        speeds_snapshot = {motor: current_speeds.get(motor, 0) for motor in MOTORS.keys()}
        
        json_response = JsonResponse({"log": "Velocità motori (stato attuale placeholder) recuperate", "speeds": speeds_snapshot})
        log_json_response(json_response)
        return json_response
    except Exception as e:
        log_error(f"Errore durante il recupero delle velocità: {str(e)}")
        response = JsonResponse({"log": "Errore durante il recupero delle velocità", "error": str(e)}, status=500)
        log_json_response(response)
        return response

@api_view(['POST'])
def start_simulation_view(request):
    """
    API per avviare una simulazione di un percorso predefinito.
    """
    global pi
    if not pi or not pi.connected:
        log_error("start_simulation_view: pigpio non connesso.")
        return JsonResponse({"log": "Errore: pigpio non connesso", "error": "Pigpio connection issue"}, status=503)

    try:
        # Definizione del percorso simulato
        simulation_steps = []
        extruder_direction = 1  # Direzione iniziale positiva

        for _ in range(5):  # Ripeti 5 volte
            for _ in range(3):  # Ripeti 3 volte
                simulation_steps.append({"syringe": 5})  # Syringe si sposta di 5mm
                simulation_steps.append({"extruder": 50 * extruder_direction})  # Extruder si sposta di 50mm nella direzione corrente
            extruder_direction *= -1  # Inverti la direzione dell'extruder
            simulation_steps.append({"conveyor": 50})  # Conveyor si sposta di 50mm

        # Esegui ogni step della simulazione
        for step in simulation_steps:
            wave_ids = generate_waveform(step)  # Genera waveform per il movimento
            if wave_ids:
                execute_wave_chain(wave_ids)  # Esegui il movimento
            else:
                log_error(f"Errore nella generazione della waveform per il passo: {step}")
                return JsonResponse({"log": "Errore nella simulazione", "error": "Waveform non generata"}, status=500)

        return JsonResponse({"log": "Simulazione completata con successo", "status": "success"})
    except Exception as e:
        log_error(f"Errore durante la simulazione: {str(e)}")
        return handle_exception(e)

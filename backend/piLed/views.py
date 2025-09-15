from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt  # Import aggiunto
import json
import time
import threading
import requests
import logging
try:
    import board  # type: ignore
    import neopixel  # type: ignore
except ImportError:
    from unittest.mock import MagicMock
    board = MagicMock()
    neopixel = MagicMock()

# Configurazione della strip LED
LED_COUNT = 100  # Numero di LED
# LED_PIN = board.D19  # GPIO dei dati
LED_PIN = board.D21  # GPIO dei dati
LED_BRIGHTNESS = 0.5  # Luminosità (da 0.0 a 1.0)
ORDER = neopixel.GRB  # Ordine dei colori

# Inizializza la strip LED
strip = neopixel.NeoPixel(LED_PIN, LED_COUNT, brightness=LED_BRIGHTNESS, auto_write=False, pixel_order=ORDER)

# Definizione di stop_event prima del suo utilizzo
stop_event = threading.Event()

current_effect = "none"
ESPurl = f"http://192.168.159.130:80/"

@csrf_exempt
def control_leds(request):
    if request.method == 'POST':
        try:
            body = json.loads(request.body)
            effect = body.get('effect', '')
            debug_logs = []  # Lista per accumulare i messaggi di debug

            # Ferma eventuali effetti in esecuzione
            stop_event.set()  # Imposta il flag per fermare i thread
            time.sleep(0.1)  # Attendi un breve intervallo per assicurarti che i thread si fermino
            stop_event.clear()  # Resetta il flag per consentire nuovi effetti
            global current_effect
            if effect == 'wave':
                current_effect = "wave"
                # threading.Thread(target=wave_effect_with_logs, args=(strip, debug_logs), daemon=True).start()
                response = wave_effect_with_logs()
                if response[0]:
                    debug_logs.append("Wave effect started successfully.")
                    return JsonResponse({'status': 'success', 'message': 'Wave effect started in loop', 'logs': debug_logs})
                else:
                    debug_logs.append(f"Failed to start wave effect: {response[1]}")
                    return JsonResponse({'status': 'error', 'message': 'Failed to start wave effect', 'logs': debug_logs}, status=500)
            elif effect == 'green_loading':
                current_effect = "green_loading"
                debug_logs.append("Starting green loading effect...")
                # threading.Thread(target=green_loading_with_logs, args=(strip, debug_logs), daemon=True).start()
                response = green_loading_with_logs(debug_logs)
                if response[0]:
                    debug_logs.append("green effect started successfully.")
                    return JsonResponse({'status': 'success', 'message': 'green effect started in loop', 'logs': debug_logs})
                else:
                    debug_logs.append(f"Failed to start green effect: {response[1]}")
                    return JsonResponse({'status': 'error', 'message': 'Failed to start greenn effect', 'logs': debug_logs}, status=500)
            elif effect == 'yellow_blink':
                current_effect = "yellow_blink"
                debug_logs.append("Starting yellow blink effect...")
                # threading.Thread(target=yellow_blink_with_logs, args=(strip, debug_logs), daemon=True).start()
                response = yellow_blink_with_logs(debug_logs)
                if response[0]:
                    debug_logs.append("yrllow effect started successfully.")
                    return JsonResponse({'status': 'success', 'message': 'yellow effect started in loop', 'logs': debug_logs})
                else:
                    debug_logs.append(f"Failed to start yellow effect: {response[1]}")
                    return JsonResponse({'status': 'error', 'message': 'Failed to start yellow effect', 'logs': debug_logs}, status=500)
            elif effect == 'red_static':
                current_effect = "red_static"
                debug_logs.append("Starting red static effect...")
                # threading.Thread(target=red_static_with_logs, args=(strip, debug_logs), daemon=True).start()
                response = red_static_with_logs(debug_logs)
                if response[0]:
                    debug_logs.append("red effect started successfully.")
                    return JsonResponse({'status': 'success', 'message': 'red effect started in loop', 'logs': debug_logs})
                else:
                    debug_logs.append(f"Failed to start red effect: {response[1]}")
                    return JsonResponse({'status': 'error', 'message': 'Failed to start red effect', 'logs': debug_logs}, status=500)
            elif effect == 'stop':
                current_effect = "none"
                # Ferma gli effetti LED
                response = stop_led_effect()
                if response[0]:
                    debug_logs.append("LED effect stopped successfully.")
                    return JsonResponse({'status': 'success', 'message': 'LED effect stopped', 'logs': debug_logs})
                else:
                    debug_logs.append(f"Failed to stop LED effect: {response[1]}")
                    return JsonResponse({'status': 'error', 'message': 'Failed to stop LED effect', 'logs': debug_logs}, status=500)
            else:
                return JsonResponse({'status': 'error', 'message': 'Invalid effect'}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON'}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)

def get_led_status(request):
    if request.method == 'GET':
        status = {
            'current_effect': current_effect,
        }
        return JsonResponse({'status': 'success', 'led_status': status})
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)

def stop_led_effect():
    try:
        response = requests.get(f'{ESPurl}off', timeout=5)
        if response.status_code == 200:
            print(f"Successo! Effetto LED spento.")
            return True, response.text
        else:
            print(f"Errore: l'ESP32 ha risposto con codice {response.status_code}")
            return False, f"Status Code: {response.status_code}"
    except requests.exceptions.RequestException as e:
        print(f"Errore di connessione all'ESP32: {e}")
        return False, str(e)

def wave_effect_with_logs():
    # debug_logs.append("Wave effect thread started")
    try:
        response = requests.get(f'{ESPurl}rainbow', timeout=5)
        if response.status_code == 200:
            print(f"Successo! Effetto rainbow impostato.")
            logging.info("Successo! Effetto rainbow impostato.")
            return True, response.text
        else:
            print(f"Errore: l'ESP32 ha risposto con codice {response.status_code}")
            logging.error(f"Errore: l'ESP32 ha risposto con codice {response.status_code}")
            return False, f"Status Code: {response.status_code}"
    except requests.exceptions.RequestException as e:
        print(f"Errore di connessione all'ESP32: {e}")
        logging.error(f"Errore di connessione all'ESP32: {e}")
        return False, str(e)
    

def green_loading_with_logs():
    try:
        response = requests.get(f'{ESPurl}greenloading', timeout=5)
        if response.status_code == 200:
            print(f"Successo! Effetto green wave impostato.")
            logging.info("Successo! Effetto green wave impostato.")
            return True, response.text
        else:
            print(f"Errore: l'ESP32 ha risposto con codice {response.status_code}")
            logging.error(f"Errore: l'ESP32 ha risposto con codice {response.status_code}")
            return False, f"Status Code: {response.status_code}"
    except requests.exceptions.RequestException as e:
        print(f"Errore di connessione all'ESP32: {e}")
        logging.error(f"Errore di connessione all'ESP32: {e}")
        return False, str(e)

def yellow_blink_with_logs():
    try:
        response = requests.get(f'{ESPurl}yellowblink', timeout=5)
        if response.status_code == 200:
            print(f"Successo! Effetto yelloblink impostato.")
            logging.info("Successo! Effetto yelloblink impostato.")
            return True, response.text
        else:
            print(f"Errore: l'ESP32 ha risposto con codice {response.status_code}")
            logging.error(f"Errore: l'ESP32 ha risposto con codice {response.status_code}")
            return False, f"Status Code: {response.status_code}"
    except requests.exceptions.RequestException as e:
        print(f"Errore di connessione all'ESP32: {e}")
        logging.error(f"Errore di connessione all'ESP32: {e}")
        
        return False, str(e)

def red_static_with_logs():
    try:
        response = requests.get(f'{ESPurl}redstatic', timeout=5)
        if response.status_code == 200:
            print(f"Successo! Effetto static red impostato.")
            logging.info("Successo! Effetto static red impostato.")
            return True, response.text
        else:
            print(f"Errore: l'ESP32 ha risposto con codice {response.status_code}")
            logging.error(f"Errore: l'ESP32 ha risposto con codice {response.status_code}")
            return False, f"Status Code: {response.status_code}"
    except requests.exceptions.RequestException as e:
        print(f"Errore di connessione all'ESP32: {e}")
        logging.error(f"Errore di connessione all'ESP32: {e}")
        return False, str(e)
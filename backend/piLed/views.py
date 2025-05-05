from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import time
import threading

try:
    import board  # type: ignore
    import neopixel  # type: ignore
except ImportError:
    from unittest.mock import MagicMock
    board = MagicMock()
    neopixel = MagicMock()

# Configurazione della strip LED
LED_COUNT = 100  # Numero di LED
LED_PIN = board.D19  # GPIO dei dati
LED_BRIGHTNESS = 0.5  # Luminosità (da 0.0 a 1.0)
ORDER = neopixel.GRB  # Ordine dei colori

# Inizializza la strip LED
strip = neopixel.NeoPixel(LED_PIN, LED_COUNT, brightness=LED_BRIGHTNESS, auto_write=False, pixel_order=ORDER)

def wave_effect(strip):
     print("Wave effect thread started") 
     while True:  # Loop infinito
          for i in range(LED_COUNT):
               strip[i] = (0, 0, 255)  # Blu
               strip.show()
               time.sleep(0.02)
               strip[i] = (0, 0, 0)  # Spegni il LED
          for i in reversed(range(LED_COUNT)):
               strip[i] = (0, 0, 255)  # Blu
               strip.show()
               time.sleep(0.02)
               strip[i] = (0, 0, 0)  # Spegni il LED

def green_loading(strip):
    for i in range(LED_COUNT):
        strip[i] = (0, 255, 0)  # Verde
        strip.show()
        time.sleep(0.02)
    for i in range(LED_COUNT):
        strip[i] = (0, 0, 0)  # Spegni il LED
        strip.show()
        time.sleep(0.02)

def yellow_blink(strip):
    for _ in range(5):  # Lampeggia 5 volte
        for i in range(LED_COUNT):
            strip[i] = (255, 255, 0)  # Giallo
        strip.show()
        time.sleep(0.5)
        for i in range(LED_COUNT):
            strip[i] = (0, 0, 0)  # Spegni il LED
        strip.show()
        time.sleep(0.5)

def red_static(strip):
    for i in range(LED_COUNT):
        strip[i] = (255, 0, 0)  # Rosso
    strip.show()

@csrf_exempt
def control_leds(request):
    if request.method == 'POST':
        try:
            body = json.loads(request.body)
            effect = body.get('effect', '')
            debug_logs = []  # Lista per accumulare i messaggi di debug

            if effect == 'wave':
                debug_logs.append("Starting wave effect...")
                threading.Thread(target=wave_effect_with_logs, args=(strip, debug_logs), daemon=True).start()
                return JsonResponse({'status': 'success', 'message': 'Wave effect started in loop', 'logs': debug_logs})
            elif effect == 'green_loading':
                debug_logs.append("Starting green loading effect...")
                green_loading_with_logs(strip, debug_logs)
                return JsonResponse({'status': 'success', 'message': 'Green loading started', 'logs': debug_logs})
            elif effect == 'yellow_blink':
                debug_logs.append("Starting yellow blink effect...")
                yellow_blink_with_logs(strip, debug_logs)
                return JsonResponse({'status': 'success', 'message': 'Yellow blinking started', 'logs': debug_logs})
            elif effect == 'red_static':
                debug_logs.append("Starting red static effect...")
                red_static_with_logs(strip, debug_logs)
                return JsonResponse({'status': 'success', 'message': 'Red static started', 'logs': debug_logs})
            else:
                return JsonResponse({'status': 'error', 'message': 'Invalid effect'}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON'}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)

# Funzioni con logging
def wave_effect_with_logs(strip, debug_logs):
    debug_logs.append("Wave effect thread started")
    while True:  # Loop infinito
        debug_logs.append("Wave effect running...")
        for i in range(LED_COUNT):
            strip[i] = (0, 0, 255)  # Blu
            strip.show()
            time.sleep(0.02)
            strip[i] = (0, 0, 0)  # Spegni il LED
        for i in reversed(range(LED_COUNT)):
            strip[i] = (0, 0, 255)  # Blu
            strip.show()
            time.sleep(0.02)
            strip[i] = (0, 0, 0)  # Spegni il LED

def green_loading_with_logs(strip, debug_logs):
    debug_logs.append("Green loading effect started")
    for i in range(LED_COUNT):
        strip[i] = (0, 255, 0)  # Verde
        strip.show()
        time.sleep(0.02)
    for i in range(LED_COUNT):
        strip[i] = (0, 0, 0)  # Spegni il LED
        strip.show()
        time.sleep(0.02)
    debug_logs.append("Green loading effect completed")

def yellow_blink_with_logs(strip, debug_logs):
    debug_logs.append("Yellow blink effect started")
    for _ in range(5):  # Lampeggia 5 volte
        for i in range(LED_COUNT):
            strip[i] = (255, 255, 0)  # Giallo
        strip.show()
        time.sleep(0.5)
        for i in range(LED_COUNT):
            strip[i] = (0, 0, 0)  # Spegni il LED
        strip.show()
        time.sleep(0.5)
    debug_logs.append("Yellow blink effect completed")

def red_static_with_logs(strip, debug_logs):
    debug_logs.append("Red static effect started")
    for i in range(LED_COUNT):
        strip[i] = (255, 0, 0)  # Rosso
    strip.show()
    debug_logs.append("Red static effect completed")
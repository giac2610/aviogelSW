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

            if effect == 'wave':
                threading.Thread(target=wave_effect, args=(strip,), daemon=True).start()
                return JsonResponse({'status': 'success', 'message': 'Wave effect started in loop'})
            elif effect == 'green_loading':
                green_loading(strip)
                return JsonResponse({'status': 'success', 'message': 'Green loading started'})
            elif effect == 'yellow_blink':
                yellow_blink(strip)
                return JsonResponse({'status': 'success', 'message': 'Yellow blinking started'})
            elif effect == 'red_static':
                red_static(strip)
                return JsonResponse({'status': 'success', 'message': 'Red static started'})
            else:
                return JsonResponse({'status': 'error', 'message': 'Invalid effect'}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON'}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)

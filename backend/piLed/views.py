from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

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

# Inizializza la strip LED
strip = neopixel.NeoPixel(LED_PIN, LED_COUNT, brightness=LED_BRIGHTNESS, auto_write=False)

@csrf_exempt
def control_leds(request):
    if request.method == 'POST':
        try:
            body = json.loads(request.body)
            effect = body.get('effect', '')

            if effect == 'wave':
                # Logica per l'effetto ondeggiamento
                return JsonResponse({'status': 'success', 'message': 'Wave effect started'})
            elif effect == 'green_loading':
                # Logica per il caricamento verde
                return JsonResponse({'status': 'success', 'message': 'Green loading started'})
            elif effect == 'yellow_blink':
                # Logica per lampeggiare giallo
                return JsonResponse({'status': 'success', 'message': 'Yellow blinking started'})
            elif effect == 'red_static':
                # Logica per accendere rosso fisso
                return JsonResponse({'status': 'success', 'message': 'Red static started'})
            else:
                return JsonResponse({'status': 'error', 'message': 'Invalid effect'}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON'}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)

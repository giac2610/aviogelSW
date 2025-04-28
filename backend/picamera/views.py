import sys
import json
import os
from unittest.mock import MagicMock

# Percorso relativo al file setup.json
SETUP_JSON_PATH = os.path.join(os.path.dirname(__file__), '../config/setup.json')

# Carica i parametri iniziali da setup.json
with open(SETUP_JSON_PATH, 'r') as f:
    config = json.load(f)
camera_settings = config["camera"]

# Switch automatico per macOS
if sys.platform == "darwin":
    import cv2
    from django.http import JsonResponse
    import numpy as np

    # Inizializza la webcam del Mac
    mac_camera = cv2.VideoCapture(0)  # 0 indica la webcam predefinita

    def gen_frames_normal():
        while True:
            try:
                frame = mac_camera.read()[1]
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Errore durante la generazione del frame normale: {e}")
                break

    def gen_frames_greyscale():
        while True:
            try:
                frame = mac_camera.read()[1]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, buffer = cv2.imencode('.jpg', gray)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Errore durante la generazione del frame in scala di grigi: {e}")
                break

    def gen_frames_threshold():
        while True:
            try:
                with open(SETUP_JSON_PATH, 'r') as f:
                    config = json.load(f)
                    camera_settings = config["camera"]

                frame = mac_camera.read()[1]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, camera_settings["minThreshold"], camera_settings["maxThreshold"], cv2.THRESH_BINARY_INV)
                _, buffer = cv2.imencode('.jpg', thresh)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Errore durante la generazione del frame threshold: {e}")
                break

else:
    from picamera2 import Picamera2  # type: ignore
    import cv2
    import numpy as np

    # Inizializza Picamera2
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
    picam2.start()

    def gen_frames_normal():
        while True:
            try:
                frame = picam2.capture_array()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                _, buffer = cv2.imencode('.jpg', frame_bgr)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Errore durante la generazione del frame normale: {e}")
                break

    def gen_frames_greyscale():
        while True:
            try:
                frame = picam2.capture_array()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                _, buffer = cv2.imencode('.jpg', gray)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Errore durante la generazione del frame in scala di grigi: {e}")
                break

    def gen_frames_threshold():
        while True:
            try:
                with open(SETUP_JSON_PATH, 'r') as f:
                    config = json.load(f)
                    camera_settings = config["camera"]

                frame = picam2.capture_array()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, camera_settings["minThreshold"], camera_settings["maxThreshold"], cv2.THRESH_BINARY_INV)
                _, buffer = cv2.imencode('.jpg', thresh)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Errore durante la generazione del frame threshold: {e}")
                break

from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def update_camera_settings(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            global camera_settings
            camera_settings.update(data)  # Aggiorna i parametri globali
            config["camera"] = camera_settings  # Aggiorna il file di configurazione
            with open(SETUP_JSON_PATH, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"Impostazioni aggiornate: {camera_settings}")  # Log per debug
            return JsonResponse({"status": "success", "updated_settings": camera_settings})
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)
    elif request.method == "GET":
        return JsonResponse({"status": "success", "camera_settings": camera_settings})
    return JsonResponse({"status": "error", "message": "Invalid request method"}, status=405)

def camera_feed(request):
    # Controlla i parametri della richiesta per determinare il tipo di stream
    is_greyscale = request.GET.get('isGreyscale', 'false').lower() == 'true'
    is_threshold = request.GET.get('isThreshold', 'false').lower() == 'true'

    if is_threshold:
        return StreamingHttpResponse(gen_frames_threshold(), content_type='multipart/x-mixed-replace; boundary=frame')
    elif is_greyscale:
        return StreamingHttpResponse(gen_frames_greyscale(), content_type='multipart/x-mixed-replace; boundary=frame')
    else:
        return StreamingHttpResponse(gen_frames_normal(), content_type='multipart/x-mixed-replace; boundary=frame')

def camera_feed_greyscale(request):
    # Controlla se il parametro 'show_threshold' è presente nella richiesta
    show_threshold = request.GET.get('show_threshold', 'false').lower() == 'true'
    return StreamingHttpResponse(gen_frames_greyscale(show_threshold=show_threshold), content_type='multipart/x-mixed-replace; boundary=frame')

def camera_feed_threshold(request):
    return StreamingHttpResponse(gen_frames_greyscale(show_threshold=True), content_type='multipart/x-mixed-replace; boundary=frame')
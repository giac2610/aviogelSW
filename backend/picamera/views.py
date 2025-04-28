import sys
import json
from unittest.mock import MagicMock

# Carica i parametri iniziali da setup.json
with open('/Users/ale2610/Documents/Startup/Aviogel/aviogelSW/backend/config/setup.json', 'r') as f:
    config = json.load(f)
camera_settings = config["camera"]

# Switch automatico per macOS
if sys.platform == "darwin":
    import cv2
    from django.http import JsonResponse
    import numpy as np

    # Inizializza la webcam del Mac
    mac_camera = cv2.VideoCapture(0)  # 0 indica la webcam predefinita

    def gen_frames():
        while True:
            try:
                # Ricarica i valori aggiornati di camera_settings
                with open('/Users/ale2610/Documents/Startup/Aviogel/aviogelSW/backend/config/setup.json', 'r') as f:
                    config = json.load(f)
                    camera_settings = config["camera"]

                # Cattura un frame dalla telecamera
                frame = mac_camera.read()[1]
                
                # Converti in scala di grigi
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Applica la soglia inversa
                _, thresh = cv2.threshold(gray, camera_settings["minThreshold"], camera_settings["maxThreshold"], cv2.THRESH_BINARY_INV)
                
                # Aggiungi overlay di debug
                cv2.putText(frame, f"MinThreshold: {camera_settings['minThreshold']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(frame, f"MaxThreshold: {camera_settings['maxThreshold']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Configura il rilevatore di blob
                params = cv2.SimpleBlobDetector_Params()
                params.filterByArea = camera_settings["filterByArea"]
                params.minArea = camera_settings["minArea"]
                params.maxArea = camera_settings["maxArea"]
                params.filterByCircularity = camera_settings["filterByCircularity"]
                params.minCircularity = camera_settings["minCircularity"]
                params.filterByConvexity = camera_settings["filterByConvexity"]
                params.minConvexity = camera_settings["minConvexity"]
                params.filterByInertia = camera_settings["filterByInertia"]
                params.minInertiaRatio = camera_settings["minInertiaRatio"]
                
                # Rileva i blob
                detector = cv2.SimpleBlobDetector_create(params)
                keypoints = detector.detect(thresh)
                
                # Disegna i blob rilevati
                frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                
                # Codifica il frame in JPEG
                _, buffer = cv2.imencode('.jpg', frame_with_keypoints)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Errore durante la generazione del frame: {e}")
                break

    def gen_frames_greyscale(show_threshold=False):
        while True:
            try:
                # Ricarica i valori aggiornati di camera_settings
                with open('/Users/ale2610/Documents/Startup/Aviogel/aviogelSW/backend/config/setup.json', 'r') as f:
                    config = json.load(f)
                    camera_settings = config["camera"]

                # Cattura un frame dalla telecamera
                frame = mac_camera.read()[1]
                
                # Converti in scala di grigi
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Applica la soglia inversa
                _, thresh = cv2.threshold(gray, camera_settings["minThreshold"], camera_settings["maxThreshold"], cv2.THRESH_BINARY_INV)
                
                # Se show_threshold è True, mostra il frame binarizzato
                frame_to_show = thresh if show_threshold else gray
                
                # Aggiungi overlay di debug
                cv2.putText(frame_to_show, f"MinThreshold: {camera_settings['minThreshold']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame_to_show, f"MaxThreshold: {camera_settings['maxThreshold']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Codifica il frame in JPEG
                _, buffer = cv2.imencode('.jpg', frame_to_show)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Errore durante la generazione del frame in scala di grigi: {e}")
                break
else:
    from picamera2 import Picamera2  # type: ignore
    import cv2
    import numpy as np

    # Inizializza Picamera2
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
    picam2.start()

    # Configura la fotocamera per abilitare l'autofocus (se supportato)
    picam2.set_controls({"AfMode": 1})  # 1 abilita l'autofocus, 0 lo disabilita

    def gen_frames():
        while True:
            try:
                # Ricarica i valori aggiornati di camera_settings
                with open('/Users/ale2610/Documents/Startup/Aviogel/aviogelSW/backend/config/setup.json', 'r') as f:
                    config = json.load(f)
                    camera_settings = config["camera"]

                # Cattura un frame dalla telecamera
                frame = picam2.capture_array() if sys.platform != "darwin" else mac_camera.read()[1]
                
                # Converti i colori da RGB a BGR (solo per Raspberry Pi)
                if sys.platform != "darwin":
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                
                # Converti in scala di grigi
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                
                # Applica la soglia inversa
                _, thresh = cv2.threshold(gray, camera_settings["minThreshold"], camera_settings["maxThreshold"], cv2.THRESH_BINARY_INV)
                
                # Aggiungi overlay di debug
                cv2.putText(frame_bgr, f"MinThreshold: {camera_settings['minThreshold']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame_bgr, f"MaxThreshold: {camera_settings['maxThreshold']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Configura il rilevatore di blob
                params = cv2.SimpleBlobDetector_Params()
                params.filterByArea = camera_settings["areaFilter"]
                params.minArea = camera_settings["minArea"]
                params.maxArea = camera_settings["maxArea"]
                params.filterByCircularity = camera_settings["circularityFilter"]
                params.minCircularity = camera_settings["minCircularity"]
                params.filterByConvexity = camera_settings["filterByConvexity"]
                params.minConvexity = camera_settings["minConvexity"]
                params.filterByInertia = camera_settings["inertiaFilter"]
                params.minInertiaRatio = camera_settings["minInertia"]
                
                # Rileva i blob
                detector = cv2.SimpleBlobDetector_create(params)
                keypoints = detector.detect(thresh)
                
                # Disegna i blob rilevati
                frame_with_keypoints = cv2.drawKeypoints(frame_bgr, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                
                # Codifica il frame in JPEG
                _, buffer = cv2.imencode('.jpg', frame_with_keypoints)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Errore durante la generazione del frame: {e}")
                break

    def gen_frames_greyscale(show_threshold=False):
        while True:
            try:
                # Ricarica i valori aggiornati di camera_settings
                with open('/Users/ale2610/Documents/Startup/Aviogel/aviogelSW/backend/config/setup.json', 'r') as f:
                    config = json.load(f)
                    camera_settings = config["camera"]

                # Cattura un frame dalla telecamera
                frame = picam2.capture_array() if sys.platform != "darwin" else mac_camera.read()[1]
                
                # Converti i colori da RGB a BGR (solo per Raspberry Pi)
                if sys.platform != "darwin":
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                
                # Converti in scala di grigi
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                
                # Applica la soglia inversa
                _, thresh = cv2.threshold(gray, camera_settings["minThreshold"], camera_settings["maxThreshold"], cv2.THRESH_BINARY_INV)
                
                # Se show_threshold è True, mostra il frame binarizzato
                frame_to_show = thresh if show_threshold else gray
                
                # Aggiungi overlay di debug
                cv2.putText(frame_to_show, f"MinThreshold: {camera_settings['minThreshold']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame_to_show, f"MaxThreshold: {camera_settings['maxThreshold']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Codifica il frame in JPEG
                _, buffer = cv2.imencode('.jpg', frame_to_show)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Errore durante la generazione del frame in scala di grigi: {e}")
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
            with open('/Users/ale2610/Documents/Startup/Aviogel/aviogelSW/backend/config/setup.json', 'w') as f:
                json.dump(config, f, indent=4)
            print(f"Impostazioni aggiornate: {camera_settings}")  # Log per debug
            return JsonResponse({"status": "success", "updated_settings": camera_settings})
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)
    elif request.method == "GET":
        return JsonResponse({"status": "success", "camera_settings": camera_settings})
    return JsonResponse({"status": "error", "message": "Invalid request method"}, status=405)

def camera_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def camera_feed_greyscale(request):
    # Controlla se il parametro 'show_threshold' è presente nella richiesta
    show_threshold = request.GET.get('show_threshold', 'false').lower() == 'true'
    return StreamingHttpResponse(gen_frames_greyscale(show_threshold=show_threshold), content_type='multipart/x-mixed-replace; boundary=frame')

def camera_feed_threshold(request):
    return StreamingHttpResponse(gen_frames_greyscale(show_threshold=True), content_type='multipart/x-mixed-replace; boundary=frame')
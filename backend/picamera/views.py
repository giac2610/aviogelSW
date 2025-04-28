import sys
from unittest.mock import MagicMock

# Switch automatico per macOS
if sys.platform == "darwin":
    import cv2
    from django.http import JsonResponse

    # Parametri configurabili
    camera_settings = {
        "minThreshold": 160,
        "maxThreshold": 210,
        "filterByArea": True,
        "minArea": 2000,
        "maxArea": 11000,
        "filterByCircularity": True,
        "minCircularity": 0.001,
        "filterByConvexity": True,
        "minConvexity": 0.001,
        "filterByInertia": False,
        "minInertiaRatio": 0.01
    }

    # Inizializza la webcam del Mac
    mac_camera = cv2.VideoCapture(0)  # 0 indica la webcam predefinita

    def gen_frames():
        while True:
            ret, frame = mac_camera.read()
            if not ret:
                print("Errore durante la lettura del frame dalla webcam")
                break
            
            # Converti in JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def gen_frames_greyscale(show_threshold=False):
        global camera_settings  # Assicura l'accesso alla variabile globale
        while True:
            ret, frame = mac_camera.read()
            if not ret:
                print("Errore durante la lettura del frame dalla webcam")
                continue  # Prova a leggere il prossimo frame
            
            try:
                # Converti in scala di grigi
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if show_threshold:
                    # Applica la soglia se richiesto
                    _, frame_to_show = cv2.threshold(gray, camera_settings["minThreshold"], camera_settings["maxThreshold"], cv2.THRESH_BINARY)
                else:
                    # Altrimenti mostra il frame in scala di grigi
                    frame_to_show = gray
                
                # Codifica il frame in JPEG
                _, buffer = cv2.imencode('.jpg', frame_to_show)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Errore durante la conversione o codifica del frame: {e}")
                continue
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

    # Parametri configurabili
    camera_settings = {
        "minThreshold": 160,
        "maxThreshold": 210,
        "filterByArea": True,
        "minArea": 2000,
        "maxArea": 11000,
        "filterByCircularity": True,
        "minCircularity": 0.001,
        "filterByConvexity": True,
        "minConvexity": 0.001,
        "filterByInertia": False,
        "minInertiaRatio": 0.01
    }

    def gen_frames():
        while True:
            # Cattura un frame dalla telecamera
            frame = picam2.capture_array()
            
            # Converti i colori da RGB a BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Converti in scala di grigi
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            
            # Applica la soglia
            _, thresh = cv2.threshold(gray, camera_settings["minThreshold"], camera_settings["maxThreshold"], cv2.THRESH_BINARY)
            
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
            frame_with_keypoints = cv2.drawKeypoints(frame_bgr, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            # Codifica il frame in JPEG
            _, buffer = cv2.imencode('.jpg', frame_with_keypoints)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def gen_frames_greyscale(show_threshold=False):
        global camera_settings  # Assicura l'accesso alla variabile globale
        while True:
            try:
                # Cattura un frame dalla telecamera
                frame = picam2.capture_array()
                
                # Converti i colori da RGB a BGR
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Converti in scala di grigi
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                
                # Applica la soglia
                _, thresh = cv2.threshold(gray, camera_settings["minThreshold"], camera_settings["maxThreshold"], cv2.THRESH_BINARY)
                
                # Se show_threshold è True, mostra il frame binarizzato
                frame_to_show = thresh if show_threshold else gray
                
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
            camera_settings.update(data)
            return JsonResponse({"status": "success", "updated_settings": camera_settings})
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)
    return JsonResponse({"status": "error", "message": "Invalid request method"}, status=405)

def camera_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def camera_feed_greyscale(request):
    # Controlla se il parametro 'show_threshold' è presente nella richiesta
    show_threshold = request.GET.get('show_threshold', 'false').lower() == 'true'
    return StreamingHttpResponse(gen_frames_greyscale(show_threshold=show_threshold), content_type='multipart/x-mixed-replace; boundary=frame')

def camera_feed_threshold(request):
    return StreamingHttpResponse(gen_frames_greyscale(show_threshold=True), content_type='multipart/x-mixed-replace; boundary=frame')
import sys
import json
import os
from unittest.mock import MagicMock

# Percorsi dei file di configurazione
CONFIG_DIR = os.path.join(os.path.dirname(__file__), '../config')
SETUP_JSON_PATH = os.path.join(CONFIG_DIR, 'setup.json')
EXAMPLE_JSON_PATH = os.path.join(CONFIG_DIR, 'setup.example.json')

# Verifica ed eventualmente rigenera setup.json
if not os.path.exists(SETUP_JSON_PATH):
    if not os.path.exists(EXAMPLE_JSON_PATH):
        raise FileNotFoundError(f"File di esempio mancante: {EXAMPLE_JSON_PATH}")
    from shutil import copyfile
    copyfile(EXAMPLE_JSON_PATH, SETUP_JSON_PATH)
    print(f"[INFO] File di configurazione creato da setup.example.json")

# Caricamento configurazione
with open(SETUP_JSON_PATH, 'r') as f:
    config = json.load(f)
camera_settings = config.get("camera", {})
# Switch automatico per macOS
if sys.platform == "darwin":
    import cv2
    from django.http import JsonResponse
    import numpy as np

    # Inizializza la webcam del Mac
    mac_camera = cv2.VideoCapture(0)  # 0 indica la webcam predefinita
    if not mac_camera.isOpened():
        raise RuntimeError("La webcam non è disponibile o è in uso da un altro processo.")

    def process_blob_detection():
        """Process the camera feed to detect blobs on the binary (threshold) image."""
        while True:
            try:
                with open(SETUP_JSON_PATH, 'r') as f:
                    config = json.load(f)
                    camera_settings = config["camera"]

                frame = mac_camera.read()[1]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, camera_settings["minThreshold"], camera_settings["maxThreshold"], cv2.THRESH_BINARY)
                keypoints = detect_blobs(thresh)
                yield frame, thresh, keypoints
            except Exception as e:
                print(f"Errore durante il processamento dei blob: {e}")
                break

else:
    from picamera2 import Picamera2  # type: ignore
    import cv2
    import numpy as np

    try:
        # Inizializza Picamera2
        picam2 = Picamera2()
        picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
        picam2.start()
    except RuntimeError as e:
        print(f"Errore durante l'inizializzazione della videocamera: {e}")
        picam2 = None  # Imposta a None se la videocamera non è disponibile

    def process_blob_detection():
        """Process the camera feed to detect blobs on the binary (threshold) image."""
        if picam2 is None:
            print("La videocamera non è disponibile. Interrompo il processamento.")
            return
        while True:
            try:
                with open(SETUP_JSON_PATH, 'r') as f:
                    config = json.load(f)
                    camera_settings = config["camera"]

                frame = picam2.capture_array()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, camera_settings["minThreshold"], camera_settings["maxThreshold"], cv2.THRESH_BINARY)
                keypoints = detect_blobs(thresh)
                yield frame_bgr, thresh, keypoints
            except Exception as e:
                print(f"Errore durante il processamento dei blob: {e}")
                break

def detect_blobs(binary_image):
    """Detect blobs on the binary (threshold) image."""
    with open(SETUP_JSON_PATH, 'r') as f:
        config = json.load(f)
        camera_settings = config["camera"]

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = camera_settings.get("areaFilter", True)
    params.minArea = camera_settings.get("minArea", 150)
    params.maxArea = camera_settings.get("maxArea", 5000)
    params.filterByCircularity = camera_settings.get("circularityFilter", True)
    params.minCircularity = camera_settings.get("minCircularity", 0.1)
    params.filterByConvexity = camera_settings.get("filterByConvexity", True)
    params.minConvexity = camera_settings.get("minConvexity", 0.87)
    params.filterByInertia = camera_settings.get("inertiaFilter", True)
    params.minInertiaRatio = camera_settings.get("minInertia", 0.01)

    detector = cv2.SimpleBlobDetector_create(params)
    return detector.detect(binary_image)

def gen_frames_visualization(mode="normal"):
    """Generate frames for visualization with blobs overlaid."""
    for frame, thresh, keypoints in process_blob_detection():
        try:
            if mode == "normal":
                frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            elif mode == "threshold":
                frame_with_keypoints = cv2.drawKeypoints(thresh, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            else:
                raise ValueError("Invalid visualization mode")

            # Aggiungi keyframe
            # cv2.putText(frame_with_keypoints, "Keyframe", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Codifica l'immagine come JPEG
            _, buffer = cv2.imencode('.jpg', frame_with_keypoints)
            frame_bytes = buffer.tobytes()

            # Genera il frame per lo streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Errore durante la generazione del frame: {e}")  # Log per debug
            break

# Funzione per applicare il rilevamento dei blob
def apply_blob_detection(frame, is_greyscale=False, is_threshold=False):
    with open(SETUP_JSON_PATH, 'r') as f:
        config = json.load(f)
        camera_settings = config["camera"]

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

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(frame)

    # Disegna i blob rilevati
    frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Aggiungi overlay di debug
    overlay_color = (255, 255, 255) if is_greyscale or is_threshold else (0, 255, 0)
    cv2.putText(frame_with_keypoints, f"MinThreshold: {camera_settings['minThreshold']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, overlay_color, 2)
    cv2.putText(frame_with_keypoints, f"MaxThreshold: {camera_settings['maxThreshold']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, overlay_color, 2)

    return frame_with_keypoints

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
    """Stream the camera feed with blob visualization."""
    mode = request.GET.get('mode', 'normal').lower()
    if mode not in ["normal", "threshold"]:
        mode = "normal"
    print(f"Modalità richiesta: {mode}")  # Log per debug
    return StreamingHttpResponse(gen_frames_visualization(mode=mode), content_type='multipart/x-mixed-replace; boundary=frame')

def release_resources():
    if sys.platform == "darwin":
        mac_camera.release()
    else:
        picam2.stop()
import sys
import json
import os
import numpy as np
import cv2
from django.http import StreamingHttpResponse, JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET
import threading

# --- Configurazione file ---
CONFIG_DIR = os.path.join(os.path.dirname(__file__), '../config')
SETUP_JSON_PATH = os.path.join(CONFIG_DIR, 'setup.json')
EXAMPLE_JSON_PATH = os.path.join(CONFIG_DIR, 'setup.example.json')
MEDIA_DIR = os.path.join(os.path.dirname(__file__), '../media')
MEDIA_URL = '/media/'

if not os.path.exists(SETUP_JSON_PATH):
    if not os.path.exists(EXAMPLE_JSON_PATH):
        raise FileNotFoundError(f"File di esempio mancante: {EXAMPLE_JSON_PATH}")
    from shutil import copyfile
    copyfile(EXAMPLE_JSON_PATH, SETUP_JSON_PATH)
    print(f"[INFO] File di configurazione creato da setup.example.json")

with open(SETUP_JSON_PATH, 'r') as f:
    config = json.load(f)
camera_settings = config.get("camera", {})

# --- Camera Init ---
if sys.platform == "darwin":
    mac_camera = cv2.VideoCapture(0)
    if not mac_camera.isOpened():
        raise RuntimeError("La webcam non è disponibile o è in uso da un altro processo.")

    def get_frame():
        ret, frame = mac_camera.read()
        if not ret:
            raise RuntimeError("Impossibile leggere il frame dalla webcam.")
        return frame
else:
    from picamera2 import Picamera2  # type: ignore
    try:
        picam2 = Picamera2()
        picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
        picam2.start()
    except RuntimeError as e:
        print(f"Errore durante l'inizializzazione della videocamera: {e}")
        picam2 = None

    def get_frame():
        frame = picam2.capture_array()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# --- Utility ---
def load_homography():
    with open(SETUP_JSON_PATH, 'r') as f:
        config = json.load(f)
    H = config.get("camera", {}).get("homography", None)
    if H is not None:
        return np.array(H, dtype=np.float32)
    return None

def save_homography(H):
    with open(SETUP_JSON_PATH, 'r') as f:
        config = json.load(f)
    config.setdefault("camera", {})
    config["camera"]["homography"] = H.tolist()
    with open(SETUP_JSON_PATH, 'w') as f:
        json.dump(config, f, indent=4)

def detect_blobs(binary_image):
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

def get_current_frame_and_keypoints():
    frame = get_frame()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray,
        camera_settings["minThreshold"],
        camera_settings["maxThreshold"],
        cv2.THRESH_BINARY
    )
    keypoints = detect_blobs(thresh)
    return keypoints

def get_grid_points(righe=6, colonne=8, dx=50, dy=50):
    return [[j*dx, i*dy] for i in range(righe) for j in range(colonne)]

# --- API PRINCIPALI ---

@csrf_exempt
@require_POST
def update_camera_settings(request):
    """
    Aggiorna le impostazioni della camera (POST, JSON).
    """
    try:
        data = json.loads(request.body)
        global camera_settings
        camera_settings.update(data)
        config["camera"] = camera_settings
        with open(SETUP_JSON_PATH, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Impostazioni aggiornate: {camera_settings}")
        return JsonResponse({"status": "success", "updated_settings": camera_settings})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=400)

@csrf_exempt
@require_GET
def get_camera_settings(request):
    """
    Restituisce le impostazioni attuali della camera (GET).
    """
    return JsonResponse({"status": "success", "camera_settings": camera_settings})

@csrf_exempt
def camera_feed(request):
    """
    Stream MJPEG della camera (normale o threshold, a seconda dei parametri GET).
    """
    mode = request.GET.get("mode", "normal")
    def gen_frames():
        while True:
            frame = get_frame()
            if mode == "threshold":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, frame = cv2.threshold(
                    gray,
                    camera_settings["minThreshold"],
                    camera_settings["maxThreshold"],
                    cv2.THRESH_BINARY
                )
            keypoints = detect_blobs(frame if mode == "threshold" else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            frame_with_keypoints = cv2.drawKeypoints(
                frame if mode == "normal" else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR),
                keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            _, buffer = cv2.imencode('.jpg', frame_with_keypoints)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

@csrf_exempt
@require_GET
def get_frame_api(request):
    """
    Restituisce un singolo frame JPEG (non streaming).
    """
    try:
        frame = get_frame()
        _, buffer = cv2.imencode('.jpg', frame)
        return HttpResponse(buffer.tobytes(), content_type='image/jpeg')
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

@csrf_exempt
@require_GET
def get_keypoints(request):
    """
    Restituisce SOLO i keypoints rilevati (coordinate immagine).
    """
    try:
        keypoints = get_current_frame_and_keypoints()
        keypoints_list = [[kp.pt[0], kp.pt[1]] for kp in keypoints]
        return JsonResponse({"status": "success", "keypoints": keypoints_list})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

@csrf_exempt
@require_GET
def get_keypoints_all(request):
    """
    Restituisce tutti i keypoints: quelli rilevati e quelli interpolati (reticolo 6x8, prima del warping).
    """
    try:
        righe, colonne, dx, dy = 6, 8, 50, 50
        keypoints = get_current_frame_and_keypoints()
        keypoints_img = [[kp.pt[0], kp.pt[1]] for kp in keypoints]
        grid_points = get_grid_points(righe, colonne, dx, dy)
        soglia = 30
        missing = []
        for pt in grid_points:
            dists = [np.linalg.norm(np.array(pt)-np.array(kp)) for kp in keypoints_img]
            if not dists or min(dists) > soglia:
                missing.append(pt)
        return JsonResponse({
            "status": "success",
            "keypoints_detected": keypoints_img,
            "keypoints_grid": grid_points,
            "keypoints_interpolated": missing
        })
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

@csrf_exempt
@require_POST
def set_camera_origin(request):
    """
    Imposta l'origine della camera (POST, JSON: x, y).
    """
    try:
        data = json.loads(request.body)
        x = float(data.get("x", 0))
        y = float(data.get("y", 0))
        with open(SETUP_JSON_PATH, 'r') as f:
            config = json.load(f)
        config.setdefault("camera", {})
        config["camera"]["x"] = x
        config["camera"]["y"] = y
        with open(SETUP_JSON_PATH, 'w') as f:
            json.dump(config, f, indent=4)
        return JsonResponse({"status": "success", "origin": {"x": x, "y": y}})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=400)

@csrf_exempt
@require_GET
def get_homography(request):
    """
    Calcola e restituisce la matrice di omografia ricavata dai keypoints (reticolo 6x8) e quelli rilevati.
    """
    try:
        righe, colonne, dx, dy = 6, 8, 50, 50
        keypoints = get_current_frame_and_keypoints()
        keypoints_img = [[kp.pt[0], kp.pt[1]] for kp in keypoints]
        grid_points = get_grid_points(righe, colonne, dx, dy)
        if len(keypoints_img) >= 4:
            pts_img = np.array(keypoints_img[:righe*colonne], dtype=np.float32)
            pts_mm = np.array(grid_points[:len(pts_img)], dtype=np.float32)
            H, _ = cv2.findHomography(pts_mm, pts_img, method=0)
            if H is not None:
                return JsonResponse({"homography": H.tolist()})
        return JsonResponse({"homography": None})
    except Exception as e:
        return JsonResponse({"homography": None, "error": str(e)})

@csrf_exempt
@require_GET
def dynamic_warped_stream(request):
    """
    Stream MJPEG: calcola la matrice omografica dai blob rilevati in tempo reale e applica il warp a ogni frame.
    """
    def gen_frames():
        righe, colonne, dx, dy = 6, 8, 50, 50
        grid_points = get_grid_points(righe, colonne, dx, dy)
        while True:
            frame = get_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(
                gray,
                camera_settings["minThreshold"],
                camera_settings["maxThreshold"],
                cv2.THRESH_BINARY
            )
            keypoints = detect_blobs(thresh)
            keypoints_img = [[kp.pt[0], kp.pt[1]] for kp in keypoints]
            if len(keypoints_img) >= 4:
                pts_img = np.array(keypoints_img[:righe*colonne], dtype=np.float32)
                pts_mm = np.array(grid_points[:len(pts_img)], dtype=np.float32)
                H, _ = cv2.findHomography(pts_mm, pts_img, method=0)
                if H is not None:
                    h, w = frame.shape[:2]
                    warped = cv2.warpPerspective(frame, np.linalg.inv(H), (w, h))
                    _, buffer = cv2.imencode('.jpg', warped)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    continue
            # fallback: frame originale se non si può calcolare l'omografia
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

@csrf_exempt
@require_GET
def capture_and_warp_frame(request):
    """
    Cattura un frame, annota i keypoints sull'originale, calcola la matrice omografica,
    applica il warp e salva entrambe le immagini (originale e warpata).
    Restituisce i path delle immagini statiche.
    """
    frame = get_frame()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray,
        camera_settings["minThreshold"],
        camera_settings["maxThreshold"],
        cv2.THRESH_BINARY
    )
    keypoints = detect_blobs(thresh)
    keypoints_img = [[kp.pt[0], kp.pt[1]] for kp in keypoints]

    # Annotazione keypoints sull'immagine originale
    annotated = frame.copy()
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(annotated, (x, y), 8, (0,0,255), 2)
        cv2.putText(annotated, f"({int(kp.pt[0])},{int(kp.pt[1])})", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    # Calcola omografia e warpa
    righe, colonne, dx, dy = 6, 8, 50, 50
    grid_points = get_grid_points(righe, colonne, dx, dy)
    warped = None
    H = None
    if len(keypoints_img) >= 4:
        pts_img = np.array(keypoints_img[:righe*colonne], dtype=np.float32)
        pts_mm = np.array(grid_points[:len(pts_img)], dtype=np.float32)
        H, _ = cv2.findHomography(pts_mm, pts_img, method=0)
        if H is not None:
            h, w = frame.shape[:2]
            warped = cv2.warpPerspective(frame, np.linalg.inv(H), (w, h))

    # Salva immagini statiche nella cartella media, sovrascrivendo sempre gli stessi file
    os.makedirs(MEDIA_DIR, exist_ok=True)
    orig_filename = "frame_orig.jpg"
    warped_filename = "frame_warped.jpg"
    orig_path = os.path.join(MEDIA_DIR, orig_filename)
    warped_path = os.path.join(MEDIA_DIR, warped_filename)
    saved_orig = cv2.imwrite(orig_path, annotated)
    if warped is not None:
        saved_warped = cv2.imwrite(warped_path, warped)
        warped_url = MEDIA_URL + warped_filename if saved_warped else None
    else:
        warped_url = None
    orig_url = MEDIA_URL + orig_filename if saved_orig else None

    # Log di debug
    print(f"[DEBUG] Immagine originale salvata: {orig_path} ({'OK' if saved_orig else 'FALLITO'})")
    if warped is not None:
        print(f"[DEBUG] Immagine warping salvata: {warped_path} ({'OK' if saved_warped else 'FALLITO'})")

    return JsonResponse({
        "status": "success",
        "original_url": orig_url,
        "warped_url": warped_url,
        "homography": H.tolist() if H is not None else None
    })

@csrf_exempt
def calculate_homography_from_points(request):
    """
    Riceve 4 punti (x, y), calcola la warp e restituisce l'immagine post-processata.
    """
    try:
        data = json.loads(request.body)
        points = data.get("points", [])
        if not isinstance(points, list) or len(points) != 4:
            return JsonResponse({"status": "error", "message": "Devi fornire 4 punti"}, status=400)
        pts_src = np.array([[float(p['x']), float(p['y'])] for p in points], dtype=np.float32)
        # Ordina i punti (top-left, top-right, bottom-right, bottom-left)
        s = pts_src.sum(axis=1)
        diff = np.diff(pts_src, axis=1)
        rect = np.zeros((4,2), dtype="float32")
        rect[0] = pts_src[np.argmin(s)]
        rect[2] = pts_src[np.argmax(s)]
        rect[1] = pts_src[np.argmin(diff)]
        rect[3] = pts_src[np.argmax(diff)]
        w = int(max(np.linalg.norm(rect[0]-rect[1]), np.linalg.norm(rect[2]-rect[3])))
        h = int(max(np.linalg.norm(rect[0]-rect[3]), np.linalg.norm(rect[1]-rect[2])))
        dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype="float32")
        H, _ = cv2.findHomography(rect, dst)
        frame = get_frame()
        warped = cv2.warpPerspective(frame, H, (w, h))
        # Salva l'immagine warpata
        os.makedirs(MEDIA_DIR, exist_ok=True)
        warped_filename = "custom_warped.jpg"
        warped_path = os.path.join(MEDIA_DIR, warped_filename)
        cv2.imwrite(warped_path, warped)
        warped_url = MEDIA_URL + warped_filename
        return JsonResponse({"status": "success", "warped_url": warped_url})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

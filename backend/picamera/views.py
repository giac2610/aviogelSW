from contextlib import contextmanager
import cv2
import numpy as np
import glob
import sys
import json
import os
import time
import threading
from django.http import StreamingHttpResponse, JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET
# from django.conf import settings # Non usato direttamente qui, ma potresti averlo per MEDIA_ROOT ecc.

# --- Configurazione file ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Assumendo che views.py sia in una subdir dell'app
CONFIG_DIR = os.path.join(BASE_DIR, 'config')
SETUP_JSON_PATH = os.path.join(CONFIG_DIR, 'setup.json')
EXAMPLE_JSON_PATH = os.path.join(CONFIG_DIR, 'setup.example.json')
CALIBRATION_MEDIA_DIR = os.path.join(BASE_DIR, 'calibrationMedia') # Metti le immagini di calibrazione qui
os.makedirs(CALIBRATION_MEDIA_DIR, exist_ok=True) # Crea la cartella se non esiste

# MEDIA_URL = '/media/' # Per Django settings, non direttamente usato qui

if not os.path.exists(SETUP_JSON_PATH):
    from shutil import copyfile
    os.makedirs(CONFIG_DIR, exist_ok=True)
    copyfile(EXAMPLE_JSON_PATH, SETUP_JSON_PATH)
    print(f"[INFO] File di configurazione creato da {EXAMPLE_JSON_PATH}")

# --- Caricamento configurazione globale ---
try:
    with open(SETUP_JSON_PATH, 'r') as f:
        config = json.load(f)
    camera_settings = config.get("camera", {})
except Exception as e:
    print(f"Errore critico nel caricamento di setup.json all'avvio: {e}")
    config = {} # Fallback
    camera_settings = {}

# --- Camera Init ---
camera_instance = None
camera_lock = threading.Lock()  # Lock globale per accesso thread-safe
active_streams = 0

def initialize_camera():
    global camera_instance
    with camera_lock:
        if camera_instance is not None:
            # Tenta di rilasciare la camera se già inizializzata, per sicurezza
            if hasattr(camera_instance, 'release'):
                camera_instance.release()
            elif hasattr(camera_instance, 'stop'): # Per Picamera2
                # Rilascia solo se esiste il metodo stop
                try:
                    camera_instance.stop()
                except Exception:
                    pass
            camera_instance = None

        if sys.platform == "darwin":
            mac_cam = cv2.VideoCapture(1)
            if not mac_cam.isOpened():
                mac_cam = cv2.VideoCapture(0)
                if not mac_cam.isOpened():
                    print("ATTENZIONE: Webcam non disponibile o in uso su macOS.")
                    return None
            camera_instance = mac_cam
            print("[INFO] Camera macOS inizializzata.")
        else:
            try:
                from picamera2 import Picamera2
                picam2 = Picamera2()
                cfg_data = load_config_data()
                # picam_config = cfg_data.get("camera", {}).get("picamera_config", {"main": {"size": (640, 480)}})
                picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
                picam2.start()
                camera_instance = picam2
                print("[INFO] Picamera2 inizializzata.")
            except Exception as e:
                print(f"Errore durante l'inizializzazione della Picamera2: {e}")
                camera_instance = None
                return None
        return camera_instance

# Inizializza la camera subito all'avvio del modulo, così è pronta per il primo accesso
initialize_camera()

def get_frame(release_after=False):
    global camera_instance, active_streams
    with camera_lock:
        if camera_instance is None:
            print("get_frame: Tentativo di reinizializzare la camera.")
            camera_instance = initialize_camera()
            if camera_instance is None:
                print("get_frame: Camera non disponibile, restituisco frame vuoto.")
                return np.zeros((480, 640, 3), dtype=np.uint8)

        # Se c'è almeno uno stream attivo, non chiudere la camera
        should_release = release_after and active_streams == 0

        if sys.platform == "darwin":
            if not camera_instance.isOpened():
                print("get_frame (macOS): Camera non aperta, restituisco frame vuoto.")
                return np.zeros((480, 640, 3), dtype=np.uint8)
            ret, frame = camera_instance.read()
            if should_release:
                camera_instance.release()
                camera_instance = None
            if not ret:
                print("get_frame (macOS): Impossibile leggere il frame.")
                return np.zeros((480, 640, 3), dtype=np.uint8)
            return frame
        else: # Picamera2
            # Nuovo controllo: la camera è pronta se esiste il metodo capture_array
            if not camera_instance or not hasattr(camera_instance, 'capture_array'):
                print("get_frame (Pi): Picamera2 non pronta, restituisco frame vuoto.")
                return np.zeros((480, 640, 3), dtype=np.uint8)
            try:
                frame = camera_instance.capture_array()
            except Exception as e:
                print(f"get_frame (Pi): Errore nella cattura del frame: {e}")
                return np.zeros((480, 640, 3), dtype=np.uint8)
            if should_release:
                try:
                    camera_instance.stop()
                except Exception:
                    pass
                camera_instance = None
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)



# --- Utility ---
def load_config_data():
    try:
        with open(SETUP_JSON_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Errore durante il caricamento di {SETUP_JSON_PATH}: {e}")
        return {}

def save_config_data(new_config_data):
    try:
        with open(SETUP_JSON_PATH, 'w') as f:
            json.dump(new_config_data, f, indent=4)
        global config, camera_settings # Aggiorna variabili globali
        config = new_config_data
        camera_settings = config.get("camera", {})
        print(f"Configurazione salvata e ricaricata.")
        return True
    except Exception as e:
        print(f"Errore durante il salvataggio in {SETUP_JSON_PATH}: {e}")
        return False

def load_fixed_perspective_homography():
    current_config = load_config_data()
    H_list = current_config.get("camera", {}).get("fixed_perspective", {}).get("homography_matrix", None)
    if H_list and isinstance(H_list, list): # Check, controlla che sia una lista
        try:
            return np.array(H_list, dtype=np.float32)
        except Exception as e:
            print(f"Errore nel convertire fixed_perspective_homography in array numpy: {e}")
            return None
    return None

def save_fixed_perspective_homography(H_matrix_ref):
    current_config = load_config_data()
    # Assicura che i dizionari intermedi esistano
    current_config.setdefault("camera", {}).setdefault("fixed_perspective", {})
    if H_matrix_ref is not None:
        current_config["camera"]["fixed_perspective"]["homography_matrix"] = H_matrix_ref.tolist()
    else:
        current_config["camera"]["fixed_perspective"]["homography_matrix"] = None
    return save_config_data(current_config)

def detect_blobs(binary_image): # Tua funzione originale
    cfg = load_config_data().get("camera", {})
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = cfg.get("areaFilter", True)
    params.minArea = cfg.get("minArea", 150)
    params.maxArea = cfg.get("maxArea", 5000)
    params.filterByCircularity = cfg.get("circularityFilter", True)
    params.minCircularity = cfg.get("minCircularity", 0.1)
    params.filterByConvexity = cfg.get("filterByConvexity", True)
    params.minConvexity = cfg.get("minConvexity", 0.87)
    params.filterByInertia = cfg.get("inertiaFilter", True)
    params.minInertiaRatio = cfg.get("minInertia", 0.01)
    detector = cv2.SimpleBlobDetector_create(params)
    return detector.detect(binary_image)

def get_current_frame_and_keypoints():
    cfg = load_config_data().get("camera", {})
    # Usa release_after=True per endpoint che fanno una sola acquisizione
    frame = get_frame(release_after=True)
    if frame is None or frame.size == 0:
        print("get_current_frame_and_keypoints: Ricevuto frame non valido.")
        return np.zeros((480, 640, 3), dtype=np.uint8), []
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray,
        cfg.get("minThreshold", 127),
        cfg.get("maxThreshold", 255),
        cv2.THRESH_BINARY
    )
    keypoints = detect_blobs(thresh)
    return frame, keypoints

# --- Logica Chiave per la Vista Fissa (Adattata da Script) ---
def get_board_and_canonical_homography_for_django(undistorted_frame, new_camera_matrix_cv, calibration_cfg_dict):
    cs_cols = calibration_cfg_dict.get("chessboard_cols", 7)
    cs_rows = calibration_cfg_dict.get("chessboard_rows", 9)
    sq_size = calibration_cfg_dict.get("square_size_mm", 15.0)
    chessboard_dim_cv = (cs_cols, cs_rows)

    objp_cv = np.zeros((cs_cols * cs_rows, 3), np.float32)
    objp_cv[:, :2] = np.mgrid[0:cs_cols, 0:cs_rows].T.reshape(-1, 2)
    objp_cv *= sq_size
    # objp_cv[:, 1] = (cs_rows - 1) * sq_size - objp_cv[:, 1]
    
    criteria_cv = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_dim_cv, None)

    if not ret: return None, None

    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria_cv)
    success, rvec, tvec = cv2.solvePnP(objp_cv, corners2, new_camera_matrix_cv, None)

    if not success: return None, None

    board_w_obj = (cs_cols - 1) * sq_size
    board_h_obj = (cs_rows - 1) * sq_size
    
    obj_board_perimeter_pts = np.array([
        [0,0,0], [board_w_obj,0,0], 
        [board_w_obj,board_h_obj,0], [0,board_h_obj,0]
    ], dtype=np.float32)

    img_board_perimeter_pts, _ = cv2.projectPoints(obj_board_perimeter_pts, rvec, tvec, new_camera_matrix_cv, None)
    img_board_perimeter_pts = img_board_perimeter_pts.reshape(-1, 2)

    canonical_dst_pts = np.array([
        [0,0], [board_w_obj-1,0], 
        [board_w_obj-1,board_h_obj-1], [0,board_h_obj-1]
    ], dtype=np.float32)
    
    H_canonical = cv2.getPerspectiveTransform(img_board_perimeter_pts, canonical_dst_pts)
    canonical_board_size = (int(board_w_obj), int(board_h_obj))
    
    return H_canonical, canonical_board_size

@contextmanager
def stream_context():
    global active_streams
    active_streams += 1
    try:
        yield
    finally:
        active_streams -= 1


# --- Endpoint Django ---
@csrf_exempt
@require_POST
def update_camera_settings(request): # Tua funzione originale
    try:
        data = json.loads(request.body)
        current_config_data = load_config_data()
        current_config_data.setdefault("camera", {})
        # Conserva le sotto-chiavi esistenti come "calibration" se non sono in data
        for key, value in data.items():
            if isinstance(value, dict) and key in current_config_data["camera"] and isinstance(current_config_data["camera"][key], dict):
                current_config_data["camera"][key].update(value)
            else:
                current_config_data["camera"][key] = value
        
        if save_config_data(current_config_data):
            return JsonResponse({"status": "success", "updated_settings": config.get("camera", {})})
        else:
            return JsonResponse({"status": "error", "message": "Failed to save updated settings."}, status=500)
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=400)

@csrf_exempt
def camera_feed(request): # Tua funzione originale per i blob
    mode = request.GET.get("mode", "normal")
    def gen_frames():
        with stream_context():  # Usa il context manager per gestire gli stream
            while True:
                try:
                    cfg_cam = load_config_data().get("camera", {})
                    frame_orig = get_frame()
                    if frame_orig is None or frame_orig.size == 0:
                        frame_orig = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(frame_orig, "No Frame", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

                    gray = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
                    _, processed_for_blobs = cv2.threshold(
                        gray, cfg_cam.get("minThreshold", 127),
                        cfg_cam.get("maxThreshold", 255), cv2.THRESH_BINARY
                    )

                    display_frame_feed = frame_orig.copy()
                    if mode == "threshold":
                        display_frame_feed = cv2.cvtColor(processed_for_blobs, cv2.COLOR_GRAY2BGR)

                    keypoints_blob = detect_blobs(processed_for_blobs)
                    frame_with_keypoints = cv2.drawKeypoints(
                        display_frame_feed, keypoints_blob, np.array([]), (0, 0, 255), 
                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                    # Parallelepipedo (dal tuo script)
                    # if keypoints_blob and len(keypoints_blob) >= 4:
                    #     pts = np.array([kp.pt for kp in keypoints_blob], dtype=np.float32)
                    #     s = pts.sum(axis=1)
                    #     diff = np.diff(pts, axis=1)
                    #     corners = np.zeros((4,2), dtype=np.float32)
                    #     corners[0] = pts[np.argmin(s)]
                    #     corners[2] = pts[np.argmax(s)]
                    #     corners[1] = pts[np.argmin(diff)]
                    #     corners[3] = pts[np.argmax(diff)]
                    #     corners_int = corners.astype(np.intp)
                    #     inside = all(cv2.pointPolygonTest(corners, (pt[0], pt[1]), False) >= 0 for pt in pts)
                    #     if inside:
                    #         cv2.polylines(frame_with_keypoints, [corners_int], isClosed=True, color=(0,255,0), thickness=2)

                    _, buffer = cv2.imencode('.jpg', frame_with_keypoints)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    time.sleep(0.03) # Limita framerate
                except Exception as e:
                    error_f = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(error_f, f"Error: {str(e)[:50]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    _, buffer = cv2.imencode('.jpg', error_f)
                    frame_b = buffer.tobytes()
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_b + b'\r\n')
                    time.sleep(1)
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

@csrf_exempt
@require_GET
def get_keypoints(request): # Tua funzione originale
    try:
        # Usa release_after=True per evitare di lasciare la camera aperta inutilmente
        _, keypoints_data = get_current_frame_and_keypoints()
        keypoints_list = [[kp.pt[0], kp.pt[1]] for kp in keypoints_data]
        
        rect_vertices = []
        if len(keypoints_data) >= 2: # cv2.minAreaRect richiede almeno 2 punti, ma 3 per un rettangolo non degenere
            pts_rect = np.array(keypoints_list, dtype=np.float32)
            if pts_rect.shape[0] >=3 : # Assicurati ci siano abbastanza punti per un rettangolo
                 rect = cv2.minAreaRect(pts_rect)
                 box = cv2.boxPoints(rect)
                 rect_vertices = box.astype(float).tolist()

        parallelepiped_vertices = []
        parallelepiped_ok = False
        if len(keypoints_data) >= 4:
            pts_para = np.array(keypoints_list, dtype=np.float32)
            s = pts_para.sum(axis=1)
            diff = np.diff(pts_para, axis=1)
            corners = np.zeros((4,2), dtype=np.float32)
            corners[0] = pts_para[np.argmin(s)]
            corners[2] = pts_para[np.argmax(s)]
            corners[1] = pts_para[np.argmin(diff)]
            corners[3] = pts_para[np.argmax(diff)]
            inside = all(cv2.pointPolygonTest(corners, (pt[0], pt[1]), False) >= 0 for pt in pts_para)
            if inside:
                parallelepiped_vertices = corners.tolist()
                parallelepiped_ok = True

        return JsonResponse({
            "status": "success", "keypoints": keypoints_list,
            "bounding_box_vertices": rect_vertices,
            "parallelepiped_vertices": parallelepiped_vertices,
            "parallelepiped_ok": parallelepiped_ok
        })
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

@csrf_exempt
@require_POST
def set_camera_origin(request): # Tua funzione originale
    try:
        data = json.loads(request.body)
        x_val = float(data.get("origin_x", 0.0))
        y_val = float(data.get("origin_y", 0.0))
        
        current_config_data = load_config_data()
        current_config_data.setdefault("camera", {})
        current_config_data["camera"]["origin_x"] = x_val
        current_config_data["camera"]["origin_y"] = y_val
        
        if save_config_data(current_config_data):
            return JsonResponse({"status": "success", "origin": {"origin_x": x_val, "origin_y": y_val}})
        else:
            return JsonResponse({"status": "error", "message": "Failed to save origin."}, status=500)
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=400)

@csrf_exempt
@require_POST
def save_frame_calibration(request): # Tua funzione originale
    try:
        # Usa release_after=True per acquisizione singola
        frame_to_save = get_frame(release_after=True)
        if frame_to_save is None or frame_to_save.size == 0:
            return JsonResponse({"status": "error", "message": "Frame non valido"}, status=500)
        
        filename = f"calib_{int(time.time())}.jpg"
        filepath = os.path.join(CALIBRATION_MEDIA_DIR, filename) # Usa la costante definita
        cv2.imwrite(filepath, frame_to_save)
        print(f"Frame salvato per calibrazione: {filepath}")
        return JsonResponse({"status": "success", "filename": filename, "path": filepath})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

@csrf_exempt
@require_POST
def calibrate_camera_endpoint(request): # Rinomina per chiarezza, era calibrate_camera
    current_config_data = load_config_data()
    calib_settings = current_config_data.get("camera", {}).get("calibration_settings", {})
    
    cs_cols = calib_settings.get("chessboard_cols", 7)
    cs_rows = calib_settings.get("chessboard_rows", 9)
    square_size_mm = calib_settings.get("square_size_mm", 15.0)
    chessboard_dim_config = (cs_cols, cs_rows)
    criteria_config = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp_config = np.zeros((cs_cols * cs_rows, 3), np.float32)
    objp_config[:, :2] = np.mgrid[0:cs_cols, 0:cs_rows].T.reshape(-1, 2)
    objp_config *= square_size_mm

    objpoints_list = []
    imgpoints_list = []

    image_files = glob.glob(os.path.join(CALIBRATION_MEDIA_DIR, '*.jpg'))
    print(f"Trovate {len(image_files)} immagini per la calibrazione in {CALIBRATION_MEDIA_DIR}.")

    if not image_files:
        return JsonResponse({"status": "error", "message": f"Nessuna immagine .jpg trovata in {CALIBRATION_MEDIA_DIR}."}, status=400)
    
    last_gray_shape = None
    images_processed_count = 0

    for image_path in image_files:
        img = cv2.imread(image_path)
        if img is None: continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if images_processed_count == 0: # Prendi le dimensioni dalla prima immagine valida
            last_gray_shape = gray.shape[::-1]
        elif last_gray_shape != gray.shape[::-1]:
             print(f"ATTENZIONE: Dimensione immagine {image_path} ({gray.shape[::-1]}) diversa da precedente ({last_gray_shape}).")
             # Continua, ma la calibrazione potrebbe essere imprecisa se le dimensioni variano molto.
             # `cv2.calibrateCamera` usa la dimensione fornita principalmente per inizializzare K.

        images_processed_count +=1
        ret_corners, corners = cv2.findChessboardCorners(gray, chessboard_dim_config, None)
        if ret_corners:
            print(f"Scacchiera trovata in: {image_path}")
            objpoints_list.append(objp_config)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria_config)
            imgpoints_list.append(corners2)
        else:
            print(f"Scacchiera NON trovata in: {image_path}")
    
    if not objpoints_list or not imgpoints_list:
        return JsonResponse({"status": "error", "message": "Nessun punto scacchiera valido trovato nelle immagini fornite."}, status=400)
    if last_gray_shape is None: # Nessuna immagine è stata processata con successo
         return JsonResponse({"status": "error", "message": "Nessuna immagine processata, impossibile determinare dimensioni per calibrazione."}, status=400)

    print(f"Calcolo parametri di calibrazione usando {len(objpoints_list)} set di punti. Dim. immagine: {last_gray_shape}")
    ret_calib, camera_matrix_calib, dist_coeffs_calib, _, _ = cv2.calibrateCamera(
        objpoints_list, imgpoints_list, last_gray_shape, None, None
    )

    if ret_calib:
        calibration_data_tosave = {
            "camera_matrix": camera_matrix_calib.tolist(),
            "distortion_coefficients": dist_coeffs_calib.tolist()
        }
        current_config_data.setdefault("camera", {}).setdefault("calibration", {})
        current_config_data["camera"]["calibration"] = calibration_data_tosave
        
        if save_config_data(current_config_data):
            return JsonResponse({"status": "success", "message": "Calibrazione completata e salvata.", "calibration": calibration_data_tosave})
        else:
            return JsonResponse({"status": "error", "message": "Calibrazione completata ma fallito salvataggio configurazione."}, status=500)
    else:
        return JsonResponse({"status": "error", "message": "cv2.calibrateCamera fallita."}, status=500)

@csrf_exempt
@require_POST
def set_fixed_perspective_view(request):
    current_config_data = load_config_data()
    cam_calib_data = current_config_data.get("camera", {}).get("calibration", None)
    calib_settings_dict = current_config_data.get("camera", {}).get("calibration_settings", {})
    fixed_perspective_cfg = current_config_data.get("camera", {}).get("fixed_perspective", {})

    if not (cam_calib_data and cam_calib_data.get("camera_matrix") and cam_calib_data.get("distortion_coefficients")):
        return JsonResponse({"status": "error", "message": "Calibrazione camera non trovata. Eseguire prima la calibrazione."}, status=400)

    camera_matrix_cv = np.array(cam_calib_data["camera_matrix"], dtype=np.float32)
    dist_coeffs_cv = np.array(cam_calib_data["distortion_coefficients"], dtype=np.float32)

    FIXED_WIDTH = fixed_perspective_cfg.get("output_width", 1000)
    FIXED_HEIGHT = fixed_perspective_cfg.get("output_height", 800)

    with stream_context():  # Usa il context manager per gestire gli stream
        try:
            # Usa release_after=True per acquisizione singola
            frame_cap = get_frame(release_after=True)
            if frame_cap is None or frame_cap.size == 0: 
                return JsonResponse({"status": "error", "message": "Impossibile ottenere frame dalla camera."}, status=500)

            h_cam_cap, w_cam_cap = frame_cap.shape[:2]
            new_camera_matrix_cv, _ = cv2.getOptimalNewCameraMatrix(camera_matrix_cv, dist_coeffs_cv, (w_cam_cap,h_cam_cap), 1.0, (w_cam_cap,h_cam_cap))
            undistorted_frame_cap = cv2.undistort(frame_cap, camera_matrix_cv, dist_coeffs_cv, None, new_camera_matrix_cv)

            H_canonical, canonical_dims = get_board_and_canonical_homography_for_django(
                undistorted_frame_cap, new_camera_matrix_cv, calib_settings_dict
            )

            if H_canonical is not None:
                cb_w, cb_h = canonical_dims
                offset_x = (FIXED_WIDTH - cb_w) / 2.0
                offset_y = (FIXED_HEIGHT - cb_h) / 2.0
                M_translate = np.array([[1,0,offset_x], [0,1,offset_y], [0,0,1]], dtype=np.float32)
                H_ref = M_translate @ H_canonical

                if save_fixed_perspective_homography(H_ref):
                    return JsonResponse({"status": "success", "message": "Vista fissa del piano stabilita e salvata."})
                else:
                    return JsonResponse({"status": "error", "message": "Errore salvataggio omografia vista fissa."}, status=500)
            else:
                return JsonResponse({"status": "error", "message": "Scacchiera non trovata. Impossibile definire vista fissa."}, status=400)
        except Exception as e:
            print(f"Eccezione in set_fixed_perspective_view: {e}")
            import traceback
            traceback.print_exc()
            return JsonResponse({"status": "error", "message": str(e)}, status=500)

@csrf_exempt
@require_GET
def fixed_perspective_stream(request):
    def gen_frames():
        # Carica config una volta all'inizio per questo worker dello stream
        # Nota: se la config cambia mentre lo stream è attivo, non vedrà le modifiche
        # fino a quando lo stream non viene riavviato. Per modifiche dinamiche, `load_config_data()`
        # dovrebbe essere chiamato dentro il loop, ma può avere impatto sulle performance.
        cfg_snapshot = load_config_data()
        H_ref = load_fixed_perspective_homography() # Carica da cfg_snapshot o direttamente da file
        
        cam_calib = cfg_snapshot.get("camera", {}).get("calibration", None)
        fixed_persp_cfg = cfg_snapshot.get("camera", {}).get("fixed_perspective", {})
        
        OUT_W = fixed_persp_cfg.get("output_width", 1000)
        OUT_H = fixed_persp_cfg.get("output_height", 800)

        if not (cam_calib and cam_calib.get("camera_matrix") and cam_calib.get("distortion_coefficients")):
            error_msg = "Calibrazione camera mancante"
            print(f"fixed_perspective_stream: {error_msg}")
            while True:
                err_f = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
                cv2.putText(err_f, error_msg, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                _, buf = cv2.imencode('.jpg', err_f); yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'); time.sleep(1)

        cam_matrix = np.array(cam_calib["camera_matrix"], dtype=np.float32)
        dist_coeffs = np.array(cam_calib["distortion_coefficients"], dtype=np.float32)
        
        # Calcola new_camera_matrix una volta se le dimensioni del frame sono costanti
        new_cam_matrix_stream = None
        try:
            # Prendi un frame per determinare le dimensioni per new_camera_matrix
            # Questo è importante se la risoluzione della camera può cambiare o non è nota a priori
            # per questo specifico stream.
            sample_frame_for_dims = get_frame()
            if sample_frame_for_dims is not None and sample_frame_for_dims.size > 0 :
                h_str, w_str = sample_frame_for_dims.shape[:2]
                new_cam_matrix_stream, _ = cv2.getOptimalNewCameraMatrix(cam_matrix, dist_coeffs, (w_str,h_str), 1.0, (w_str,h_str))
            else: # Fallback se non si ottiene un frame valido
                 raise ValueError("Impossibile ottenere dimensioni frame per new_camera_matrix")
        except Exception as e_stream_setup:
            error_msg = f"Setup stream fallito: {e_stream_setup}"
            print(f"fixed_perspective_stream: {error_msg}")
            while True:
                err_f = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
                cv2.putText(err_f, error_msg[:70], (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
                _, buf = cv2.imencode('.jpg', err_f); yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'); time.sleep(1)
        
        while True:
            try:
                frame_live = get_frame()
                if frame_live is None or frame_live.size == 0: 
                    # Se il frame è perso, invia un frame nero o l'ultimo valido (più complesso)
                    # Per ora, un frame nero con messaggio
                    err_f_loop = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
                    cv2.putText(err_f_loop, "Frame perso", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,100,255),2)
                    if H_ref is None: cv2.putText(err_f_loop, "Vista Fissa Non Impostata", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),1)
                    _, buf_err = cv2.imencode('.jpg', err_f_loop)
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf_err.tobytes() + b'\r\n')
                    time.sleep(0.1) # Pausa per non sovraccaricare
                    continue

                undistorted_live = cv2.undistort(frame_live, cam_matrix, dist_coeffs, None, new_cam_matrix_stream)
                
                if H_ref is not None:
                    output_img = cv2.warpPerspective(undistorted_live, H_ref, (OUT_W, OUT_H))
                    # --- DISEGNA I BLOB SULLA VISTA RETTIFICATA ---
                    # 1. Trova i keypoints nel frame originale (non thresholdato)
                    gray_for_blobs = cv2.cvtColor(undistorted_live, cv2.COLOR_BGR2GRAY)
                    cfg_cam = load_config_data().get("camera", {})
                    _, thresh_for_blobs = cv2.threshold(
                        gray_for_blobs,
                        cfg_cam.get("minThreshold", 127),
                        cfg_cam.get("maxThreshold", 255),
                        cv2.THRESH_BINARY
                    )
                    keypoints_blob = detect_blobs(thresh_for_blobs)
                    
                    # 2. Trasforma i keypoints (coordinate immagine) nella vista rettificata
                    if keypoints_blob:
                        pts = np.array([kp.pt for kp in keypoints_blob], dtype=np.float32).reshape(-1,1,2)
                        # Applica undistortion
                        pts_undist = cv2.undistortPoints(pts, cam_matrix, dist_coeffs, P=new_cam_matrix_stream)
                        # Applica la stessa omografia usata per la warp
                        pts_homog = np.concatenate([pts_undist.squeeze(), np.ones((pts_undist.shape[0],1), dtype=np.float32)], axis=1)
                        pts_warped = (H_ref @ pts_homog.T).T
                        pts_warped = pts_warped[:,:2] / pts_warped[:,2,np.newaxis]

                        # --- Calcola world coordinates (in mm) ---
                        # Recupera parametri per eventuale shift/inversione asse y se vuoi
                        # (qui sono le coordinate nella vista rettificata, già in mm se l'omografia è coerente)
                        for i, (x, y) in enumerate(pts_warped):
                            cv2.circle(output_img, (int(round(x)), int(round(y))), 8, (0,0,255), 2)
                            # Disegna la coordinata numerica (arrotondata)
                            cv2.putText(
                                output_img,
                                f"{x:.1f},{y:.1f}",
                                (int(round(x))+10, int(round(y))-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0,255,0),
                                1
                            )
                else:
                    output_img = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8) # Placeholder
                    cv2.putText(output_img, "Vista Fissa Non Impostata", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),1)
                    cv2.putText(output_img, "Usa endpoint per settarla", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),1)


                _, buffer_ok = cv2.imencode('.jpg', output_img)
                frame_bytes_ok = buffer_ok.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes_ok + b'\r\n')
                time.sleep(0.03) # Circa 30fps
            except Exception as e_loop_stream:
                print(f"Errore nel loop gen_frames (fixed_perspective_stream): {e_loop_stream}")
                # Invia un frame di errore specifico per il loop
                err_f_loop = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
                cv2.putText(err_f_loop, f"Stream Loop Err: {str(e_loop_stream)[:50]}", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,100,255),2)
                if H_ref is None: cv2.putText(err_f_loop, "Vista Fissa Non Impostata", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),1)
                _, buf_err = cv2.imencode('.jpg', err_f_loop)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf_err.tobytes() + b'\r\n')
                time.sleep(1) # Pausa prima di ritentare o se l'errore è persistente
                # Potresti voler aggiungere una logica per uscire dal loop se l'errore si ripete troppo

    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')


@csrf_exempt
@require_GET
def get_world_coordinates(request): # Modificato per usare la corretta omografia e undistortion
    H_fixed_ref = load_fixed_perspective_homography()
    if H_fixed_ref is None:
        return JsonResponse({"status": "error", "message": "Omografia vista fissa non disponibile."}, status=400)
    
    cfg_snapshot_wc = load_config_data()
    cam_calib_wc = cfg_snapshot_wc.get("camera", {}).get("calibration", None)
    if not (cam_calib_wc and cam_calib_wc.get("camera_matrix") and cam_calib_wc.get("distortion_coefficients")):
        return JsonResponse({"status": "error", "message": "Dati calibrazione camera mancanti."}, status=400)

    cam_matrix_wc = np.array(cam_calib_wc["camera_matrix"], dtype=np.float32)
    dist_coeffs_wc = np.array(cam_calib_wc["distortion_coefficients"], dtype=np.float32)

    frame_for_coords, keypoints_for_coords = get_current_frame_and_keypoints()
    if frame_for_coords is None or frame_for_coords.size == 0:
         return JsonResponse({"status": "error", "message": "Impossibile ottenere frame per coordinate."}, status=500)


    if not keypoints_for_coords:
        return JsonResponse({"status": "success", "coordinates": []})

    img_pts_distorted = np.array([kp.pt for kp in keypoints_for_coords], dtype=np.float32).reshape(-1,1,2)
    
    # È cruciale usare la STESSA new_camera_matrix (o una derivata nello stesso modo)
    # che è stata usata implicitamente quando H_fixed_ref è stata calcolata
    # (cioè, quella usata in get_board_and_canonical_homography_for_django via set_fixed_perspective_view)
    h_frame_wc, w_frame_wc = frame_for_coords.shape[:2]
    new_cam_matrix_wc, _ = cv2.getOptimalNewCameraMatrix(cam_matrix_wc, dist_coeffs_wc, (w_frame_wc,h_frame_wc), 1.0, (w_frame_wc,h_frame_wc))
    
    # Non distorce i punti e li proietta nel sistema di coordinate di new_cam_matrix_wc
    img_pts_undistorted = cv2.undistortPoints(img_pts_distorted, cam_matrix_wc, dist_coeffs_wc, P=new_cam_matrix_wc)
    
    world_coords_list = []
    for pt_undist in img_pts_undistorted:
        pt_homog = np.array([pt_undist[0][0], pt_undist[0][1], 1.0], dtype=np.float32)
        world_pt_homog = H_fixed_ref @ pt_homog # Equivalente a np.dot(H_fixed_ref, pt_homog)
        
        if world_pt_homog[2] == 0: # Evita divisione per zero
            world_coords_list.append([float('inf'), float('inf')]) 
            continue
        world_pt_cartesian = world_pt_homog[:2] / world_pt_homog[2]
        world_coords_list.append(world_pt_cartesian.tolist())
    
    return JsonResponse({"status": "success", "coordinates": world_coords_list})
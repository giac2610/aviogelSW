import glob
import sys
import json
import os
import time
import numpy as np
import cv2
from django.http import StreamingHttpResponse, JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET
import threading #
from django.conf import settings # Per MEDIA_ROOT e MEDIA_URL se usati direttamente qui

# --- Configurazione file ---
CONFIG_DIR = os.path.join(os.path.dirname(__file__), '../config') #
SETUP_JSON_PATH = os.path.join(CONFIG_DIR, 'setup.json') #
EXAMPLE_JSON_PATH = os.path.join(CONFIG_DIR, 'setup.example.json') #

MEDIA_DIR = os.path.join(os.path.dirname(__file__), '..', 'media') # Opzione 2: percorso relativo
MEDIA_URL = '/media/' #


if not os.path.exists(SETUP_JSON_PATH): #
    if not os.path.exists(EXAMPLE_JSON_PATH): #
        raise FileNotFoundError(f"File di esempio mancante: {EXAMPLE_JSON_PATH}") #
    from shutil import copyfile #
    os.makedirs(CONFIG_DIR, exist_ok=True) # Crea la dir config se non esiste
    copyfile(EXAMPLE_JSON_PATH, SETUP_JSON_PATH) #
    print(f"[INFO] File di configurazione creato da setup.example.json") #

# Carica la configurazione globale all'avvio, ma le funzioni specifiche ricaricheranno se necessario
try:
    with open(SETUP_JSON_PATH, 'r') as f:
        config = json.load(f) #
    camera_settings = config.get("camera", {}) #
except Exception as e:
    print(f"Errore critico nel caricamento di setup.json all'avvio: {e}")
    config = {}
    camera_settings = {}


# --- Camera Init ---
# (Il tuo codice di inizializzazione della camera rimane invariato)
if sys.platform == "darwin": #
    mac_camera = cv2.VideoCapture(0) #
    if not mac_camera.isOpened(): #
        # Prova con un altro indice se lo 0 fallisce
        mac_camera = cv2.VideoCapture(1)
        if not mac_camera.isOpened():
            raise RuntimeError("La webcam non è disponibile o è in uso da un altro processo.") #

    def get_frame(): #
        ret, frame = mac_camera.read() #
        if not ret: #
            raise RuntimeError("Impossibile leggere il frame dalla webcam.") #
        return frame #
else: #
    try:
        from picamera2 import Picamera2 #
        picam2 = Picamera2() #
        picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)})) #
        picam2.start() #
    except RuntimeError as e: #
        print(f"Errore durante l'inizializzazione della Picamera2: {e}") #
        picam2 = None #
    except ImportError:
        print("Modulo Picamera2 non trovato. L'applicazione potrebbe non funzionare correttamente su Raspberry Pi.")
        picam2 = None


    def get_frame(): #
        if picam2:
            frame = picam2.capture_array() #
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) #
        else:
            # Restituisce un frame nero o un'immagine placeholder se la camera non è inizializzata
            print("Attenzione: Picamera2 non inizializzata, restituisco un frame vuoto.")
            return np.zeros((480, 640, 3), dtype=np.uint8)


# --- Utility ---
def load_config_data():
    """Carica i dati di configurazione completi da setup.json."""
    try:
        with open(SETUP_JSON_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Errore durante il caricamento di {SETUP_JSON_PATH}: {e}")
        return {} # Restituisce un dizionario vuoto in caso di errore

def save_config_data(new_config_data):
    """Salva i dati di configurazione completi in setup.json."""
    try:
        with open(SETUP_JSON_PATH, 'w') as f:
            json.dump(new_config_data, f, indent=4)
        # Aggiorna le variabili globali 'config' e 'camera_settings' dopo il salvataggio
        global config, camera_settings
        config = new_config_data
        camera_settings = config.get("camera", {})
        print(f"Configurazione salvata e ricaricata.")
        return True
    except Exception as e:
        print(f"Errore durante il salvataggio in {SETUP_JSON_PATH}: {e}")
        return False

def load_homography_from_config(): # Modificata da load_homography per evitare conflitti di nome
    """Carica la matrice di omografia da setup.json."""
    current_config = load_config_data()
    H_list = current_config.get("camera", {}).get("homography", None)
    if H_list is not None:
        try:
            return np.array(H_list, dtype=np.float32)
        except Exception as e:
            print(f"Errore nel convertire l'omografia caricata in array numpy: {e}")
            return None
    return None

def save_homography_to_config(H_matrix): # Modificata da save_homography
    """Salva la matrice di omografia (come lista) in setup.json."""
    current_config = load_config_data()
    current_config.setdefault("camera", {})
    if H_matrix is not None:
        current_config["camera"]["homography"] = H_matrix.tolist()
    else:
        current_config["camera"]["homography"] = None # O una matrice identità di default
    return save_config_data(current_config)


def detect_blobs(binary_image): #
    cfg = load_config_data().get("camera", {}) # Ricarica per avere i parametri più aggiornati
    
    params = cv2.SimpleBlobDetector_Params() #
    params.filterByArea = cfg.get("areaFilter", True) #
    params.minArea = cfg.get("minArea", 150) #
    params.maxArea = cfg.get("maxArea", 5000) #
    params.filterByCircularity = cfg.get("circularityFilter", True) #
    params.minCircularity = cfg.get("minCircularity", 0.1) #
    # filterByConvexity e minConvexity sono usati nella view originale, aggiungiamoli
    params.filterByConvexity = cfg.get("filterByConvexity", True) #
    params.minConvexity = cfg.get("minConvexity", 0.87) #
    params.filterByInertia = cfg.get("inertiaFilter", True) #
    params.minInertiaRatio = cfg.get("minInertia", 0.01) #
    
    detector = cv2.SimpleBlobDetector_create(params) #
    return detector.detect(binary_image) #

def get_current_frame_and_keypoints(): # ma aggiorniamo il caricamento config
    cfg = load_config_data().get("camera", {}) # Ricarica config
    frame = get_frame() #
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #
    _, thresh = cv2.threshold( #
        gray, #
        cfg.get("minThreshold", 127), #
        cfg.get("maxThreshold", 255), #
        cv2.THRESH_BINARY #
    )
    keypoints = detect_blobs(thresh) #
    return frame, keypoints # Restituisce anche il frame per usi successivi

@csrf_exempt
@require_POST
def update_camera_settings(request):
    try:
        data = json.loads(request.body)
        current_config = load_config_data()
        current_config.setdefault("camera", {})
        current_config["camera"].update(data) # Aggiorna solo i campi forniti
        
        if save_config_data(current_config):
            # Le variabili globali 'config' e 'camera_settings' sono aggiornate da save_config_data
            print(f"Impostazioni camera aggiornate: {config.get('camera', {})}")
            return JsonResponse({"status": "success", "updated_settings": config.get("camera", {})})
        else:
            return JsonResponse({"status": "error", "message": "Failed to save updated settings."}, status=500)
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=400)

@csrf_exempt
def camera_feed(request): # assicuriamoci che usi config aggiornata
    mode = request.GET.get("mode", "normal") #
    def gen_frames():
        while True:
            try:
                cfg_cam = load_config_data().get("camera", {}) # Carica config ad ogni frame per dinamicità
                frame = get_frame() #
                if frame is None or not hasattr(frame, "shape"):
                    # Frame non valido, restituisci un'immagine nera
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                # processed_frame_for_blobs = frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #
                _, processed_frame_for_blobs = cv2.threshold( #
                    gray, #
                    cfg_cam.get("minThreshold", 127), #
                    cfg_cam.get("maxThreshold", 255), #
                    cv2.THRESH_BINARY #
                )
                if mode == "threshold": #
                    # Per la visualizzazione, converti l'immagine threshold in BGR
                    display_frame = cv2.cvtColor(processed_frame_for_blobs, cv2.COLOR_GRAY2BGR)
                else: # mode == "normal"
                    # processed_frame_for_blobs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    display_frame = frame.copy()

                keypoints = detect_blobs(processed_frame_for_blobs) #
                frame_with_keypoints = cv2.drawKeypoints( #
                    display_frame, #
                    keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) #
                

                # ----------------------------------------------------------

                # --- Parallelepipedo: verifica che tutti i keypoints siano dentro ---
                parallelepiped_ok = False
                if keypoints is not None and len(keypoints) >= 4:
                    pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
                    s = pts.sum(axis=1)
                    diff = np.diff(pts, axis=1)
                    corners = np.zeros((4,2), dtype=np.float32)
                    corners[0] = pts[np.argmin(s)]      # top-left
                    corners[2] = pts[np.argmax(s)]      # bottom-right
                    corners[1] = pts[np.argmin(diff)]   # top-right
                    corners[3] = pts[np.argmax(diff)]   # bottom-left
                    corners_int = corners.astype(np.intp)
                    # Verifica che tutti i keypoints siano dentro il parallelepipedo
                    inside = True
                    for pt in pts:
                        # pointPolygonTest restituisce >0 se dentro, 0 se sul bordo, <0 se fuori
                        if cv2.pointPolygonTest(corners, (pt[0], pt[1]), False) < 0:
                            inside = False
                            break
                    if inside:
                        parallelepiped_ok = True
                        cv2.polylines(frame_with_keypoints, [corners_int], isClosed=True, color=(0,255,0), thickness=2)
                # ----------------------------------------------------------

                _, buffer = cv2.imencode('.jpg', frame_with_keypoints) #
                frame_bytes = buffer.tobytes() #
                yield (b'--frame\r\n' #
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n') #
            except Exception as e:
                # In caso di errore, restituisci un frame nero con errore stampato
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, f"Error: {str(e)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                _, buffer = cv2.imencode('.jpg', error_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

@csrf_exempt
@require_GET
def get_keypoints(request): #
    try:
        _, keypoints = get_current_frame_and_keypoints() # (modificato per restituire anche frame)
        keypoints_list = [[kp.pt[0], kp.pt[1]] for kp in keypoints] #
        # Calcola i vertici del rettangolo che racchiude i keypoints
        rect_vertices = []
        if len(keypoints) >= 2:
            pts = np.array(keypoints_list, dtype=np.float32)
            rect = cv2.minAreaRect(pts)
            box = cv2.boxPoints(rect)
            rect_vertices = box.astype(float).tolist()  # <-- usa float per serializzare bene

        parallelepiped_vertices = []
        parallelepiped_ok = False
        if len(keypoints) >= 4:
            pts = np.array(keypoints_list, dtype=np.float32)
            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)
            corners = np.zeros((4,2), dtype=np.float32)
            corners[0] = pts[np.argmin(s)]      # top-left
            corners[2] = pts[np.argmax(s)]      # bottom-right
            corners[1] = pts[np.argmin(diff)]   # top-right
            corners[3] = pts[np.argmax(diff)]   # bottom-left
            # Verifica che tutti i keypoints siano dentro il parallelepipedo
            inside = True
            for pt in pts:
                if cv2.pointPolygonTest(corners, (pt[0], pt[1]), False) < 0:
                    inside = False
                    break
            if inside:
                parallelepiped_vertices = corners.tolist()
                parallelepiped_ok = True

        return JsonResponse({
            "status": "success",
            "keypoints": keypoints_list,
            "bounding_box_vertices": rect_vertices,
            "parallelepiped_vertices": parallelepiped_vertices,
            "parallelepiped_ok": parallelepiped_ok
        })
    except Exception as e: #
        return JsonResponse({"status": "error", "message": str(e)}, status=500) 


@csrf_exempt
@require_POST
def set_camera_origin(request): # ma aggiornata per i nuovi campi
    try:
        data = json.loads(request.body)
        # Ora ci aspettiamo 'origin_x' e 'origin_y' per coerenza
        # Mantengo anche 'x' e 'y' per retrocompatibilità se altre parti del codice li usano
        # ma la logica della griglia userà 'origin_x' e 'origin_y'.
        x_val = float(data.get("origin_x", 0.0))
        y_val = float(data.get("origin_y", 0.0))
        
        current_config = load_config_data()
        current_config.setdefault("camera", {})
        current_config["camera"]["origin_x"] = x_val # Per la nuova logica griglia
        current_config["camera"]["origin_y"] = y_val # Per la nuova logica griglia
        
        if save_config_data(current_config):
            return JsonResponse({"status": "success", "origin": {"origin_x": x_val, "origin_y": y_val}})
        else:
            return JsonResponse({"status": "error", "message": "Failed to save origin."}, status=500)

    except Exception as e: #
        return JsonResponse({"status": "error", "message": str(e)}, status=400) #


@csrf_exempt
@require_GET
def dynamic_warped_stream(request): # ma ora dovrebbe usare l'omografia salvata
    def gen_frames():
        H_matrix = load_homography_from_config() # Carica l'omografia una volta all'inizio dello stream
        if H_matrix is None:
            print("dynamic_warped_stream: Omografia non disponibile. Verrà mostrato il frame originale.")
        
        while True:
            frame = get_frame() #
            output_frame = frame.copy()

            if H_matrix is not None:
                try:
                    # L'omografia salvata (H_matrix) mappa mondo -> immagine.
                    # Per raddrizzare l'immagine (da immagine a "vista dall'alto del mondo"),
                    # abbiamo bisogno dell'inversa, e dobbiamo definire le dimensioni del frame raddrizzato.
                    # Qui l'intenzione originale di dynamic_warped_stream era forse di *calcolare* H dinamicamente.
                    # Se invece vogliamo applicare una H *fissa* per raddrizzare:
                    h_img, w_img = frame.shape[:2]
                    # Per fare il warp corretto, dovremmo definire le coordinate mondo degli angoli
                    # della vista raddrizzata desiderata, e poi trasformarle in coordinate immagine
                    # usando H_matrix per ottenere i punti sorgente, oppure usare l'inversa.
                    # Per semplicità, se H_matrix è vista come img -> mondo_raddrizzato, allora:
                    # warped = cv2.warpPerspective(frame, H_matrix, (w_img, h_img))
                    # Ma se H_matrix è mondo -> immagine (come calcolata da noi):
                    H_inv = np.linalg.inv(H_matrix)
                    output_frame = cv2.warpPerspective(frame, H_inv, (w_img, h_img)) #
                except np.linalg.LinAlgError:
                    print("dynamic_warped_stream: Errore nell'invertire l'omografia. Mostro frame originale.")
                    # output_frame rimane il frame originale
                except Exception as e_warp:
                    print(f"dynamic_warped_stream: Errore durante il warping: {e_warp}. Mostro frame originale.")
                    # output_frame rimane il frame originale
            
            _, buffer = cv2.imencode('.jpg', output_frame) #
            frame_bytes = buffer.tobytes() #
            yield (b'--frame\r\n' #
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n') #
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame') #

@csrf_exempt
@require_POST
def calibrate_camera(request):
    try:
        # Parametri della scacchiera: modifica se necessario
        chessboard_size = (7, 9)  # 8 caselle interne per riga, 10 per colonna
        frameSize = (1920, 1080)  # dimensioni del frame (larghezza, altezza)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        square_size = 15  # dimensione reale della casella (in mm, cm, o unità arbitrarie)
        max_frames = 20  # massimo numero di frame da acquisire

        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        # objp *= square_size

        objpoints = []  # 3D points nel mondo reale
        imgpoints = []  # 2D points nell'immagine

        calib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../calibrationMedia'))
        path = os.path.join(calib_dir, '*.jpg')
        images = glob.glob(path)  # Cambia il percorso se necessario
        print(f"Trovate {len(images)} immagini per la calibrazione.")

        print("Avvio calibrazione...")

        for image in images:
            print(f"Caricamento immagine: {image}")
            img = cv2.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            if ret:
                # print(f"Trovata scacchiera nell'immagine {image}")
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    criteria=criteria
                )
                imgpoints.append(corners2)
                cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
                # cv2.imshow('Chessboard', img)
                # cv2.waitKey(500)

        if ret:
            print("Calcolo parametri di calibrazione...")

            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None
            )

            calibration_data = {
                "camera_matrix": camera_matrix.tolist(),
                "distortion_coefficients": dist_coeffs.tolist()
            }
            current_config = load_config_data()
            current_config["camera"]["calibration"] = calibration_data
            save_config_data(current_config)
            
            
            return JsonResponse({
                "status": "success",
                "message": "Calibrazione completata.",
                "calibration": calibration_data
            })
                
        if not ret:
            return JsonResponse({
                "status": "error",
                "message": "Calibrazione non riuscita."
            }, status=500)

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

@csrf_exempt
@require_POST
def save_frame_calibration(request):
    try:
        frame = get_frame()
        if frame is None or not hasattr(frame, "shape"):
            return JsonResponse({"status": "error", "message": "Frame non valido"}, status=500)
        # Crea la cartella se non esiste
        calib_dir = os.path.join(os.path.dirname(__file__), '..', 'calibrationMedia')
        os.makedirs(calib_dir, exist_ok=True)
        # Salva il frame con timestamp
        filename = f"calib_{int(time.time())}.jpg"
        filepath = os.path.join(calib_dir, filename)
        cv2.imwrite(filepath, frame)
        return JsonResponse({"status": "success", "filename": filename})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)
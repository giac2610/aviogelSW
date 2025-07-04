# -*- coding: utf-8 -*-

# ==============================================================================
# IMPORTS
# ==============================================================================
from contextlib import contextmanager
from io import BytesIO
import cv2
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import glob
import sys
import json
import os
import time
import threading
import traceback

from django.http import StreamingHttpResponse, JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET
import requests #type: ignore
from sklearn.cluster import DBSCAN

# ==============================================================================
# CONFIGURAZIONE E VARIABILI GLOBALI
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, 'config')
SETUP_JSON_PATH = os.path.join(CONFIG_DIR, 'setup.json')
CALIBRATION_MEDIA_DIR = os.path.join(BASE_DIR, 'calibrationMedia')
os.makedirs(CALIBRATION_MEDIA_DIR, exist_ok=True)

# --- Caricamento Configurazione Globale ---
config = {}
camera_settings = {}

def save_config_data_to_file(data):
    """Salva il dizionario di configurazione e ricarica le variabili globali."""
    try:
        with open(SETUP_JSON_PATH, 'w') as f:
            json.dump(data, f, indent=4)
        _load_global_config_from_file() # Ricarica subito le modifiche
        return True
    except Exception as e:
        print(f"Errore salvataggio {SETUP_JSON_PATH}: {e}")
        return False

def _load_global_config_from_file():
    """Carica la configurazione da file o crea un default se non esiste."""
    global config, camera_settings
    try:
        with open(SETUP_JSON_PATH, 'r') as f:
            config = json.load(f)
        camera_settings = config.get("camera", {})
        camera_settings.setdefault("picamera_config", {}).setdefault("main", {}).setdefault("size", [640, 480])
        print("[INFO] Configurazione globale caricata.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[WARN] setup.json non trovato o malformato ({e}). Creo configurazione di default.")
        config = {
            "camera": {
                "picamera_config": {"main": {"size": [640, 480]}},
                "calibration_settings": {"point_spacing_mm": 50.0}
            }
        }
        camera_settings = config["camera"]
        save_config_data_to_file(config)

_load_global_config_from_file()

# ==============================================================================
# FUNZIONI INTERNE E LOGICA DI VISIONE
# ==============================================================================

# --- Gestione Camera ---
camera_instance = None
camera_lock = threading.Lock()
active_streams = 0

def _initialize_camera_internally():
    global camera_instance
    if camera_instance is not None:
        try:
            if hasattr(camera_instance, 'release'): camera_instance.release()
            elif hasattr(camera_instance, 'stop'): camera_instance.stop(); camera_instance.close()
        except Exception: pass
    
    cfg = camera_settings
    size = cfg.get("picamera_config", {}).get("main", {}).get("size", [640, 480])
    width, height = size

    try:
        if sys.platform == "darwin":
            mac_cam = cv2.VideoCapture(cfg.get("mac_camera_index", 0))
            if not mac_cam.isOpened(): raise IOError("Webcam non trovata.")
            mac_cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            mac_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            camera_instance = mac_cam
        else:
            from picamera2 import Picamera2
            picam2 = Picamera2()
            config = picam2.create_video_configuration(main={"size": (width, height), "format": "RGB888"})
            picam2.configure(config)
            picam2.start()
            camera_instance = picam2
        print("[INFO] Camera inizializzata.")
    except Exception as e:
        print(f"[ERROR] Inizializzazione camera fallita: {e}")
        camera_instance = None
    return camera_instance

def get_frame():
    global camera_instance
    with camera_lock:
        if camera_instance is None and _initialize_camera_internally() is None:
            size = camera_settings.get("picamera_config", {}).get("main", {}).get("size", [480, 640])
            return np.zeros((size[1], size[0], 3), dtype=np.uint8)
        try:
            if sys.platform == "darwin":
                ret, frame = camera_instance.read()
                if not ret: raise IOError("Lettura frame fallita.")
            else:
                frame = camera_instance.capture_array()
            return frame
        except Exception as e:
            print(f"Errore cattura frame: {e}.")
            if camera_instance:
                try:
                    if sys.platform == "darwin": camera_instance.release()
                    else: camera_instance.stop(); camera_instance.close()
                except: pass
            camera_instance = None
            size = camera_settings.get("picamera_config", {}).get("main", {}).get("size", [480, 640])
            return np.zeros((size[1], size[0], 3), dtype=np.uint8)

# --- Funzioni Helper di Configurazione e Utilità ---

def load_config_data_from_file():
    """Carica l'intero dizionario di configurazione dal file JSON."""
    try:
        with open(SETUP_JSON_PATH, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def get_fixed_perspective_homography_from_config():
    H_list = camera_settings.get("fixed_perspective", {}).get("homography_matrix")
    return np.array(H_list, dtype=np.float32) if H_list else None

def save_fixed_perspective_homography_to_config(H_matrix):
    config_data = load_config_data_from_file()
    config_data.setdefault("camera", {}).setdefault("fixed_perspective", {})
    if H_matrix is not None:
        config_data["camera"]["fixed_perspective"]["homography_matrix"] = H_matrix.tolist()
    else:
        config_data["camera"]["fixed_perspective"]["homography_matrix"] = None
    return save_config_data_to_file(config_data)

def get_current_motor_speeds():
    try:
        resp = requests.get("http://localhost:8000/motors/maxSpeeds/")
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "success":
                return data["speeds"]
    except Exception as e:
        print(f"Errore richiesta velocità motori: {e}")
    return {"extruder": 4.0, "conveyor": 1.0}

# --- Funzioni di Visione e Calcolo ---
def detect_blobs_from_params(binary_image, params, scale_x=1.0, scale_y=1.0):
    cv2_params = cv2.SimpleBlobDetector_Params()
    area_scale = scale_x * scale_y
    cv2_params.filterByArea = params.get("areaFilter", True)
    cv2_params.minArea = max(1, params.get("minArea", 150) * area_scale)
    cv2_params.maxArea = max(1, params.get("maxArea", 5000) * area_scale)
    cv2_params.filterByCircularity = params.get("circularityFilter", True)
    cv2_params.minCircularity = params.get("minCircularity", 0.1)
    cv2_params.filterByConvexity = params.get("filterByConvexity", True)
    cv2_params.minConvexity = params.get("minConvexity", 0.87)
    cv2_params.filterByInertia = params.get("inertiaFilter", True)
    cv2_params.minInertiaRatio = params.get("minInertia", 0.01)
    return cv2.SimpleBlobDetector_create(cv2_params).detect(binary_image)

def get_current_frame_and_keypoints_from_config():
    frame = get_frame()
    if frame.size == 0:
        size = camera_settings.get("picamera_config", {}).get("main", {}).get("size", [480, 640])
        return np.zeros((size[1], size[0], 3), dtype=np.uint8), []
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    proc_w = 640
    orig_h, orig_w = frame.shape[:2]
    proc_h = int(orig_h * (proc_w / orig_w))
    scale_x, scale_y = orig_w / proc_w, orig_h / proc_h
    resized_gray = cv2.resize(gray, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

    _, thresh = cv2.threshold(resized_gray, camera_settings.get("minThreshold", 127), camera_settings.get("maxThreshold", 255), cv2.THRESH_BINARY)
    kps_resized = detect_blobs_from_params(thresh, camera_settings, scale_x, scale_y)
    
    kps_orig = [cv2.KeyPoint(kp.pt[0] * scale_x, kp.pt[1] * scale_y, kp.size * ((scale_x + scale_y) / 2)) for kp in kps_resized]
    return frame, kps_orig

def get_world_coordinates_data():
    H_fixed = get_fixed_perspective_homography_from_config()
    if H_fixed is None: return {"status": "error", "message": "Omografia non disponibile."}
    
    cam_calib = camera_settings.get("calibration")
    if not (cam_calib and "camera_matrix" in cam_calib and "distortion_coefficients" in cam_calib):
        return {"status": "error", "message": "Dati di calibrazione mancanti."}
    
    cam_matrix = np.array(cam_calib["camera_matrix"], dtype=np.float32)
    dist_coeffs = np.array(cam_calib["distortion_coefficients"], dtype=np.float32)
    
    frame, keypoints = get_current_frame_and_keypoints_from_config()
    if not keypoints: return {"status": "success", "coordinates": []}

    img_pts = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
    h, w = frame.shape[:2]
    new_cam_matrix, _ = cv2.getOptimalNewCameraMatrix(cam_matrix, dist_coeffs, (w, h), 1.0, (w, h))
    img_pts_undistorted = cv2.undistortPoints(img_pts, cam_matrix, dist_coeffs, P=new_cam_matrix)
    
    if img_pts_undistorted is None: return {"status": "error", "message": "Undistortion fallita."}
    
    world_coords = cv2.perspectiveTransform(img_pts_undistorted, H_fixed).reshape(-1, 2).tolist() if img_pts_undistorted.size > 0 else []
    return {"status": "success", "coordinates": world_coords}

# SOSTITUISCI LA VECCHIA FUNZIONE CON QUESTA NUOVA VERSIONE
def generate_adaptive_grid_from_cluster(points, config=None):
    """
    Versione Riscostruita e Corretta:
    Costruisce una griglia geometricamente perfetta usando la spaziatura e l'angolo
    rilevati, garantendo che la distanza tra i punti sia sempre quella specificata.
    """
    if config is None: config = camera_settings
    spacing = config.get("calibration_settings", {}).get("point_spacing_mm", 50.0)
    
    if len(points) < 3: return None, None
    
    points_np = np.array(points, dtype=np.float32)
    db = DBSCAN(eps=spacing * 1.5, min_samples=2).fit(points_np)
    labels = db.labels_
    if not np.any(labels != -1): return None, None
    
    unique, counts = np.unique(labels[labels != -1], return_counts=True)
    main_cluster_points = points_np[labels == unique[np.argmax(counts)]]
    if len(main_cluster_points) < 3: return None, None

    # 1. Ottieni il rettangolo di contorno per avere dimensioni e angolo
    rect = cv2.minAreaRect(main_cluster_points)
    (center, (width, height), angle_deg) = rect
    
    # Assicura che la larghezza sia la dimensione maggiore per coerenza
    if width < height:
        width, height = height, width
        angle_deg += 90

    # 2. Calcola le dimensioni della griglia (questa logica è corretta)
    num_cols = int(round(width / spacing)) + 1
    num_rows = int(round(height / spacing)) + 1
    grid_dims = (num_cols, num_rows)
    print(f"[INFO] Griglia adattiva calcolata: {grid_dims[0]}x{grid_dims[1]}")

    # 3. Trova l'origine della griglia in modo robusto
    box = cv2.boxPoints(rect)
    # Ruota virtualmente i 4 angoli del box per trovare quello in alto a sinistra
    angle_rad_rot = np.deg2rad(angle_deg)
    c, s = np.cos(angle_rad_rot), np.sin(angle_rad_rot)
    rotation_matrix = np.array([[c, s], [-s, c]]) # Matrice per ruotare "indietro"
    rotated_box_points = np.dot(box - center, rotation_matrix) + center
    # L'origine è l'angolo del box con le coordinate x e y minimi nel sistema ruotato
    origin_point = box[np.argmin(rotated_box_points[:, 0] + rotated_box_points[:, 1])]

    # 4. Calcola i vettori di spostamento CORRETTI usando la trigonometria
    # La distanza è sempre `spacing`, l'orientamento è dato dall'angolo del rettangolo.
    angle_rad = np.deg2rad(angle_deg)
    col_vector = np.array([np.cos(angle_rad), np.sin(angle_rad)]) * spacing
    row_vector = np.array([-np.sin(angle_rad), np.cos(angle_rad)]) * spacing # Vettore perpendicolare

    # 5. Genera la griglia finale partendo dall'origine e usando i vettori corretti
    final_grid = []
    for i in range(num_cols):      # <-- Ciclo delle colonne all'esterno
        for j in range(num_rows):  # <-- Ciclo delle righe all'interno
            point = origin_point + i * col_vector + j * row_vector
            final_grid.append(tuple(point))
            
    return final_grid, grid_dims

def generate_serpentine_path(nodes, grid_dims):
    if not nodes or not all(grid_dims): return []
    cols, rows = grid_dims
    indexed_nodes = list(enumerate(nodes))
    path_indices = []
    for c in range(cols):
        column_nodes = indexed_nodes[c * rows : (c + 1) * rows]
        if not column_nodes: continue
        column_nodes.sort(key=lambda item: item[1][1], reverse=(c % 2 != 0))
        path_indices.extend([item[0] for item in column_nodes])
    return path_indices

def get_board_and_canonical_homography_for_django(undistorted_frame, new_camera_matrix, calib_cfg):
    cs_cols = calib_cfg.get("chessboard_cols", 9)
    cs_rows = calib_cfg.get("chessboard_rows", 7)
    sq_size = calib_cfg.get("square_size_mm", 15.0)
    chessboard_dim = (cs_cols, cs_rows)
    objp = np.zeros((cs_cols * cs_rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cs_cols, 0:cs_rows].T.reshape(-1, 2) * sq_size
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_dim, None)
    if not ret: return None, None

    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    _, rvec, tvec = cv2.solvePnP(objp, corners2, new_camera_matrix, None)
    
    obj_board_pts = np.float32([[0,0,0], [ (cs_cols-1)*sq_size, 0, 0], [(cs_cols-1)*sq_size, (cs_rows-1)*sq_size, 0], [0, (cs_rows-1)*sq_size, 0]])
    img_board_pts, _ = cv2.projectPoints(obj_board_pts, rvec, tvec, new_camera_matrix, None)
    
    w, h = int(round((cs_cols-1) * sq_size)), int(round((cs_rows-1) * sq_size))
    canonical_pts = np.float32([[0,0], [w-1,0], [w-1,h-1], [0,h-1]])
    
    H = cv2.getPerspectiveTransform(img_board_pts, canonical_pts)
    return H, (w, h)


def _calculate_common_route_data():
    """
    Funzione interna che esegue tutti i calcoli pesanti comuni
    a 'compute_route' e 'plot_graph'.
    """
    response = get_world_coordinates_data()
    if response.get("status") != "success":
        return None, None, None, response

    coordinates = response.get("coordinates", [])
    if not coordinates:
        return None, None, None, {"status": "error", "message": "Nessun punto rilevato per il calcolo."}
        
    nodi, grid_dims = generate_adaptive_grid_from_cluster(coordinates, config=config)
    if not nodi:
        return None, None, None, {"status": "error", "message": "Impossibile calcolare griglia adattiva."}

    path_indices = generate_serpentine_path(nodi, grid_dims)
    
    return nodi, grid_dims, path_indices, {"status": "success"}


# ==============================================================================
# ENDPOINT API DJANGO
# ==============================================================================

@contextmanager
def stream_context():
    global active_streams
    active_streams += 1
    print(f"[STREAM] Stream avviato. Attivi: {active_streams}")
    try: yield
    finally:
        active_streams -= 1
        print(f"[STREAM] Stream terminato. Attivi: {active_streams}")

# --- Endpoint di Controllo e Streaming ---
def fixed_perspective_stream(request):
    """Funzione wrapper per mantenere la vecchia API, chiama camera_feed in modalità 'fixed'."""
    request.mode_override = 'fixed'
    return camera_feed(request)

@csrf_exempt
@require_GET
def camera_feed(request):
    if hasattr(request, 'mode_override'):
        mode = request.mode_override
    else:
        mode = request.GET.get("mode", "normal")
    
    def gen_frames():
        stream_cfg = camera_settings
        H_ref, cam_matrix, dist_coeffs, new_cam_matrix_stream = None, None, None, None
        OUT_W, OUT_H = 0, 0
        
        blob_detection_interval = stream_cfg.get("blob_detection_interval", 5)
        frame_count = 0
        last_keypoints_for_drawing = []
        blob_processing_width = stream_cfg.get("blob_processing_width", 640)

        if mode == "fixed":
            H_ref = get_fixed_perspective_homography_from_config()
            cam_calib = stream_cfg.get("calibration")
            fixed_persp_cfg = stream_cfg.get("fixed_perspective", {})
            OUT_W = fixed_persp_cfg.get("output_width", 1000)
            OUT_H = fixed_persp_cfg.get("output_height", 800)
            if cam_calib and cam_calib.get("camera_matrix") and cam_calib.get("distortion_coefficients"):
                cam_matrix = np.array(cam_calib["camera_matrix"], dtype=np.float32)
                dist_coeffs = np.array(cam_calib["distortion_coefficients"], dtype=np.float32)
                try:
                    sample_frame = get_frame()
                    if sample_frame.size > 0:
                        h_str, w_str = sample_frame.shape[:2]
                        new_cam_matrix_stream, _ = cv2.getOptimalNewCameraMatrix(cam_matrix, dist_coeffs, (w_str,h_str), 1.0, (w_str,h_str))
                except Exception as e:
                    print(f"Errore setup fixed stream: {e}")
        
        with stream_context():
            while True:
                try:
                    frame_orig = get_frame()
                    if frame_orig is None or frame_orig.size == 0:
                        time.sleep(0.1)
                        continue
                    
                    if frame_count % blob_detection_interval == 0:
                        gray_for_blobs = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
                        orig_h, orig_w = gray_for_blobs.shape
                        proc_h = int(orig_h * (blob_processing_width / orig_w))
                        scale_x, scale_y = orig_w / blob_processing_width, orig_h / proc_h
                        resized_gray = cv2.resize(gray_for_blobs, (blob_processing_width, proc_h), interpolation=cv2.INTER_AREA)
                        
                        _, thresh = cv2.threshold(resized_gray, stream_cfg.get("minThreshold", 127), stream_cfg.get("maxThreshold", 255), cv2.THRESH_BINARY)
                        kps_resized = detect_blobs_from_params(thresh, stream_cfg, scale_x, scale_y)
                        last_keypoints_for_drawing = [cv2.KeyPoint(kp.pt[0]*scale_x, kp.pt[1]*scale_y, kp.size*((scale_x+scale_y)/2)) for kp in kps_resized]
                    
                    frame_count += 1
                    display_frame = frame_orig.copy()

                    if mode == "normal" or mode == "threshold":
                        if mode == "threshold":
                            gray_disp = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
                            _, thresh_disp = cv2.threshold(gray_disp, stream_cfg.get("minThreshold", 127), stream_cfg.get("maxThreshold", 255), cv2.THRESH_BINARY)
                            display_frame = cv2.cvtColor(thresh_disp, cv2.COLOR_GRAY2BGR)
                        
                        display_frame = cv2.drawKeypoints(display_frame, last_keypoints_for_drawing, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    
                    elif mode == "fixed":
                        if H_ref is not None and new_cam_matrix_stream is not None and cam_matrix is not None and dist_coeffs is not None:
                            undistorted = cv2.undistort(frame_orig, cam_matrix, dist_coeffs, None, new_cam_matrix_stream)
                            display_frame = cv2.warpPerspective(undistorted, H_ref, (OUT_W, OUT_H))
                            if last_keypoints_for_drawing:
                                pts = np.array([kp.pt for kp in last_keypoints_for_drawing], dtype=np.float32).reshape(-1,1,2)
                                pts_undist = cv2.undistortPoints(pts, cam_matrix, dist_coeffs, P=new_cam_matrix_stream)
                                if pts_undist is not None:
                                    pts_warped = cv2.perspectiveTransform(pts_undist, H_ref)
                                    if pts_warped is not None:
                                        for pt_w in pts_warped.reshape(-1,2):
                                            cv2.circle(display_frame, (int(round(pt_w[0])), int(round(pt_w[1]))), 8, (0,0,255), 2)
                        else:
                            display_frame = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
                            cv2.putText(display_frame, "Fixed View Not Set", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),1)

                    _, buffer = cv2.imencode('.jpg', display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

                except Exception as e:
                    print(f"Errore nel loop di streaming (mode: {mode}): {e}")
                    traceback.print_exc()
                    time.sleep(1)
    
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')


@csrf_exempt
@require_POST
def initialize_camera_endpoint(request):
    instance = _initialize_camera_internally()
    return JsonResponse({"status": "success" if instance else "error"}, status=200 if instance else 500)

@csrf_exempt
@require_POST
def deinitialize_camera_endpoint(request):
    global camera_instance
    with camera_lock:
        if camera_instance:
            try:
                if sys.platform == "darwin": camera_instance.release()
                else: camera_instance.stop(); camera_instance.close()
                camera_instance = None
            except Exception as e: return JsonResponse({"status": "error", "message": str(e)}, status=500)
    return JsonResponse({"status": "success", "message": "Camera rilasciata."})
        
@csrf_exempt
@require_GET
def get_world_coordinates(request):
    data = get_world_coordinates_data()
    return JsonResponse(data, status=200 if data.get("status") == "success" else 400)

@csrf_exempt
@require_GET
def get_keypoints(request):
    try:
        _, keypoints = get_current_frame_and_keypoints_from_config()
        return JsonResponse({"status": "success", "keypoints": [[kp.pt[0], kp.pt[1]] for kp in keypoints]})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

@csrf_exempt
@require_GET
def compute_route(request):
    try:
        nodi, grid_dims, path_indices, status = _calculate_common_route_data()
        if status.get("status") != "success":
            return JsonResponse(status, status=400)

        path_nodes = [nodi[i] for i in path_indices]
        
        origin_x = camera_settings.get("origin_x", 0.0)
        origin_y = camera_settings.get("origin_y", 0.0)
        last_pos = (origin_x, origin_y)
        motor_commands = []
        for pos in path_nodes:
            # Calcola i valori
            extruder_val = pos[0] - last_pos[0]
            conveyor_val = pos[1] - last_pos[1]

            # SOLUZIONE: Arrotonda E POI converti a float standard
            motor_commands.append({
                "extruder": float(round(extruder_val, 4)),
                "conveyor": float(round(conveyor_val, 4))
            })
            last_pos = pos


        plt.figure(figsize=(8, 6))
        graph, pos = nx.Graph(), {i: node for i, node in enumerate(nodi)}
        graph.add_nodes_from(pos.keys())
        nx.draw_networkx_nodes(graph, pos, node_color='skyblue', node_size=150)
        if len(path_indices) > 1:
            edges = [(path_indices[i], path_indices[i+1]) for i in range(len(path_indices)-1)]
            nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color='red', width=2)
        plt.title(f"Percorso Adattivo su Griglia {grid_dims[0]}x{grid_dims[1]}")
        plt.axis('equal'); plt.gca().invert_yaxis()
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight'); plt.close()
        
        return JsonResponse({
            "status": "success", "message": f"Percorso per griglia {grid_dims} calcolato.",
            "grid_dimensions": grid_dims, "route_indices": path_indices,
            "motor_commands": motor_commands, "plot_graph_base64": base64.b64encode(buf.getvalue()).decode('utf-8')
        })
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"status": "error", "message": f"Errore interno: {e}"}, status=500)

@csrf_exempt
@require_GET
def plot_graph(request):
    try:
        nodi, grid_dims, path_indices, status = _calculate_common_route_data()
        if status.get("status") != "success":
            return HttpResponse(f"Errore: {status.get('message')}", status=400)

        plt.figure(figsize=(8, 6))
        graph, pos = nx.Graph(), {i: node for i, node in enumerate(nodi)}
        graph.add_nodes_from(pos.keys())
        nx.draw_networkx_nodes(graph, pos, node_color='skyblue', node_size=150)
        
        if len(path_indices) > 1:
            edges = [(path_indices[i], path_indices[i+1]) for i in range(len(path_indices)-1)]
            nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color='red', width=2)

        plt.title(f"Percorso Calcolato su Griglia {grid_dims[0]}x{grid_dims[1]}")
        plt.axis('equal'); plt.gca().invert_yaxis()
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight'); plt.close()
        buf.seek(0)
        return HttpResponse(buf.getvalue(), content_type='image/png')
    except Exception as e:
        traceback.print_exc()
        return HttpResponse(f"Errore interno: {e}", status=500)


# --- Endpoint di Configurazione e Calibrazione ---
@csrf_exempt
@require_POST
def update_camera_settings(request):
    try:
        data = json.loads(request.body)
        config_data = load_config_data_from_file()
        config_data.setdefault("camera", {}).update(data)
        if save_config_data_to_file(config_data):
            return JsonResponse({"status": "success"})
        else:
            return JsonResponse({"status": "error", "message": "Salvataggio fallito."}, status=500)
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=400)

@csrf_exempt
@require_POST
def set_camera_origin(request):
    try:
        data = json.loads(request.body)
        config_data = load_config_data_from_file()
        config_data.setdefault("camera", {})
        config_data["camera"]["origin_x"] = float(data.get("origin_x", 0.0))
        config_data["camera"]["origin_y"] = float(data.get("origin_y", 0.0))
        if save_config_data_to_file(config_data):
            return JsonResponse({"status": "success"})
        else:
            return JsonResponse({"status": "error", "message": "Salvataggio origine fallito."}, status=500)
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=400)

@csrf_exempt
@require_POST
def save_frame_calibration(request):
    try:
        frame = get_frame()
        if frame.size == 0:
            return JsonResponse({"status": "error", "message": "Frame non valido."}, status=500)
        
        filename = f"calib_{int(time.time())}.jpg"
        filepath = os.path.join(CALIBRATION_MEDIA_DIR, filename)
        cv2.imwrite(filepath, frame)
        return JsonResponse({"status": "success", "filename": filename})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

@csrf_exempt
@require_POST
def reset_camera_calibration(request):
    try:
        config_data = load_config_data_from_file()
        config_data.setdefault("camera", {})
        config_data["camera"]["calibration"] = None
        config_data["camera"]["fixed_perspective"] = None
        if save_config_data_to_file(config_data):
            return JsonResponse({"status": "success"})
        else:
            return JsonResponse({"status": "error", "message": "Salvataggio fallito."}, status=500)
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

@csrf_exempt
@require_POST
def calibrate_camera_endpoint(request):
    config_data = load_config_data_from_file()
    calib_settings = config_data.get("camera", {}).get("calibration_settings", {})
    cs_cols = calib_settings.get("chessboard_cols", 9)
    cs_rows = calib_settings.get("chessboard_rows", 7)
    chessboard_dim = (cs_cols, cs_rows)
    objp = np.zeros((cs_cols * cs_rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cs_cols, 0:cs_rows].T.reshape(-1, 2)
    
    objpoints, imgpoints = [], []
    images = glob.glob(os.path.join(CALIBRATION_MEDIA_DIR, '*.jpg'))
    if not images: return JsonResponse({"status": "error", "message": "Nessuna immagine trovata."}, status=400)
    
    gray_shape = None
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray_shape is None: gray_shape = gray.shape[::-1]
        
        ret, corners = cv2.findChessboardCorners(gray, chessboard_dim, None)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001))
            imgpoints.append(corners2)
    
    if not objpoints: return JsonResponse({"status": "error", "message": "Nessun pattern trovato."}, status=400)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)
    if not ret: return JsonResponse({"status": "error", "message": "Calibrazione fallita."}, status=500)

    config_data.setdefault("camera", {})["calibration"] = {"camera_matrix": mtx.tolist(), "distortion_coefficients": dist.tolist()}
    if save_config_data_to_file(config_data):
        return JsonResponse({"status": "success"})
    else:
        return JsonResponse({"status": "error", "message": "Salvataggio calibrazione fallito."}, status=500)

@csrf_exempt
@require_POST
def set_fixed_perspective_view(request):
    cam_calib = camera_settings.get("calibration")
    if not (cam_calib and "camera_matrix" in cam_calib and "distortion_coefficients" in cam_calib):
        return JsonResponse({"status": "error", "message": "Calibrazione non trovata."}, status=400)
    
    mtx, dist = np.array(cam_calib["camera_matrix"]), np.array(cam_calib["distortion_coefficients"])
    frame = get_frame()
    if frame.size == 0: return JsonResponse({"status": "error", "message": "Frame non valido."}, status=500)
    
    h, w = frame.shape[:2]
    new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    undistorted = cv2.undistort(frame, mtx, dist, None, new_mtx)
    
    calib_settings = camera_settings.get("calibration_settings", {})
    H, dims = get_board_and_canonical_homography_for_django(undistorted, new_mtx, calib_settings)
    
    if H is None: return JsonResponse({"status": "error", "message": "Pattern scacchiera non rilevato."}, status=400)
    
    out_w = camera_settings.get("fixed_perspective", {}).get("output_width", 1000)
    out_h = camera_settings.get("fixed_perspective", {}).get("output_height", 800)
    offset_x = max(0, (out_w - dims[0]) / 2.0)
    offset_y = max(0, (out_h - dims[1]) / 2.0)
    H_final = np.float32([[1,0,offset_x],[0,1,offset_y],[0,0,1]]) @ H
    
    if save_fixed_perspective_homography_to_config(H_final):
        return JsonResponse({"status": "success"})
    else:
        return JsonResponse({"status": "error", "message": "Salvataggio vista fallito."}, status=500)
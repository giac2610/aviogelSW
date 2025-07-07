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

config = {}
camera_settings = {}

def save_config_data_to_file(data):
    """Salva il dizionario di configurazione e ricarica le variabili globali."""
    try:
        with open(SETUP_JSON_PATH, 'w') as f:
            json.dump(data, f, indent=4)
        _load_global_config_from_file()
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
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[WARN] setup.json non trovato o malformato ({e}). Creo configurazione di default.")
        config = {
            "camera": {
                "picamera_config": {"main": {"size": [640, 480]}},
                "calibration_settings": {"point_spacing_mm": 50.0},
                "origin_x": 0.0,
                "origin_y": 0.0
            }
        }
        camera_settings = config["camera"]
        save_config_data_to_file(config)

_load_global_config_from_file()

# ==============================================================================
# FUNZIONI INTERNE E LOGICA DI VISIONE
# ==============================================================================
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

def get_frame(release_after=False):
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

def load_config_data_from_file():
    try:
        with open(SETUP_JSON_PATH, 'r') as f:
            return json.load(f)
    except Exception:
        return {"camera": {}}

def get_fixed_perspective_homography_from_config():
    H_list = camera_settings.get("fixed_perspective", {}).get("homography_matrix", None)
    return np.array(H_list, dtype=np.float32) if H_list and isinstance(H_list, list) else None

def save_fixed_perspective_homography_to_config(H_matrix_ref):
    current_disk_config = load_config_data_from_file()
    current_disk_config.setdefault("camera", {}).setdefault("fixed_perspective", {})
    if H_matrix_ref is not None:
        current_disk_config["camera"]["fixed_perspective"]["homography_matrix"] = H_matrix_ref.tolist()
    else:
        current_disk_config["camera"]["fixed_perspective"]["homography_matrix"] = None
    return save_config_data_to_file(current_disk_config)

def detect_blobs_from_params(binary_image, blob_params, scale_x=1.0, scale_y=1.0):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = blob_params.get("areaFilter", True)
    area_scale_factor = scale_x * scale_y
    params.minArea = max(1, blob_params.get("minArea", 150) * area_scale_factor)
    params.maxArea = max(1, blob_params.get("maxArea", 5000) * area_scale_factor)
    params.filterByCircularity = blob_params.get("circularityFilter", True)
    params.minCircularity = blob_params.get("minCircularity", 0.1)
    params.filterByConvexity = blob_params.get("filterByConvexity", True)
    params.minConvexity = blob_params.get("minConvexity", 0.87)
    params.filterByInertia = blob_params.get("inertiaFilter", True)
    params.minInertiaRatio = blob_params.get("minInertia", 0.01)
    return cv2.SimpleBlobDetector_create(params).detect(binary_image)

# ==============================================================================
# ==== SEZIONE DI FUNZIONI REFACTORIZZATE PER RISOLVERE IL DEADLOCK ====
# ==============================================================================

def get_current_frame_and_keypoints_from_config(frame):
    """Elabora un frame esistente per trovare i keypoint, non ne cattura uno nuovo."""
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

def get_world_coordinates_data(frame):
    """Calcola le coordinate mondo a partire da un frame esistente."""
    H_fixed = get_fixed_perspective_homography_from_config()
    if H_fixed is None: return {"status": "error", "message": "Omografia non disponibile."}
    cam_calib = camera_settings.get("calibration")
    if not (cam_calib and "camera_matrix" in cam_calib and "distortion_coefficients" in cam_calib):
        return {"status": "error", "message": "Dati di calibrazione mancanti."}
    
    cam_matrix = np.array(cam_calib["camera_matrix"])
    dist_coeffs = np.array(cam_calib["distortion_coefficients"])
    
    _, keypoints = get_current_frame_and_keypoints_from_config(frame) 
    
    if not keypoints: return {"status": "success", "coordinates": []}

    img_pts = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
    h, w = frame.shape[:2]
    new_cam_matrix, _ = cv2.getOptimalNewCameraMatrix(cam_matrix, dist_coeffs, (w, h), 1.0, (w, h))
    img_pts_undistorted = cv2.undistortPoints(img_pts, cam_matrix, dist_coeffs, P=new_cam_matrix)
    if img_pts_undistorted is None: return {"status": "error", "message": "Undistortion fallita."}
    
    world_coords = cv2.perspectiveTransform(img_pts_undistorted, H_fixed).reshape(-1, 2).tolist() if img_pts_undistorted.size > 0 else []
    return {"status": "success", "coordinates": world_coords}

def _calculate_serpentine_path_data(frame):
    """Funzione helper che centralizza la logica di calcolo del percorso a partire da un frame."""
    response = get_world_coordinates_data(frame) 
    
    if response.get("status") != "success":
        return None, None, response

    coordinates = response.get("coordinates", [])
    if len(coordinates) > 48:
        print(f"[INFO] Rilevati {len(coordinates)} punti, limitati a 48.")
        coordinates = sorted(coordinates, key=lambda p: (p[1], p[0]))[:48]

    if not coordinates:
        return None, None, {"status": "error", "message": "Nessun punto rilevato per il calcolo."}
        
    nodi, grid_dims = generate_adaptive_grid_from_cluster(coordinates, config_data=camera_settings)
    if not nodi:
        return None, None, {"status": "error", "message": "Impossibile calcolare griglia adattiva."}

    path_indices = generate_serpentine_path(nodi, grid_dims)
    
    graph = nx.Graph()
    for i, pos in enumerate(nodi):
        graph.add_node(i, pos=pos)

    origin_x = camera_settings.get("origin_x", 0.0)
    origin_y = camera_settings.get("origin_y", 0.0)
    origin_pos = np.array([origin_x, origin_y])
    
    if path_indices:
        path_nodes = [nodi[i] for i in path_indices]
        distances_sq = [((node[0] - origin_pos[0])**2 + (node[1] - origin_pos[1])**2) for node in path_nodes]
        start_index_in_path = np.argmin(distances_sq)
        final_path_indices = path_indices[start_index_in_path:] + path_indices[:start_index_in_path]
    else:
        final_path_indices = []
        
    return graph, final_path_indices, {"status": "success", "nodi": nodi, "grid_dims": grid_dims}

# ==============================================================================
# ==== FINE SEZIONE REFACTORIZZATA ====
# ==============================================================================

def generate_adaptive_grid_from_cluster(points, config_data=None):
    if config_data is None:
        config_data = camera_settings
    spacing = config_data.get("calibration_settings", {}).get("point_spacing_mm", 50.0)

    MAX_COLS = 6
    MAX_ROWS = 8

    if len(points) < 2:
        return None, None

    points_np = np.array(points, dtype=np.float32)

    db = DBSCAN(eps=spacing * 1.4, min_samples=3).fit(points_np)
    labels = db.labels_

    if not np.any(labels != -1):
        return None, None

    unique, counts = np.unique(labels[labels != -1], return_counts=True)
    main_label = unique[np.argmax(counts)]
    main_cluster_points = points_np[labels == main_label]
    if len(main_cluster_points) < 2:
        return None, None

    rect = cv2.minAreaRect(main_cluster_points)
    angle = rect[2]
    width, height = rect[1]

    if width < height:
        orientation_angle = 90 + angle
    else:
        orientation_angle = angle

    centroid = np.mean(main_cluster_points, axis=0)
    M = cv2.getRotationMatrix2D(tuple(centroid), orientation_angle, 1.0)
    rotated = cv2.transform(main_cluster_points[None, :, :], M)[0]

    min_x, min_y = np.min(rotated, axis=0)
    max_x, max_y = np.max(rotated, axis=0)

    cols = min(MAX_COLS, int(round((max_x - min_x) / spacing)) + 1)
    rows = min(MAX_ROWS, int(round((max_y - min_y) / spacing)) + 1)

    grid_points = []
    for c in range(cols):
        for r in range(rows):
            x = min_x + c * spacing
            y = min_y + r * spacing
            grid_points.append([x, y])
    grid_points = np.array(grid_points, dtype=np.float32)

    M_inv = cv2.getRotationMatrix2D(tuple(centroid), -orientation_angle, 1.0)
    grid_points = cv2.transform(grid_points[None, :, :], M_inv)[0]

    return grid_points.tolist(), (cols, rows)

def generate_serpentine_path(nodes, grid_dims):
    if not nodes or not all(grid_dims): return []
    cols, rows = grid_dims
    path_indices = []
    for c in range(cols):
        column_nodes_with_indices = list(enumerate(nodes[c*rows : (c+1)*rows]))

        if c % 2 == 0:
            column_nodes_with_indices.sort(key=lambda item: item[1][1])
        else:
            column_nodes_with_indices.sort(key=lambda item: item[1][1], reverse=True)

        path_indices.extend([c*rows + idx for idx, node in column_nodes_with_indices])

    return path_indices

def _generate_plot_image(graph, nodi, grid_dims, path_indices):
    """Funzione helper che genera l'immagine del grafico."""
    plt.figure(figsize=(8, 6))
    pos = nx.get_node_attributes(graph, 'pos')

    origin_x = camera_settings.get("origin_x", 0.0)
    origin_y = camera_settings.get("origin_y", 0.0)
    graph.add_node('origin'); pos['origin'] = (origin_x, origin_y)

    nx.draw_networkx_nodes(graph, pos, nodelist=list(range(len(nodi))), node_color='skyblue', node_size=150)
    nx.draw_networkx_nodes(graph, pos, nodelist=['origin'], node_color='limegreen', node_size=400, node_shape='s')

    if path_indices:
        edges = [('origin', path_indices[0])] + [(path_indices[i], path_indices[i+1]) for i in range(len(path_indices)-1)]
        nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color='red', width=2)

    plt.title(f"Percorso a Serpentina su Griglia {grid_dims[0]}x{grid_dims[1]}")
    plt.axis('equal'); plt.gca().invert_yaxis()

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight'); plt.close()
    buf.seek(0)
    return buf.getvalue()

def get_board_and_canonical_homography_for_django(undistorted_frame, new_camera_matrix_cv, calibration_cfg_dict):
    cs_cols = calibration_cfg_dict.get("chessboard_cols", 9)
    cs_rows = calibration_cfg_dict.get("chessboard_rows", 7)
    sq_size = calibration_cfg_dict.get("square_size_mm", 15.0)
    chessboard_dim_cv = (cs_cols, cs_rows)
    objp_cv = np.zeros((cs_cols * cs_rows, 3), np.float32)
    objp_cv[:,:2] = np.mgrid[0:cs_cols, 0:cs_rows].T.reshape(-1, 2) * sq_size
    criteria_cv = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_dim_cv, None)
    if not ret: return None, None
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria_cv)
    success, rvec, tvec = cv2.solvePnP(objp_cv, corners2, new_camera_matrix_cv, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success: return None, None
    obj_board_perimeter_pts = np.float32([[0,0,0], [(cs_cols-1)*sq_size, 0, 0], [(cs_cols-1)*sq_size, (cs_rows-1)*sq_size, 0], [0, (cs_rows-1)*sq_size, 0]])
    img_board_perimeter_pts, _ = cv2.projectPoints(obj_board_perimeter_pts, rvec, tvec, new_camera_matrix_cv, None)
    img_board_perimeter_pts = img_board_perimeter_pts.reshape(-1, 2)
    w, h = int(round((cs_cols-1) * sq_size)), int(round((cs_rows-1) * sq_size))
    canonical_dst_pts = np.float32([[0,0], [w-1,0], [w-1,h-1], [0,h-1]])
    H_canonical = cv2.getPerspectiveTransform(img_board_perimeter_pts, canonical_dst_pts)
    return H_canonical, (w, h)

# SOSTITUISCI QUESTA FUNZIONE CON LA VERSIONE AGGIORNATA
def get_frame_with_world_grid():
    """
    Cattura un frame, vi disegna sopra i blob rilevati, la griglia del mondo
    e il percorso a serpentina.
    """
    # 1. Cattura il frame UNA SOLA VOLTA
    frame = get_frame()

    # 2. Esegui il rilevamento dei blob grezzi per la visualizzazione
    # Questa è la nuova parte che rileva i "contorni"
    # Lo facciamo qui per poterli disegnare subito sul frame corretto.
    _, raw_keypoints = get_current_frame_and_keypoints_from_config(frame)

    # 3. Recupera i dati di calibrazione e omografia
    cam_calib = camera_settings.get("calibration")
    H_fixed = get_fixed_perspective_homography_from_config()

    if H_fixed is None or not cam_calib:
        # Se non siamo calibrati, disegna almeno i blob grezzi e restituisci
        frame_with_keypoints = cv2.drawKeypoints(frame, raw_keypoints, np.array([]), (0, 255, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return frame_with_keypoints

    # 4. Correggi la distorsione del frame catturato
    cam_matrix = np.array(cam_calib["camera_matrix"])
    dist_coeffs = np.array(cam_calib["distortion_coefficients"])
    h, w = frame.shape[:2]
    new_cam_matrix, _ = cv2.getOptimalNewCameraMatrix(cam_matrix, dist_coeffs, (w, h), 1.0, (w, h))
    undistorted_frame = cv2.undistort(frame, cam_matrix, dist_coeffs, None, new_cam_matrix)

    # === NUOVO: Disegna i keypoint grezzi sul frame non distorto ===
    # Usiamo un colore CIANO (0, 255, 255) per distinguerli
    cv2.drawKeypoints(undistorted_frame, raw_keypoints, undistorted_frame, (0, 255, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # 5. Calcola la griglia e il percorso (la logica non cambia)
    graph, path_indices, info = _calculate_serpentine_path_data(frame) 
    
    if graph is None:
        return undistorted_frame # Restituisci il frame con i soli keypoint gialli
    
    nodi_mondo = info.get("nodi", [])
    if not nodi_mondo:
        return undistorted_frame

    # 6. Proietta la griglia ideale sull'immagine (puntini rossi e linee verdi)
    try:
        H_inversa = np.linalg.inv(H_fixed)
    except np.linalg.LinAlgError:
        return undistorted_frame
        
    nodi_mondo_np = np.array(nodi_mondo, dtype=np.float32).reshape(-1, 1, 2)
    nodi_immagine = cv2.perspectiveTransform(nodi_mondo_np, H_inversa)
    
    if nodi_immagine is None:
        return undistorted_frame
        
    nodi_immagine = nodi_immagine.reshape(-1, 2).astype(int)

    # 7. Disegna la griglia finale e il percorso
    if path_indices:
        for i in range(len(path_indices) - 1):
            start_node_index = path_indices[i]
            end_node_index = path_indices[i+1]
            if start_node_index < len(nodi_immagine) and end_node_index < len(nodi_immagine):
                start_point = tuple(nodi_immagine[start_node_index])
                end_point = tuple(nodi_immagine[end_node_index])
                cv2.line(undistorted_frame, start_point, end_point, (0, 255, 0), 2) # Linee Verdi

    for point in nodi_immagine:
        cv2.circle(undistorted_frame, tuple(point), 5, (0, 0, 255), -1) # Puntini Rossi
        
    return undistorted_frame

# ==============================================================================
# ENDPOINT API DJANGO
# ==============================================================================
@contextmanager
def stream_context():
    global active_streams
    active_streams += 1; yield; active_streams -= 1

@csrf_exempt
@require_POST
def update_camera_settings(request):
    try:
        data = json.loads(request.body)
        config_data = load_config_data_from_file()
        config_data.setdefault("camera", {}).update(data)
        if save_config_data_to_file(config_data): return JsonResponse({"status": "success"})
        return JsonResponse({"status": "error", "message": "Salvataggio fallito."}, status=500)
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=400)

@csrf_exempt
@require_GET
def camera_feed(request):
    def gen_frames():
        with stream_context():
            while True:
                frame = get_frame()
                if frame.size == 0: time.sleep(0.1); continue
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

@csrf_exempt
@require_GET
def get_world_coordinates(request):
    # NOTA: Questa funzione ora richiede un frame.
    # Per una chiamata API singola, catturiamo un frame qui.
    # Questo non causa deadlock perché non è in un loop di streaming che detiene già il lock.
    frame = get_frame()
    data = get_world_coordinates_data(frame)
    return JsonResponse(data, status=200 if data.get("status") == "success" else 400)

@csrf_exempt
@require_GET
def get_keypoints(request):
    try:
        # Anche qui, catturiamo un frame per la singola elaborazione.
        frame = get_frame()
        _, keypoints = get_current_frame_and_keypoints_from_config(frame)
        return JsonResponse({"status": "success", "keypoints": [[kp.pt[0], kp.pt[1]] for kp in keypoints]})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

@csrf_exempt
@require_POST
def set_camera_origin(request):
    try:
        data = json.loads(request.body)
        config_data = load_config_data_from_file()
        config_data.setdefault("camera", {})
        config_data["camera"]["origin_x"] = float(data.get("origin_x", 0.0))
        config_data["camera"]["origin_y"] = float(data.get("origin_y", 0.0))
        if save_config_data_to_file(config_data): return JsonResponse({"status": "success"})
        return JsonResponse({"status": "error", "message": "Salvataggio fallito."}, status=500)
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=400)

@csrf_exempt
@require_GET
def compute_route(request):
    """Calcola un percorso a serpentina, partendo dal punto più vicino all'origine."""
    try:
        # Per una singola computazione, cattura un frame.
        frame = get_frame()
        graph, path_indices, info = _calculate_serpentine_path_data(frame)
        if graph is None:
            return JsonResponse(info, status=400, safe=False)

        nodi, grid_dims = info["nodi"], info["grid_dims"]
        origin_x = camera_settings.get("origin_x", 0.0)
        origin_y = camera_settings.get("origin_y", 0.0)

        path_nodes = [nodi[i] for i in path_indices]
        last_pos = (origin_x, origin_y)
        motor_commands = []
        for pos in path_nodes:
            motor_commands.append({
                "extruder": float(round(pos[0] - last_pos[0], 4)),
                "conveyor": float(round(pos[1] - last_pos[1], 4))
            })
            last_pos = pos

        plot_image_bytes = _generate_plot_image(graph, nodi, grid_dims, path_indices)

        return JsonResponse({
            "status": "success", "route": path_indices,
            "motor_commands": motor_commands, "plot_graph_base64": base64.b64encode(plot_image_bytes).decode('utf-8')
        })
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"status": "error", "message": f"Errore interno: {e}"}, status=500)

@csrf_exempt
@require_GET
def plot_graph(request):
    """Genera un'immagine del percorso a serpentina."""
    try:
        frame = get_frame()
        graph, path_indices, info = _calculate_serpentine_path_data(frame)
        if graph is None:
            return HttpResponse(f"Errore: {info.get('message')}", status=400)

        plot_image_bytes = _generate_plot_image(graph, info["nodi"], info["grid_dims"], path_indices)
        return HttpResponse(plot_image_bytes, content_type='image/png')
    except Exception as e:
        traceback.print_exc()
        return HttpResponse(f"Errore interno: {e}", status=500)

@csrf_exempt
def fixed_perspective_stream(request):
    request.mode_override = 'fixed'
    return camera_feed(request)

# --- Endpoint di Setup e Calibrazione ---
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
@require_POST
def save_frame_calibration(request):
    try:
        frame = get_frame()
        if frame.size == 0: return JsonResponse({"status": "error", "message": "Frame non valido."}, status=500)
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
        if save_config_data_to_file(config_data): return JsonResponse({"status": "success"})
        return JsonResponse({"status": "error", "message": "Salvataggio fallito."}, status=500)
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

@csrf_exempt
@require_POST
def calibrate_camera_endpoint(request):
    config_data = load_config_data_from_file()
    calib_settings = config_data.get("camera", {}).get("calibration_settings", {})
    cs_cols, cs_rows = calib_settings.get("chessboard_cols", 9), calib_settings.get("chessboard_rows", 7)
    chessboard_dim = (cs_cols, cs_rows)
    objp = np.zeros((cs_cols * cs_rows, 3), np.float32)
    objp[:,:2] = np.mgrid[0:cs_cols, 0:cs_rows].T.reshape(-1,2) * calib_settings.get("square_size_mm", 15.0)
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
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
    if not objpoints: return JsonResponse({"status": "error", "message": "Nessun pattern trovato."}, status=400)
    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)
    if not ret: return JsonResponse({"status": "error", "message": "Calibrazione fallita."}, status=500)
    config_data.setdefault("camera", {})["calibration"] = {"camera_matrix": mtx.tolist(), "distortion_coefficients": dist.tolist()}
    if save_config_data_to_file(config_data):
        return JsonResponse({"status": "success"})
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
    return JsonResponse({"status": "error", "message": "Salvataggio vista fallito."}, status=500)

@csrf_exempt
@require_GET
def debug_camera_feed(request):
    """
    Fornisce uno streaming video di debug con la griglia del mondo disegnata sopra.
    """
    def gen_debug_frames():
        with stream_context():
            while True:
                frame = get_frame_with_world_grid()
                if frame.size == 0:
                    time.sleep(0.1)
                    continue

                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return StreamingHttpResponse(gen_debug_frames(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')
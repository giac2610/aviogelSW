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
# Rimuoviamo DBSCAN perché non è usato in questa versione del codice
# from sklearn.cluster import DBSCAN 

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
    if H_list and isinstance(H_list, list):
        return np.array(H_list, dtype=np.float32)
    return None

def save_fixed_perspective_homography_to_config(H_matrix_ref):
    current_disk_config = load_config_data_from_file()
    current_disk_config.setdefault("camera", {}).setdefault("fixed_perspective", {})
    if H_matrix_ref is not None:
        current_disk_config["camera"]["fixed_perspective"]["homography_matrix"] = H_matrix_ref.tolist()
    else:
        current_disk_config["camera"]["fixed_perspective"]["homography_matrix"] = None
    return save_config_data_to_file(current_disk_config)

def detect_blobs_from_params(binary_image, blob_detection_params, scale_x=1.0, scale_y=1.0):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = blob_detection_params.get("areaFilter", True)
    area_scale_factor = scale_x * scale_y
    params.minArea = max(1, blob_detection_params.get("minArea", 150) * area_scale_factor)
    params.maxArea = max(1, blob_detection_params.get("maxArea", 5000) * area_scale_factor)
    params.filterByCircularity = blob_detection_params.get("circularityFilter", True)
    params.minCircularity = blob_detection_params.get("minCircularity", 0.1)
    params.filterByConvexity = blob_detection_params.get("filterByConvexity", True)
    params.minConvexity = blob_detection_params.get("minConvexity", 0.87)
    params.filterByInertia = blob_detection_params.get("inertiaFilter", True)
    params.minInertiaRatio = blob_detection_params.get("minInertia", 0.01)
    detector = cv2.SimpleBlobDetector_create(params)
    return detector.detect(binary_image)

def get_current_frame_and_keypoints_from_config():
    frame = get_frame(release_after=False)
    if frame is None or frame.size == 0:
        configured_height, configured_width = camera_settings.get("picamera_config", {}).get("main", {}).get("size", [480, 640])
        return np.zeros((configured_height, configured_width, 3), dtype=np.uint8), []
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processing_width_for_single_shot = 640
    original_height, original_width = frame.shape[:2]
    processing_height_for_single_shot = int(original_height * (processing_width_for_single_shot / original_width))
    scale_x = original_width / processing_width_for_single_shot
    scale_y = original_height / processing_height_for_single_shot
    resized_gray = cv2.resize(gray, (processing_width_for_single_shot, processing_height_for_single_shot), interpolation=cv2.INTER_AREA)
    _, thresh = cv2.threshold(resized_gray, camera_settings.get("minThreshold", 127), camera_settings.get("maxThreshold", 255), cv2.THRESH_BINARY)
    keypoints_resized = detect_blobs_from_params(thresh, camera_settings, scale_x, scale_y)
    keypoints_original_coords = [cv2.KeyPoint(kp.pt[0] * scale_x, kp.pt[1] * scale_y, kp.size * ((scale_x + scale_y) / 2)) for kp in keypoints_resized]
    return frame, keypoints_original_coords

# === FUNZIONE CHE MANCAVA, ORA REINSERITA ===
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

def get_current_motor_speeds():
    try:
        resp = requests.get("http://localhost:8000/motors/maxSpeeds/")
        data = resp.json()
        if data.get("status") == "success": return data["speeds"]
    except Exception as e:
        print(f"Errore richiesta velocità motori: {e}")
    return {"extruder": 4.0, "conveyor": 1.0}

def get_world_coordinates_data():
    H_fixed_ref = get_fixed_perspective_homography_from_config()
    if H_fixed_ref is None: return {"status": "error", "message": "Omografia non disponibile."}
    cam_calib_wc = camera_settings.get("calibration", None)
    if not (cam_calib_wc and cam_calib_wc.get("camera_matrix") and cam_calib_wc.get("distortion_coefficients")):
        return {"status": "error", "message": "Dati di calibrazione mancanti."}
    cam_matrix = np.array(cam_calib_wc["camera_matrix"], dtype=np.float32)
    dist_coeffs = np.array(cam_calib_wc["distortion_coefficients"], dtype=np.float32)
    frame, keypoints = get_current_frame_and_keypoints_from_config()
    if not keypoints: return {"status": "success", "coordinates": []}
    img_pts = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1,1,2)
    h, w = frame.shape[:2]
    new_cam_matrix, _ = cv2.getOptimalNewCameraMatrix(cam_matrix, dist_coeffs, (w,h), 1.0, (w,h))
    img_pts_undistorted = cv2.undistortPoints(img_pts, cam_matrix, dist_coeffs, P=new_cam_matrix)
    if img_pts_undistorted is None: return {"status": "error", "message": "Undistortion fallita."}
    world_coords = cv2.perspectiveTransform(img_pts_undistorted, H_fixed_ref).reshape(-1, 2).tolist() if img_pts_undistorted.size > 0 else []
    return {"status": "success", "coordinates": world_coords}

def construct_graph(nodi, velocita_x=4.0, velocita_y=1.0):
    G = nx.Graph()
    for i, pos in enumerate(nodi):
        G.add_node(i, pos=pos)
    for i in range(len(nodi)):
        for j in range(i + 1, len(nodi)):
            dx = nodi[i][0] - nodi[j][0]
            dy = nodi[i][1] - nodi[j][1]
            tempo = max(abs(dx) / velocita_x, abs(dy) / velocita_y)
            G.add_edge(i, j, weight=round(tempo, 4))
    return G

def get_graph_and_tsp_path(velocita_x=4.0, velocita_y=1.0):
    response = get_world_coordinates_data()
    if response.get("status") != "success": return None, None, response
    coordinates = response.get("coordinates", [])
    origin_x = camera_settings.get("origin_x", 0.0)
    filtered_coords = [coord for coord in coordinates if 0 <= (coord[0] - origin_x) <= 250]
    if len(filtered_coords) > 48:
        print(f"[INFO] Rilevati {len(filtered_coords)} punti, limitati a 48.")
        filtered_coords = sorted(filtered_coords, key=lambda p: (p[1], p[0]))[:48]
    nodi = [tuple(coord) for coord in filtered_coords]
    if not nodi: return None, None, {"status": "error", "message": "Punti insufficienti per il percorso."}
    graph = construct_graph(nodi, velocita_x, velocita_y)
    origin_pos = np.array([origin_x, camera_settings.get("origin_y", 0.0)])
    distances_sq = [((node[0] - origin_pos[0])**2 + (node[1] - origin_pos[1])**2) for node in nodi]
    closest_node_index = np.argmin(distances_sq)
    print(f"[INFO] Punto di partenza più vicino trovato: nodo #{closest_node_index}")
    path = [closest_node_index] if len(nodi) == 1 else nx.algorithms.approximation.traveling_salesman_problem(graph, cycle=False, method=nx.algorithms.approximation.greedy_tsp, source=closest_node_index)
    return graph, path, {"status": "success", "nodi": nodi}

def get_graph_and_tsp_path_with_speeds(velocita_x=4.0, velocita_y=1.0):
    return get_graph_and_tsp_path(velocita_x, velocita_y)

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
        current_disk_config = load_config_data_from_file()
        current_disk_config.setdefault("camera", {}).update(data)
        if save_config_data_to_file(current_disk_config):
            return JsonResponse({"status": "success"})
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
    data = get_world_coordinates_data()
    return JsonResponse(data, status=200 if data.get("status") == "success" else 400)

@csrf_exempt
@require_GET
def get_keypoints(request):
    try:
        _, keypoints_data = get_current_frame_and_keypoints_from_config()
        keypoints_list = [[kp.pt[0], kp.pt[1]] for kp in keypoints_data]
        return JsonResponse({"status": "success", "keypoints": keypoints_list})
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
        if save_config_data_to_file(config_data):
            return JsonResponse({"status": "success"})
        return JsonResponse({"status": "error", "message": "Salvataggio fallito."}, status=500)
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=400)

@csrf_exempt
@require_GET
def compute_route(request):
    try:
        velocita_x, velocita_y = 4.0, 1.0
        graph, hamiltonian_path, info = get_graph_and_tsp_path_with_speeds(velocita_x, velocita_y)
        if graph is None: return JsonResponse(info, status=500)
        
        nodi = info["nodi"]
        origin_x = camera_settings.get("origin_x", 0.0)
        origin_y = camera_settings.get("origin_y", 0.0)
        
        motor_commands = []
        if hamiltonian_path:
            primo_nodo_pos = nodi[hamiltonian_path[0]]
            motor_commands.append({
                "extruder": float(round(primo_nodo_pos[0] - origin_x, 4)),
                "conveyor": float(round(primo_nodo_pos[1] - origin_y, 4))
            })
            for i in range(len(hamiltonian_path) - 1):
                pos_attuale = nodi[hamiltonian_path[i]]
                pos_successiva = nodi[hamiltonian_path[i+1]]
                motor_commands.append({
                    "extruder": float(round(pos_successiva[0] - pos_attuale[0], 4)),
                    "conveyor": float(round(pos_successiva[1] - pos_attuale[1], 4))
                })

        plt.figure(figsize=(8, 6))
        pos = nx.get_node_attributes(graph, 'pos')
        graph.add_node('origin'); pos['origin'] = (origin_x, origin_y)
        
        nx.draw_networkx_nodes(graph, pos, nodelist=list(range(len(nodi))), node_color='skyblue', node_size=500)
        nx.draw_networkx_labels(graph, pos, font_size=10)
        nx.draw_networkx_nodes(graph, pos, nodelist=['origin'], node_color='limegreen', node_size=700, node_shape='s')
        if hamiltonian_path:
            tsp_edges = [('origin', hamiltonian_path[0])] + [(hamiltonian_path[i], hamiltonian_path[i+1]) for i in range(len(hamiltonian_path)-1)]
            nx.draw_networkx_edges(graph, pos, edgelist=tsp_edges, edge_color='red', width=2)
        
        plt.title("Percorso TSP (in rosso) con Origine"); plt.axis('off')
        buf = BytesIO(); plt.savefig(buf, format='png'); plt.close()
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return JsonResponse({
            "status": "success", "route": hamiltonian_path,
            "motor_commands": motor_commands, "plot_graph_base64": img_base64
        })
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"status": "error", "message": f"Errore interno: {e}"}, status=500)

@csrf_exempt
@require_GET
def plot_graph(request):
    try:
        velocita_x, velocita_y = 4.0, 1.0
        graph, hamiltonian_path, info = get_graph_and_tsp_path(velocita_x, velocita_y)
        if graph is None: return HttpResponse(f"Errore: {info.get('message')}", status=500)
        
        nodi = info["nodi"]
        plt.figure(figsize=(8, 6))
        pos = nx.get_node_attributes(graph, 'pos')
        if not pos: return HttpResponse("Nessuna posizione nodo trovata.", status=400)
        
        origin_x = camera_settings.get("origin_x", 0.0)
        origin_y = camera_settings.get("origin_y", 0.0)
        graph.add_node('origin'); pos['origin'] = (origin_x, origin_y)
        
        nx.draw_networkx_nodes(graph, pos, nodelist=list(range(len(nodi))), node_color='skyblue', node_size=500)
        nx.draw_networkx_labels(graph, pos, font_size=10)
        nx.draw_networkx_nodes(graph, pos, nodelist=['origin'], node_color='limegreen', node_size=700, node_shape='s')
        if hamiltonian_path:
            tsp_edges = [('origin', hamiltonian_path[0])] + [(hamiltonian_path[i], hamiltonian_path[i+1]) for i in range(len(hamiltonian_path)-1)]
            nx.draw_networkx_edges(graph, pos, edgelist=tsp_edges, edge_color='red', width=2)
        
        plt.title("Percorso TSP (in rosso) con Origine"); plt.axis('off')
        buf = BytesIO(); plt.savefig(buf, format='png'); plt.close(); buf.seek(0)
        return HttpResponse(buf.getvalue(), content_type='image/png')
    except Exception as e:
        traceback.print_exc()
        return HttpResponse(f"Errore interno: {e}", status=500)

@csrf_exempt
def fixed_perspective_stream(request):
    request.mode_override = 'fixed'
    return camera_feed(request) 

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
        if save_config_data_to_file(config_data):
            return JsonResponse({"status": "success"})
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
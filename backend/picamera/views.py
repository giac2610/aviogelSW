from contextlib import contextmanager
from io import BytesIO
import cv2
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-GUI environments
import matplotlib.pyplot as plt
import base64
import glob
import sys
import json
import os
import time
import threading
from django.http import StreamingHttpResponse, JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET
import traceback
import requests #type: ignore

# --- File Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, 'config')
SETUP_JSON_PATH = os.path.join(CONFIG_DIR, 'setup.json')
EXAMPLE_JSON_PATH = os.path.join(CONFIG_DIR, 'setup.example.json')
CALIBRATION_MEDIA_DIR = os.path.join(BASE_DIR, 'calibrationMedia')
os.makedirs(CALIBRATION_MEDIA_DIR, exist_ok=True)

if not os.path.exists(SETUP_JSON_PATH):
    from shutil import copyfile
    os.makedirs(CONFIG_DIR, exist_ok=True)
    if os.path.exists(EXAMPLE_JSON_PATH):
        copyfile(EXAMPLE_JSON_PATH, SETUP_JSON_PATH)
        print(f"[INFO] Configuration file created from {EXAMPLE_JSON_PATH}")
    else:
        default_config_content = {"camera": {"capture_width": 640, "capture_height": 480}}
        with open(SETUP_JSON_PATH, 'w') as f_default:
            json.dump(default_config_content, f_default, indent=4)
        print(f"[INFO] Minimal configuration file created at {SETUP_JSON_PATH} as example was missing.")

# --- Global Configuration Loading ---
config = {}
camera_settings = {}

def _load_global_config_from_file():
    global config, camera_settings
    try:
        with open(SETUP_JSON_PATH, 'r') as f:
            config = json.load(f)
        camera_settings = config.get("camera", {})
        camera_settings.setdefault("picamera_config", {}).setdefault("main", {}).setdefault("size", [640, 480])
        print("[INFO] Global configuration loaded successfully.")
    except Exception as e:
        print(f"Critical error loading setup.json at startup: {e}. Falling back to empty config.")
        config = {"camera": {"picamera_config": {"main": {"size": [640, 480]}}}}
        camera_settings = config["camera"]

_load_global_config_from_file()

# --- Camera Init ---
camera_instance = None
camera_lock = threading.Lock()
active_streams = 0

def _initialize_camera_internally():
    global camera_instance
    if camera_instance is not None:
        try:
            if hasattr(camera_instance, 'release'): camera_instance.release()
            elif hasattr(camera_instance, 'stop'): camera_instance.stop(); camera_instance.close()
            print("[INFO] Previous camera instance released.")
        except Exception as e:
            print(f"[WARN] Error releasing previous camera: {e}")
        camera_instance = None

    cfg_data_for_init = camera_settings
    picam_main_size = cfg_data_for_init.get("picamera_config", {}).get("main", {}).get("size", [640, 480])
    if not isinstance(picam_main_size, list) or len(picam_main_size) != 2:
        print(f"[WARN] picamera_config.main.size malformed: {picam_main_size}. Using default [640, 480].")
        picam_main_size = [640, 480]
    capture_width, capture_height = picam_main_size

    if sys.platform == "darwin":
        print("[INFO] Attempting macOS camera initialization...")
        mac_cam = cv2.VideoCapture(cfg_data_for_init.get("mac_camera_index", 0))
        if not mac_cam.isOpened():
            print(f"[WARN] macOS camera {cfg_data_for_init.get('mac_camera_index', 0)} not open. Trying index 1.")
            mac_cam = cv2.VideoCapture(1)
            if not mac_cam.isOpened():
                print("[ERROR] No webcam available on macOS.")
                return None
        mac_cam.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
        mac_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
        camera_instance = mac_cam
        print(f"[INFO] macOS camera initialized ({int(mac_cam.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(mac_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))}).")
    else:
        print("[INFO] Attempting Picamera2 initialization...")
        try:
            from picamera2 import Picamera2
            picam2 = Picamera2()
            video_config = picam2.create_video_configuration(main={"size": (capture_width, capture_height), "format": "RGB888"})
            picam2.configure(video_config)
            picam2.start()
            camera_instance = picam2
            print(f"[INFO] Picamera2 initialized ({capture_width}x{capture_height}, RGB888).")
        except Exception as e:
            print(f"[ERROR] Error during Picamera2 initialization: {e}")
            if 'picam2' in locals() and hasattr(picam2, 'close'):
                try: picam2.close()
                except Exception as e_close: print(f"[WARN] Error closing picam2 after failed init: {e_close}")
            return None
    return camera_instance

def get_frame(release_after=False):
    global camera_instance
    with camera_lock:
        if camera_instance is None:
            print("get_frame: Camera not initialized. Attempting to initialize.")
            _initialize_camera_internally()
            if camera_instance is None:
                print("get_frame: Camera unavailable, returning blank frame.")
                cfg_height = camera_settings.get("picamera_config", {}).get("main", {}).get("size", [640, 480])[1]
                cfg_width = camera_settings.get("picamera_config", {}).get("main", {}).get("size", [640, 480])[0]
                return np.zeros((cfg_height, cfg_width, 3), dtype=np.uint8)
        
        frame = None
        try:
            if sys.platform == "darwin":
                if not camera_instance.isOpened(): raise IOError("macOS camera failed to open.")
                ret, frame = camera_instance.read()
                if not ret: raise IOError("macOS camera failed to read frame.")
            else:
                if not hasattr(camera_instance, 'capture_array'): raise IOError("Picamera2 not ready or failed to initialize.")
                frame = camera_instance.capture_array()
        except Exception as e:
            print(f"get_frame: Error capturing frame: {e}. Returning blank frame.")
            if camera_instance is not None:
                try:
                    if sys.platform == "darwin": camera_instance.release()
                    else: camera_instance.stop(); camera_instance.close()
                except Exception as e_rel: print(f"Error releasing camera after capture error: {e_rel}")
                camera_instance = None
            cfg_height = camera_settings.get("picamera_config", {}).get("main", {}).get("size", [640, 480])[1]
            cfg_width = camera_settings.get("picamera_config", {}).get("main", {}).get("size", [640, 480])[0]
            return np.zeros((cfg_height, cfg_width, 3), dtype=np.uint8)

        if release_after and active_streams == 0:
            try:
                if sys.platform == "darwin": camera_instance.release()
                else: camera_instance.stop(); camera_instance.close()
                camera_instance = None
                print("get_frame: Camera released after single capture.")
            except Exception as e:
                print(f"get_frame: Error releasing camera: {e}")
        return frame

# --- Utility Functions ---
def load_config_data_from_file():
    try:
        with open(SETUP_JSON_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {SETUP_JSON_PATH}: {e}")
        return {"camera": {}}
    
def save_config_data_to_file(new_config_data):
    try:
        with open(SETUP_JSON_PATH, 'w') as f:
            json.dump(new_config_data, f, indent=4)
        _load_global_config_from_file()
        print(f"Configuration saved to {SETUP_JSON_PATH} and globals reloaded.")
        return True
    except Exception as e:
        print(f"Error saving to {SETUP_JSON_PATH}: {e}")
        return False

def get_fixed_perspective_homography_from_config():
    H_list = camera_settings.get("fixed_perspective", {}).get("homography_matrix", None)
    if H_list and isinstance(H_list, list):
        try:
            return np.array(H_list, dtype=np.float32)
        except Exception as e:
            print(f"Error converting fixed_perspective_homography to numpy array: {e}")
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
    area_scale_factor = (scale_x * scale_y)
    params.minArea = blob_detection_params.get("minArea", 150) * area_scale_factor
    params.maxArea = blob_detection_params.get("maxArea", 5000) * area_scale_factor
    params.minArea = max(1, params.minArea) 
    params.maxArea = max(1, params.maxArea)
    params.filterByCircularity = blob_detection_params.get("circularityFilter", True)
    params.minCircularity = blob_detection_params.get("minCircularity", 0.1)
    params.filterByConvexity = blob_detection_params.get("filterByConvexity", True)
    params.minConvexity = blob_detection_params.get("minConvexity", 0.87)
    params.filterByInertia = blob_detection_params.get("inertiaFilter", True)
    params.minInertiaRatio = blob_detection_params.get("minInertia", 0.01)
    detector = cv2.SimpleBlobDetector_create(params)
    return detector.detect(binary_image)

def get_current_frame_and_keypoints_from_config():
    frame = get_frame()
    if frame is None or frame.size == 0:
        return np.zeros((480, 640, 3), dtype=np.uint8), []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processing_width = 640
    original_height, original_width = frame.shape[:2]
    processing_height = int(original_height * (processing_width / original_width))
    scale_x = original_width / processing_width
    scale_y = original_height / processing_height
    resized_gray = cv2.resize(gray, (processing_width, processing_height), interpolation=cv2.INTER_AREA)
    _, thresh = cv2.threshold(resized_gray, camera_settings.get("minThreshold", 127), camera_settings.get("maxThreshold", 255), cv2.THRESH_BINARY)
    keypoints_resized = detect_blobs_from_params(thresh, camera_settings, scale_x, scale_y)
    keypoints_original_coords = [cv2.KeyPoint(kp.pt[0] * scale_x, kp.pt[1] * scale_y, kp.size * (scale_x + scale_y) / 2) for kp in keypoints_resized]
    return frame, keypoints_original_coords

def get_world_coordinates_data():
    H_fixed_ref = get_fixed_perspective_homography_from_config()
    if H_fixed_ref is None:
        return {"status": "error", "message": "Homography not set."}
    cam_calib = camera_settings.get("calibration", {})
    cam_matrix = cam_calib.get("camera_matrix")
    dist_coeffs = cam_calib.get("distortion_coefficients")
    if not (cam_matrix and dist_coeffs):
        return {"status": "error", "message": "Camera not calibrated."}
    
    frame, keypoints = get_current_frame_and_keypoints_from_config()
    if not keypoints:
        return {"status": "success", "coordinates": []}
        
    img_pts = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
    h, w = frame.shape[:2]
    cam_matrix_np = np.array(cam_matrix, dtype=np.float32)
    dist_coeffs_np = np.array(dist_coeffs, dtype=np.float32)
    new_cam_matrix, _ = cv2.getOptimalNewCameraMatrix(cam_matrix_np, dist_coeffs_np, (w, h), 1.0, (w, h))
    
    undistorted_pts = cv2.undistortPoints(img_pts, cam_matrix_np, dist_coeffs_np, P=new_cam_matrix)
    if undistorted_pts is None:
        return {"status": "error", "message": "Point undistortion failed."}
        
    world_coords_tl = cv2.perspectiveTransform(undistorted_pts, H_fixed_ref)
    if world_coords_tl is None:
        return {"status": "error", "message": "Perspective transformation failed."}
    
    world_coords_tl = world_coords_tl.reshape(-1, 2).tolist()
    
    fixed_persp_cfg = camera_settings.get("fixed_perspective", {})
    OUTPUT_WIDTH = fixed_persp_cfg.get("output_width", 1000)
    OUTPUT_HEIGHT = fixed_persp_cfg.get("output_height", 800)
    
    world_coords_br = [[OUTPUT_WIDTH - x, OUTPUT_HEIGHT - y] for x, y in world_coords_tl]
    
    return {"status": "success", "coordinates": world_coords_br}

@contextmanager
def stream_context():
    global active_streams
    active_streams += 1
    print(f"[STREAM] Stream started. Active streams: {active_streams}")
    try:
        yield
    finally:
        active_streams -= 1
        print(f"[STREAM] Stream ended. Active streams: {active_streams}")

def get_board_and_canonical_homography_for_django(undistorted_frame, new_camera_matrix_cv, calibration_cfg_dict):
    cs_cols = calibration_cfg_dict.get("chessboard_cols", 9)
    cs_rows = calibration_cfg_dict.get("chessboard_rows", 7)
    sq_size = calibration_cfg_dict.get("square_size_mm", 15.0)
    chessboard_dim_cv = (cs_cols, cs_rows)
    objp_cv = np.zeros((cs_cols * cs_rows, 3), np.float32)
    objp_cv[:, :2] = np.mgrid[0:cs_cols, 0:cs_rows].T.reshape(-1, 2)
    objp_cv *= sq_size
    
    criteria_cv = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_dim_cv, None)
    if not ret: return None, None
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria_cv)
    success, rvec, tvec = cv2.solvePnP(objp_cv, corners2, new_camera_matrix_cv, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success: return None, None
    
    obj_board_perimeter_pts = np.array([
        [0,0,0], [ (cs_cols-1)*sq_size, 0, 0], 
        [(cs_cols-1)*sq_size, (cs_rows-1)*sq_size, 0], [0, (cs_rows-1)*sq_size, 0]
    ], dtype=np.float32)
    img_board_perimeter_pts, _ = cv2.projectPoints(obj_board_perimeter_pts, rvec, tvec, new_camera_matrix_cv, None)
    img_board_perimeter_pts = img_board_perimeter_pts.reshape(-1, 2)
    
    canonical_width = int(round((cs_cols-1) * sq_size))
    canonical_height = int(round((cs_rows-1) * sq_size))
    canonical_dst_pts = np.array([
        [0,0], [canonical_width-1,0], 
        [canonical_width-1,canonical_height-1], [0,canonical_height-1]
    ], dtype=np.float32)
    
    H_canonical = cv2.getPerspectiveTransform(img_board_perimeter_pts, canonical_dst_pts)
    canonical_board_size_tuple = (canonical_width, canonical_height) 
    
    return H_canonical, canonical_board_size_tuple

# --- Sezione Logica Principale per Griglia e Percorso ---

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

def rotate_points(points, angle_deg, center):
    angle_rad = np.deg2rad(angle_deg)
    R = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    return np.dot(points - center, R.T) + center

def _generate_grid_and_path(world_coords, camera_settings, velocita_x=4.0, velocita_y=1.0):
    GRID_ROWS, GRID_COLS = 8, 6
    SPACING_X_MM, SPACING_Y_MM = 50.0, 50.0

    points = np.array(world_coords, dtype=np.float32)
    if len(points) < 3:
        # fallback: griglia statica
        anchor_point = np.array([np.min(points[:, 0]), np.min(points[:, 1])])
        ideal_grid_world = []
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                x = anchor_point[0] + c * SPACING_X_MM
                y = anchor_point[1] + r * SPACING_Y_MM
                ideal_grid_world.append([x, y])
        ordered_grid_points = ideal_grid_world
    else:
        # 1. Stima angolo griglia con minAreaRect
        rect = cv2.minAreaRect(points)
        width, height = rect[1]
        angle = rect[2]
        if width > height:
            angle = 90 + angle  # Allinea sempre all'asse delle righe (8)
        print(f"[INFO] Angolo rispetto all'asse delle 8 righe: {angle:.2f}°")

        # 2. Ruota i punti per allinearli all'asse X
        center = np.mean(points, axis=0)
        def rotate_points(pts, angle_deg, center):
            angle_rad = np.deg2rad(angle_deg)
            R = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad),  np.cos(angle_rad)]
            ])
            return np.dot(pts - center, R.T) + center

        points_rot = rotate_points(points, -angle, center)

        # 3. Trova punto di ancoraggio nella base ruotata
        min_x, min_y = np.min(points_rot, axis=0)
        anchor_point_rot = np.array([min_x, min_y])

        # 4. Genera la griglia ideale ruotata
        ideal_grid_rot = []
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                x = anchor_point_rot[0] + c * SPACING_X_MM
                y = anchor_point_rot[1] + r * SPACING_Y_MM
                ideal_grid_rot.append([x, y])
        ideal_grid_rot = np.array(ideal_grid_rot, dtype=np.float32)

        # 5. Ruota indietro la griglia per riportarla nel sistema originale
        ideal_grid_world = rotate_points(ideal_grid_rot, angle, center).tolist()
        ordered_grid_points = ideal_grid_world

    # 6. Costruisci grafo e percorso
    graph = construct_graph([tuple(p) for p in ordered_grid_points], velocita_x, velocita_y)
    path_indices_grid_only = nx.algorithms.approximation.greedy_tsp(graph, source=0)
    ordered_grid_points = [ordered_grid_points[i] for i in path_indices_grid_only]

    origin_x = camera_settings.get("origin_x", 0.0)
    origin_y = camera_settings.get("origin_y", 0.0)
    final_ordered_path = [[origin_x, origin_y]] + ordered_grid_points

    return ideal_grid_world, final_ordered_path

def get_graph_and_tsp_path_with_speeds(velocita_x=4.0, velocita_y=1.0):
    response = get_world_coordinates_data()
    if response.get("status") != "success" or not response.get("coordinates"):
        return None, None, {"status": "error", "message": "Nessun punto rilevato."}

    _, final_ordered_path = _generate_grid_and_path(response["coordinates"], camera_settings, velocita_x, velocita_y)

    final_nodes = [tuple(p) for p in final_ordered_path]
    final_graph = construct_graph(final_nodes, velocita_x, velocita_y)
    final_path_indices = list(range(len(final_nodes)))

    return final_graph, final_path_indices, {"status": "success", "nodi": final_nodes}

def get_graph_and_tsp_path(velocita_x=4.0, velocita_y=1.0):
    return get_graph_and_tsp_path_with_speeds(velocita_x, velocita_y)


# --- Django Endpoints ---

@csrf_exempt
@require_POST
def initialize_camera_endpoint(request):
    print("[ENDPOINT] HTTP request to initialize camera.")
    with camera_lock:
        instance = _initialize_camera_internally()
    if instance is not None:
        return JsonResponse({"status": "success", "message": "Camera initialization attempt completed."})
    else:
        return JsonResponse({"status": "error", "message": "Camera initialization failed."}, status=500)

@csrf_exempt
@require_POST
def deinitialize_camera_endpoint(request):
    global camera_instance
    with camera_lock:
        if camera_instance is not None:
            try:
                if sys.platform == "darwin":
                    camera_instance.release()
                else:
                    if hasattr(camera_instance, 'stop'): camera_instance.stop()
                    if hasattr(camera_instance, 'close'): camera_instance.close()
                camera_instance = None
                print("[INFO] Camera deinitialized.")
                return JsonResponse({"status": "success", "message": "Camera deinitialized."})
            except Exception as e:
                print(f"[ERROR] Error during camera deinitialization: {e}")
                return JsonResponse({"status": "error", "message": str(e)}, status=500)
        else:
            return JsonResponse({"status": "success", "message": "Camera was already released."})

@csrf_exempt
@require_POST
def update_camera_settings(request):
    try:
        data = json.loads(request.body)
        current_disk_config = load_config_data_from_file()
        if "camera" not in current_disk_config: current_disk_config["camera"] = {}
        for key, value in data.items():
            if isinstance(value, dict) and key in current_disk_config["camera"] and isinstance(current_disk_config["camera"][key], dict):
                current_disk_config["camera"][key].update(value)
            else:
                current_disk_config["camera"][key] = value
        if save_config_data_to_file(current_disk_config):
            return JsonResponse({"status": "success", "updated_settings": camera_settings})
        else:
            return JsonResponse({"status": "error", "message": "Failed to save updated settings."}, status=500)
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=400)

@csrf_exempt
@require_GET
def get_world_coordinates(request):
    data = get_world_coordinates_data()
    status_code = 200 if data.get("status") == "success" else 400
    return JsonResponse(data, status=status_code)

@csrf_exempt
@require_GET
def compute_route(request):
    try:
        resp = requests.get("http://localhost:8000/motors/maxSpeeds/")
        data = resp.json()
        velocita_x = data.get("speeds", {}).get("extruder", 4.0)
        velocita_y = data.get("speeds", {}).get("conveyor", 1.0)
    except Exception as e:
        print(f"Errore richiesta velocità motori: {e}")
        velocita_x, velocita_y = 4.0, 1.0

    # Usa la stessa funzione di plot_graph!
    graph, hamiltonian_path, info = get_graph_and_tsp_path(velocita_x, velocita_y)
    if graph is None or hamiltonian_path is None:
        return JsonResponse({"status": "error", "message": info.get("message", "Errore generico")}, status=500)
    
    nodi = info["nodi"]
    motor_commands = []
    for i in range(1, len(nodi)):
        extruder_mm = nodi[i][0] - nodi[i-1][0]
        conveyor_mm = nodi[i][1] - nodi[i-1][1]
        motor_commands.append({"extruder": round(extruder_mm, 4), "conveyor": round(conveyor_mm, 4)})

    # Genera il plot come immagine base64 (opzionale)
    plt.figure(figsize=(8, 6))
    pos = nx.get_node_attributes(graph, 'pos')
    nx.draw_networkx_nodes(graph, pos, node_color='skyblue', node_size=500)
    nx.draw_networkx_labels(graph, pos, font_size=10)
    tsp_edges = [(hamiltonian_path[i], hamiltonian_path[i+1]) for i in range(len(hamiltonian_path)-1)]
    nx.draw_networkx_edges(graph, pos, edgelist=tsp_edges, edge_color='red', width=2)
    plt.title("Percorso TSP (in rosso)")
    plt.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return JsonResponse({
        "status": "success",
        "route": hamiltonian_path,
        "motor_commands": motor_commands,
        "plot_graph_base64": img_base64
    })
    
@csrf_exempt
@require_GET
def camera_feed(request):
    def gen_frames():
        with stream_context():
            while True:
                try:
                    frame = get_frame()
                    if frame is None or frame.size == 0:
                        time.sleep(0.1)
                        continue
                    
                    _, keypoints = get_current_frame_and_keypoints_from_config()
                    frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    
                    _, buffer = cv2.imencode('.jpg', frame_with_keypoints, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                except Exception as e:
                    print(f"Error in camera_feed loop: {e}")
                    time.sleep(1)
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

@csrf_exempt
@require_GET
def reproject_points_feed(request):
    def gen_frames():
        H_fixed = get_fixed_perspective_homography_from_config()
        if H_fixed is None:
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(dummy_frame, "Homography not set", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', dummy_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            return

        ret, H_inv = cv2.invert(H_fixed)
        if not ret: return

        cam_calib = camera_settings.get("calibration", {})
        cam_matrix = np.array(cam_calib.get("camera_matrix"))
        dist_coeffs = np.array(cam_calib.get("distortion_coefficients"))
        try:
            sample_frame = get_frame()
            h, w = sample_frame.shape[:2]
            new_cam_matrix, _ = cv2.getOptimalNewCameraMatrix(cam_matrix, dist_coeffs, (w,h), 1.0, (w,h))
        except Exception as e:
            print(f"[ERRORE] Nello stream: {e}")
            return

        with stream_context():
            while True:
                try:
                    frame = get_frame()
                    undistorted_frame = cv2.undistort(frame, cam_matrix, dist_coeffs, None, new_cam_matrix)
                    world_coords_data = get_world_coordinates_data()
                    
                    if world_coords_data.get('status') == 'success' and world_coords_data.get('coordinates'):
                        ideal_grid_world, ordered_path_world = _generate_grid_and_path(world_coords_data['coordinates'], camera_settings)

                        grid_world_pts_np = np.array(ideal_grid_world, dtype=np.float32).reshape(-1, 1, 2)
                        grid_pixels_np = cv2.perspectiveTransform(grid_world_pts_np, H_inv)
                        if grid_pixels_np is not None:
                            for pt in grid_pixels_np:
                                cv2.circle(undistorted_frame, tuple(pt[0].astype(int)), 5, (0, 255, 0), -1)

                        path_world_pts_np = np.array(ordered_path_world, dtype=np.float32).reshape(-1, 1, 2)
                        path_pixels_np = cv2.perspectiveTransform(path_world_pts_np, H_inv)
                        if path_pixels_np is not None:
                            path_pixel_coords = [tuple(p[0].astype(int)) for p in path_pixels_np]
                            for i in range(len(path_pixel_coords) - 1):
                                cv2.line(undistorted_frame, path_pixel_coords[i], path_pixel_coords[i+1], (255, 0, 0), 2)

                    _, buffer = cv2.imencode('.jpg', undistorted_frame)
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                except Exception as e:
                    print(f"[ERRORE] Nel loop di disegno: {e}")
                    traceback.print_exc()
                    time.sleep(1)

    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

@csrf_exempt
@require_POST
def reset_camera_calibration(request):
    try:
        current_disk_config = load_config_data_from_file()
        if "camera" not in current_disk_config: current_disk_config["camera"] = {}
        current_disk_config["camera"].setdefault("calibration", {})
        current_disk_config["camera"].setdefault("fixed_perspective", {})
        current_disk_config["camera"]["calibration"]["camera_matrix"] = None
        current_disk_config["camera"]["calibration"]["distortion_coefficients"] = None
        current_disk_config["camera"]["fixed_perspective"]["homography_matrix"] = None
        if save_config_data_to_file(current_disk_config):
            return JsonResponse({"status": "success", "message": "Camera calibration and fixed perspective reset."})
        else:
            return JsonResponse({"status": "error", "message": "Failed to save the reset configuration."}, status=500)
    except Exception as e:
        print(f"Error resetting camera calibration: {e}")
        traceback.print_exc()
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

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
        x_val, y_val = float(data.get("origin_x", 0.0)), float(data.get("origin_y", 0.0))
        current_disk_config = load_config_data_from_file()
        current_disk_config.setdefault("camera", {})
        current_disk_config["camera"]["origin_x"] = x_val
        current_disk_config["camera"]["origin_y"] = y_val
        if save_config_data_to_file(current_disk_config):
            return JsonResponse({"status": "success", "origin": {"origin_x": x_val, "origin_y": y_val}})
        else:
            return JsonResponse({"status": "error", "message": "Failed to save origin."}, status=500)
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=400)

@csrf_exempt
@require_POST
def save_frame_calibration(request):
    try:
        frame_to_save = get_frame(release_after=True)
        if frame_to_save is None or frame_to_save.size == 0:
            return JsonResponse({"status": "error", "message": "Invalid frame from camera."}, status=500)
        filename = f"calib_{int(time.time())}.jpg"
        filepath = os.path.join(CALIBRATION_MEDIA_DIR, filename)
        cv2.imwrite(filepath, frame_to_save)
        print(f"Frame saved for calibration: {filepath}")
        return JsonResponse({"status": "success", "filename": filename, "path": filepath})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

@csrf_exempt
@require_POST
def calibrate_camera_endpoint(request):
    calib_settings = camera_settings.get("calibration_settings", {})
    cs_cols, cs_rows = calib_settings.get("chessboard_cols", 9), calib_settings.get("chessboard_rows", 7)
    square_size_mm = calib_settings.get("square_size_mm", 15.0)
    chessboard_dim = (cs_cols, cs_rows)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((cs_cols * cs_rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cs_cols, 0:cs_rows].T.reshape(-1, 2) * square_size_mm
    
    objpoints, imgpoints = [], []
    image_files = glob.glob(os.path.join(CALIBRATION_MEDIA_DIR, '*.jpg'))
    if not image_files:
        return JsonResponse({"status": "error", "message": f"No JPG images in {CALIBRATION_MEDIA_DIR}."}, status=400)
        
    gray_shape = None
    for fname in image_files:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray_shape is None: gray_shape = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, chessboard_dim, None)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
    if not objpoints:
        return JsonResponse({"status": "error", "message": "No valid chessboard points found."}, status=400)
        
    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)
    if ret:
        calib_data = {"camera_matrix": mtx.tolist(), "distortion_coefficients": dist.tolist()}
        current_config = load_config_data_from_file()
        current_config.setdefault("camera", {})["calibration"] = calib_data
        if save_config_data_to_file(current_config):
            return JsonResponse({"status": "success", "message": "Calibration saved.", "calibration": calib_data})
        else:
            return JsonResponse({"status": "error", "message": "Calibration failed to save."}, status=500)
    else:
        return JsonResponse({"status": "error", "message": "cv2.calibrateCamera failed."}, status=500)

@csrf_exempt
@require_POST
def set_fixed_perspective_view(request):
    cam_calib = camera_settings.get("calibration", {})
    cam_matrix = np.array(cam_calib.get("camera_matrix"))
    dist_coeffs = np.array(cam_calib.get("distortion_coefficients"))
    if cam_matrix.size == 0 or dist_coeffs.size == 0:
        return JsonResponse({"status": "error", "message": "Camera not calibrated."}, status=400)
        
    frame = get_frame(release_after=True)
    if frame is None: return JsonResponse({"status": "error", "message": "Could not get frame."}, status=500)
    
    h, w = frame.shape[:2]
    new_cam_matrix, _ = cv2.getOptimalNewCameraMatrix(cam_matrix, dist_coeffs, (w,h), 1.0, (w,h))
    undistorted_frame = cv2.undistort(frame, cam_matrix, dist_coeffs, None, new_cam_matrix)
    
    calib_settings = camera_settings.get("calibration_settings", {})
    H_canonical, canonical_dims = get_board_and_canonical_homography_for_django(undistorted_frame, new_cam_matrix, calib_settings)
    
    if H_canonical is not None:
        fixed_persp_cfg = camera_settings.get("fixed_perspective", {})
        FIXED_WIDTH = fixed_persp_cfg.get("output_width", 1000)
        FIXED_HEIGHT = fixed_persp_cfg.get("output_height", 800)
        cb_w, cb_h = canonical_dims
        offset_x = max(0, (FIXED_WIDTH - cb_w) / 2.0)
        offset_y = max(0, (FIXED_HEIGHT - cb_h) / 2.0)
        M_translate = np.array([[1,0,offset_x], [0,1,offset_y], [0,0,1]], dtype=np.float32)
        H_ref = M_translate @ H_canonical
        if save_fixed_perspective_homography_to_config(H_ref):
            return JsonResponse({"status": "success", "message": "Fixed perspective view established."})
        else:
            return JsonResponse({"status": "error", "message": "Error saving homography."}, status=500)
    else:
        return JsonResponse({"status": "error", "message": "Chessboard pattern not detected."}, status=400)

@csrf_exempt
@require_GET
def plot_graph(request):
    graph, hamiltonian_path, info = get_graph_and_tsp_path()
    if graph is None: return HttpResponse("Error: " + info.get("message", "Unknown"), status=500)
    
    plt.figure(figsize=(8, 6))
    pos = nx.get_node_attributes(graph, 'pos')
    nx.draw_networkx_nodes(graph, pos, node_color='skyblue', node_size=500)
    nx.draw_networkx_labels(graph, pos, font_size=10)
    path_edges = list(zip(hamiltonian_path, hamiltonian_path[1:]))
    nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='red', width=2)
    plt.title("Percorso TSP (in rosso)")
    plt.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return HttpResponse(buf.getvalue(), content_type='image/png')

@csrf_exempt
@require_POST
def set_fixed_perspective_view(request):
    cam_calib_data = camera_settings.get("calibration", None)
    calib_settings_dict = camera_settings.get("calibration_settings", {})
    fixed_perspective_cfg = camera_settings.get("fixed_perspective", {})
    if not (cam_calib_data and cam_calib_data.get("camera_matrix") and cam_calib_data.get("distortion_coefficients")):
        return JsonResponse({"status": "error", "message": "Camera calibration data not found. Please calibrate first."}, status=400)
    camera_matrix_cv = np.array(cam_calib_data["camera_matrix"], dtype=np.float32)
    dist_coeffs_cv = np.array(cam_calib_data["distortion_coefficients"], dtype=np.float32)
    FIXED_WIDTH = fixed_perspective_cfg.get("output_width", 1000)
    FIXED_HEIGHT = fixed_perspective_cfg.get("output_height", 800)
    
    try:
        frame_cap = get_frame(release_after=False)
        if frame_cap is None or frame_cap.size == 0: 
            return JsonResponse({"status": "error", "message": "Could not get frame from camera."}, status=500)
        h_cam_cap, w_cam_cap = frame_cap.shape[:2]
        new_camera_matrix_cv, _ = cv2.getOptimalNewCameraMatrix(camera_matrix_cv, dist_coeffs_cv, (w_cam_cap,h_cam_cap), 1.0, (w_cam_cap,h_cam_cap))
        undistorted_frame_cap = cv2.undistort(frame_cap, camera_matrix_cv, dist_coeffs_cv, None, new_camera_matrix_cv)
        
        H_canonical, canonical_dims = get_board_and_canonical_homography_for_django(
            undistorted_frame_cap, new_camera_matrix_cv, calib_settings_dict
        )
        
        if H_canonical is not None and canonical_dims is not None and canonical_dims[0] > 0 and canonical_dims[1] > 0:
            cb_w, cb_h = canonical_dims
            offset_x = max(0, (FIXED_WIDTH - cb_w) / 2.0)
            offset_y = max(0, (FIXED_HEIGHT - cb_h) / 2.0)
            M_translate = np.array([[1,0,offset_x], [0,1,offset_y], [0,0,1]], dtype=np.float32)
            H_ref = M_translate @ H_canonical
            if save_fixed_perspective_homography_to_config(H_ref):
                return JsonResponse({
                    "status": "success",
                    "message": "Fixed perspective view established and saved."
                })
            else:
                print("[ERROR] set_fixed_perspective_view: Failed to save homography to config file.")
                return JsonResponse({
                    "status": "error",
                    "message": "Error saving fixed perspective homography to configuration file.",
                    "error_code": "SAVE_HOMOGRAPHY_FAILED"
                }, status=500)
        else:
            error_message = "Cannot define fixed view. An unknown error occurred."
            error_code = "UNKNOWN_FIXED_VIEW_ERROR"
            status_code = 400
            if H_canonical is None:
                error_message = "Chessboard pattern not detected in the current camera view. Ensure the full pattern is clearly visible and well-lit."
                error_code = "CHESSBOARD_NOT_DETECTED"
                print(f"[ERROR] set_fixed_perspective_view ({error_code}): {error_message}")
            elif canonical_dims is None:
                error_message = "Internal error: Chessboard detected, but its dimensions could not be determined."
                error_code = "CANONICAL_DIMS_MISSING_UNEXPECTEDLY"
                status_code = 500
                print(f"[ERROR] set_fixed_perspective_view ({error_code}): {error_message}")
            elif canonical_dims[0] <= 0 or canonical_dims[1] <= 0:
                error_message = (f"Invalid canonical dimensions calculated for the chessboard: {canonical_dims}. "
                                 f"This might indicate an issue with the chessboard configuration (e.g., square size, pattern size in settings) "
                                 f"or a highly distorted detection.")
                error_code = "INVALID_CANONICAL_DIMS_CALCULATED"
                print(f"[ERROR] set_fixed_perspective_view ({error_code}): {error_message} - Dimensions: {canonical_dims}")
            
            return JsonResponse({
                "status": "error",
                "message": error_message,
                "error_code": error_code
            }, status=status_code)
    except Exception as e:
        print(f"Exception in set_fixed_perspective_view: {e}")
        traceback.print_exc()
        return JsonResponse({"status": "error", "message": str(e)}, status=500)
    
@csrf_exempt
@require_GET
def fixed_perspective_stream(request):
    def gen_frames():
        stream_cfg = camera_settings 
        
        H_ref = get_fixed_perspective_homography_from_config()
        
        cam_calib = stream_cfg.get("calibration", None)
        fixed_persp_cfg = stream_cfg.get("fixed_perspective", {})
        blob_params_for_stream = stream_cfg
        
        OUT_W = fixed_persp_cfg.get("output_width", 1000)
        OUT_H = fixed_persp_cfg.get("output_height", 800)
        error_template_frame = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)

        blob_detection_interval = stream_cfg.get("blob_detection_interval", 5)
        frame_count = 0
        last_keypoints_undistorted_for_drawing = [] # Keypoint nell'immagine undistorta, prima di essere warpatati

        blob_processing_width = stream_cfg.get("blob_processing_width", 640)

        if not (cam_calib and cam_calib.get("camera_matrix") and cam_calib.get("distortion_coefficients")):
            error_msg = "Camera calibration missing"
            print(f"fixed_perspective_stream: {error_msg}")
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            while True:
                err_f = error_template_frame.copy()
                cv2.putText(err_f, error_msg, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                _, buf = cv2.imencode('.jpg', err_f, encode_param)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n') 
                time.sleep(1)
        cam_matrix = np.array(cam_calib["camera_matrix"], dtype=np.float32)
        dist_coeffs = np.array(cam_calib["distortion_coefficients"], dtype=np.float32)
        
        new_cam_matrix_stream = None
        try:
            sample_frame_for_dims = get_frame(release_after=False) 
            if sample_frame_for_dims is not None and sample_frame_for_dims.size > 0:
                h_str, w_str = sample_frame_for_dims.shape[:2]
                new_cam_matrix_stream, _ = cv2.getOptimalNewCameraMatrix(cam_matrix, dist_coeffs, (w_str,h_str), 1.0, (w_str,h_str))
            else:
                 raise ValueError("Could not get valid sample frame dimensions for new_camera_matrix")
        except Exception as e_stream_setup:
            error_msg = f"Stream setup failed: {e_stream_setup}"
            print(f"fixed_perspective_stream: {error_msg}")
            while True:
                err_f = error_template_frame.copy()
                cv2.putText(err_f, error_msg[:70], (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
                
                _, buf = cv2.imencode('.jpg', err_f)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
                time.sleep(1)
        
        with stream_context():
            while True:
                try:
                    frame_live = get_frame()
                    if frame_live is None or frame_live.size == 0: 
                        err_f_loop = error_template_frame.copy()
                        cv2.putText(err_f_loop, "Frame lost", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,100,255),2)
                        if H_ref is None: cv2.putText(err_f_loop, "Fixed View Not Set", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),1)
                        _, buf_err = cv2.imencode('.jpg', err_f_loop)
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf_err.tobytes() + b'\r\n')
                        # time.sleep(0.1)
                        continue

                    undistorted_live = cv2.undistort(frame_live, cam_matrix, dist_coeffs, None, new_cam_matrix_stream)
                    output_img = error_template_frame.copy()
                    
                    if H_ref is not None:
                        output_img = cv2.warpPerspective(undistorted_live, H_ref, (OUT_W, OUT_H))
                        
                        # LOGICA BLOB DETECTION A INTERVALLI (MODO FIXED)
                        if frame_count % blob_detection_interval == 0:
                            original_height_undistorted, original_width_undistorted = undistorted_live.shape[:2]
                            blob_processing_height = int(original_height_undistorted * (blob_processing_width / original_width_undistorted))
                            
                            scale_x = original_width_undistorted / blob_processing_width
                            scale_y = original_height_undistorted / blob_processing_height

                            gray_for_blobs = cv2.cvtColor(undistorted_live, cv2.COLOR_BGR2GRAY)
                            resized_gray_for_blobs = cv2.resize(gray_for_blobs, (blob_processing_width, blob_processing_height), interpolation=cv2.INTER_AREA)

                            _, thresh_for_blobs = cv2.threshold(
                                resized_gray_for_blobs,
                                blob_params_for_stream.get("minThreshold", 127),
                                blob_params_for_stream.get("maxThreshold", 255),
                                cv2.THRESH_BINARY
                            )
                            # Passiamo i fattori di scala a detect_blobs_from_params
                            keypoints_resized = detect_blobs_from_params(thresh_for_blobs, blob_params_for_stream, scale_x, scale_y)
                            
                            # Riscala i keypoint alle coordinate dell'immagine undistorta originale
                            last_keypoints_undistorted_for_drawing = []
                            for kp in keypoints_resized:
                                new_x = kp.pt[0] * scale_x
                                new_y = kp.pt[1] * scale_y
                                new_size = kp.size * ((scale_x + scale_y) / 2)
                                last_keypoints_undistorted_for_drawing.append(cv2.KeyPoint(new_x, new_y, new_size, kp.angle, kp.response, kp.octave, kp.class_id))
                        
                        # Usa gli ultimi keypoint rilevati (o i nuovi) per disegnare
                        if last_keypoints_undistorted_for_drawing:
                            # Questi keypoint sono già nelle coordinate dell'immagine undistorta.
                            # Quindi non serve undistortPoints su di essi. Basta trasformarli con H_ref.
                            pts_undist = np.array([kp.pt for kp in last_keypoints_undistorted_for_drawing], dtype=np.float32).reshape(-1,1,2)
                            pts_warped = cv2.perspectiveTransform(pts_undist, H_ref)
                            
                            if pts_warped is not None:
                                for i, pt_w in enumerate(pts_warped.reshape(-1,2)):
                                    x, y = pt_w[0], pt_w[1]
                                    cv2.circle(output_img, (int(round(x)), int(round(y))), 8, (0,0,255), 2)
                                    cv2.putText(
                                        output_img, f"{x:.1f},{y:.1f}",
                                        (int(round(x))+10, int(round(y))-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1
                                    )
                        
                        frame_count += 1

                    else: # H_ref is None
                        cv2.putText(output_img, "Fixed View Not Set", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),1)
                        cv2.putText(output_img, "Use endpoint to set it", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),1)
                    
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                    _, buffer_ok = cv2.imencode('.jpg', output_img, encode_param)
                    frame_bytes_ok = buffer_ok.tobytes()
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes_ok + b'\r\n')
                    # time.sleep(0.03) # Commentato per massimizzare il framerate, se rallenta decommenta con valore > 0.03
                except Exception as e_loop_stream:
                    print(f"Error in fixed_perspective_stream loop: {e_loop_stream}")
                    traceback.print_exc()
                    err_f_loop = error_template_frame.copy()
                    cv2.putText(err_f_loop, f"Stream Loop Err: {str(e_loop_stream)[:50]}", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,100,255),2)
                    if H_ref is None: cv2.putText(err_f_loop, "Fixed View Not Set", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),1)
                    _, buf_err = cv2.imencode('.jpg', err_f_loop)
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf_err.tobytes() + b'\r\n')
                    time.sleep(1)
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

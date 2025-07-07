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
import requests #type: ignore[import-untyped]

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
                if not camera_instance.isOpened():
                    raise IOError("macOS camera failed to open.")
                ret, frame = camera_instance.read()
                if not ret: raise IOError("macOS camera failed to read frame.")
            else:
                if not hasattr(camera_instance, 'capture_array'):
                    raise IOError("Picamera2 not ready or failed to initialize.")
                frame = camera_instance.capture_array()
        except Exception as e:
            print(f"get_frame: Error capturing frame: {e}. Returning blank frame.")
            # Reset camera instance on error
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

def _generate_grid_and_path(world_coords, camera_settings):
    """
    Funzione unificata che genera la griglia ideale e calcola il percorso.
    Questa è l'unica "fonte di verità" per la geometria.
    """
    GRID_ROWS, GRID_COLS = 8, 6
    SPACING_X_MM, SPACING_Y_MM = 50.0, 50.0

    points = np.array(world_coords)
    anchor_point = np.array([np.min(points[:, 0]), np.min(points[:, 1])])
    print(f"[INFO] Ancoraggio griglia calcolato: {anchor_point}")

    ideal_grid_world = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            x = anchor_point[0] - c * SPACING_X_MM
            y = anchor_point[1] - r * SPACING_Y_MM
            ideal_grid_world.append([x, y])
    
    print(f"[INFO] Griglia ideale {GRID_ROWS}x{GRID_COLS} generata.")

    graph = construct_graph([tuple(p) for p in ideal_grid_world])
    path_indices_grid_only = nx.algorithms.approximation.greedy_tsp(graph, source=0)
    ordered_grid_points = [ideal_grid_world[i] for i in path_indices_grid_only]

    origin_x = camera_settings.get("origin_x", 0.0)
    origin_y = camera_settings.get("origin_y", 0.0)
    final_ordered_path = [[origin_x, origin_y]] + ordered_grid_points

    return ideal_grid_world, final_ordered_path

def get_graph_and_tsp_path_with_speeds(velocita_x=4.0, velocita_y=1.0):
    """
    Prepara i dati per i comandi motore chiamando la funzione master.
    """
    response = get_world_coordinates_data()
    if response.get("status") != "success" or not response.get("coordinates"):
        return None, None, {"status": "error", "message": "Nessun punto rilevato."}

    _, final_ordered_path = _generate_grid_and_path(response["coordinates"], camera_settings)

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

    graph, hamiltonian_path, info = get_graph_and_tsp_path_with_speeds(velocita_x, velocita_y)
    if graph is None or hamiltonian_path is None:
        return JsonResponse({"status": "error", "message": info.get("message", "Errore generico")}, status=500)
    
    nodi = info["nodi"]
    motor_commands = []
    for i in range(1, len(nodi)):
        extruder_mm = nodi[i][0] - nodi[i-1][0]
        conveyor_mm = nodi[i][1] - nodi[i-1][1]
        motor_commands.append({"extruder": round(extruder_mm, 4), "conveyor": round(conveyor_mm, 4)})

    return JsonResponse({
        "status": "success",
        "route": hamiltonian_path,
        "motor_commands": motor_commands
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
    """
    Stream video che disegna ESATTAMENTE la stessa griglia e percorso usati per i comandi motore.
    """
    def gen_frames():
        H_fixed = get_fixed_perspective_homography_from_config()
        if H_fixed is None:
            # Gestione errore
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

                        # Disegna la griglia (punti verdi)
                        grid_world_pts_np = np.array(ideal_grid_world, dtype=np.float32).reshape(-1, 1, 2)
                        grid_pixels_np = cv2.perspectiveTransform(grid_world_pts_np, H_inv)
                        if grid_pixels_np is not None:
                            for pt in grid_pixels_np:
                                cv2.circle(undistorted_frame, tuple(pt[0].astype(int)), 5, (0, 255, 0), -1)

                        # Disegna il percorso (linee blu)
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
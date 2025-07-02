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
import traceback # Added for more detailed error logging if needed
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
        # Create a minimal default config if example is missing
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
        # Imposta un default per picamera_config.main.size se non esiste
        camera_settings.setdefault("picamera_config", {}).setdefault("main", {}).setdefault("size", [640, 480])
        print("[INFO] Global configuration loaded successfully.")
    except Exception as e:
        print(f"Critical error loading setup.json at startup: {e}. Falling back to empty config.")
        config = {"camera": {"picamera_config": {"main": {"size": [640, 480]}}}} # Ensure config is a dict with defaults
        camera_settings = config["camera"]

_load_global_config_from_file() # Load it once at startup

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
    
    # Assicurati che picam_main_size sia una lista di due elementi
    picam_main_size = cfg_data_for_init.get("picamera_config", {}).get("main", {}).get("size", [640, 480])
    if not isinstance(picam_main_size, list) or len(picam_main_size) != 2:
        print(f"[WARN] picamera_config.main.size in setup.json is malformed: {picam_main_size}. Using default [640, 480].")
        picam_main_size = [640, 480]

    capture_width = picam_main_size[0]
    capture_height = picam_main_size[1]

    if sys.platform == "darwin":
        print("[INFO] Attempting macOS camera initialization...")
        mac_cam = cv2.VideoCapture(cfg_data_for_init.get("mac_camera_index", 0)) # Use configured index or default 0
        if not mac_cam.isOpened():
            print(f"[WARN] macOS camera {cfg_data_for_init.get('mac_camera_index', 0)} not open. Trying index 1 if different.")
            if cfg_data_for_init.get("mac_camera_index", 0) != 1: # Avoid trying index 1 twice
                mac_cam = cv2.VideoCapture(1)
            if not mac_cam.isOpened():
                print("[ERROR] No webcam available or in use on macOS.")
                return None
        
        # Set capture resolution for macOS camera (if supported)
        mac_cam.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
        mac_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)

        camera_instance = mac_cam
        print(f"[INFO] macOS camera initialized ({int(mac_cam.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(mac_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))}).")
    else:
        print("[INFO] Attempting Picamera2 initialization...")
        try:
            from picamera2 import Picamera2
            picam2 = Picamera2()
            
            # Use 'L8' for grayscale to reduce data if color isn't needed for raw capture.
            # Change to 'RGB888' if color is always required for the base frame.
            video_config = picam2.create_video_configuration(
                main={"size": (capture_width, capture_height), "format": "RGB888"} # Adjusted based on previous discussion
            )
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
    """
    Endpoint per deinizializzare la camera e rilasciare la risorsa.
    """
    global camera_instance
    with camera_lock:
        if camera_instance is not None:
            try:
                if sys.platform == "darwin":
                    camera_instance.release()
                else:
                    if hasattr(camera_instance, 'stop'):
                        camera_instance.stop()
                    if hasattr(camera_instance, 'close'):
                        camera_instance.close()
                camera_instance = None
                print("[INFO] Camera deinitialized and released.")
                return JsonResponse({"status": "success", "message": "Camera deinitialized and released."})
            except Exception as e:
                print(f"[ERROR] Error during camera deinitialization: {e}")
                return JsonResponse({"status": "error", "message": str(e)}, status=500)
        else:
            return JsonResponse({"status": "success", "message": "Camera was already released."})
        
def get_frame(release_after=False):
    global camera_instance
    with camera_lock:
        if camera_instance is None:
            print("get_frame: Camera not initialized. Attempting to initialize.")
            _initialize_camera_internally()
            if camera_instance is None:
                print("get_frame: Camera unavailable, returning blank frame.")
                # Usa le dimensioni configurate o un default ragionevole per il frame vuoto
                configured_height = camera_settings.get("picamera_config", {}).get("main", {}).get("size", [640, 480])[1]
                configured_width = camera_settings.get("picamera_config", {}).get("main", {}).get("size", [640, 480])[0]
                return np.zeros((configured_height, configured_width, 3), dtype=np.uint8)
        
        should_release_now = release_after and active_streams == 0
        frame = None
        try:
            if sys.platform == "darwin":
                if not camera_instance.isOpened():
                    print("get_frame (macOS): Camera not open. Re-initializing.")
                    _initialize_camera_internally()
                    if camera_instance is None or not camera_instance.isOpened():
                         raise IOError("macOS camera failed to open.")
                ret, frame = camera_instance.read()
                if not ret: raise IOError("macOS camera failed to read frame.")
            else: # Picamera2
                if not hasattr(camera_instance, 'capture_array'):
                    print("get_frame (Pi): Picamera2 not ready. Re-initializing.")
                    _initialize_camera_internally()
                    if camera_instance is None or not hasattr(camera_instance, 'capture_array'):
                        raise IOError("Picamera2 not ready or failed to initialize.")
                frame = camera_instance.capture_array()
        except Exception as e:
            print(f"get_frame: Error capturing frame: {e}. Returning blank frame.")
            if camera_instance is not None:
                try:
                    if sys.platform == "darwin": camera_instance.release()
                    else: camera_instance.stop(); camera_instance.close()
                except Exception as e_rel: print(f"Error releasing camera after capture error: {e_rel}")
                camera_instance = None
            
            configured_height = camera_settings.get("picamera_config", {}).get("main", {}).get("size", [640, 480])[1]
            configured_width = camera_settings.get("picamera_config", {}).get("main", {}).get("size", [640, 480])[0]
            return np.zeros((configured_height, configured_width, 3), dtype=np.uint8)

        if should_release_now: 
            try:
                if sys.platform == "darwin": camera_instance.release()
                else: camera_instance.stop(); camera_instance.close()
                camera_instance = None
                print("get_frame: Camera released after single capture.")
            except Exception as e:
                print(f"get_frame: Error releasing camera: {e}")
        return frame

# --- Utility ---
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

# MODIFICATO: detect_blobs_from_params ora accetta anche i fattori di scala
def detect_blobs_from_params(binary_image, blob_detection_params, scale_x=1.0, scale_y=1.0):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = blob_detection_params.get("areaFilter", True)
    
    # Adattamento dinamico di minArea e maxArea
    area_scale_factor = (scale_x * scale_y) # Scala bidimensionale per l'area
    
    params.minArea = blob_detection_params.get("minArea", 150) * area_scale_factor
    params.maxArea = blob_detection_params.get("maxArea", 5000) * area_scale_factor
    
    # Assicurati che i valori non siano negativi o troppo piccoli se la scala è estrema
    params.minArea = max(1, params.minArea) 
    params.maxArea = max(1, params.maxArea) # Puoi mettere un limite superiore sensato se necessario

    params.filterByCircularity = blob_detection_params.get("circularityFilter", True)
    params.minCircularity = blob_detection_params.get("minCircularity", 0.1) # Non scalato
    
    params.filterByConvexity = blob_detection_params.get("filterByConvexity", True)
    params.minConvexity = blob_detection_params.get("minConvexity", 0.87) # Non scalato
    
    params.filterByInertia = blob_detection_params.get("inertiaFilter", True)
    params.minInertiaRatio = blob_detection_params.get("minInertia", 0.01) # Non scalato
    
    detector = cv2.SimpleBlobDetector_create(params)
    return detector.detect(binary_image)

def get_current_frame_and_keypoints_from_config():
    frame = get_frame(release_after=False)
    if frame is None or frame.size == 0:
        print("get_current_frame_and_keypoints: Invalid frame received.")
        configured_height = camera_settings.get("picamera_config", {}).get("main", {}).get("size", [640, 480])[1]
        configured_width = camera_settings.get("picamera_config", {}).get("main", {}).get("size", [640, 480])[0]
        return np.zeros((configured_height, configured_width, 3), dtype=np.uint8), []
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # >>> Implementazione ridimensionamento per get_current_frame_and_keypoints_from_config <<<
    processing_width_for_single_shot = 640 # Risoluzione target per il rilevamento blob
    
    # Calcola processing_height mantenendo l'aspect ratio
    original_height, original_width = frame.shape[:2]
    processing_height_for_single_shot = int(original_height * (processing_width_for_single_shot / original_width))

    scale_x_for_single_shot = original_width / processing_width_for_single_shot
    scale_y_for_single_shot = original_height / processing_height_for_single_shot

    resized_gray = cv2.resize(gray, (processing_width_for_single_shot, processing_height_for_single_shot), interpolation=cv2.INTER_AREA)

    _, thresh = cv2.threshold(
        resized_gray, # Usa l'immagine ridimensionata
        camera_settings.get("minThreshold", 127),
        camera_settings.get("maxThreshold", 255),
        cv2.THRESH_BINARY
    )
    
    keypoints_resized = detect_blobs_from_params(thresh, camera_settings, scale_x_for_single_shot, scale_y_for_single_shot)
    
    # Riscala i keypoint all'immagine originale
    keypoints_original_coords = []
    for kp in keypoints_resized:
        new_x = kp.pt[0] * scale_x_for_single_shot
        new_y = kp.pt[1] * scale_y_for_single_shot
        new_size = kp.size * ((scale_x_for_single_shot + scale_y_for_single_shot) / 2) # Scala la dimensione
        keypoints_original_coords.append(cv2.KeyPoint(new_x, new_y, new_size, kp.angle, kp.response, kp.octave, kp.class_id))
    # >>> Fine implementazione ridimensionamento <<<

    return frame, keypoints_original_coords

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

def get_current_motor_speeds():
    try:
        resp = requests.get("http://localhost:8000/motors/maxSpeeds/")
        data = resp.json()
        if data.get("status") == "success":
            return data["speeds"]
    except Exception as e:
        print(f"Errore richiesta velocità motori: {e}")
    return {"extruder": 4.0, "conveyor": 1.0}  # fallback


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
        # Optional: Conditional camera release if no streams are active
        # if active_streams == 0 and camera_instance is not None:
        #     print("[STREAM] Last active stream closed. Considering camera release.")
        #     with camera_lock:
        #         if camera_instance is not None:
        #             try:
        #                 if sys.platform == "darwin": camera_instance.release()
        #                 else: camera_instance.stop(); camera_instance.close()
        #                 camera_instance = None
        #                 print("[STREAM] Camera released after all streams closed.")
        #             except Exception as e:
        #                 print(f"[WARN] Error releasing camera after stream context: {e}")

def get_world_coordinates_data():
    H_fixed_ref = get_fixed_perspective_homography_from_config()
    if H_fixed_ref is None:
        return {"status": "error", "message": "Fixed perspective homography not available. Please set it first via its endpoint."}
    cam_calib_wc = camera_settings.get("calibration", None)
    if not (cam_calib_wc and cam_calib_wc.get("camera_matrix") and cam_calib_wc.get("distortion_coefficients")):
        return {"status": "error", "message": "Camera calibration data missing. Please calibrate the camera first."}
    cam_matrix_wc = np.array(cam_calib_wc["camera_matrix"], dtype=np.float32)
    dist_coeffs_wc = np.array(cam_calib_wc["distortion_coefficients"], dtype=np.float32)
    frame_for_coords, keypoints_for_coords_original_coords = get_current_frame_and_keypoints_from_config()
    if frame_for_coords is None or frame_for_coords.size == 0:
         return {"status": "error", "message": "Could not get frame for coordinates calculation."}
    if not keypoints_for_coords_original_coords:
        return {"status": "success", "coordinates": []}
    img_pts_original_coords = np.array([kp.pt for kp in keypoints_for_coords_original_coords], dtype=np.float32).reshape(-1,1,2)
    h_frame_wc, w_frame_wc = frame_for_coords.shape[:2]
    new_cam_matrix_wc, _ = cv2.getOptimalNewCameraMatrix(cam_matrix_wc, dist_coeffs_wc, (w_frame_wc,h_frame_wc), 1.0, (w_frame_wc,h_frame_wc))
    img_pts_undistorted_remapped = cv2.undistortPoints(img_pts_original_coords, cam_matrix_wc, dist_coeffs_wc, P=new_cam_matrix_wc)
    if img_pts_undistorted_remapped is None:
        return {"status": "error", "message": "Point undistortion failed unexpectedly."}
    world_coords_top_left_origin = []
    if img_pts_undistorted_remapped.size > 0:
        transformed_pts = cv2.perspectiveTransform(img_pts_undistorted_remapped, H_fixed_ref)
        if transformed_pts is not None:
             world_coords_top_left_origin = transformed_pts.reshape(-1, 2).tolist()
        else:
            return {"status": "error", "message": "Perspective transformation of points failed."}
    fixed_persp_cfg = camera_settings.get("fixed_perspective", {})
    OUTPUT_WIDTH = fixed_persp_cfg.get("output_width", 1000)
    OUTPUT_HEIGHT = fixed_persp_cfg.get("output_height", 800)
    world_coords_bottom_right_origin = []
    for x_tl, y_tl in world_coords_top_left_origin:
        x_br = OUTPUT_WIDTH - x_tl
        y_br = OUTPUT_HEIGHT - y_tl
        world_coords_bottom_right_origin.append([x_br, y_br])
    return {"status": "success", "coordinates": world_coords_bottom_right_origin}

def get_graph_and_tsp_path(velocita_x=4.0, velocita_y=1.0):
    response = get_world_coordinates_data()
    if response.get("status", []) != "success" and response.get("status") != "success":
        return None, None, response
    coordinates = response.get("coordinates", [])
    origin_x = camera_settings.get("origin_x", 0.0)
    origin_y = camera_settings.get("origin_y", 0.0)
    origin_coord = [origin_x, origin_y]
    coordinates_with_origin = [origin_coord] + coordinates

    filtered_coords = []
    for coord in coordinates_with_origin:
        x_rel = coord[0] - origin_x
        if 5 <= x_rel <= 250:
            filtered_coords.append(coord)
    nodi = [tuple(coord) for coord in filtered_coords]

    if len(nodi) < 2:
        return None, None, {"status": "error", "message": "Nessun punto da plottare."}
    graph = construct_graph(nodi, velocita_x, velocita_y)
    source = 0
    hamiltonian_path = nx.algorithms.approximation.traveling_salesman_problem(
        graph, cycle=False, method=nx.algorithms.approximation.greedy_tsp, source=source
    )
    return graph, hamiltonian_path, {"status": "success", "nodi": nodi}

# --- Django Endpoints ---
@csrf_exempt
@require_POST
def update_camera_settings(request):
    try:
        data = json.loads(request.body)
        current_disk_config = load_config_data_from_file()
        
        if "camera" not in current_disk_config:
            current_disk_config["camera"] = {}
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
def camera_feed(request):
    mode = request.GET.get("mode", "normal")
    
    def gen_frames():
        stream_cfg_cam = camera_settings
        
        H_ref = None
        cam_matrix = None
        dist_coeffs = None
        new_cam_matrix_stream = None
        OUT_W = 0
        OUT_H = 0
        blob_params_for_stream = stream_cfg_cam

        blob_detection_interval = stream_cfg_cam.get("blob_detection_interval", 5) # Default a 5 frame
        frame_count = 0
        last_keypoints_for_drawing = [] # Per memorizzare gli ultimi keypoint rilevati (per normal/threshold mode)

        # Dimensioni target per il ridimensionamento dell'immagine per la rilevazione blob
        # Puoi rendere queste configurabili in setup.json sotto "camera"
        blob_processing_width = stream_cfg_cam.get("blob_processing_width", 640)
        
        if mode == "fixed":
            H_ref = get_fixed_perspective_homography_from_config()
            cam_calib = stream_cfg_cam.get("calibration", None)
            fixed_persp_cfg = stream_cfg_cam.get("fixed_perspective", {})
            
            OUT_W = fixed_persp_cfg.get("output_width", 1000)
            OUT_H = fixed_persp_cfg.get("output_height", 800)
            
            if not (cam_calib and cam_calib.get("camera_matrix") and cam_calib.get("distortion_coefficients")):
                error_msg = "Camera calibration missing for fixed view"
                print(f"camera_feed (fixed mode): {error_msg}")
            else:
                cam_matrix = np.array(cam_calib["camera_matrix"], dtype=np.float32)
                dist_coeffs = np.array(cam_calib["distortion_coefficients"], dtype=np.float32)
                
                try:
                    sample_frame_for_dims = get_frame(release_after=False) 
                    if sample_frame_for_dims is not None and sample_frame_for_dims.size > 0:
                        h_str, w_str = sample_frame_for_dims.shape[:2]
                        new_cam_matrix_stream, _ = cv2.getOptimalNewCameraMatrix(cam_matrix, dist_coeffs, (w_str,h_str), 1.0, (w_str,h_str))
                    else:
                        raise ValueError("Could not get valid sample frame dimensions for new_camera_matrix in fixed mode.")
                except Exception as e_stream_setup:
                    print(f"camera_feed (fixed mode) setup failed: {e_stream_setup}")
        
        with stream_context():
            while True:
                try:
                    frame_orig = get_frame()
                    if frame_orig is None or frame_orig.size == 0:
                        current_height = stream_cfg_cam.get("picamera_config", {}).get("main", {}).get("size", [640, 480])[1]
                        current_width = stream_cfg_cam.get("picamera_config", {}).get("main", {}).get("size", [640, 480])[0]
                        if mode == "fixed":
                            current_height = OUT_H if OUT_H > 0 else 480
                            current_width = OUT_W if OUT_W > 0 else 640

                        blank_frame = np.zeros((current_height, current_width, 3), dtype=np.uint8)
                        cv2.putText(blank_frame, "No Frame", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
                        if mode == "fixed" and H_ref is None:
                             cv2.putText(blank_frame, "Fixed View Not Set", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),1)
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                        _, buffer = cv2.imencode('.jpg', blank_frame, encode_param)
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        time.sleep(0.1)
                        continue
                    
                    display_frame_feed = frame_orig.copy()
                    
                    # LOGICA BLOB DETECTION A INTERVALLI
                    if frame_count % blob_detection_interval == 0:
                        original_height, original_width = frame_orig.shape[:2]
                        # Calcola processing_height mantenendo l'aspect ratio
                        blob_processing_height = int(original_height * (blob_processing_width / original_width))
                        
                        scale_x = original_width / blob_processing_width
                        scale_y = original_height / blob_processing_height

                        gray_for_blobs = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
                        resized_gray_for_blobs = cv2.resize(gray_for_blobs, (blob_processing_width, blob_processing_height), interpolation=cv2.INTER_AREA)

                        _, processed_for_blobs = cv2.threshold(
                            resized_gray_for_blobs,
                            stream_cfg_cam.get("minThreshold", 127),
                            stream_cfg_cam.get("maxThreshold", 255),
                            cv2.THRESH_BINARY
                        )
                        keypoints_resized = detect_blobs_from_params(processed_for_blobs, blob_params_for_stream, scale_x, scale_y)
                        
                        # Riscala i keypoint all'immagine originale
                        last_keypoints_for_drawing = []
                        for kp in keypoints_resized:
                            new_x = kp.pt[0] * scale_x
                            new_y = kp.pt[1] * scale_y
                            new_size = kp.size * ((scale_x + scale_y) / 2) # Scala la dimensione
                            last_keypoints_for_drawing.append(cv2.KeyPoint(new_x, new_y, new_size, kp.angle, kp.response, kp.octave, kp.class_id))
                    
                    frame_count += 1
                    
                    if mode == "normal" or mode == "threshold":
                        if mode == "threshold":
                             # Per visualizzare la threshold sull'immagine originale, rifacciamo il threshold su frame_orig
                            gray_display = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
                            _, processed_for_blobs_display = cv2.threshold(
                                gray_display, stream_cfg_cam.get("minThreshold", 127),
                                stream_cfg_cam.get("maxThreshold", 255), cv2.THRESH_BINARY
                            )
                            display_frame_feed = cv2.cvtColor(processed_for_blobs_display, cv2.COLOR_GRAY2BGR)

                        frame_with_keypoints = cv2.drawKeypoints(
                            display_frame_feed, last_keypoints_for_drawing, np.array([]), (0, 0, 255), 
                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                        
                        # NUOVO: Stampa l'area (o dimensione) dei blob
                        for kp in last_keypoints_for_drawing:
                            x, y = int(kp.pt[0]), int(kp.pt[1])
                            estimated_area = np.pi * (kp.size / 2) ** 2
                            text_to_display = f"D:{kp.size:.1f} A:{estimated_area:.0f}" 
                            cv2.putText(frame_with_keypoints, text_to_display, 
                                        (x + 15, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1) # Blu
                        # FINE NUOVO

                        _, buffer = cv2.imencode('.jpg', frame_with_keypoints, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                    elif mode == "fixed":
                        output_img = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
                        
                        if H_ref is not None and cam_matrix is not None and dist_coeffs is not None and new_cam_matrix_stream is not None:
                            undistorted_live = cv2.undistort(frame_orig, cam_matrix, dist_coeffs, None, new_cam_matrix_stream)
                            output_img = cv2.warpPerspective(undistorted_live, H_ref, (OUT_W, OUT_H))
                            
                            if last_keypoints_for_drawing:
                                # Questi keypoint sono nelle coordinate originali (distorte).
                                # Dobbiamo prima undistortarli e poi trasformarli con H_ref.
                                
                                # Converti i keypoint in un array di punti per undistortPoints e perspectiveTransform
                                pts_original_coords = np.array([kp.pt for kp in last_keypoints_for_drawing], dtype=np.float32).reshape(-1,1,2)

                                # UndistortPoints i keypoint nella stessa prospettiva di new_cam_matrix_stream
                                pts_undistorted_remapped_for_display = cv2.undistortPoints(
                                    pts_original_coords, cam_matrix, dist_coeffs, P=new_cam_matrix_stream
                                )

                                if pts_undistorted_remapped_for_display is not None:
                                    pts_warped = cv2.perspectiveTransform(pts_undistorted_remapped_for_display, H_ref)
                                    
                                    if pts_warped is not None:
                                        for i, pt_w in enumerate(pts_warped.reshape(-1,2)):
                                            kp_original = last_keypoints_for_drawing[i] # Ottieni il keypoint originale per la sua dimensione
                                            x, y = pt_w[0], pt_w[1]
                                            cv2.circle(output_img, (int(round(x)), int(round(y))), 8, (0,0,255), 2)
                                            cv2.putText(
                                                output_img, f"{x:.1f},{y:.1f}",
                                                (int(round(x))+10, int(round(y))-10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1
                                            )
                                            # NUOVO: Stampa l'area (o dimensione) dei blob nel fixed mode
                                            estimated_area = np.pi * (kp_original.size / 2) ** 2
                                            text_to_display_fixed = f"D:{kp_original.size:.1f} A:{estimated_area:.0f}"
                                            cv2.putText(output_img, text_to_display_fixed,
                                                        (int(round(x)) + 15, int(round(y)) + 5), # Posizione leggermente diversa
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1) # Giallo-verde
                                            # FINE NUOVO

                        else: # H_ref is None
                            cv2.putText(output_img, "Fixed View Not Set", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),1)
                            cv2.putText(output_img, "Set via endpoint", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),1)
                        
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                        _, buffer_ok = cv2.imencode('.jpg', output_img, encode_param)
                        frame_bytes_ok = buffer_ok.tobytes()
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes_ok + b'\r\n')
                    
                    # time.sleep(0.03) # Commentato per massimizzare il framerate, se rallenta decommenta con valore > 0.03
                except Exception as e:
                    print(f"Error in camera_feed loop (mode: {mode}): {e}")
                    traceback.print_exc()
                    
                    current_height = stream_cfg_cam.get("picamera_config", {}).get("main", {}).get("size", [640, 480])[1]
                    current_width = stream_cfg_cam.get("picamera_config", {}).get("main", {}).get("size", [640, 480])[0]
                    if mode == "fixed":
                        current_height = OUT_H if OUT_H > 0 else 480
                        current_width = OUT_W if OUT_W > 0 else 640

                    error_frame = np.zeros((current_height, current_width, 3), dtype=np.uint8)
                    cv2.putText(error_frame, f"Stream Err: {str(e)[:50]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    _, buffer = cv2.imencode('.jpg', error_frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    time.sleep(1)
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

@csrf_exempt
@require_POST
def reset_camera_calibration(request):
    try:
        current_disk_config = load_config_data_from_file()

        if "camera" not in current_disk_config:
            current_disk_config["camera"] = {}
        
        # Ensure 'calibration' and 'fixed_perspective' keys exist as dicts if they don't
        current_disk_config["camera"].setdefault("calibration", {})
        current_disk_config["camera"].setdefault("fixed_perspective", {})

        current_disk_config["camera"]["calibration"]["camera_matrix"] = None
        current_disk_config["camera"]["calibration"]["distortion_coefficients"] = None
        current_disk_config["camera"]["fixed_perspective"]["homography_matrix"] = None
        
        if save_config_data_to_file(current_disk_config):
            return JsonResponse({
                "status": "success",
                "message": "Camera calibration and fixed perspective reset successfully."
            })
        else:
            return JsonResponse({
                "status": "error",
                "message": "Failed to save the reset configuration to file."
            }, status=500)

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
        
        rect_vertices = []
        if len(keypoints_data) >= 3: 
            pts_rect = np.array(keypoints_list, dtype=np.float32)
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
            
            if cv2.pointPolygonTest(corners.reshape(-1,1,2).astype(np.int32), tuple(pts_para[0]), False) >= 0:
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
def set_camera_origin(request):
    try:
        data = json.loads(request.body)
        x_val = float(data.get("origin_x", 0.0))
        y_val = float(data.get("origin_y", 0.0))
        
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
        frame_to_save = get_frame(release_after=False)
        if frame_to_save is None or frame_to_save.size == 0:
            return JsonResponse({"status": "error", "message": "Invalid frame received from camera."}, status=500)
        
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
    current_disk_config = load_config_data_from_file()
    calib_settings = current_disk_config.get("camera", {}).get("calibration_settings", {})
    
    cs_cols = calib_settings.get("chessboard_cols", 9)
    cs_rows = calib_settings.get("chessboard_rows", 7)
    square_size_mm = calib_settings.get("square_size_mm", 15.0)
    chessboard_dim_config = (cs_cols, cs_rows)
    criteria_config = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp_config = np.zeros((cs_cols * cs_rows, 3), np.float32)
    objp_config[:, :2] = np.mgrid[0:cs_cols, 0:cs_rows].T.reshape(-1, 2)
    objp_config *= square_size_mm
    objpoints_list = []
    imgpoints_list = []
    image_files = glob.glob(os.path.join(CALIBRATION_MEDIA_DIR, '*.jpg'))
    print(f"Found {len(image_files)} images for calibration in {CALIBRATION_MEDIA_DIR}.")
    if not image_files:
        return JsonResponse({"status": "error", "message": f"No .jpg images found in {CALIBRATION_MEDIA_DIR}."}, status=400)
    
    last_gray_shape = None
    images_processed_count = 0
    for image_path in image_files:
        img = cv2.imread(image_path)
        if img is None: 
            print(f"Could not read image: {image_path}")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if images_processed_count == 0:
            last_gray_shape = gray.shape[::-1]
        elif last_gray_shape != gray.shape[::-1]:
             print(f"WARNING: Image {image_path} dimensions ({gray.shape[::-1]}) differ from first image ({last_gray_shape}).")
        
        images_processed_count +=1
        ret_corners, corners = cv2.findChessboardCorners(gray, chessboard_dim_config, None)
        if ret_corners:
            print(f"Chessboard found in: {image_path}")
            objpoints_list.append(objp_config)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria_config)
            imgpoints_list.append(corners2)
        else:
            print(f"Chessboard NOT found in: {image_path}")
    
    if not objpoints_list or not imgpoints_list:
        return JsonResponse({"status": "error", "message": "No valid chessboard points found in provided images."}, status=400)
    if last_gray_shape is None:
         return JsonResponse({"status": "error", "message": "No images processed, cannot determine dimensions for calibration."}, status=400)
    print(f"Calculating calibration parameters using {len(objpoints_list)} sets of points. Image dimensions: {last_gray_shape}")
    ret_calib, camera_matrix_calib, dist_coeffs_calib, _, _ = cv2.calibrateCamera(
        objpoints_list, imgpoints_list, last_gray_shape, None, None
    )
    if ret_calib:
        calibration_data_tosave = {
            "camera_matrix": camera_matrix_calib.tolist(),
            "distortion_coefficients": dist_coeffs_calib.tolist()
        }
        current_disk_config.setdefault("camera", {}).setdefault("calibration", {})
        current_disk_config["camera"]["calibration"] = calibration_data_tosave
        
        if save_config_data_to_file(current_disk_config):
            return JsonResponse({"status": "success", "message": "Calibration completed and saved.", "calibration": calibration_data_tosave})
        else:
            return JsonResponse({"status": "error", "message": "Calibration completed but failed to save configuration."}, status=500)
    else:
        return JsonResponse({"status": "error", "message": "cv2.calibrateCamera failed."}, status=500)

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

@csrf_exempt
@require_GET
def get_world_coordinates(request):
    data = get_world_coordinates_data()
    status_code = 200 if data.get("status") == "success" else 400
    return JsonResponse(data, status=status_code)

@csrf_exempt
@require_GET
def compute_route(request):
    # 1. Ottieni le velocità attuali dei motori
    try:
        resp = requests.get("http://localhost:8000/motors/maxSpeeds/")
        data = resp.json()
        if data.get("status") == "success":
            velocita_x = data["speeds"].get("extruder", 4.0)
            velocita_y = data["speeds"].get("conveyor", 1.0)
        else:
            velocita_x = 4.0
            velocita_y = 1.0
    except Exception as e:
        print(f"Errore richiesta velocità motori: {e}")
        velocita_x = 4.0
        velocita_y = 1.0

    # 2. Calcola il grafo e il percorso usando le velocità ottenute
    graph, hamiltonian_path, info = get_graph_and_tsp_path_with_speeds(velocita_x, velocita_y)
    if graph is None or hamiltonian_path is None:
        return JsonResponse({"status": "error", "message": info.get("message", "Errore generico")}, status=500)
    nodi = info["nodi"]
    origin_x = camera_settings.get("origin_x", 0.0)
    origin_y = camera_settings.get("origin_y", 0.0)
    motor_commands = []
    for i, idx in enumerate(hamiltonian_path):
        if i == 0:
            extruder_mm = nodi[idx][0] - origin_x
            conveyor_mm = nodi[idx][1] - origin_y
        else:
            extruder_mm = nodi[idx][0] - nodi[hamiltonian_path[i-1]][0]
            conveyor_mm = nodi[idx][1] - nodi[hamiltonian_path[i-1]][1]
        motor_commands.append({"extruder": extruder_mm, "conveyor": conveyor_mm})

    # --- Genera il plot come immagine base64 ---
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

# Funzione di supporto per passare le velocità a construct_graph
def get_graph_and_tsp_path_with_speeds(velocita_x=4.0, velocita_y=1.0):
    response = get_world_coordinates_data()
    if response.get("status", []) != "success" and response.get("status") != "success":
        return None, None, response
    coordinates = response.get("coordinates", [])
    origin_x = camera_settings.get("origin_x", 0.0)
    origin_y = camera_settings.get("origin_y", 0.0)
    origin_coord = [origin_x, origin_y]
    coordinates_with_origin = [origin_coord] + coordinates

    filtered_coords = []
    for coord in coordinates_with_origin:
        x_rel = coord[0] - origin_x
        if 5 <= x_rel <= 250:
            filtered_coords.append(coord)
    nodi = [tuple(coord) for coord in filtered_coords]

    if len(nodi) < 2:
        return None, None, {"status": "error", "message": "Nessun punto da plottare."}
    graph = construct_graph(nodi, velocita_x, velocita_y)
    source = 0
    hamiltonian_path = nx.algorithms.approximation.traveling_salesman_problem(
        graph, cycle=False, method=nx.algorithms.approximation.greedy_tsp, source=source
    )
    return graph, hamiltonian_path, {"status": "success", "nodi": nodi}


@csrf_exempt
@require_GET
def plot_graph(request):
    try:
        # 1. Ottieni le velocità attuali dei motori
        try:
            resp = requests.get("http://localhost:8000/motors/maxSpeeds/")
            data = resp.json()
            if data.get("status") == "success":
                velocita_x = data["speeds"].get("extruder", 4.0)
                velocita_y = data["speeds"].get("conveyor", 1.0)
            else:
                velocita_x = 4.0
                velocita_y = 1.0
        except Exception as e:
            print(f"Errore richiesta velocità motori: {e}")
            velocita_x = 4.0
            velocita_y = 1.0
        graph, hamiltonian_path, info = get_graph_and_tsp_path(velocita_x, velocita_y)
        if graph is None or hamiltonian_path is None:
            return HttpResponse("Errore: " + info.get("message", "Unknown error"), status=500)
        nodi = info["nodi"]
        plt.figure(figsize=(8, 6))
        pos = nx.get_node_attributes(graph, 'pos')
        if not pos:
            return HttpResponse("Nessuna posizione nodo trovata.", status=400)
        # Disegna i nodi
        nx.draw_networkx_nodes(graph, pos, node_color='skyblue', node_size=500)
        nx.draw_networkx_labels(graph, pos, font_size=10)
        # Disegna solo il percorso TSP in rosso
        tsp_edges = [(hamiltonian_path[i], hamiltonian_path[i+1]) for i in range(len(hamiltonian_path)-1)]
        nx.draw_networkx_edges(graph, pos, edgelist=tsp_edges, edge_color='red', width=2)
        plt.title("Percorso TSP (in rosso)")
        plt.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return HttpResponse(buf.getvalue(), content_type='image/png')
    except Exception as e:
        import traceback
        print("Errore in plot_graph:", e)
        traceback.print_exc()
        return HttpResponse("Errore interno: " + str(e), status=500)
    
def construct_graph(nodi, velocita_x=4.0, velocita_y=1.0):
    G = nx.Graph()
    for i in range(len(nodi)):
        G.add_node(i, pos=nodi[i])
    for i in range(len(nodi)):
        for j in range(i+1, len(nodi)):
            dx = nodi[i][0] - nodi[j][0]
            dy = nodi[i][1] - nodi[j][1]
            # Calcola il tempo minimo considerando le velocità
            tempo = max(abs(dx)/velocita_x, abs(dy)/velocita_y)
            G.add_edge(i, j, weight=round(tempo, 4))
    return G
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
import traceback # Added for more detailed error logging if needed
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
# These globals will be updated by save_config_data
config = {}
camera_settings = {}
def _load_global_config_from_file():
    global config, camera_settings
    try:
        with open(SETUP_JSON_PATH, 'r') as f:
            config = json.load(f)
        camera_settings = config.get("camera", {})
        print("[INFO] Global configuration loaded successfully.")
    except Exception as e:
        print(f"Critical error loading setup.json at startup: {e}. Falling back to empty config.")
        config = {"camera": {}} # Ensure config is a dict
        camera_settings = {}
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
    # Use the global camera_settings loaded at startup or after save
    cfg_data_for_init = camera_settings
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
        camera_instance = mac_cam
        print("[INFO] macOS camera initialized.")
    else:
        print("[INFO] Attempting Picamera2 initialization...")
        try:
            from picamera2 import Picamera2
            picam2 = Picamera2()
            picam_main_size = cfg_data_for_init.get("picamera_config", {}).get("main", {}).get("size", [640, 480])
            capture_width = picam_main_size[0]
            capture_height = picam_main_size[1]
            video_config = picam2.create_video_configuration(
                main={"size": (capture_width, capture_height), "format": "RGB888"}
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
# Inizializzazione non aggressiva all'avvio, l'endpoint o il primo get_frame la forzeranno se necessario
# print("[INFO] Attempting to initialize camera at module startup...")
# _initialize_camera_internally() # Commented out: prefer explicit init or lazy init
def get_frame(release_after=False):
    global camera_instance # active_streams is handled by stream_context or implicitly for singl e calls
    with camera_lock:
        if camera_instance is None:
            print("get_frame: Camera not initialized. Attempting to initialize.")
            _initialize_camera_internally()
            if camera_instance is None:
                print("get_frame: Camera unavailable, returning blank frame.")
                return np.zeros((camera_settings.get("capture_height", 480),
                                 camera_settings.get("capture_width", 640), 3), dtype=np.uint8)
        # Determine if release is needed: only if release_after is true AND no active streams
        # active_streams count is primarily for StreamingHttpResponse. Single calls don't increment it.
        # So, if release_after is true, it's usually a single-shot call.
        should_release_now = release_after and active_streams == 0
        frame = None
        try:
            if sys.platform == "darwin":
                if not camera_instance.isOpened():
                    print("get_frame (macOS): Camera not open. Re-initializing.")
                    _initialize_camera_internally() # Try to reopen
                    if camera_instance is None or not camera_instance.isOpened():
                         raise IOError("macOS camera failed to open.")
                ret, frame = camera_instance.read()
                if not ret: raise IOError("macOS camera failed to read frame.")
            else: # Picamera2
                if not hasattr(camera_instance, 'capture_array'): # Basic check if instance is valid
                    print("get_frame (Pi): Picamera2 not ready. Re-initializing.")
                    _initialize_camera_internally() # Try to re-init
                    if camera_instance is None or not hasattr(camera_instance, 'capture_array'):
                        raise IOError("Picamera2 not ready or failed to initialize.")
                frame = camera_instance.capture_array()
        except Exception as e:
            print(f"get_frame: Error capturing frame: {e}. Returning blank frame.")
            # Attempt to release the problematic camera instance before returning blank
            if camera_instance is not None:
                try:
                    if sys.platform == "darwin": camera_instance.release()
                    else: camera_instance.stop(); camera_instance.close()
                except Exception as e_rel: print(f"Error releasing camera after capture error: {e_rel}")
                camera_instance = None
            return np.zeros((camera_settings.get("capture_height", 480),
                             camera_settings.get("capture_width", 640), 3), dtype=np.uint8)
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
def load_config_data_from_file(): # Renamed to be explicit about file I/O
    try:
        with open(SETUP_JSON_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {SETUP_JSON_PATH}: {e}")
        return {"camera": {}} # Return a minimal valid structure
    
def save_config_data_to_file(new_config_data): # Renamed
    try:
        with open(SETUP_JSON_PATH, 'w') as f:
            json.dump(new_config_data, f, indent=4)
        # Update global config variables after successful save
        _load_global_config_from_file()
        print(f"Configuration saved to {SETUP_JSON_PATH} and globals reloaded.")
        return True
    except Exception as e:
        print(f"Error saving to {SETUP_JSON_PATH}: {e}")
        return False
def get_fixed_perspective_homography_from_config(): # Reads from global config
    H_list = camera_settings.get("fixed_perspective", {}).get("homography_matrix", None)
    if H_list and isinstance(H_list, list):
        try:
            return np.array(H_list, dtype=np.float32)
        except Exception as e:
            print(f"Error converting fixed_perspective_homography to numpy array: {e}")
    return None
def save_fixed_perspective_homography_to_config(H_matrix_ref): # Modifies and saves
    current_disk_config = load_config_data_from_file() # Load fresh from disk before modifying
    current_disk_config.setdefault("camera", {}).setdefault("fixed_perspective", {})
    if H_matrix_ref is not None:
        current_disk_config["camera"]["fixed_perspective"]["homography_matrix"] = H_matrix_ref.tolist()
    else:
        current_disk_config["camera"]["fixed_perspective"]["homography_matrix"] = None
    return save_config_data_to_file(current_disk_config)
def detect_blobs_from_params(binary_image, blob_detection_params): # Takes params
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = blob_detection_params.get("areaFilter", True)
    params.minArea = blob_detection_params.get("minArea", 150)
    params.maxArea = blob_detection_params.get("maxArea", 5000)
    params.filterByCircularity = blob_detection_params.get("circularityFilter", True)
    params.minCircularity = blob_detection_params.get("minCircularity", 0.1)
    params.filterByConvexity = blob_detection_params.get("filterByConvexity", True) # Corrected param name from original
    params.minConvexity = blob_detection_params.get("minConvexity", 0.87)
    params.filterByInertia = blob_detection_params.get("inertiaFilter", True)
    params.minInertiaRatio = blob_detection_params.get("minInertia", 0.01) # Corrected param name from original
    detector = cv2.SimpleBlobDetector_create(params)
    return detector.detect(binary_image)
def get_current_frame_and_keypoints_from_config(): # Uses global config
    # Uses global camera_settings for thresholds and blob params
    frame = get_frame(release_after=True) # Single acquisition
    if frame is None or frame.size == 0:
        print("get_current_frame_and_keypoints: Invalid frame received.")
        return np.zeros((camera_settings.get("capture_height", 480),
                         camera_settings.get("capture_width", 640), 3), dtype=np.uint8), []
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray,
        camera_settings.get("minThreshold", 127),
        camera_settings.get("maxThreshold", 255),
        cv2.THRESH_BINARY
    )
    # Pass camera_settings directly as it contains blob parameters
    keypoints = detect_blobs_from_params(thresh, camera_settings)
    return frame, keypoints
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
    # Use cv2.solvePnP without flags if not needed, or specify cv2.SOLVEPNP_ITERATIVE
    success, rvec, tvec = cv2.solvePnP(objp_cv, corners2, new_camera_matrix_cv, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success: return None, None
    board_w_obj = (cs_cols) * sq_size # Use full columns for width calculation for perimeter
    board_h_obj = (cs_rows) * sq_size # Use full rows for height 
    
    # Define object points for the board perimeter (outer corners of the squares)
    # These points define the rectangle encompassing the entire grid in 3D object space.
    obj_board_perimeter_pts = np.array([
        [0,0,0], [ (cs_cols-1)*sq_size, 0, 0], 
        [(cs_cols-1)*sq_size, (cs_rows-1)*sq_size, 0], [0, (cs_rows-1)*sq_size, 0]
    ], dtype=np.float32)
    img_board_perimeter_pts, _ = cv2.projectPoints(obj_board_perimeter_pts, rvec, tvec, new_camera_matrix_cv, None)
    img_board_perimeter_pts = img_board_perimeter_pts.reshape(-1, 2)
    # Destination points for the canonical view should match the object dimensions
    # Using cs_cols * sq_size implies pixel per mm if sq_size is in mm.
    # The canonical board size should represent the "real" dimensions.
    canonical_width = int(round((cs_cols-1) * sq_size)) # Width of the pattern area
    canonical_height = int(round((cs_rows-1) * sq_size)) # Height of the pattern area
    canonical_dst_pts = np.array([
        [0,0], [canonical_width-1,0], 
        [canonical_width-1,canonical_height-1], [0,canonical_height-1]
    ], dtype=np.float32)
    
    H_canonical = cv2.getPerspectiveTransform(img_board_perimeter_pts, canonical_dst_pts)
    # The canonical_board_size should be the size of the image you're warping *to*
    # This should be the size of the pattern area, not the full board width/height from earlier.
    canonical_board_size_tuple = (canonical_width, canonical_height) 
    
    return H_canonical, canonical_board_size_tuple
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
        # Consider if get_frame's release_after is sufficient or if explicit release here is desired
        # with camera_lock:
        #     if active_streams == 0 and camera_instance is not None:
        #         print("[STREAM] Last active stream closed. Releasing camera.")
        #         # Similar release logic as in get_frame
        #         # This part needs careful thought to avoid conflicts with get_frame's own release logic
        #         pass
# --- Django Endpoints ---
@csrf_exempt
@require_POST
def update_camera_settings(request):
    try:
        data = json.loads(request.body)
        current_disk_config = load_config_data_from_file() # Load fresh for update
        
        # Ensure "camera" key exists
        if "camera" not in current_disk_config:
            current_disk_config["camera"] = {}
        for key, value in data.items():
            if isinstance(value, dict) and key in current_disk_config["camera"] and isinstance(current_disk_config["camera"][key], dict):
                current_disk_config["camera"][key].update(value)
            else:
                current_disk_config["camera"][key] = value
        
        if save_config_data_to_file(current_disk_config):
            # Global 'camera_settings' is updated by save_config_data_to_file via _load_global_config_from_file
            return JsonResponse({"status": "success", "updated_settings": camera_settings})
        else:
            return JsonResponse({"status": "error", "message": "Failed to save updated settings."}, status=500)
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=400)
@csrf_exempt
def camera_feed(request):
    mode = request.GET.get("mode", "normal")
    
    def gen_frames():
        # Load config for this stream worker once
        stream_cfg_cam = camera_settings
        
        # Fixed perspective stream initialization
        H_ref = None
        cam_matrix = None
        dist_coeffs = None
        new_cam_matrix_stream = None
        OUT_W = 0
        OUT_H = 0
        blob_params_for_stream = stream_cfg_cam

        if mode == "fixed":
            H_ref = get_fixed_perspective_homography_from_config()
            cam_calib = stream_cfg_cam.get("calibration", None)
            fixed_persp_cfg = stream_cfg_cam.get("fixed_perspective", {})
            
            OUT_W = fixed_persp_cfg.get("output_width", 1000)
            OUT_H = fixed_persp_cfg.get("output_height", 800)
            
            if not (cam_calib and cam_calib.get("camera_matrix") and cam_calib.get("distortion_coefficients")):
                error_msg = "Camera calibration missing for fixed view"
                print(f"camera_feed (fixed mode): {error_msg}")
                # This error will be handled by the outer loop with a blank frame
                # and error text.
            else:
                cam_matrix = np.array(cam_calib["camera_matrix"], dtype=np.float32)
                dist_coeffs = np.array(cam_calib["distortion_coefficients"], dtype=np.float32)
                
                try:
                    sample_frame_for_dims = get_frame(release_after=True) 
                    if sample_frame_for_dims is not None and sample_frame_for_dims.size > 0:
                        h_str, w_str = sample_frame_for_dims.shape[:2]
                        new_cam_matrix_stream, _ = cv2.getOptimalNewCameraMatrix(cam_matrix, dist_coeffs, (w_str,h_str), 1.0, (w_str,h_str))
                    else:
                        raise ValueError("Could not get valid sample frame dimensions for new_camera_matrix in fixed mode.")
                except Exception as e_stream_setup:
                    print(f"camera_feed (fixed mode) setup failed: {e_stream_setup}")
                    # Error will be caught in the main loop and displayed on frame.
        
        # End of fixed perspective stream initialization
        
        with stream_context():
            while True:
                try:
                    frame_orig = get_frame() # release_after=False by default
                    if frame_orig is None or frame_orig.size == 0:
                        current_height = stream_cfg_cam.get("capture_height", 480)
                        current_width = stream_cfg_cam.get("capture_width", 640)
                        if mode == "fixed":
                            current_height = OUT_H if OUT_H > 0 else 480
                            current_width = OUT_W if OUT_W > 0 else 640

                        blank_frame = np.zeros((current_height, current_width, 3), dtype=np.uint8)
                        cv2.putText(blank_frame, "No Frame", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
                        if mode == "fixed" and H_ref is None:
                             cv2.putText(blank_frame, "Fixed View Not Set", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),1)
                        _, buffer = cv2.imencode('.jpg', blank_frame)
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        time.sleep(0.1) # Shorter sleep for responsiveness to camera coming online
                        continue
                    
                    display_frame_feed = frame_orig.copy()
                    
                    if mode == "normal" or mode == "threshold":
                        gray = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
                        _, processed_for_blobs = cv2.threshold(
                            gray, stream_cfg_cam.get("minThreshold", 127),
                            stream_cfg_cam.get("maxThreshold", 255), cv2.THRESH_BINARY
                        )
                        if mode == "threshold":
                            display_frame_feed = cv2.cvtColor(processed_for_blobs, cv2.COLOR_GRAY2BGR)
                        
                        keypoints_blob = detect_blobs_from_params(processed_for_blobs, blob_params_for_stream)
                        frame_with_keypoints = cv2.drawKeypoints(
                            display_frame_feed, keypoints_blob, np.array([]), (0, 0, 255), 
                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                        
                        _, buffer = cv2.imencode('.jpg', frame_with_keypoints)
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                    elif mode == "fixed":
                        output_img = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8) # Start with a blank output slate
                        
                        if H_ref is not None and cam_matrix is not None and dist_coeffs is not None and new_cam_matrix_stream is not None:
                            undistorted_live = cv2.undistort(frame_orig, cam_matrix, dist_coeffs, None, new_cam_matrix_stream)
                            output_img = cv2.warpPerspective(undistorted_live, H_ref, (OUT_W, OUT_H))
                            
                            # --- BLOB DETECTION ON UNDISTORTED-NORMALIZED, THEN TRANSFORM TO WARPED ---
                            gray_for_blobs = cv2.cvtColor(undistorted_live, cv2.COLOR_BGR2GRAY)
                            _, thresh_for_blobs = cv2.threshold(
                                gray_for_blobs,
                                blob_params_for_stream.get("minThreshold", 127),
                                blob_params_for_stream.get("maxThreshold", 255),
                                cv2.THRESH_BINARY
                            )
                            keypoints_on_undistorted = detect_blobs_from_params(thresh_for_blobs, blob_params_for_stream)
                            
                            if keypoints_on_undistorted:
                                pts_undist = np.array([kp.pt for kp in keypoints_on_undistorted], dtype=np.float32).reshape(-1,1,2)
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
                        else: # Fixed perspective not ready or H_ref is None
                            cv2.putText(output_img, "Fixed View Not Set", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),1)
                            cv2.putText(output_img, "Set via endpoint", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),1)

                        _, buffer_ok = cv2.imencode('.jpg', output_img)
                        frame_bytes_ok = buffer_ok.tobytes()
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes_ok + b'\r\n')
                    
                    time.sleep(0.03) # Control frame rate
                except Exception as e:
                    print(f"Error in camera_feed loop (mode: {mode}): {e}")
                    traceback.print_exc()
                    
                    current_height = stream_cfg_cam.get("capture_height", 480)
                    current_width = stream_cfg_cam.get("capture_width", 640)
                    if mode == "fixed":
                        current_height = OUT_H if OUT_H > 0 else 480
                        current_width = OUT_W if OUT_W > 0 else 640

                    error_frame = np.zeros((current_height, current_width, 3), dtype=np.uint8)
                    cv2.putText(error_frame, f"Stream Err: {str(e)[:50]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    _, buffer = cv2.imencode('.jpg', error_frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    time.sleep(1) # Pause on error
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

@csrf_exempt
@require_POST
def reset_camera_calibration(request):
    try:
        # Carica la configurazione corrente dal disco
        current_disk_config = load_config_data_from_file()

        # Ottieni la sezione 'camera' in modo sicuro
        # Questo garantisce che 'camera' sia un dizionario su cui possiamo operare.
        if "camera" not in current_disk_config:
            current_disk_config["camera"] = {}
        
        # Resetta i parametri di calibrazione a None (o rimuovi le chiavi)
        # Impostare a None è una chiara indicazione dello stato "non calibrato".
        current_disk_config["camera"]["calibration"]["camera_matrix"] = None
        current_disk_config["camera"]["calibration"]["distortion_coefficients"] = None
        current_disk_config["camera"]["fixed_perspective"]["homography_matrix"] = None
        
        if save_config_data_to_file(current_disk_config): # Passa l'intero oggetto config modificato
            return JsonResponse({
                "status": "success",
                "message": "Camera calibration and fixed perspective reset successfully."
            })
        else:
            # save_config_data_to_file stampa già un errore, quindi qui gestiamo solo la risposta JSON.
            return JsonResponse({
                "status": "error",
                "message": "Failed to save the reset configuration to file."
            }, status=500)

    except Exception as e:
        print(f"Error resetting camera calibration: {e}")
        traceback.print_exc() # Per un log più dettagliato sul server
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

@csrf_exempt
@require_GET
def get_keypoints(request):
    try:
        # Uses global config via get_current_frame_and_keypoints_from_config
        _, keypoints_data = get_current_frame_and_keypoints_from_config()
        keypoints_list = [[kp.pt[0], kp.pt[1]] for kp in keypoints_data]
        
        rect_vertices = []
        # cv2.minAreaRect requires at least 3 points for a non-degenerate rectangle
        if len(keypoints_data) >= 3: 
            pts_rect = np.array(keypoints_list, dtype=np.float32)
            rect = cv2.minAreaRect(pts_rect)
            box = cv2.boxPoints(rect)
            rect_vertices = box.astype(float).tolist()
        parallelepiped_vertices = []
        parallelepiped_ok = False
        if len(keypoints_data) >= 4:
            pts_para = np.array(keypoints_list, dtype=np.float32)
            # This logic for parallelepiped corners is a common heuristic
            s = pts_para.sum(axis=1)
            diff = np.diff(pts_para, axis=1) # Difference between y and x
            
            corners = np.zeros((4,2), dtype=np.float32)
            corners[0] = pts_para[np.argmin(s)]     # Top-left
            corners[2] = pts_para[np.argmax(s)]     # Bottom-right
            corners[1] = pts_para[np.argmin(diff)]  # Top-right
            corners[3] = pts_para[np.argmax(diff)]  # Bottom-left
            
            # Basic check if all original points are within the found polygon
            # This is not a perfect check for convexity or a true parallelepiped.
            if cv2.pointPolygonTest(corners.reshape(-1,1,2).astype(np.int32), tuple(pts_para[0]), False) >= 0: # Test one point
                 parallelepiped_vertices = corners.tolist()
                 parallelepiped_ok = True # Mark as "ok" based on this heuristic
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
        frame_to_save = get_frame(release_after=True) # Single acquisition
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
    # Uses global 'camera_settings' for calibration parameters
    # but loads the whole config from disk for modification.
    current_disk_config = load_config_data_from_file()
    calib_settings = current_disk_config.get("camera", {}).get("calibration_settings", {}) # Use latest from disk
    
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
            last_gray_shape = gray.shape[::-1] # (width, height)
        elif last_gray_shape != gray.shape[::-1]:
             print(f"WARNING: Image {image_path} dimensions ({gray.shape[::-1]}) differ from first image ({last_gray_shape}).")
             # This can affect calibration accuracy.
        
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
    # Uses global 'camera_settings' for various parameters but loads full config for save
    # This function reads calibration data, calibration settings, and fixed_perspective settings
    # All of these should be available in the global 'camera_settings'
    
    cam_calib_data = camera_settings.get("calibration", None)
    calib_settings_dict = camera_settings.get("calibration_settings", {})
    fixed_perspective_cfg = camera_settings.get("fixed_perspective", {})
    if not (cam_calib_data and cam_calib_data.get("camera_matrix") and cam_calib_data.get("distortion_coefficients")):
        return JsonResponse({"status": "error", "message": "Camera calibration data not found. Please calibrate first."}, status=400)
    camera_matrix_cv = np.array(cam_calib_data["camera_matrix"], dtype=np.float32)
    dist_coeffs_cv = np.array(cam_calib_data["distortion_coefficients"], dtype=np.float32)
    FIXED_WIDTH = fixed_perspective_cfg.get("output_width", 1000)
    FIXED_HEIGHT = fixed_perspective_cfg.get("output_height", 800)
    # No stream_context here as it's a single operation
    try:
        frame_cap = get_frame(release_after=True) # Single acquisition
        if frame_cap is None or frame_cap.size == 0: 
            return JsonResponse({"status": "error", "message": "Could not get frame from camera."}, status=500)
        h_cam_cap, w_cam_cap = frame_cap.shape[:2]
        new_camera_matrix_cv, _ = cv2.getOptimalNewCameraMatrix(camera_matrix_cv, dist_coeffs_cv, (w_cam_cap,h_cam_cap), 1.0, (w_cam_cap,h_cam_cap))
        undistorted_frame_cap = cv2.undistort(frame_cap, camera_matrix_cv, dist_coeffs_cv, None, new_camera_matrix_cv)
        H_canonical, canonical_dims = get_board_and_canonical_homography_for_django(
            undistorted_frame_cap, new_camera_matrix_cv, calib_settings_dict # calib_settings_dict from global
        )
        # Check if H_canonical and canonical_dims are valid for proceeding
        if H_canonical is not None and canonical_dims is not None and canonical_dims[0] > 0 and canonical_dims[1] > 0:
            # --- SUCCESS CASE: Proceed with homography calculation and saving ---
            cb_w, cb_h = canonical_dims
            offset_x = max(0, (FIXED_WIDTH - cb_w) / 2.0) # Ensure offset is not negative
            offset_y = max(0, (FIXED_HEIGHT - cb_h) / 2.0)
            M_translate = np.array([[1,0,offset_x], [0,1,offset_y], [0,0,1]], dtype=np.float32)
            H_ref = M_translate @ H_canonical
            if save_fixed_perspective_homography_to_config(H_ref): # This function handles file I/O
                return JsonResponse({
                    "status": "success",
                    "message": "Fixed perspective view established and saved."
                })
            else:
                # This is an error during the saving process itself
                print("[ERROR] set_fixed_perspective_view: Failed to save homography to config file.")
                return JsonResponse({
                    "status": "error",
                    "message": "Error saving fixed perspective homography to configuration file.",
                    "error_code": "SAVE_HOMOGRAPHY_FAILED"
                }, status=500)
        else:
            # --- ERROR CASE: Determine the specific reason for failure ---
            error_message = "Cannot define fixed view. An unknown error occurred." # Default message
            error_code = "UNKNOWN_FIXED_VIEW_ERROR" # Default error code
            status_code = 400 # Default HTTP status
            if H_canonical is None:
                # This is the most common failure: chessboard not found.
                # get_board_and_canonical_homography_for_django returns (None, None) in this case.
                error_message = "Chessboard pattern not detected in the current camera view. Ensure the full pattern is clearly visible and well-lit."
                error_code = "CHESSBOARD_NOT_DETECTED"
                print(f"[ERROR] set_fixed_perspective_view ({error_code}): {error_message}")
            elif canonical_dims is None:
                # This case should ideally not be reached if H_canonical is not None,
                # as get_board_and_canonical_homography_for_django should return both or neither.
                # If it does, it might indicate an internal logic issue in that helper function.
                error_message = "Internal error: Chessboard detected, but its dimensions could not be determined."
                error_code = "CANONICAL_DIMS_MISSING_UNEXPECTEDLY"
                status_code = 500 # Internal server error
                print(f"[ERROR] set_fixed_perspective_view ({error_code}): {error_message}")
            elif canonical_dims[0] <= 0 or canonical_dims[1] <= 0:
                # Chessboard was found, H_canonical exists, but the calculated dimensions are invalid (e.g., negative or zero).
                error_message = (f"Invalid canonical dimensions calculated for the chessboard: {canonical_dims}. "
                                 f"This might indicate an issue with the chessboard configuration (e.g., square size, pattern size in settings) "
                                 f"or a highly distorted detection.")
                error_code = "INVALID_CANONICAL_DIMS_CALCULATED"
                print(f"[ERROR] set_fixed_perspective_view ({error_code}): {error_message} - Dimensions: {canonical_dims}")
            
            return JsonResponse({
                "status": "error",
                "message": error_message,
                "error_code": error_code # Adding an error_code can be useful for frontend handling
            }, status=status_code)
    except Exception as e:
        print(f"Exception in set_fixed_perspective_view: {e}")
        traceback.print_exc()
        return JsonResponse({"status": "error", "message": str(e)}, status=500)
    
@csrf_exempt
@require_GET
def fixed_perspective_stream(request):
    def gen_frames():
        # Use global 'camera_settings' as a snapshot for this stream worker
        # This avoids repeated file access within the loop.
        # If settings change, the stream needs to be restarted to see them.
        stream_cfg = camera_settings 
        
        H_ref = get_fixed_perspective_homography_from_config() # Reads from global config via helper
        
        cam_calib = stream_cfg.get("calibration", None)
        fixed_persp_cfg = stream_cfg.get("fixed_perspective", {})
        blob_params_for_stream = stream_cfg # Pass the whole camera_settings for blob detection
        
        OUT_W = fixed_persp_cfg.get("output_width", 1000)
        OUT_H = fixed_persp_cfg.get("output_height", 800)
        error_template_frame = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
        if not (cam_calib and cam_calib.get("camera_matrix") and cam_calib.get("distortion_coefficients")):
            error_msg = "Camera calibration missing"
            print(f"fixed_perspective_stream: {error_msg}")
            while True:
                err_f = error_template_frame.copy()
                cv2.putText(err_f, error_msg, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                _, buf = cv2.imencode('.jpg', err_f); yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'); time.sleep(1)
        cam_matrix = np.array(cam_calib["camera_matrix"], dtype=np.float32)
        dist_coeffs = np.array(cam_calib["distortion_coefficients"], dtype=np.float32)
        
        new_cam_matrix_stream = None
        try:
            # Get a sample frame to determine dimensions for new_camera_matrix
            # Using get_frame with release_after=True to ensure camera is free if it was the first call
            sample_frame_for_dims = get_frame(release_after=True) 
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
                _, buf = cv2.imencode('.jpg', err_f); yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'); time.sleep(1)
        
        with stream_context():
            while True:
                try:
                    frame_live = get_frame() # release_after=False default
                    if frame_live is None or frame_live.size == 0: 
                        err_f_loop = error_template_frame.copy()
                        cv2.putText(err_f_loop, "Frame lost", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,100,255),2)
                        if H_ref is None: cv2.putText(err_f_loop, "Fixed View Not Set", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),1)
                        _, buf_err = cv2.imencode('.jpg', err_f_loop)
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf_err.tobytes() + b'\r\n')
                        time.sleep(0.1)
                        continue
                    # Undistort using the new_cam_matrix_stream calculated once for this stream
                    undistorted_live = cv2.undistort(frame_live, cam_matrix, dist_coeffs, None, new_cam_matrix_stream)
                    
                    output_img = error_template_frame.copy() # Start with a blank output slate
                    if H_ref is not None:
                        output_img = cv2.warpPerspective(undistorted_live, H_ref, (OUT_W, OUT_H))
                        
                        # --- BLOB DETECTION ON UNDISTORTED-NORMALIZED, THEN TRANSFORM TO WARPED ---
                        # 1. Create binary image for blob detection from undistorted_live
                        gray_for_blobs = cv2.cvtColor(undistorted_live, cv2.COLOR_BGR2GRAY)
                        _, thresh_for_blobs = cv2.threshold(
                            gray_for_blobs,
                            blob_params_for_stream.get("minThreshold", 127),
                            blob_params_for_stream.get("maxThreshold", 255),
                            cv2.THRESH_BINARY
                        )
                        keypoints_on_undistorted = detect_blobs_from_params(thresh_for_blobs, blob_params_for_stream)
                        
                        if keypoints_on_undistorted:
                            # Points are relative to 'undistorted_live' which is in the coordinate system
                            # defined by 'new_cam_matrix_stream'. H_ref transforms from this system to the warped output.
                            pts_undist = np.array([kp.pt for kp in keypoints_on_undistorted], dtype=np.float32).reshape(-1,1,2)
                            
                            # No need for cv2.undistortPoints here as points are already in the new_cam_matrix_stream system.
                            # We just need to apply the homography H_ref.
                            # Convert to homogeneous coordinates for perspectiveTransform
                            pts_warped = cv2.perspectiveTransform(pts_undist, H_ref)
                            
                            if pts_warped is not None: # perspectiveTransform can return None
                                for i, pt_w in enumerate(pts_warped.reshape(-1,2)):
                                    x, y = pt_w[0], pt_w[1]
                                    cv2.circle(output_img, (int(round(x)), int(round(y))), 8, (0,0,255), 2) # Red circle
                                    cv2.putText(
                                        output_img, f"{x:.1f},{y:.1f}",
                                        (int(round(x))+10, int(round(y))-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1 # Green text
                                    )
                    else: # H_ref is None
                        cv2.putText(output_img, "Fixed View Not Set", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),1)
                        cv2.putText(output_img, "Use endpoint to set it", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),1)
                    _, buffer_ok = cv2.imencode('.jpg', output_img)
                    frame_bytes_ok = buffer_ok.tobytes()
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes_ok + b'\r\n')
                    time.sleep(0.03)
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
    H_fixed_ref = get_fixed_perspective_homography_from_config() # From global config
    if H_fixed_ref is None:
        return JsonResponse({"status": "error", "message": "Fixed perspective homography not available. Please set it first via its endpoint."}, status=400)
    cam_calib_wc = camera_settings.get("calibration", None)
    if not (cam_calib_wc and cam_calib_wc.get("camera_matrix") and cam_calib_wc.get("distortion_coefficients")):
        return JsonResponse({"status": "error", "message": "Camera calibration data missing. Please calibrate the camera first."}, status=400)
    cam_matrix_wc = np.array(cam_calib_wc["camera_matrix"], dtype=np.float32)
    dist_coeffs_wc = np.array(cam_calib_wc["distortion_coefficients"], dtype=np.float32)
    # Gets keypoints detected on the *original possibly distorted* frame
    frame_for_coords, keypoints_for_coords = get_current_frame_and_keypoints_from_config()
    if frame_for_coords is None or frame_for_coords.size == 0:
         return JsonResponse({"status": "error", "message": "Could not get frame for coordinates calculation."}, status=500)
    if not keypoints_for_coords:
        # Se non ci sono keypoint, restituisci una lista vuota di coordinate
        return JsonResponse({"status": "success", "coordinates": []})
    # These are points from the original, possibly distorted image
    img_pts_distorted = np.array([kp.pt for kp in keypoints_for_coords], dtype=np.float32).reshape(-1,1,2)
    h_frame_wc, w_frame_wc = frame_for_coords.shape[:2]
    # This new_cam_matrix_wc should ideally be the same as the one used when H_fixed_ref was calculated
    # or derived in the exact same way.
    new_cam_matrix_wc, _ = cv2.getOptimalNewCameraMatrix(cam_matrix_wc, dist_coeffs_wc, (w_frame_wc,h_frame_wc), 1.0, (w_frame_wc,h_frame_wc))
    # Undistort points and map them to the coordinate system of new_cam_matrix_wc
    img_pts_undistorted_remapped = cv2.undistortPoints(img_pts_distorted, cam_matrix_wc, dist_coeffs_wc, P=new_cam_matrix_wc)
    if img_pts_undistorted_remapped is None: # Should not happen if inputs are valid
        return JsonResponse({"status": "error", "message": "Point undistortion failed unexpectedly."}, status=500)
    world_coords_top_left_origin = []
    if img_pts_undistorted_remapped.size > 0: # Check if there are points
        # cv2.perspectiveTransform expects Nx1x2 array
        transformed_pts = cv2.perspectiveTransform(img_pts_undistorted_remapped, H_fixed_ref)
        if transformed_pts is not None:
             world_coords_top_left_origin = transformed_pts.reshape(-1, 2).tolist()
        else: # Should not happen if H_fixed_ref is valid 3x3
            print("[WARN] cv2.perspectiveTransform returned None in get_world_coordinates. This might indicate an issue with H_fixed_ref.")
            # Restituisce comunque una lista vuota o un errore a seconda della gravità percepita
            return JsonResponse({"status": "error", "message": "Perspective transformation of points failed."}, status=500)
    # --- TRASFORMAZIONE COORDINATE A ORIGINE BASSO-DESTRA ---
    # Ottieni le dimensioni del frame di output della prospettiva fissa dalla configurazione globale
    # Queste dovrebbero corrispondere alle dimensioni usate per generare H_fixed_ref
    # e per lo stream fixed_perspective_stream.
    fixed_persp_cfg = camera_settings.get("fixed_perspective", {})
    OUTPUT_WIDTH = fixed_persp_cfg.get("output_width", 1000) # Default se non specificato
    OUTPUT_HEIGHT = fixed_persp_cfg.get("output_height", 800) # Default se non specificato
    world_coords_bottom_right_origin = []
    for x_tl, y_tl in world_coords_top_left_origin:
        # x_nuovo aumenta verso sinistra dalla nuova origine (basso-destra)
        # y_nuovo aumenta verso l'alto dalla nuova origine (basso-destra)
        x_br = OUTPUT_WIDTH - x_tl
        y_br = OUTPUT_HEIGHT - y_tl
        world_coords_bottom_right_origin.append([x_br, y_br])
    # --- FINE TRASFORMAZIONE ---
    return JsonResponse({"status": "success", "coordinates": world_coords_bottom_right_origin})
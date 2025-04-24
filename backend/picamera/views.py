from django.http import StreamingHttpResponse
import sys
from unittest.mock import MagicMock
if sys.platform == "darwin":
    sys.modules["picamera2"] = MagicMock()
from picamera2 import Picamera2 # type: ignore
import cv2

def gen_frames():
    # Inizializza Picamera2
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
    
    # Abilita l'autofocus
    picam2.set_controls({"AfMode": 2})  # 2 = Continuous autofocus
    picam2.start()

    try:
        while True:
            # Cattura un frame dalla telecamera
            frame = picam2.capture_array()
            
            # Converti i colori da RGB a BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Codifica il frame in JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        picam2.stop()

def camera_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
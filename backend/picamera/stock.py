# import the necessary packages
from picamera2 import Picamera2, Preview
from libcamera import ColorSpace 
import numpy as np
import time
import cv2

# initialize the camera and grab a reference to the raw camera capture
camera = Picamera2()

# create the configuration for the preview with a lower resolution
config = camera.create_still_configuration(
    colour_space=ColorSpace.Sycc()
)

# set the configuration
camera.configure(config)

# start the camera
camera.start()

# allow the camera to warm up
time.sleep(1)

# capture the image as an array
try:
    rawCapture = camera.capture_array("main")
    print(f"Captured image shape: {rawCapture.shape}")  # Stampa la forma dell'immagine
except Exception as e:
    print(f"Error capturing image: {e}")
    camera.stop()
    exit()

# convert to the correct color space
#image = cv2.cvtColor(rawCapture, cv2.COLOR_BGR2RGB)

# save the image
#cv2.imwrite('test1.jpg', rawCapture)

# Read image
imRgb = cv2.imread('test1.jpg',cv2.COLOR_BGR2RGB)
imGrayScale = cv2.imread('test1.jpg', cv2.IMREAD_GRAYSCALE)

(thresh, im) = cv2.threshold(imGrayScale, 160, 210, cv2.THRESH_BINARY)
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
 
# Change thresholds
params.minThreshold = 0
params.maxThreshold = 280
 
# Filter by Area.
params.filterByArea = True
params.minArea = 2000
params.maxArea = 11000
 
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.001
 
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.001
 
# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
 detector = cv2.SimpleBlobDetector(params)
else : 
 detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(im)
#for i, kp in enumerate(keypoints):
#    x, y = kp.pt  # Estrai le coordinate x e y
#    print(f'Keypoint {i}:')
#    print(f'  Coordinate: (x={x}, y={y})')
# Draw detected blobs as red circles.
im_with_keypoints = cv2.drawKeypoints(imRgb, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
for i, kp in enumerate(keypoints):
    x, y = int(kp.pt[0]), int(kp.pt[1])  # Ottieni le coordinate del keypoint
    cv2.putText(im_with_keypoints, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
# display the image on screen and wait for a keypress
cv2.imshow("key points", im_with_keypoints)
cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()
camera.stop()
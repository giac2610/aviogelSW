from django.conf import settings
from django.urls import path
from django.conf.urls.static import static
from .views import (
    camera_feed,
    get_keypoints,
    set_camera_origin,
    update_camera_settings,
    dynamic_warped_stream, 
    calibrate_camera,
    save_frame_calibration,
)

urlpatterns = [
    path('stream/', camera_feed, name='camera_feed'),
    path('keypoints/', get_keypoints, name='get_keypoints'),
    path('set-origin/', set_camera_origin, name='set_camera_origin'),
    path('update-camera-settings/', update_camera_settings, name='update_camera_settings'),
    path('dynamic-warped-stream/', dynamic_warped_stream, name='dynamic_warped_stream'),
    path('calibrate_camera/', calibrate_camera, name='calibrate_camera'),
    path('save-frame-calibration/', save_frame_calibration, name='save_frame_calibration'),

]

# Serve i file media solo in sviluppo (DEBUG=True)
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
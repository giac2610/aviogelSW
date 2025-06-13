from django.conf import settings
from django.urls import path
from django.conf.urls.static import static
from .views import (
    camera_feed,
    fixed_perspective_stream,
    get_keypoints,
    initialize_camera_endpoint,
    set_camera_origin,
    set_fixed_perspective_view,
    update_camera_settings,
    save_frame_calibration,
    get_world_coordinates,
    calibrate_camera_endpoint,
    reset_camera_calibration,
    compute_route,
    plot_graph,
    deinitialize_camera_endpoint,
)

urlpatterns = [
    path('stream/', camera_feed, name='camera_feed'),
    path('keypoints/', get_keypoints, name='get_keypoints'),
    path('set-origin/', set_camera_origin, name='set_camera_origin'),
    path('update-camera-settings/', update_camera_settings, name='update_camera_settings'),
    path('save-frame-calibration/', save_frame_calibration, name='save_frame_calibration'),
    path('get_coordinates/',get_world_coordinates, name='get_world_coordinates'),
    path('calibrate_camera/', calibrate_camera_endpoint, name='calibrate_camera_view'),
    path('fixed-perspective-stream/', fixed_perspective_stream, name='fixed_perspective_stream'),
    path('set-fixed-perspective/', set_fixed_perspective_view, name='set_fixed_perspective_view'),
    path('reset-camera-calibration/', reset_camera_calibration, name='reset_camera_calibration'),
    path('initialize-camera/', initialize_camera_endpoint, name='initialize_camera'),
    path('deinitialize_camera_endpoint/', deinitialize_camera_endpoint, name='deinitialize_camera_endpoint'),
    path('get_route/', compute_route, name='compute_route'),
    path('plot_graph/', plot_graph, name='plot_graph'),
]

# Serve i file media solo in sviluppo (DEBUG=True)
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
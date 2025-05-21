from django.conf import settings
from django.urls import path
from django.conf.urls.static import static
from .views import (
    camera_feed,
    get_keypoints,
    get_keypoints_all,
    set_camera_origin,
    update_camera_settings,
    get_homography,
    get_frame_api,
    dynamic_warped_stream,
    capture_and_warp_frame,
    contour_params,  # Cambia qui: importa la funzione unificata
    contour_stream,
    contour_homography,
    # debug_keypoints_view,
    # detect_grid_points,
)

urlpatterns = [
    path('stream/', camera_feed, name='camera_feed'),
    path('keypoints/', get_keypoints, name='get_keypoints'),
    path('keypoints-all/', get_keypoints_all, name='get_keypoints_all'),
    path('set-origin/', set_camera_origin, name='set_camera_origin'),
    path('update-camera-settings/', update_camera_settings, name='update_camera_settings'),
    path('homography/', get_homography, name='get_homography'),
    path('frame/', get_frame_api, name='get_frame_api'),
    path('dynamic-warped-stream/', dynamic_warped_stream, name='dynamic_warped_stream'),
    path('capture-and-warp-frame/', capture_and_warp_frame, name='capture_and_warp_frame'),
    path('contour-params/', contour_params, name='contour_params'),  # Solo una view per GET/POST
    path('contour-stream/', contour_stream, name='contour_stream'),
    path('contour-homography/', contour_homography, name='contour_homography'),
    # path('camera/detect-grid/', detect_grid_points, name='detect_grid'),
    # path('camera/debug-keypoints/', debug_keypoints_view, name='debug_keypoints'),
]

# Serve i file media solo in sviluppo (DEBUG=True)
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

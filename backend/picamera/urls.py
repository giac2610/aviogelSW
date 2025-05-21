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
    calculate_homography_from_points,
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
    path('calculate-homography-from-points/', calculate_homography_from_points, name='calculate_homography_from_points'),
]

# Serve i file media solo in sviluppo (DEBUG=True)
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

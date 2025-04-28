from django.urls import path
from .views import camera_feed, camera_feed_greyscale, camera_feed_threshold

urlpatterns = [
    path('stream/', camera_feed, name='camera_feed'),
    path('stream/greyscale/', camera_feed_greyscale, name='camera_feed_greyscale'),
    path('stream/threshold/', camera_feed_threshold, name='camera_feed_threshold'),
]
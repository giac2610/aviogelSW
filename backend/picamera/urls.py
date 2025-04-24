from django.urls import path
from .views import camera_feed

urlpatterns = [
    path('stream/', camera_feed, name='camera_feed'),
]
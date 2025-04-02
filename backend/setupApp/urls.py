from django.urls import path
from .views import get_settings, update_settings

urlpatterns = [
    path('get/', get_settings, name='get_settings'),
    path('update/', update_settings, name='update_settings'),
]

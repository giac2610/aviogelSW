from django.urls import path
from .views import get_settings, update_settings, get_ip_address

urlpatterns = [
    path('get/', get_settings, name='get_settings'),
    path('update/', update_settings, name='update_settings'),
    path('get_ip/', get_ip_address, name='get_ip_address'),  # Aggiunta la nuova rotta per ottenere l'IP
]

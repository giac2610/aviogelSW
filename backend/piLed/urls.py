from django.urls import path
from . import views

urlpatterns = [
    path('control-leds/', views.control_leds, name='control_leds'),
    path('get-led-status/', views.get_led_status, name='get_led_status'),
]

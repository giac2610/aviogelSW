from django.urls import path
from . import views

urlpatterns = [
    path('control-leds/', views.control_leds, name='control_leds'),
]

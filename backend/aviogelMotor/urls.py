from django.urls import path
from .views import update_config, move_motor, stop_motor, save_motor_config

urlpatterns = [
    path('update/', update_config, name='update_config'),
    path('move/', move_motor, name='move_motor'),
    path('stop/', stop_motor, name='stop_motor'),
    path('save/', save_motor_config, name='save_motor_config'),
]

from django.urls import path
from .views import update_config, move_motor, stop_motor, save_motor_config, get_motor_speeds

urlpatterns = [
    path('update/', update_config, name='update_config'),
    path('move/', move_motor, name='move_motor'),
    path('stop/', stop_motor, name='stop_motor'),
    path('save/', save_motor_config, name='save_motor_config'),
    path('speeds/', get_motor_speeds, name='get_motor_speeds'),
]

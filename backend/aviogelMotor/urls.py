from django.urls import path
from .views import update_config_view, execute_route_view, move_motor_view, stop_motor_view, save_motor_config_view, get_motor_speeds_view, start_simulation_view

urlpatterns = [
    path('update/', update_config_view, name='update_config'),
    path('move/', move_motor_view, name='move_motor'),
    path('stop/', stop_motor_view, name='stop_motor'),
    path('save/', save_motor_config_view, name='save_motor_config'),
    path('speeds/', get_motor_speeds_view, name='get_motor_speeds'),
    path('simulate/', start_simulation_view, name='start_simulation'),
    path('execute_route/', execute_route_view, name='execute_route'),
]

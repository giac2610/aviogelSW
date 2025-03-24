from django.urls import path
from .views import get_users, add_user, delete_user

urlpatterns = [
    path('list/', get_users, name='get_users'),
    path('add/', add_user, name='add_user'),
    path('delete/<int:user_id>/', delete_user, name='delete_user'),

]

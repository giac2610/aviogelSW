from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
from .models import UserProfile
from .serializers import UserProfileSerializer

@api_view(['GET'])
def get_users(request):
    """Restituisce la lista degli utenti"""
    users = UserProfile.objects.all()
    serializer = UserProfileSerializer(users, many=True)
    return Response(serializer.data)

@api_view(['POST'])
def add_user(request):
    name = request.data.get('name')
    gender = request.data.get('gender', None)  # Può essere "male", "female" o None
    expertUser = request.data.get('expertUser', None)
    
    user = UserProfile(name=name, gender=gender, expertUser=expertUser)
    user.save()

    return Response(UserProfileSerializer(user).data)

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

@api_view(['DELETE'])
def delete_user(request, user_id):
    """Elimina un utente dato il suo ID"""
    try:
        user = UserProfile.objects.get(id=user_id)
        user.delete()
        return Response({"message": "Utente eliminato con successo"}, status=status.HTTP_204_NO_CONTENT)
    except UserProfile.DoesNotExist:
        return Response({"error": "Utente non trovato"}, status=status.HTTP_404_NOT_FOUND)
    
@api_view(['PUT'])
def update_user(request, user_id):
    """Aggiorna un utente dato il suo ID"""
    try:
        user = UserProfile.objects.get(id=user_id)
        serializer = UserProfileSerializer(user, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    except UserProfile.DoesNotExist:
        return Response({"error": "Utente non trovato"}, status=status.HTTP_404_NOT_FOUND)
from rest_framework import serializers
from .models import UserProfile

class UserProfileSerializer(serializers.ModelSerializer):
    avatar_url = serializers.SerializerMethodField()  # ✅ Definisce avatar_url
    expertUser = serializers.SerializerMethodField()
    class Meta:
        model = UserProfile
        fields = ['id', 'name', 'gender', 'avatar_url', 'expertUser']  # ✅ Includiamo avatar_url nei campi serializzati

    def get_avatar_url(self, obj):
        """Genera il percorso dell'avatar basato sul nome del file memorizzato nel modello."""
        return f"assets/avatars/{obj.avatar}"  # ✅ Il frontend Ionic carica l'immagine da questa posizione
    
    def get_expertUser(self, obj):
        return obj.expertUser

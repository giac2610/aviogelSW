from rest_framework import serializers

class MotorSettingsSerializer(serializers.Serializer):
    stepOneRev = serializers.FloatField()
    microstep = serializers.ChoiceField(choices=[2, 4, 8, 16, 32, 64, 128])
    maxSpeed = serializers.FloatField()
    acceleration = serializers.FloatField()
    deceleration = serializers.FloatField()

class CameraSettingsSerializer(serializers.Serializer):
    minThreshold = serializers.IntegerField()
    maxThreshold = serializers.IntegerField()
    areaFilter = serializers.BooleanField()
    minArea = serializers.FloatField()
    maxArea = serializers.FloatField()
    circularityFilter = serializers.BooleanField()
    minCircularity = serializers.FloatField()
    maxCircularity = serializers.FloatField()
    inertiaFilter = serializers.BooleanField()
    minInertia = serializers.FloatField()
    maxInertia = serializers.FloatField()

class SettingsSerializer(serializers.Serializer):
    motors = MotorSettingsSerializer()
    camera = CameraSettingsSerializer()

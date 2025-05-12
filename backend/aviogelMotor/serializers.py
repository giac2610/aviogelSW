from rest_framework import serializers

class MotorSettingsSerializer(serializers.Serializer):
    homeDir = serializers.IntegerField()  # 0 o 1
    stepOneRev = serializers.FloatField()
    microstep = serializers.ChoiceField(choices=[1, 2, 4, 8, 16, 32, 64, 128])
    pitch = serializers.FloatField()
    maxSpeed = serializers.FloatField()
    acceleration = serializers.FloatField()
    deceleration = serializers.FloatField()
    maxTravel = serializers.FloatField()
    hertz = serializers.FloatField()

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

class MotorsSerializer(serializers.Serializer):
    extruder = MotorSettingsSerializer()
    conveyor = MotorSettingsSerializer()
    syringe = MotorSettingsSerializer()

class SettingsSerializer(serializers.Serializer):
    motors = MotorsSerializer()
    camera = CameraSettingsSerializer()
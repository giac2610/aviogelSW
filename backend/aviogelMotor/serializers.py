from rest_framework import serializers

class CalibrationSerializer(serializers.Serializer):
    camera_matrix = serializers.ListField(child=serializers.ListField(child=serializers.FloatField()))
    distortion_coefficients = serializers.ListField(child=serializers.ListField(child=serializers.FloatField()))

class CalibrationSettingsSerializer(serializers.Serializer):
    chessboard_cols = serializers.IntegerField()
    chessboard_rows = serializers.IntegerField()
    square_size_mm = serializers.FloatField()

class FixedPerspectiveSerializer(serializers.Serializer):
    output_width = serializers.IntegerField()
    output_height = serializers.IntegerField()
    homography_matrix = serializers.ListField(child=serializers.ListField(child=serializers.FloatField()))

class PicameraConfigSerializer(serializers.Serializer):
    main = serializers.DictField()
    lores = serializers.DictField()
    controls = serializers.DictField()

class MotorSettingsSerializer(serializers.Serializer):
    homeDir = serializers.IntegerField()
    stepOneRev = serializers.FloatField()
    microstep = serializers.ChoiceField(choices=[1, 2, 4, 8, 16, 32, 64, 128])
    pitch = serializers.FloatField()
    maxSpeed = serializers.FloatField()
    acceleration = serializers.FloatField()
    deceleration = serializers.FloatField(required=False)
    maxTravel = serializers.FloatField()
    hertz = serializers.FloatField()
    doseVolume = serializers.FloatField()
    retractVolume = serializers.FloatField()

class CameraSettingsSerializer(serializers.Serializer):
    minThreshold = serializers.IntegerField()
    maxThreshold = serializers.IntegerField()
    areaFilter = serializers.BooleanField()
    minArea = serializers.FloatField()
    maxArea = serializers.FloatField()
    circularityFilter = serializers.BooleanField()
    minCircularity = serializers.FloatField()
    maxCircularity = serializers.FloatField(required=False)
    filterByConvexity = serializers.BooleanField()
    minConvexity = serializers.FloatField()
    inertiaFilter = serializers.BooleanField()
    minInertia = serializers.FloatField()
    origin_x = serializers.FloatField()
    origin_y = serializers.FloatField()
    picamera_config = PicameraConfigSerializer()
    calibration = CalibrationSerializer()
    calibration_settings = CalibrationSettingsSerializer()
    fixed_perspective = FixedPerspectiveSerializer()

class MotorsSerializer(serializers.Serializer):
    extruder = MotorSettingsSerializer()
    conveyor = MotorSettingsSerializer()
    syringe = MotorSettingsSerializer()

class SettingsSerializer(serializers.Serializer):
    motors = MotorsSerializer()
    camera = CameraSettingsSerializer()
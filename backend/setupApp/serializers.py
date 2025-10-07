from rest_framework import serializers

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
    doseVolume = serializers.FloatField(required=False)
    retractVolume = serializers.FloatField(required=False)

class PicameraConfigSerializer(serializers.Serializer):
    main = serializers.DictField(required=False)
    lores = serializers.DictField(required=False)
    controls = serializers.DictField(required=False)

class CalibrationSerializer(serializers.Serializer):
    camera_matrix = serializers.ListField(child=serializers.ListField(child=serializers.FloatField()), required=False)
    distortion_coefficients = serializers.ListField(child=serializers.ListField(child=serializers.FloatField()), required=False)

class CalibrationSettingsSerializer(serializers.Serializer):
    chessboard_cols = serializers.IntegerField(required=False)
    chessboard_rows = serializers.IntegerField(required=False)
    square_size_mm = serializers.FloatField(required=False)

class FixedPerspectiveSerializer(serializers.Serializer):
    output_width = serializers.IntegerField(required=False)
    output_height = serializers.IntegerField(required=False)
    homography_matrix = serializers.ListField(child=serializers.ListField(child=serializers.FloatField()), required=False)

class CameraSettingsSerializer(serializers.Serializer):
    minThreshold = serializers.IntegerField()
    maxThreshold = serializers.IntegerField()
    areaFilter = serializers.BooleanField()
    minArea = serializers.FloatField()
    maxArea = serializers.FloatField()
    circularityFilter = serializers.BooleanField()
    minCircularity = serializers.FloatField()
    maxCircularity = serializers.FloatField()
    filterByConvexity = serializers.BooleanField(required=False)
    minConvexity = serializers.FloatField(required=False)
    inertiaFilter = serializers.BooleanField()
    minInertia = serializers.FloatField()
    maxInertia = serializers.FloatField(required=False)
    origin_x = serializers.FloatField(required=False)
    origin_y = serializers.FloatField(required=False)
    picamera_config = PicameraConfigSerializer(required=False)
    calibration = CalibrationSerializer(required=False)
    calibration_settings = CalibrationSettingsSerializer(required=False)
    fixed_perspective = FixedPerspectiveSerializer(required=False)

class MotorsSerializer(serializers.Serializer):
    extruder = MotorSettingsSerializer()
    conveyor = MotorSettingsSerializer()
    syringe = MotorSettingsSerializer()

class SettingsSerializer(serializers.Serializer):
    motors = MotorsSerializer()
    camera = CameraSettingsSerializer()
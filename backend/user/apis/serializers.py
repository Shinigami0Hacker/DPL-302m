from rest_framework import serializers

class UserSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField()
    is_doctor = serializers.BooleanField()



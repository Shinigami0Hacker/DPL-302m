from rest_framework import serializers
from user.models import Patient, Doctor

class PatientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Patient
        fileds = '__all__'

class DoctorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Doctor
        fields = '__all__'




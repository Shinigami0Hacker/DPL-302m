from rest_framework import serializers
from record.models import ModelResultRecord
from user.apis.serializers import DoctorSerializer, PatientSerializer


class ModelResultRecordSerializer(serializers.ModelSerializer):
    doctor = DoctorSerializer()
    patient = PatientSerializer()
    class Meta:
        model = ModelResultRecord
        field = '__all__'

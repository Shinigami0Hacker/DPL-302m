from django.db import models
from django.db.models import ForeignKey

from user.models import Doctor, Patient

class ModelResultRecord(models):
    doctor = ForeignKey(Doctor, on_delete=models.CASCADE)
    patient = ForeignKey(Patient, on_delete=models.CASCADE)
    


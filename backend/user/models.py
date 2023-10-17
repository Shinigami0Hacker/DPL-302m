from django.db import models
from django.contrib.auth.models import AbstractUser
from django.contrib.auth.base_user import BaseUserManager
from django.db import models
from django.db.models import ForeignKey, CharField, DateField

class User(AbstractUser):
    name = CharField()
    days_of_birth = DateField(null=True, blank=True)
    

class Doctor(User):
    """
    
    """
    pass

class Patient(User):
    """
    
    """
    
    responsible_by = ForeignKey(on_delete=models.CASCADE)




class UserManagement(BaseUserManager):
    """
    
    
    """
    pass

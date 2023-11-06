from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin, UserManager
from django.db import models
from django.utils import timezone

class CustomUserManager(UserManager):
    def _create_user(self, email, password, **extra_fields):
        if not email:
            raise ValueError("You have not provided a valid e-mail address")
        
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)

        return user
    
    def create_user(self, email=None, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', False)
        extra_fields.setdefault('is_superuser', False)
        return self._create_user(email, password, **extra_fields)
    
    def create_superuser(self, email=None, password=None, **extra_fields):
        """
        
        """
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        return self._create_user(email, password, **extra_fields)
    
    def authenticate(self, email=None, password=None, **kwrags):
        """

        """
        try:
            if email:
                user = User.objects.get(email = email)
        except User.DoesNotExist as e:
            return (-1, "The user is not exist")
        if not user.check_password(password):
            return (-1, "Wrong password")
        return (1, user)
    
    def get_user(self, user_id):
        try:
            return User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return None
    
class User(AbstractBaseUser, PermissionsMixin):
    """
    
    
    """
    GENDER_CHOICES = [
        ('M', 'Nam'),
        ('F', 'Nữ'),
        ('O', 'Khác'),
    ]
    email = models.EmailField(unique=True)
    
    lname = models.CharField(max_length=255, blank=True, default='')
    fname = models.CharField(max_length=255, blank=True, default='')

    days_of_birth = models.DateTimeField(default=timezone.now)
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES, default='O')

    is_active = models.BooleanField(default=True)
    is_superuser = models.BooleanField(default=False)
    is_staff = models.BooleanField(default=False)

    date_joined = models.DateTimeField(default=timezone.now)
    last_login = models.DateTimeField(blank=True, null=True)

    objects = CustomUserManager()

    USERNAME_FIELD = 'email'
    EMAIL_FIELD = 'email'
    REQUIRED_FIELDS = []

    class Meta:
        verbose_name = 'User'
        verbose_name_plural = 'Users'
    
    def get_full_name(self):
        return self.name
    

class Doctor(models.Model):
    """
    The Doctor model which role to manage the permission.
    @attributes:
        user: mapping to the user.
    """
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    def __str__(self) -> str:
        return self.user

class Patient(models.Model):
    """
    The patient model
    
    """
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    respone_by = models.ForeignKey(Doctor, on_delete=models.CASCADE)





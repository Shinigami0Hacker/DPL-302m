from django.db import models
import uuid
from user.models import User

class Room(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    own = models.ForeignKey(User, on_delete=models.CASCADE)
    path = models.FilePathField()

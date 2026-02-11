from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    class Role(models.TextChoices):
        ADMIN = 'admin', 'Admin'
        ANNOTATOR = 'annotator', 'Annotator'

    role = models.CharField(max_length=20, choices=Role.choices, default=Role.ANNOTATOR)

    REQUIRED_FIELDS = ['email', 'role']

    def is_admin(self):
        return self.role == self.Role.ADMIN

    def is_annotator(self):
        return self.role == self.Role.ANNOTATOR

from django.db import models

# Create your models here.
# models.py
from django.db import models


class User(models.Model):
    user_id = models.CharField(max_length=60)
    first_name = models.CharField(max_length=60)
    last_name = models.CharField(max_length=60)
    password = models.CharField(max_length=60)
    def __str__(self):
        return self.user_id

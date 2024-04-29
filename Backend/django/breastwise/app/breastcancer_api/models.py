""" This class declares the models for use in the API. """

from django.db import models
from django.contrib.postgres.fields import ArrayField


# pylint: disable=E0307
class File(models.Model):
    """ Model for the scan results. """
    file = models.FileField(blank=False, null=False)


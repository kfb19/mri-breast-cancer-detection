""" This class declares the models for use in the API. """

from django.db import models
from django.contrib.postgres.fields import ArrayField


# pylint: disable=E0307
class Results(models.Model):
    """ Model for the scan results."""
    cancerous_slices = ArrayField(models.IntegerField())

    def __str__(self):
        return self.cancerous_slices


class Scan(models.Model):
    """ Model for the uploaded scan."""
    dicom_file = models.FileField()

    def __str__(self):
        return self.dicom_file

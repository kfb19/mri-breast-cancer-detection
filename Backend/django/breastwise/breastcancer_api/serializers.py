""" The serialiser module. """
from rest_framework import serializers

from .models import Scan
from .models import Results


class ScanSerializer(serializers.HyperlinkedModelSerializer):
    """ Defines the scan serialiser. """
    class Meta:
        """ Meta class for scan serialiser."""
        model = Scan
        fields = 'dicom_file'


class ResultsSerializer(serializers.HyperlinkedModelSerializer):
    """ Defines the scan serialiser. """
    class Meta:
        """ Meta class for results serialiser."""
        model = Results
        fields = 'cancerous_slices'

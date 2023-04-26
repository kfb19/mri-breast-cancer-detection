""" Defines the Views class. """

from rest_framework import viewsets
from .serializers import ScanSerializer, ResultsSerializer
from .models import Scan, Results


class ScanViewSet(viewsets.ModelViewSet):
    """ Class for scan viws. """
    queryset = Scan.objects.all().order_by('dicom_file')
    serializer_class = ScanSerializer


class ResultsViewSet(viewsets.ModelViewSet):
    """ Class for results views. """
    queryset = Results.objects.all().order_by('dicom_file')
    serializer_class = ResultsSerializer

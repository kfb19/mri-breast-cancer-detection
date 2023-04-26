""" This module is to register the models. """

from django.contrib import admin
from .models import Scan
from .models import Results

admin.site.register(Scan)
admin.site.register(Results)

""" Defines the required URLs. """

from django.urls import include, path
from .views import FileView
from django.urls import re_path as url

# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path('api-auth/', include('rest_framework.urls',
                              namespace='rest_framework')),
    url(r'^upload/$', FileView.as_view(), name='file-upload')
]

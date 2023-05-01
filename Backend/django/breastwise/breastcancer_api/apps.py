""" This module lists the apps. """

from django.apps import AppConfig


class BreastcancerApiConfig(AppConfig):
    """ Defines the app.

        Args:
            default_auto_field: the auto field.
            name: the app name.
        """
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'breastcancer_api'

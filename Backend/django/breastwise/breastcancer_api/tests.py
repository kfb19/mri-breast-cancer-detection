""" DOCSTRING """

from unittest.mock import patch
from django.test import TestCase
from rest_framework import status
from rest_framework.test import APIClient


class TestFileView(TestCase):
    """ DOCSTRING """

    def setUp(self):
        """ DOCSTRING """
        self.client = APIClient()

    @patch('os.listdir')
    def test_post_request_with_valid_file(self, mock_listdir):
        """ DOCSTRING """
        # Arrange
        mock_listdir.return_value = ['pre', '1st_pass', '2nd_pass', '3rd_pass']
        file_path = 'path/to/valid/file.zip'
        with open(file_path, 'rb') as file:
            data = {'file': file}
            expected_response = {'result': 'expected response'}

        # Act
        response = self.client.post('/api/file/', data, format='multipart')

        # Assert
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data, expected_response)

    @patch('os.listdir')
    def test_post_request_with_invalid_file(self, mock_listdir):
        """ DOCSTRING """
        # Arrange
        mock_listdir.return_value = []
        file_path = 'path/to/invalid/file.zip'
        with open(file_path, 'rb') as file:
            data = {'file': file}
            expected_response = {
                'detail': 'Invalid file name: no series.zip found'}

        # Act
        response = self.client.post('/api/file/', data, format='multipart')

        # Assert
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response.data, expected_response)

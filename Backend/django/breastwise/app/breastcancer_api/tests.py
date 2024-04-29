""" Test file. """

import os
import shutil
import unittest
from unittest import mock

from .views import FileView


class TestFileView(unittest.TestCase):
    """ Test class. """
    def setUp(self):
        """ Test set up. """
        self.file_view = FileView()
        self.test_data = {
            'file': open('media/test.zip', 'rb'),
            'name': 'test.zip'
        }

    def tearDown(self):
        """ Test case. """
        shutil.rmtree('media')

    @mock.patch('os.listdir')
    def test_post_missing_zip_file(self, mock_listdir):
        """ Test case. """
        mock_listdir.return_value = []
        response = self.file_view.post(None)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data, "Invalid file name: no series.zip found")

    @mock.patch('os.listdir')
    def test_post_missing_folders(self, mock_listdir):
        """ Test case. """
        mock_listdir.side_effect = [['series.zip'], []]
        response = self.file_view.post(None)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data, "Invalid folder structure.")

    @mock.patch('os.listdir')
    def test_post_multiple_folders(self, mock_listdir):
        """ Test case. """
        mock_listdir.side_effect = [['series.zip'], ['1st_pass', '2nd_pass', '3rd_pass', 'pre'], ['1st_pass', '2nd_pass', 'pre'], ['pre']]
        response = self.file_view.post(None)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data, "Invalid folder structure.")

    @mock.patch('os.listdir')
    @mock.patch('zipfile.ZipFile.extractall')
    def test_post_valid_file(self, mock_extractall, mock_listdir):
        """ Test case. """
        mock_listdir.side_effect = [['series.zip'], ['1st_pass', '2nd_pass', '3rd_pass', 'pre']]
        response = self.file_view.post(None)
        self.assertEqual(response.status_code, 201)

    def test_delete_folders(self):
        """ Test case. """
        os.mkdir('media')
        os.mkdir('media/temp')
        os.mkdir('media/scantype_bmp')
        os.mkdir('media/scantype_bmp/folder1')
        os.mkdir('media/scantype_bmp/folder2')
        with open('media/temp/test.txt', 'w') as f:
            f.write('test')
        with open('media/scantype_bmp/folder1/test.txt', 'w') as f:
            f.write('test')
        with open('media/scantype_bmp/folder2/test.txt', 'w') as f:
            f.write('test')
        FileView.delete_folders()
        self.assertFalse(os.path.exists('media/temp'))
        self.assertFalse(os.path.exists('media/scantype_bmp'))

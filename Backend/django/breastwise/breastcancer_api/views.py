""" Defines the Views class. """

import zipfile
import os


def scan(request):
    """ DOCSTRING """
     
    uploads_folder = "uploads/"
    zip_name = "uploads/series.zip"

    # Open the zip file for reading.
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        # Extract all files to the specified directory.
        zip_ref.extractall("uploads/")
        
    # Get a list of all files in the folder
    uploads_list = os.listdir(uploads_folder)

    results = []





    # Loop through the file list and delete each file
    for folder in uploads_list:
        if folder != "series.zip":  # DELETE ME LATER
            folder_dir = uploads_folder + folder + "/"
            os.chmod(folder_dir, 0o777)
            files_in_folder = os.listdir(folder_dir)
            for file in files_in_folder:
                path = os.path.join(folder_dir, file)
                print(path)
                os.remove(path)
            os.removedirs(uploads_folder + folder)
    
    return results
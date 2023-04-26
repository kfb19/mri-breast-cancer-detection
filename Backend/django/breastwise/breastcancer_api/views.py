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

    results = []  # The array in which to store the results.

    process_single()
    process_scantype()
    # assert same lengths??
    single_results = [0, 0, 1, 0, 1, 1, 0, 1]  # analyse_single()
    scantype_results = [0, 1, 1, 0, 1, 1, 0, 0]  # analyse_scantype()

    results = average_results(single_results, scantype_results)

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


def process_single():
    """ DOCSTRING """
    print()


def process_scantype():
    """ DOCSTRING """
    print()
    return []


def analyse_single():
    """ DOCSTRING """
    print()
    return []


def analyse_scantype():
    """ DOCSTRING """
    print()
    return []


def average_results(single_results, scantype_results):
    """ Looks at the two arrays and works out an average of
    results based on matching and surrounding values.

    Args:
        single_results: the results from the single CNN in an array
        scantype_results: the results from the scantype CNN in an array
    Returns:
        final_results: an array of final results for the uploaded scan
    """

    final_results = []
    for index, value in enumerate(single_results):
        if scantype_results[index] == value:  # Sets classification values.
            final_results.append(value)  # Values match so correct value.
        else:
            final_results.append(-1)  # "Not possible" value to change.

    for num, result in enumerate(final_results):
        if result == -1:  # If the result was unsure.
            pre_index = num - 1
            post_index = num + 1
            if pre_index >= 0:  # Error handling.
                if final_results[pre_index] == 1:  # There are nearby cells.
                    final_results[num] = 1
            if post_index < len(single_results):  # Error handling.
                if final_results[post_index] == 1:  # There are nearby cells.
                    final_results[num] = 1
            if final_results[num] == -1:  # No nearby cancerous slices.
                final_results[num] = 0

    return final_results

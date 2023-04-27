""" Defines the Views class. """

import zipfile
import os
import numpy as np
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
import pydicom
from skimage.io import imsave
from .serializers import FileSerializer


class FileView(APIView):
    """ DOCSTRING """
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        file_serializer = FileSerializer(data=request.data)

        if file_serializer.is_valid():
            file_serializer.save()
            return Response(file_serializer.data,
                            status=status.HTTP_201_CREATED)
        else:
            return Response(file_serializer.data,
                            status=status.HTTP_400_BAD_REQUEST)


def scan(request):
    """ DOCSTRING """

    uploads_folder = "uploads/"
    single_folder = "uploads/pre"
    pass1_folder = "uploads/1st_pass"
    pass2_folder = "uploads/2nd_pass"
    pass3_folder = "uploads/3rd_pass"
    zip_name = "uploads/series.zip"

    # Open the zip file for reading.
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        # Extract all files to the specified directory.
        zip_ref.extractall("uploads/")

    # Get a list of all files in the folder
    uploads_list = os.listdir(uploads_folder)

    results = []  # The array in which to store the results.

    process_single(single_folder)
    process_scantype(pass1_folder, pass2_folder, pass3_folder)

    single_results = analyse_single()
    scantype_results = analyse_scantype()

    results = average_results(single_results, scantype_results)

    # Loop through the file list and delete each file.
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


def process_single(single_folder):
    """ DOCSTRING """

    counter = 0
    single_bmp_path = "uploads/single_bmp/"
    if not os.path.exists(single_bmp_path):
        os.makedirs(single_bmp_path)
    for dicom_img in single_folder:
        # Create a path to save the slice .bmp file in.
        bmp_path = os.path.join(single_bmp_path, f'{counter}.bmp')
        # Only make the bmp image if it doesn't already exist.
        if not os.path.exists(bmp_path):
            # Load DICOM file with pydicom library.
            dicom = pydicom.dcmread(dicom_img)

            # Convert DICOM into numerical array of pixel intensity values.
            img = dicom.pixel_array

            # Convert uint16 datatype to float, scaled properly for uint8.
            img = img.astype(np.float) * 255. / img.max()

            # Convert from float -> uint8.
            img = img.astype(np.uint8)

            # Invert image if necessary, according to DICOM metadata.
            img_type = dicom.PhotometricInterpretation
            if img_type == "MONOCHROME1":
                img = np.invert(img)

            # Save final .bmp.
            imsave(bmp_path, img)
            counter += 1


def process_scantype(pass1_folder, pass2_folder, pass3_folder):
    """ DOCSTRING """

    array_of_three = []
    counter = 0
    scantype_bmp_path = "uploads/scantype_bmp/"
    if not os.path.exists(scantype_bmp_path):
        os.makedirs(scantype_bmp_path)

    pass2_filenames = set(os.path.basename(pass2_img) for pass2_img in
                          pass2_folder)
    pass3_filenames = set(os.path.basename(pass3_img) for pass3_img in
                          pass3_folder)

    for dicom_img in pass1_folder:
        filename = os.path.basename(dicom_img)
        array_of_three.append(dicom_img)  # CHECK WHETHER FILENAME OR IMG
        if filename in pass2_filenames:
            array_of_three.append(os.path.join(pass2_folder, filename))
        if filename in pass3_filenames:
            array_of_three.append(os.path.join(pass3_folder, filename))

        mini_folder = os.path.join(scantype_bmp_path, f'{counter}')
        if not os.path.exists(mini_folder):
            os.makedirs(mini_folder)
        img_no = 0
        for pass_scan in array_of_three:
            # Create a path to save the slice .bmp file in.
            bmp_path = os.path.join(mini_folder, f'{img_no}.bmp')
            # Only make the bmp image if it doesn't already exist.
            if not os.path.exists(bmp_path):
                # Load DICOM file with pydicom library.
                dicom = pydicom.dcmread(pass_scan)

                # Convert DICOM into numerical array of pixel intensity values.
                img = dicom.pixel_array

                # Convert uint16 datatype to float, scaled properly for uint8.
                img = img.astype(np.float) * 255. / img.max()

                # Convert from float -> uint8.
                img = img.astype(np.uint8)

                # Invert image if necessary, according to DICOM metadata.
                img_type = dicom.PhotometricInterpretation
                if img_type == "MONOCHROME1":
                    img = np.invert(img)

                # Save final .bmp.
                imsave(bmp_path, img)
                img_no += 1
        counter += 1
        array_of_three = []


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

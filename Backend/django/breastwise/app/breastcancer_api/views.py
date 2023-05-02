""" Defines the Views class. """

import zipfile
import os
import numpy as np
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
import pydicom
import torch
from torchvision.models import vgg19
from torchvision import transforms
from torchvision.models import VGG19_Weights
from skimage.io import imsave
from skimage.io import imread
from torch import nn
from .serializers import FileSerializer


# pylint: disable=E1101
# pylint: disable=E1102
# pylint: disable=W0612
# pylint: disable=W0613
# pylint: disable=C0200
class FileView(APIView):
    """ This class creates an API. """
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        """ Defines what happens on a post request. """
        file_serializer = FileSerializer(data=request.data)

        if file_serializer.is_valid():
            file_serializer.save()

            flag = False
            for folder in os.listdir('media/'):
                if folder == 'series.zip':
                    flag = True
            if flag is False:
                delete_folders()
                return Response("Invalid file name: no series.zip found",
                                status=status.HTTP_400_BAD_REQUEST)

            zip_name = "media/series.zip"
            # Open the zip file for reading.
            with zipfile.ZipFile(zip_name, 'r') as zip_ref:
                # Extract all files to the specified directory.
                zip_ref.extractall("media/")

            flag = True

            folders_to_check = ['1st_pass', '2nd_pass', '3rd_pass', 'pre']

            for folder in folders_to_check:
                if not os.path.exists(f"media/{folder}"):
                    flag = False
                    break

            if flag:
                folder_lengths = set()
                for folder in folders_to_check:
                    folder_length = len(os.listdir(f"media/{folder}"))
                    folder_lengths.add(folder_length)

                if len(folder_lengths) > 1:
                    flag = False

            if not flag:
                delete_folders()
                return Response("Invalid folder structure.",
                                status=status.HTTP_400_BAD_REQUEST)

            output = scan()
            return Response(output,
                            status=status.HTTP_201_CREATED)
        else:
            delete_folders()
            return Response(file_serializer.data,
                            status=status.HTTP_400_BAD_REQUEST)


def delete_folders():
    """ Deletes the temporary folders after use. """
    # Get a list of all files in the folder
    uploads_folder = "media/"
    uploads_list = os.listdir(uploads_folder)

    # Loop through the file list and delete each file.
    for folder in uploads_list:
        if ".zip" not in folder and folder != "scantype_bmp":
            folder_dir = uploads_folder + folder + "/"
            os.chmod(folder_dir, 0o777)
            files_in_folder = os.listdir(folder_dir)
            for file in files_in_folder:
                path = os.path.join(folder_dir, file)
                os.remove(path)
            os.removedirs(uploads_folder + folder)
        elif (folder == "scantype_bmp"):
            lst = os.listdir(uploads_folder + folder + "/")
            for fol in lst:
                fol_files = os.listdir(uploads_folder + folder + "/" + fol)
                for i in fol_files:
                    os.remove(uploads_folder + folder + "/" + fol + "/" + i)
                os.chmod(uploads_folder + folder + "/" + fol, 0o777)
                os.removedirs(uploads_folder + folder + "/" + fol)
        else:
            os.remove("media/" + folder)


def scan():
    """ Runs the code for the API CNN scan analysis.

    Returns:
        results: an array of indices of cancerous slices
    """

    single_folder = "media/pre"
    pass1_folder = "media/1st_pass"
    pass2_folder = "media/2nd_pass"
    pass3_folder = "media/3rd_pass"

    results = []  # The array in which to store the results.

    single_folder = os.listdir(single_folder)
    process_single(single_folder)

    pass1_folder = os.listdir(pass1_folder)
    pass2_folder = os.listdir(pass2_folder)
    pass3_folder = os.listdir(pass3_folder)
    process_scantype(pass1_folder, pass2_folder, pass3_folder)

    single_results = analyse_single()

    scantype_results = analyse_scantype()
    results = average_results(single_results, scantype_results)

    delete_folders()

    return results


def process_single(single_folder):
    """ Processes the single data.

    Args:
        single_folder: the folder for the pre images
    """

    counter = 0
    single_bmp_path = "media/single_bmp/"
    if not os.path.exists(single_bmp_path):
        os.makedirs(single_bmp_path)
    for dicom_img in single_folder:
        # Create a path to save the slice .bmp file in.
        bmp_path = os.path.join(single_bmp_path, f'{counter}.bmp')
        # Only make the bmp image if it doesn't already exist.
        if not os.path.exists(bmp_path):
            # Load DICOM file with pydicom library.
            dicom = pydicom.dcmread("media/pre/" + dicom_img)

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
    """ Processes the scantype data.

    Args:
        pass1_folder: the folder for the 1st pass images
        pass2_folder: the folder for the 2nd pass images
        pass3_folder: the folder for the 3rd pass images
    """

    array_of_three = []
    counter = 0
    scantype_bmp_path = "media/scantype_bmp/"
    if not os.path.exists(scantype_bmp_path):
        os.makedirs(scantype_bmp_path)

    pass2_filenames = set(os.path.basename(pass2_img) for pass2_img in
                          pass2_folder)
    pass3_filenames = set(os.path.basename(pass3_img) for pass3_img in
                          pass3_folder)

    for dicom_img in pass1_folder:
        filename = os.path.basename(dicom_img)
        array_of_three.append(os.path.join("media/1st_pass", filename))
        if filename in pass2_filenames:
            array_of_three.append(os.path.join("media/2nd_pass", filename))
        if filename in pass3_filenames:
            array_of_three.append(os.path.join("media/2nd_pass", filename))

        mini_folder = os.path.join(scantype_bmp_path, f'{counter}')
        if not os.path.exists(mini_folder):
            os.umask(0)
            os.makedirs(mini_folder, mode=0o777)
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


def normalize(img):
    """ Normalises image pixel values to range [0, 255].

    Args:
        img: the array for each image/scan
    Returns:
        img: the edited array for each image/scan
    """

    # Convert uint16 -> float.
    img = img.astype(float) * 255. / img.max()
    # Convert float -> unit8.
    img = img.astype(np.uint8)

    return img


def analyse_single():
    """ Analyses the single results with the VGG19 model

    Returns:
        results: the array of classifications for the scan slices
    """

    results = []
    # Preprocess each image.
    directory = "media/single_bmp/"
    for fname in os.listdir(directory):
        if '.bmp' in fname:
            fpath = os.path.join(directory, fname)
            # Load img from file (bmp).
            img_arr = imread(fpath, as_gray=True)

            # Normalise image.
            img_arr = normalize(img_arr)

            # Convert to Tensor (PyTorch matrix).
            data_tensor = torch.from_numpy(img_arr)
            data_tensor = data_tensor.type(torch.FloatTensor)

            # Add image channel dimension (to work with the CNN).
            data_tensor = torch.unsqueeze(data_tensor, 0)

            # Resize image.
            data_tensor = transforms.Resize(
                (128, 128))(data_tensor)
            # Run through net & append results to results
            checkpoint = torch.load("nets/vgg_single_pretrained.pth",
                                    map_location=torch.device('cpu'))

            # Define the convoluted neural network.
            net = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

            # Modify the first convolutional layer to accept one channel input.
            net.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7),
                                        stride=(2, 2), padding=(3, 3),
                                        bias=False)
            # Modify all other convolutional layers to accept one channel.
            for i, layer in enumerate(net.features):
                if isinstance(layer, nn.Conv2d):
                    layer.in_channels = 1
            net.load_state_dict(checkpoint)
            net.eval()
            data_tensor = data_tensor.unsqueeze(0)
            output = net(data_tensor)
            _, predicted_class = output.max(1)
            results.append(predicted_class)

    return results


def analyse_scantype():
    """ Analyses the scantype results with the VGG19 model

    Returns:
        results: the array of classifications for the scan slices
    """

    results = []
    # Preprocess each image.
    directory = "media/scantype_bmp/"
    # Iterate over all images in the class/case type.
    for folder in os.listdir(directory):
        group = []
        for fname in os.listdir(os.path.join(directory, folder)):
            if '.bmp' in fname:
                fpath = os.path.join(directory, folder, fname)
                # Load img from file (bmp).
                img_arr = imread(fpath, as_gray=True)

                # Normalise image.
                img_arr = normalize(img_arr)

                # Convert to Tensor (PyTorch matrix).
                data_tensor = torch.from_numpy(img_arr).type(
                    torch.FloatTensor)

                # Convert to a 4D tensor for resize operation.
                data_tensor = data_tensor.unsqueeze(0).unsqueeze(0)

                # Resize image.
                data_tensor = transforms.Resize(
                    (128, 128))(data_tensor)

                # Remove extra dimensions.
                data_tensor = data_tensor.squeeze(0).squeeze(0)

                group.append(data_tensor)

        # Create RGB image tensor with the 3 images as channels.
        data = torch.stack(group, dim=0)
        data = torch.cat([data[0:1], data[1:2], data[2:3]], dim=0)
        group = []

        # Run through net & append results to results
        checkpoint = torch.load("nets/vgg_scantype.pth",
                                map_location=torch.device('cpu'))

        # Define the convoluted neural network.
        net = vgg19(weights=None)

        # This network takes a 3 channel input.
        net.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7),
                              stride=(2, 2), padding=(3, 3), bias=False)
        net.load_state_dict(checkpoint)
        net.eval()
        data = data.unsqueeze(0)
        output = net(data)
        _, predicted_class = output.max(1)
        results.append(predicted_class)

    return results


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

    result_indexes = []
    for index in range(len(final_results)):
        if final_results[index] == 1:
            result_indexes.append(index)
    return result_indexes

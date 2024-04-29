""" This module preprocesses data used for breast cancer classification
using each image as a singular 1-channel input. It processes the images
and saves them as bitmaps.
"""

import os
import pandas as pd
import numpy as np
import pydicom
from tqdm import tqdm
from skimage.io import imsave


def read_data(boxes_path, mapping_path):
    """ Reads the bounding boxes and mapping data to be stored in a 2D array.

    Args:
        boxes_path: the bounding boxes storage path
        mapping_path: the path to the file that maps bounding boxes to the data
    Returns:
        bounding_boxes: a data structure containing the bounding boxes
        scan_data: data of the MRI scans
    """

    # Reading the bounding boxes data.
    bounding_boxes = pd.read_csv(boxes_path)
    # Reading the mapping path data of fat-saturated "pre" exams.
    scan_data = pd.read_csv(mapping_path)
    # Each row refers to a different 2D slice of a 3D volume.
    scan_data = scan_data[scan_data['original_path_and_filename'].str.
                          contains('pre')]

    return bounding_boxes, scan_data


def save_dicom_to_bitmap(dicom_filename, label, patient_index, target_bmp_dir):
    """ Saves the dicom images in a bitmap file format.

    Args:
        dicom_filename: the file name for the dicom image
        label: the positive/negative label of the dicom files as 1 or 0
        patient_index: the index number of the patient
        target_bmp_dir: the target directory for the bitmap
    Raises:
        FileNotFoundError: if the DICOM file is not found
    """

    # Create a path to save the slice .bmp file in.
    # Do this according to the original DICOM filename and target label.
    bmp_path = dicom_filename.split('/')[-1].replace(
        '.dcm', f'-{patient_index}.bmp')
    if label == 1:
        cancer_status = 'pos'
    else:
        cancer_status = 'neg'
    bmp_path = os.path.join(target_bmp_dir, cancer_status, bmp_path)

    # If no path exists, make one.
    if not os.path.exists(os.path.join(target_bmp_dir, cancer_status)):
        os.makedirs(os.path.join(target_bmp_dir, cancer_status))

    # Only make the bmp image if it doesn't already exist.
    if not os.path.exists(bmp_path):
        # Load DICOM file with pydicom library.
        try:
            dicom = pydicom.dcmread(dicom_filename)
        except FileNotFoundError:
            # Fix possible errors in filename from list.
            dicom_filename_split = dicom_filename.split('/')
            dicom_filename_end = dicom_filename_split[-1]
            assert dicom_filename_end.split('-')[1][0] == '0'

            # Dicom filename end split.
            dicom_filename_e_split = dicom_filename_end.split('-')
            dicom_filename_end = '-'.join([dicom_filename_e_split[0],
                                           dicom_filename_e_split[1][1:]])

            dicom_filename_split[-1] = dicom_filename_end
            dicom_filename = '/'.join(dicom_filename_split)
            dicom = pydicom.dcmread(dicom_filename)

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

        print("New image saved")


def main():
    """ Main function to run code to collect data for
    consecutive 3 channel images.
    """
    # Setting file paths needed for using the data.
    data_path = 'PATH\\manifest-1675379375384'
    boxes_path = 'PATH\\csvs\\Annotation_Boxes.csv'
    mapping_path = 'PATH\\csvs\\Breast-Cancer-MRI-mapping.csv'
    target_bmp_dir = 'PATH\\bmp_out_single_classify'
    if not os.path.exists(target_bmp_dir):
        os.makedirs(target_bmp_dir)

    # Setting the bounding boxes and dicom data variables.
    boxes, data = read_data(boxes_path, mapping_path)

    # Image extraction.

    # Number of examples for each class.
    n_class = 22500
    # Counts of examples extracted from each class.
    neg_extracted = 0
    pos_extracted = 0

    # Initialise iteration index of each patient volume.
    vol_index = -1  # Scan index.
    for _, row in tqdm(data.iterrows(), total=n_class*2):
        # Indices start at 1 here.
        new_vol_idx = int((row['original_path_and_filename'].
                           split('/')[1]).split('_')[-1])
        slice_idx = int(((row['original_path_and_filename'].
                        split('/')[-1]).split('_')[-1]).replace('.dcm', ''))

        # New scan - get tumor bounding box.
        if new_vol_idx != vol_index:
            box_row = boxes.iloc[[new_vol_idx-1]]
            start_slice = int(box_row['Start Slice'])
            end_slice = int(box_row['End Slice'])
            assert end_slice >= start_slice
        vol_index = new_vol_idx

        # Get DICOM filename.
        dcm_fname = str(row['classic_path'])
        dcm_fname = os.path.join(data_path, dcm_fname)

        # Determine slice label -> 1 if positive.
        if slice_idx >= start_slice and slice_idx < end_slice:
            if pos_extracted >= n_class:
                continue
            save_dicom_to_bitmap(dcm_fname, 1, vol_index, target_bmp_dir)
            pos_extracted += 1

        # Determine slice label -> 0 if negative.
        # Negative is defined as at least 5 slices from a positive image.
        elif (slice_idx + 5) <= start_slice or (slice_idx - 5) > end_slice:
            if neg_extracted >= n_class:
                continue
            save_dicom_to_bitmap(dcm_fname, 0, vol_index, target_bmp_dir)
            neg_extracted += 1


if __name__ == "__main__":
    main()

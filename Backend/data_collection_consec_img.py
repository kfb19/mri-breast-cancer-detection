""" INSERT A MODULE DOCSTRING HERE """

import os
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm
from skimage.io import imsave


def read_data(boxes_path, mapping_path):
    """Reads the data to be stored in a 2D array.

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
    """Saves the dicom images in a bitmap file format.

    Args:
        dicom_filename: the name of the dicom file
        label: the positive/negative label of the dicom file as 1 or 0
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

            dicom_filename_end_split = dicom_filename_end.split('-')
            dicom_filename_end = '-'.join([dicom_filename_end_split[0],
                                           dicom_filename_end_split[1][1:]])

            dicom_filename_split[-1] = dicom_filename_end
            dicom_filename = '/'.join(dicom_filename_split)
            dicom = pydicom.dcmread(dicom_filename)

        # Convert DICOM into numerical numpy array of pixel intensity values.
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


# Setting file paths needed for using the data.
DATA_PATH = 'E:\\data\\manifest-1675379375384'
BOXES_PATH = 'E:\\data\\csvs\\Annotation_Boxes.csv'
MAPPING_PATH = 'E:\\data\\csvs\\Breast-Cancer-MRI-mapping.csv'
TARGET_BMP_DIR = 'E:\\data\\output\\bmp_out'
if not os.path.exists(TARGET_BMP_DIR):
    os.makedirs(TARGET_BMP_DIR)

# Setting the bounding boxes and dicom data variables.
boxes, data = read_data(BOXES_PATH, MAPPING_PATH)

# Image extraction.

# Number of examples for each class.
N_CLASS = 5000  # CHANGEME
# Counts of examples extracted from each class.
NEG_EXTRACTED = 0
POS_EXTRACTED = 0

# Initialise iteration index of each patient volume.
IMG_TOTAL = 160  # How many images there are per volume.
IMG_COUNT = 0  # How many images looked through.
VOL_INDEX = -1
for row_idx, row in tqdm(data.iterrows(), total=N_CLASS*2):
    # Indices start at 1 here.
    new_vol_idx = int((row['original_path_and_filename'].
                       split('/')[1]).split('_')[-1])
    slice_idx = int(((row['original_path_and_filename'].
                      split('/')[-1]).split('_')[-1]).replace('.dcm', ''))

    # New volume - get tumor bounding box.
    if new_vol_idx != VOL_INDEX:
        box_row = boxes.iloc[[new_vol_idx-1]]
        start_slice = int(box_row['Start Slice'])
        end_slice = int(box_row['End Slice'])
        assert end_slice >= start_slice
        IMG_COUNT = 0
    VOL_INDEX = new_vol_idx

    # Get DICOM filename.
    DCM_FNAME = str(row['classic_path'])
    DCM_FNAME = os.path.join(DATA_PATH, DCM_FNAME)

    # code in here???? DELME for 3channels

    if IMG_COUNT < IMG_TOTAL:
        print("Here")

    # Determine slice label:
    # If within 3D bounding box, save as positive.
    if slice_idx >= start_slice and slice_idx < end_slice:
        if POS_EXTRACTED >= N_CLASS:
            continue
        save_dicom_to_bitmap(DCM_FNAME, 1, VOL_INDEX, TARGET_BMP_DIR)
        POS_EXTRACTED += 1

    # If outside 3D box by more than 5 slices, save as negative.
    elif (slice_idx + 5) <= start_slice or (slice_idx - 5) > end_slice:
        if NEG_EXTRACTED >= N_CLASS:
            continue
        save_dicom_to_bitmap(DCM_FNAME, 0, VOL_INDEX, TARGET_BMP_DIR)
        NEG_EXTRACTED += 1

""" This module preprocesses data used for breast cancer classification
using the RGB channels for consecutive images (e.g. 1/2/3, 4/5/6...). It
processes the images, saves them as bitmaps, and groups them in threes.
"""

import os
import csv
import numpy as np
import pandas as pd
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


def save_dicom_to_bitmap(array_of_three, label, patient_index, target_bmp_dir,
                         triple_count):
    """ Saves the dicom images in a bitmap file format.

    Args:
        array_of_three: the name of the dicom files in an array
        label: the positive/negative label of the dicom files as 1 or 0
        patient_index: the index number of the patient
        target_bmp_dir: the target directory for the bitmap
        triple_count: a number used to name the folders created
    Raises:
        FileNotFoundError: if the DICOM file is not found
    """

    for dicom_filename in array_of_three:
        # Create a path to save the slice .bmp file in.
        # Do this according to the original DICOM filename and target label.
        img_path = dicom_filename.split('/')[-1].replace(
            '.dcm', f'-{patient_index}.bmp')
        if label == 1:
            cancer_status = 'pos'
        else:
            cancer_status = 'neg'
        bmp_path = os.path.join(target_bmp_dir, cancer_status,
                                str(triple_count), img_path)

        # If no path exists, make one.
        if not os.path.exists(os.path.join(target_bmp_dir, cancer_status)):
            os.makedirs(os.path.join(target_bmp_dir, cancer_status))

        if not os.path.exists(os.path.join(target_bmp_dir, cancer_status,
                                           str(triple_count))):
            os.makedirs(os.path.join(target_bmp_dir,
                        cancer_status, str(triple_count)))

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
            img = img.astype(np.float64) * 255. / img.max()

            # Convert from float -> uint8.
            img = img.astype(np.uint8)

            # Invert image if necessary, according to DICOM metadata.
            img_type = dicom.PhotometricInterpretation
            if img_type == "MONOCHROME1":
                img = np.invert(img)

            # Save final .bmp.
            imsave(bmp_path, img)

            print("New image saved")

    return img_path  # For bounding box data.


def determine_pos_neg(slice_indexes, start, end):
    """ Determines whether a grouped array of 3 slices contain a tumour.

    Args:
        slice_indexes: the indexes (number) of the slices being looked at
        start: the start slice of the bounding box
        end: the end slice of the bounding box
    Returns:
        0: if it does not contain a positive scan
        1: if it does contain a positive scan
    """
    for index in range(3):
        # Determine slice label:
        # If within 3D bounding box, save as positive.
        if slice_indexes[index] > start and slice_indexes[index] <= end:
            return 1
    return 0


def save_bounding_box_data(label, img_id, volume, boxes, csv_path):
    """ Saves data on the bounding boxes in a .csv.

    Args:
        target_dir: the target results directory
        label: 0 for no cancer, 1 for cancer
        img_id: a specific ID for that image
        volume: patient index number
        boxes: the bounding box data
        csv_path: the path for the csv file
    """
    if label == 1:
        cancer_status = 'pos'
    else:
        cancer_status = 'neg'

    annotation_info = [img_id]

    if cancer_status == 'pos':
        # Add bounding box data only if positive scan.
        box_row = boxes.iloc[[volume-1]]
        xmin = int(box_row['Start Column'])
        annotation_info.append(xmin)
        ymin = int(box_row['Start Row'])
        annotation_info.append(ymin)
        xmax = int(box_row['End Column'])
        annotation_info.append(xmax)
        ymax = int(box_row['End Row'])
        annotation_info.append(ymax)

        annotation_info.append(label)

        # Open a .csv file for writing in 'append' mode.
        with open(csv_path, mode='a', encoding='utf-8',
                  newline='') as csv_file:
            # Create a .csv writer object.
            writer = csv.writer(csv_file)
            writer.writerow(annotation_info)
            csv_file.close()


def main():
    """ Main function to run code to collect data for
    consecutive 3 channel images.
    """
    # Setting file paths needed for using the data.
    data_path = 'E:\\data\\manifest-1675379375384'
    boxes_path = 'E:\\data\\csvs\\Annotation_Boxes.csv'
    mapping_path = 'E:\\data\\csvs\\Breast-Cancer-MRI-mapping.csv'
    target_bmp_dir = 'E:\\data\\output\\bmp_out_consec_localise'
    if not os.path.exists(target_bmp_dir):
        os.makedirs(target_bmp_dir)

    # Setting the bounding boxes and dicom data variables.
    boxes, data = read_data(boxes_path, mapping_path)

    bb_path = os.path.join(target_bmp_dir, "bounding_boxes")
    # If no path exists, make one.
    if not os.path.exists(bb_path):
        os.makedirs(bb_path)
    csv_path = os.path.join(bb_path, "bounding_boxes.csv")
    # Open a .csv file for writing in 'append' mode.
    with open(csv_path, mode='a', encoding='utf-8', newline='') as csv_file:
        # Create a .csv writer object.
        header_info = ['image_id', 'xmin', 'ymin', 'xmax', 'ymax', 'label']
        writer = csv.writer(csv_file)
        writer.writerow(header_info)
    csv_file.close()

    # Image extraction.

    # Number of examples for each class.
    n_class = 500  # CHANGE ME!!!!
    # Counts of examples extracted from each class.
    neg_extracted = 0
    pos_extracted = 0

    # Initialise iteration index of each patient volume.
    img_count = 1  # How many images looked through.
    vol_index = -1  # Scan index.
    triple_count = 1
    array_of_three = []
    slice_indexes = []
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
            img_count = 1
            array_of_three = []
        vol_index = new_vol_idx

        # Get DICOM filename.
        dcm_fname = str(row['classic_path'])
        dcm_fname = os.path.join(data_path, dcm_fname)

        array_of_three.append(dcm_fname)
        slice_indexes.append(slice_idx)

        # Determine label of group of three.
        if img_count % 3 == 0:
            triple_count += 1

            status = determine_pos_neg(slice_indexes, start_slice, end_slice)

            if status == 1:
                pos_extracted += 3
                if pos_extracted <= n_class:
                    img_id = save_dicom_to_bitmap(array_of_three, status,
                                                  vol_index, target_bmp_dir,
                                                  triple_count)
                    save_bounding_box_data(status, img_id, vol_index, boxes,
                                           csv_path)
            else:
                neg_extracted += 3
                if neg_extracted <= n_class:
                    img_id = save_dicom_to_bitmap(array_of_three, status,
                                                  vol_index, target_bmp_dir,
                                                  triple_count)
                    save_bounding_box_data(status, img_id, vol_index, boxes,
                                           csv_path)

            array_of_three = []
            slice_indexes = []

        img_count += 1


if __name__ == "__main__":
    main()

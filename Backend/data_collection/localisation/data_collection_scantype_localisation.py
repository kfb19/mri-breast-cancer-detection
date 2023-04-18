""" This module preprocesses data used for breast cancer localisation
using each image as a singular 1-channel input. It processes the bounding
box information.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm


def create_box_data():
    """Creates a .csv of bounding box data ready for use.

    Args:
        
    Returns:
        
        """


def read_data(boxes_path):
    """ Reads the bounding boxes data to be stored in a 2D array.

    Args:
        boxes_path: the bounding boxes storage path
    Returns:
        bounding_boxes: a data structure containing the bounding boxes
    """

    # Reading the bounding boxes data.
    bounding_boxes = pd.read_csv(boxes_path)

    return bounding_boxes


def main():
    """ Main function to run code to collect data for
    consecutive 3 channel images on the bounding boxes.
    """
    # Setting file paths needed for using the data.
    pos_data_path = 'E:\\data\\output\\bmp_out_single_classify\\pos'
    neg_data_path = 'E:\\data\\output\\bmp_out_single_classify\\neg'
    boxes_path = 'E:\\data\\csvs\\Annotation_Boxes.csv'
    target_bmp_dir = 'E:\\data\\output\\bmp_out_single_boxes'
    if not os.path.exists(target_bmp_dir):
        os.makedirs(target_bmp_dir)

    # Setting the bounding box variable.
    boxes = read_data(boxes_path)

    # Bounding box data extraction.

    # Number of examples for each class.
    n_class = 22500
    # Counts of examples extracted from each class.
    neg_extracted = 0
    pos_extracted = 0

    # For loop to work through positive images.
    for img in range(n_class):
        
    
    
    
    
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
            create_box_data()
            pos_extracted += 1

        # Determine slice label -> 0 if negative.
        # Negative is defined as at least 5 slices from a positive image.
        # No bounding box data needed for negative images.
        elif (slice_idx + 5) <= start_slice or (slice_idx - 5) > end_slice:
            if neg_extracted >= n_class:
                continue
            create_box_data()
            neg_extracted += 1


if __name__ == "__main__":
    main()

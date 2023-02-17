from sys import displayhook
import pandas as pd
import numpy as np
import os
import pydicom
from PIL import Image 
from tqdm import tqdm
from skimage.io import imsave

def read_data(boxes_path, mapping_path):
   # Reading the bounding boxes data. 
   boxes = pd.read_csv(boxes_path)

   # Reading the mapping path data. 
   # Only consider fat-satured "pre" exams for the time being 
   data = pd.read_csv(mapping_path)
   data = data[data['original_path_and_filename'].str.contains('pre')] # Each row refers to a different 2D slice of a 3D volume. Pre = only fat saturated images. 
   return boxes, data

def save_dicom_to_bitmap(dicom_filename, label, patient_index, target_bmp_dir): # Here label is 1 or 0 (1 being contains cancer, 0 being at least 5 slides away from positive slices)
   # Create a path to save the slice .bmp file in, according to the original DICOM filename and target label
   bmp_path = dicom_filename.split('/')[-1].replace('.dcm', '-{}.bmp'.format(patient_index))
   if (label == 1):
      cancer_status = 'pos' 
   else:
      cancer_status = 'neg'
   bmp_path = os.path.join(target_bmp_dir, cancer_status, bmp_path)

   if not os.path.exists(os.path.join(target_bmp_dir, cancer_status)):
        os.makedirs(os.path.join(target_bmp_dir, cancer_status))

   if not os.path.exists(bmp_path):
      # Only make the bmp image if it doesn't already exist (if you're running this after the first time)

      # Load DICOM file with pydicom library
      try:
         dicom = pydicom.dcmread(dicom_filename)
      except FileNotFoundError:
         # Fix possible errors in filename from list
         dicom_filename_split = dicom_filename.split('/')
         dicom_filename_end = dicom_filename_split[-1]
         assert dicom_filename_end.split('-')[1][0] == '0'

         dicom_filename_end_split = dicom_filename_end.split('-')
         dicom_filename_end = '-'.join([dicom_filename_end_split[0], dicom_filename_end_split[1][1:]])

         dicom_filename_split[-1] = dicom_filename_end
         dicom_filename = '/'.join(dicom_filename_split)
         dicom = pydicom.dcmread(dicom_filename)

   # Convert DICOM into numerical numpy array of pixel intensity values
   img = dicom.pixel_array

   # Convert uint16 datatype to float, scaled properly for uint8
   img = img.astype(np.float) * 255. / img.max()

   # convert from float -> uint8
   img = img.astype(np.uint8)

   # invert image if necessary, according to DICOM metadata
   img_type = dicom.PhotometricInterpretation
   if img_type == "MONOCHROME1":
            img = np.invert(img)

   # Save final .bmp
   img.save(bmp_path)
   #imsave(bmp_path, img)

   print("Image saved")

def prepare_data(boxes, data, target_bmp_dir):
    print()

# Setting file paths needed for using the data. 
data_path = 'E:\data\manifest-1675379375384' 
boxes_path = 'E:\data\csvs\Annotation_Boxes.csv'
mapping_path = 'E:\data\csvs\Breast-Cancer-MRI-filepath_filename-mapping.csv'
target_bmp_dir = 'E:\data\output\\bmp_out'
if not os.path.exists(target_bmp_dir):
   os.makedirs(target_bmp_dir)

# Setting the bounding boxes and dicom data variables. 
boxes, data = read_data(boxes_path, mapping_path) 

# Preparing the data ready for training. 
prepare_data(boxes, data, target_bmp_dir)

# Saving the dicom image formats as bitmaps. 
#save_dicom_to_bitmap(data, 0, 1, "/Backend")

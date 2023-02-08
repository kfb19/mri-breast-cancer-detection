from sys import displayhook
import pandas as pd
import numpy as np
import os
import pydicom
from tqdm import tqdm
#from skimage.io import imsave

# Setting file paths needed for using the data. 
data_path = 'E:\data\manifest-1675379375384' 
boxes_path = 'E:\data\csvs\Annotation_Boxes.csv'
mapping_path = 'E:\data\csvs\Breast-Cancer-MRI-filepath_filename-mapping.csv'
target_png_dir = 'E:\data\output\png_out'
if not os.path.exists(target_png_dir):
   os.makedirs(target_png_dir)

# Reading the bounding boxes data. 
boxes_df = pd.read_csv(boxes_path)
print(boxes_df)

# Reading the mapping path data. 
mapping_df = pd.read_csv(mapping_path)
# Might need to change this once decided on what type of images to use? 


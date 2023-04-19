""" This module implements ResNet50 with a single
input channel (as images are greyscale). It trains,
validates and tests a ResNet50 classification CNN
on Breast Cancer MRI scan slices, then calculates
results for performance.
"""

import os
from PIL import Image
from tqdm import tqdm
import csv


# pylint: disable=E1101
# pylint: disable=E1102
def main():
    """ Runs the bulk of the CNN code.
        Implements ResNet50 with single-channel input.
        """

    # Directory information.
    pos_data_dir = 'E:\\data\\output\\bmp_out_scantype_localise\\pos'
    bb_data_dir = 'E:\\data\\output\\bmp_out_scantype_localise\\bounding_boxes'
    bb_data_path = bb_data_dir + '\\bounding_boxes.csv'
    results_path = "E:\\data\\output\\yolo_data\\scantype\\datasets"
    images_path = os.path.join(results_path, "images")
    labels_path = os.path.join(results_path, "labels")

    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    if not os.path.exists(labels_path):
        os.makedirs(labels_path)

    # set the output size of the resized images
    output_size = (128, 128)
    # total img counter CHANGE ME
    counter = 0
    total_images = 500

    # find data splits
    train_split = 0.6
    val_split = 0.2
    train_total = train_split * total_images
    val_total = val_split * total_images

    # loop over all folders in the "pos" folder
    for folder in tqdm(os.listdir(
            pos_data_dir), desc='Processing pos folders'):
        # create a list to hold the three images in the current folder
        images = []

        # loop over all files in the current folder
        for filename in os.listdir(os.path.join(pos_data_dir, folder)):
            if not filename.endswith(".bmp"):
                continue

            # open the image file
            filepath = os.path.join(os.path.join(pos_data_dir, folder),
                                    filename)
            with Image.open(filepath) as img:
                # resize the image and add it to the list
                img_resized = img.resize(output_size)
                images.append(img_resized)

        # combine the three images into a single RGB image
        combined_image = Image.merge("RGB", images)
        output_img_filename = f"{folder}.bmp"
        output_txt_filename = f"{folder}.txt"

        if counter < train_total:
            output_folderpath_img = os.path.join(images_path, "train")
            output_folderpath_lab = os.path.join(labels_path, "train")
        elif counter < train_total + val_total:
            output_folderpath_img = os.path.join(images_path, "val")
            output_folderpath_lab = os.path.join(labels_path, "val")
        else:
            output_folderpath_img = os.path.join(images_path, "test")
            output_folderpath_lab = os.path.join(labels_path, "test")
        # save the combined image to a file
        if not os.path.exists(output_folderpath_img):
            os.makedirs(output_folderpath_img)
        output_filepath = os.path.join(output_folderpath_img,
                                       output_img_filename)
        combined_image.save(output_filepath)
        # Bounding boxes.
        with open(bb_data_path, mode="r", encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file)
            current_row = None
            for i, row in enumerate(csv_reader):
                if i == counter + 1:
                    current_row = row
                    break
            if current_row:
                # Process the current row
                img_size = 448
                xmin = float(current_row[1])
                xmax = float(current_row[3])
                ymin = float(current_row[2])
                ymax = float(current_row[4])
                width = (xmax - xmin) / img_size
                height = (ymax - ymin) / img_size
                x_centre = (xmax + xmin) / (2 * img_size)
                y_centre = (ymax + ymin) / (2 * img_size)
                if not os.path.exists(output_folderpath_lab):
                    os.makedirs(output_folderpath_lab)
                output_bb_path = os.path.join(output_folderpath_lab,
                                              output_txt_filename)
                with open(output_bb_path, "w", encoding="utf-8") as txt_file:
                    txt_file.write(str(0))
                    txt_file.write(" ")
                    txt_file.write(str(x_centre))
                    txt_file.write(" ")
                    txt_file.write(str(y_centre))
                    txt_file.write(" ")
                    txt_file.write(str(width))
                    txt_file.write(" ")
                    txt_file.write(str(height))
                txt_file.close()

        # format: class x_center y_center width height
        csv_file.close()
        counter = counter + 1


if __name__ == "__main__":
    main()

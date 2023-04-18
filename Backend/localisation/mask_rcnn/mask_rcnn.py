""" This module implements Mask R-CNN with a single
input channel (as images are greyscale). It trains,
validates and tests a Mask R-CNN classification CNN
on Breast Cancer MRI scan slices, then calculates
results for performance.
"""

import random
import os
from copy import deepcopy
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from skimage.io import imread
import matplotlib.pyplot as plt
from evaluation import Evaluation
from early_stopper import EarlyStopper


# pylint: disable=E1101
# pylint: disable=E1102
class ScanDataset(Dataset):
    """ This class creates a dataset of images
    from which to train, test and validate the
    Mask R-CNN CNN.
    """

    def __init__(self, data_dir, img_size):
        """ Initialises the dataset of scans.

        Args:
            data_dir: the directory of the data
            img_size: the size of the images needed for CNN input
        """
        self.data_dir = data_dir
        self.img_size = img_size

        # Assign labels to data within this Dataset.
        self.labels = None
        self.create_labels()

    def create_labels(self):
        """ Creates and stores labels (positive or negative)
        for scans within the dataset. Converts scans to tensors.
        Each label is a tuple: (image tensor, label number (0 or 1)).

        Args:
            data_dir: the directory of the data
            img_size: the size of the images needed for CNN input
        """
        labels = []
        print('Building dataset labels...')

        # Get bounding box data.
        bbox_path = os.path.join(self.data_dir, "bounding_boxes",
                                 "bounding_boxes.csv")
        bbox_data = pd.read_csv(bbox_path)

        # Iterate over each class.
        for target, target_label in enumerate(['neg', 'pos']):
            case_dir = os.path.join(self.data_dir, target_label)
            # Iterate over all images in the class/case type.
            for fname in os.listdir(case_dir):
                if '.bmp' in fname:
                    fpath = os.path.join(case_dir, fname)
                    # Load img from file (bmp).
                    img_arr = imread(fpath, as_gray=True)

                    # Normalise image.
                    img_arr = self.normalize(img_arr)

                    # Convert to Tensor (PyTorch matrix).
                    data_tensor = torch.from_numpy(img_arr).cuda()
                    data_tensor = data_tensor.type(torch.FloatTensor)

                    # Add image channel dimension (to work with the CNN).
                    data_tensor = torch.unsqueeze(data_tensor, 0)

                    # Resize image.
                    data_tensor = transforms.Resize(
                        (self.img_size, self.img_size))(data_tensor)

                    if target_label == 'pos':

                        # Find the corresponding bounding box data for image.
                        bbox_row = bbox_data.loc[bbox_data[
                            'image_id'] == fname]

                        # Extract the bounding box coordinates.
                        xmin = bbox_row['xmin'].values[0]
                        ymin = bbox_row['ymin'].values[0]
                        xmax = bbox_row['xmax'].values[0]
                        ymax = bbox_row['ymax'].values[0]

                        # Create a dictionary to store the target data
                        target = {
                            'boxes': torch.tensor([[xmin, ymin, xmax, ymax]]),
                            'labels': torch.tensor([1]),
                            'image_id': data_tensor,
                            'area': torch.tensor([(xmax - xmin) * (
                                ymax - ymin)]),
                            'iscrowd': torch.tensor([0])  # No crowd.
                        }
                    else:
                        # Create a dictionary to store the target data
                        target = {
                            'boxes': torch.tensor([[0, 0, 0, 0]]),
                            'labels': torch.tensor([0]),
                            'image_id': data_tensor,
                            'area': torch.tensor([0]),
                            'iscrowd': torch.tensor([0])  # No crowd.
                        }

                    # Append label to list.
                    labels.append((data_tensor, target))

        self.labels = labels

    def normalize(self, img):
        """ Normalises image pixel values to range [0, 255].

        Args:
            img: the array for each image/scan.
        Returns:
            img: the edited array for each image/scan.
        """

        # Convert uint16 -> float.
        img = img.astype(float) * 255. / img.max()
        # Convert float -> unit8.
        img = img.astype(np.uint8)

        return img

    def __getitem__(self, idx):
        """ Required method for accessing data samples.

        Args:
            idx: the index of the data to be accessed.
        Returns:
            data: the data sample as a PyTorch Tensor.
            target: the target classification/labels for that data.
        """

        data, target = self.labels[idx]

        return data, target

    def __len__(self):
        """ Required method for getting the dataset size.

        Returns:
            size: the length of the dataset.
        """
        size = len(self.labels)
        return size


def plot_imgbatch(imgs, results_path):
    """ Helper function for plotting a batch of images.

    Args:
        imgs: the images to plot.
    """
    imgs = imgs.cpu()
    imgs = imgs.type(torch.IntTensor)
    plt.figure(figsize=(15, 3*(imgs.shape[0])))
    grid_img = make_grid(imgs, nrow=5)
    plt.imshow(grid_img.permute(1, 2, 0))
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    graph_path = os.path.join(results_path, "img_batch.png")
    plt.savefig(graph_path)
    plt.close()


def main():
    """ Runs the bulk of the CNN code.
        Implements Mask R-CNN with single-channel input.
        """

    # Directory information.
    data_dir = 'E:\\data\\output\\bmp_out_single_localise'
    results_path = "E:\\data\\output\\results\\mask_single"
    save_file = "E:\\data\\output\\nets\\mask_single.pth"
    file_name = "mask_single.txt"
    folder = "mask_single"

    # Length in pixels of size of image once resized for the network.
    img_size = 128
    dataset = ScanDataset(data_dir, img_size)

    # Fractions for splitting data into train/validation/test.
    # An 60/20/20 split has been chosen.
    train_fraction = 0.6
    validation_fraction = 0.2
    test_fraction = 0.2
    dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size}\n")

    # Splits the data into the chosen fractions.
    num_train = int(train_fraction * dataset_size)
    num_validation = int(validation_fraction * dataset_size)
    num_test = int(test_fraction * dataset_size)
    print(f"Training: {num_train}\nValidation: {num_validation}" +
          f"\nTesting: {num_test}\n")

    # Sets up the datasets with a random split, ensuring half/half 1/0.
    neg_dataset = []
    pos_dataset = []
    for scan_slice in dataset:
        label = scan_slice[1]['labels'][0]
        if label == 0:
            neg_dataset.append(scan_slice)
        else:
            pos_dataset.append(scan_slice)

    train_neg_dataset, validation_neg_dataset, test_neg_dataset = \
        torch.utils.data.random_split(neg_dataset,
                                      [int(num_train/2), int(num_validation/2),
                                          int(num_test/2)])

    train_pos_dataset, validation_pos_dataset, test_pos_dataset = \
        torch.utils.data.random_split(pos_dataset,
                                      [int(num_train/2), int(num_validation/2),
                                          int(num_test/2)])

    train_dataset = train_neg_dataset + train_pos_dataset
    validation_dataset = validation_neg_dataset + validation_pos_dataset
    test_dataset = test_neg_dataset + test_pos_dataset

    print(train_pos_dataset[1])

    # Makes sure CNN training runs on GPU, if available.
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")
    print(f"Running on {device}\n")

    # Defines batch sizes.
    train_batchsize = 32  # Depends on computation hardware.
    eval_batchsize = 16  # Can be small due to small dataset size.

    # Loads images for training in a random order.
    train_loader = DataLoader(train_dataset,
                              batch_size=train_batchsize,
                              shuffle=True)

    # Loads images for validation in a random order.
    validation_loader = DataLoader(validation_dataset,
                                   batch_size=eval_batchsize,
                                   shuffle=True)

    # Loads images for testing in a random order.
    test_loader = DataLoader(test_dataset,
                             batch_size=eval_batchsize,
                             shuffle=True)

    # Set random seeds for reproducibility.
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Define the convoluted neural network.
    net = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)

    # This network takes single channel input.
    net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7),
                          stride=(2, 2), padding=(3, 3), bias=False)

    # Casts Mask R-CNN to run on device.
    net = net.to(device)

    # Defines criterion to compute the Mask R-CNN loss.
    criterion = maskrcnn_loss()
    criterion = criterion.to(device)

    # Sets the error minimizer with a learning rate of 0.001.
    error_minimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # Defines epoch number.
    epochs = 100

    # Defines the "final" version of the net to save (updated later).
    net_final = deepcopy(net)
    net_final = net_final.to(device)

    # Used to pick the best-performing model on the validation set.
    best_validation_accuracy = 0.

    # For training visualiSation later.
    train_accs = []
    val_accs = []
    losses = []
    val_losses = []

    # Set early stopping variable.
    early_stopper = EarlyStopper(patience=10, min_delta=0)
    epoch_counter = 0

    # Training loop.
    for epoch in range(epochs):
        # Set network to training mode, so its parameters can be changed.
        net.train()

        # Print training info.
        print(f"### Epoch {epoch}:")

        # Statistics needed to compute classification accuracy:
        # The total number of image examples trained on.
        total_train_examples = 0

        # The number of examples classified correctly.
        num_correct_train = 0

        # Set variables needed for loss calculations.
        loss_total = 0
        counter = 0

        # Iterate over the training set once.
        for batch_index, (inputs, targets) in tqdm(enumerate(train_loader),
                                                total=len(train_dataset) //
                                                train_batchsize):
            # Load the data onto the computation device.
            # Inputs are a Tensor of shape:
            # (batch size, number of channels, image height, image width).
            # Targets are a list of dictionaries, one for each image in the batch.
            # Each dictionary contains:
            #   - "boxes": a Tensor of shape (number of objects in the image, 4)
            #              containing the ground-truth bounding boxes for each object.
            #   - "labels": a Tensor of shape (number of objects in the image,)
            #               containing the class labels for each object.
            #   - "masks": a Tensor of shape (number of objects in the image, image_height, image_width)
            #              containing the binary segmentation masks for each object.
            inputs = inputs.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Reset changes (gradients) to parameters.
            error_minimizer.zero_grad()

            # Get the network's predictions on the training set batch.
            net = net.to(device)
            predictions = net(inputs)

            # Evaluate the error.
            # Estimate how much to change the network parameters.
            loss_dict = criterion(predictions, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss = loss.to(device)
            loss_total = loss_total + loss.item()
            counter = counter +
                optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the training progress bar.
            train_bar.set_postfix(loss=loss_total / (
                batch_idx + 1), acc=100. * correct / total)

        # Evaluate the performance of the network on the validation set.
        net.eval()
        val_loss_total = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(validation_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                predictions = net(inputs)
                loss_dict = criterion(predictions, targets)
                loss = sum(loss for loss in loss_dict.values())
                loss = loss.to(device)
                val_loss_total = val_loss_total + loss.item()

                # Compute the accuracy of the network on the validation set.
                _, predicted = torch.max(predictions.data, 1)
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)

        # Update the validation progress bar.
        val_bar.set_postfix(loss=val_loss_total / len(
            validation_loader), acc=100. * val_correct / val_total)

        # Save the model checkpoint if the validation accuracy has improved.
        if val_correct > best_val_acc:
            best_val_acc = val_correct
            torch.save(net.state_dict(), 'mask.pth')


    # Sort epochs for graph.
    epoch_list = list(range(epoch_counter))

    # Plot prediction accuracy over time.
    plt.figure()
    plt.plot(epoch_list, train_accs, 'b-', label='Training Set Accuracy')
    plt.plot(epoch_list, val_accs, 'r-', label='Validation Set Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Prediction Accuracy')
    plt.ylim(0.5, 1)
    plt.title('Classifier Training Evolution:\nPrediction Accuracy Over Time')
    plt.legend()
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    graph_path = os.path.join(results_path, "pred_acc_over_time.png")
    plt.savefig(graph_path)
    plt.close()

    # Plot loss reduction.
    plt.figure()
    plt.plot(epoch_list, losses, 'b-', label='Training Loss')
    plt.plot(epoch_list, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Reduction')
    plt.ylim(0, 1)
    plt.title('Classifier Training Evolution:\nLoss Reduction')
    plt.legend()
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    graph_path = os.path.join(results_path, "loss_reduction.png")
    plt.savefig(graph_path)
    plt.close()

    # Define some requirec counts.
    total_test_examples = 0
    num_correct_test = 0

    # True and false positive counts.
    false_pos_count = 0
    true_pos_count = 0
    false_neg_count = 0
    true_neg_count = 0

    # Visualize a random batch of data with examples.
    num_viz = 10
    viz_index = random.randint(0, len(test_dataset)//eval_batchsize)

    # See how well the final trained model does on the test set.
    with torch.no_grad():
        # Don't save parameter gradients/changes,
        # since this is not for model training.
        for batch_index, (inputs, targets) in enumerate(test_loader):
            # Make predictions.
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = net_final(inputs)

            # Compute prediction statistics.
            _, predicted_class = predictions.max(1)
            total_test_examples += predicted_class.size(0)
            num_correct_test += predicted_class.eq(targets).sum().item()

            # Thanks to:
            # https://gist.github.com/the-bass/cae9f3976866776dea17a5049013258d
            confusion_vector = predicted_class / targets
            num_true_pos = torch.sum(confusion_vector == 1).item()
            num_false_pos = torch.sum(confusion_vector == float('inf')).item()
            num_true_neg = torch.sum(torch.isnan(confusion_vector)).item()
            num_false_neg = torch.sum(confusion_vector == 0).item()

            true_pos_count += num_true_pos
            false_pos_count += num_false_pos
            false_neg_count += num_false_neg
            true_neg_count += num_true_neg

            # Plot predictions.
            if batch_index == viz_index:
                print('Example Images:')
                plot_imgbatch(inputs[:num_viz], results_path)
                print('Target labels:')
                print(targets[:num_viz].tolist())
                print('Classifier predictions:')
                print(predicted_class[:num_viz].tolist())
                file_path = os.path.join(results_path, "predictions.txt")
                if not os.path.exists(file_path):
                    with open(file_path, "w", encoding="utf-8") as file:
                        file.write('Target labels:')
                        file.write(str(targets[:num_viz].tolist()))
                        file.write('\nClassifier predictions:')
                        file.write(str(predicted_class[:num_viz].tolist()))
                    file.close()

    # Get total results:
    # Total prediction accuracy of network on test set.
    # Calulcate and save evaluation metrics using the Evaluation module.
    evaluation = Evaluation(false_pos_count, false_neg_count,
                            true_pos_count, true_neg_count, file_name, folder)
    # Print the results to the screen.
    print(f"Test set accuracy: {evaluation.accuracy}")
    print(f"True positive classifications: {true_pos_count}")
    print(f"False positive classifications: {false_pos_count}")
    print(f"True negative classifications: {true_neg_count}")
    print(f"False negative classifications: {false_neg_count}")
    print(f"Negative predictive value: {evaluation.npv}")
    print(f"Positive predictive value: {evaluation.ppv}")
    print(f"Sensitivity: {evaluation.sensitivity}")
    print(f"Specificity: {evaluation.specificity}")


if __name__ == "__main__":
    main()

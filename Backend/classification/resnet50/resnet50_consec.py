""" This module implements ResNet50 with 3 channel
inputs with consecutive scan slices. It trains,
validates and tests a ResNet50 classification CNN
on Breast Cancer MRI scan thruples, then calculates
results for performance.
"""

import random
import os
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.utils import make_grid
from skimage.io import imread
import matplotlib.pyplot as plt
from evaluation import Evaluation


# pylint: disable=E1101
# pylint: disable=E1102
class ScanDataset(Dataset):
    """ This class creates a dataset of images
    from which to train, test and validate the
    Resnet50 CNN.
    """

    def __init__(self, data_dir, img_size):
        """ Initialises the dataset of scans.

        Args:
            data_dir: the directory of the data
            img_size: the size of the images needed for CNN input
        """
        self.data_dir = data_dir
        self.img_size = img_size

        # assign labels to data within this Dataset
        self.labels = None
        self.create_labels()

    def create_labels(self):
        """ Creates and stores labels (positive/1 or negative/0)
        for scans within the dataset. Each label is a tuple:
        (img folder, label number (0 or 1)), where the folder
        contains the 3 consecutive scans.

        Args:
            data_dir: the directory of the data
            img_size: the size of the images needed for CNN input
        """

        labels = []
        print('Building dataset labels...')

        # Iterate over each class.
        for target, target_label in enumerate(['neg', 'pos']):
            case_dir = os.path.join(self.data_dir, target_label)
            # Iterate over all images in the class/case type.
            for folder in os.listdir(case_dir):
                group = []
                for fname in os.listdir(case_dir + "\\" + folder):
                    if '.bmp' in fname:
                        fpath = os.path.join(case_dir, folder, fname)
                        group.append((fpath, target))
                labels.append(group)

        self.labels = labels

    def normalise(self, img):
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

        group = self.labels[idx]
        data = []
        # Loop to work with 3 channels.
        for counter in range(3):
            fpath, target = group[counter]

            # Load img from file (bmp).
            img_arr = imread(fpath, as_gray=True)

            # Normalise image.
            img_arr = self.normalise(img_arr)

            # Convert to Tensor (PyTorch matrix).
            single = torch.from_numpy(img_arr)
            single = single.type(torch.FloatTensor)

            # Add image channel dimension (to work with the CNN).
            single = torch.unsqueeze(single, 0)

            # Resize image.
            single = transforms.Resize((self.img_size, self.img_size))(single)

            data.append(torch.Tensor.numpy(single))

        # Put 3 consecutive scans in the 3 channel inputs.
        data = torch.tensor(data).permute(1, 0, 2, 3)
        data = torch.squeeze(data).type(torch.FloatTensor)

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
    plt.show()
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    graph_path = os.path.join(results_path, "img_batch.png")
    plt.savefig(graph_path)


def main():
    """ Runs the bulk of the CNN code.
        Implements ResNet50 with three-channel input
        for consecutive scans.
        """

    # Directory where data is stored.
    data_dir = 'E:\\data\\output\\bmp_out_consec_classify'
    results_path = "E:\\data\\output\\results\\resnet50_consec"

    # Length in pixels of size of image once resized for the network.
    img_size = 128
    dataset = ScanDataset(data_dir, img_size)

    # Fractions for splitting data into train/validation/test.
    # An 80/10/10 split has been chosen.
    train_fraction = 0.8
    validation_fraction = 0.1
    test_fraction = 0.1
    dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size}\n")

    # Splits the data into the chosen fractions.
    num_train = int(train_fraction * dataset_size)
    num_validation = int(validation_fraction * dataset_size)
    num_test = int(test_fraction * dataset_size)
    print(f"Training: {num_train}\nValidation: {num_validation}" +
          f"\nTesting: {num_test}\n")

    # Sets up the datasets with a random split.
    train_dataset, validation_dataset, test_dataset = \
        torch.utils.data.random_split(dataset,
                                      [num_train, num_validation, num_test])

    # Makes sure CNN training runs on GPU, if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}\n")

    # Defines batch sizes.
    train_batchsize = 16  # Depends on computation hardware.
    eval_batchsize = 8  # Can be small due to small dataset size.

    # Loads images for training in a random order.
    train_loader = DataLoader(train_dataset,
                              batch_size=train_batchsize,
                              shuffle=True)

    # Loads images for validation in a random order.
    validation_loader = DataLoader(validation_dataset,
                                   batch_size=eval_batchsize)

    # Loads images for testing in a random order.
    test_loader = DataLoader(test_dataset,
                             batch_size=eval_batchsize)

    # Set random seeds for reproducibility.
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Define the convoluted neural network.
    net = resnet50()

    # This network takes 3 channel input.
    net.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7),
                          stride=(2, 2), padding=(3, 3), bias=False)

    # Casts CNN to run on device.
    net = net.to(device)

    # Defines criterion to compute the cross-entropy loss.
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    # Sets the error minimiser with a learning rate of 0.001.
    error_minimizer = torch.optim.SGD(net.parameters(), lr=0.001)

    # Defines epoch number.
    epochs = 3

    # Defines the "final" version of the net to save (updated later).
    net_final = deepcopy(net)

    # Used to pick the best-performing model on the validation set.
    best_validation_accuracy = 0.

    # For training visualiSation later.
    train_accs = []
    val_accs = []
    losses = []

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

        # Iterate over the training set once.
        for batch_index, (inputs, targets) in tqdm(enumerate(train_loader),
                                                   total=len(train_dataset) //
                                                   train_batchsize):
            # Load the data onto the computation device.
            # Inputs are a Tensor of shape:
            # (batch size, number of channels, image height, image width).
            # Targets are a Tensor of one-hot-encoded class
            # labels for the inputs,
            # of shape (batch size, number of classes).
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Reset changes (gradients) to parameters.
            error_minimizer.zero_grad()

            # Get the network's predictions on the training set batch.
            net = net.to(device)
            predictions = net(inputs)

            # Evaluate the error.
            # Estimate how much to change the network parameters.
            loss = criterion(predictions, targets)
            loss.backward()
            losses.append(loss)
            # Free memory to avoid overload.
            del loss

            # Change parameters.
            error_minimizer.step()

            # Calculate predicted class label.
            # The .max() method returns the maximum entries, and their indices.
            # Need the index with the highest probability.
            # Not the probability itself.
            _, predicted_class = predictions.max(1)
            total_train_examples += predicted_class.size(0)
            num_correct_train += predicted_class.eq(targets).sum().item()

        # Get results:
        # Total prediction accuracy of network on training set.
        train_acc = num_correct_train / total_train_examples
        print(f"Training accuracy: {train_acc}")
        train_accs.append(train_acc)

        # Predict on validation set (similar to training set):
        total_val_examples = 0
        num_correct_val = 0

        # Switch network from training mode (parameters can be trained),
        # to evaluation mode (parameters can't be trained).
        net.eval()

        with torch.no_grad():
            # Don't save parameter changes,
            # since this is not for training.
            for batch_index, (inputs, targets) in \
                tqdm(enumerate(validation_loader),
                     total=len(validation_dataset)//eval_batchsize):
                inputs = inputs.to(device)
                targets = targets.to(device)
                net = net.to(device)
                predictions = net(inputs)
                _, predicted_class = predictions.max(1)
                total_val_examples += predicted_class.size(0)
                num_correct_val += predicted_class.eq(targets).sum().item()

        # Get results:
        # Total prediction accuracy of network on validation set.
        val_acc = num_correct_val / total_val_examples
        print(f"Validation accuracy: {val_acc}")
        val_accs.append(val_acc)

        # Finally, save model if the validation accuracy is the best so far.
        if val_acc > best_validation_accuracy:
            best_validation_accuracy = val_acc
            print("Validation accuracy improved; saving model.")
            net_final = deepcopy(net)

            # Save final CNN in the specified filepath.
            epochs_list = list(range(epochs))
            save_file = "E:\\data\\output\\nets\\resnet50_consec.pth"
            torch.save(net_final.state_dict(), save_file)

    # Plot prediction accuracy over time.
    plt.figure()
    plt.plot(epochs_list, train_accs, 'b-', label='Training Set Accuracy')
    plt.plot(epochs_list, val_accs, 'r-', label='Validation Set Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Prediction Accuracy')
    plt.ylim(0.5, 1)
    plt.title('Classifier training evolution:\nPrediction Accuracy Over Time')
    plt.legend()
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    graph_path = os.path.join(results_path, "pred_acc_over_time.png")
    plt.savefig(graph_path)
    plt.clf()

    # Plot loss reduction.
    plt.figure()
    plt.plot(epochs_list, losses, 'b-', label='Training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Reduction')
    plt.ylim(0.5, 1)
    plt.title('Classifier training evolution:\nLoss Reduction')
    plt.legend()
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    graph_path = os.path.join(results_path, "loss_reduction.png")
    plt.savefig(graph_path)
    plt.clf()

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
            num_false_neg = torch.sum(torch.isnan(confusion_vector)).item()
            num_true_neg = torch.sum(confusion_vector == 0).item()

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

    # Get total results:
    # Total prediction accuracy of network on test set.
    file_name = "resnet50_iteration_1.txt"
    folder = "resnet50_single"

    # Calulcate and save evaluation metrics using the Evaluation module.
    evaluation = Evaluation(false_pos_count, false_neg_count,
                            true_pos_count, true_neg_count, file_name, folder)
    # Print the results to the screen.
    print(f"Test set accuracy: {evaluation.accuracy}")
    print(f"{true_pos_count} true positive classifications\n")
    print(f"{false_pos_count} false positive classifications\n")
    print(f"{true_neg_count} true negative classifications\n")
    print(f"{false_neg_count} false negative classifications\n")
    print(f"Negative predictive value: {evaluation.npv}")
    print(f"Positive predictive value: {evaluation.ppv}")
    print(f"Sensitivity: {evaluation.sensitivity}")
    print(f"Specificity: {evaluation.specificity}")


if __name__ == "__main__":
    main()

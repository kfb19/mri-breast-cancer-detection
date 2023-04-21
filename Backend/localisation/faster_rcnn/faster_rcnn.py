""" This module implements Faster R-CNN with a three
channel input (3 different MRI slice scan types).
It trains, validates and tests a Faster R-CNN classification CNN
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
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from skimage.io import imread
import matplotlib.pyplot as plt
from evaluation import Evaluation
from early_stopper import EarlyStopper


# pylint: disable=E1101
# pylint: disable=E1102
# pylint: disable=W0246
class ScanDataset(Dataset):
    """ This class creates a dataset of images
    from which to train, test and validate the
    Faster R-CNN CNN.
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
        bb_path = "E:\\data\\output\\bmp_out_scantype_localise\\bounding_boxes"
        bb_filepath = os.path.join(bb_path, "bounding_boxes.csv")
        bbox_data = pd.read_csv(bb_filepath)
        out_filename = ""
        # Iterate over each class.
        case_dir = os.path.join(self.data_dir,)
        # Iterate over all images in the class/case type.
        for folder in os.listdir(case_dir):
            group = []
            for fname in os.listdir(os.path.join(case_dir, folder)):
                out_filename = fname
                if '.bmp' in fname:
                    fpath = os.path.join(case_dir, folder, fname)
                    # Load img from file (bmp).
                    img_arr = imread(fpath)

                    # Normalise image.
                    img_arr = self.normalize(img_arr)

                    # Convert to Tensor (PyTorch matrix).
                    data_tensor = torch.from_numpy(img_arr).type(
                        torch.FloatTensor)

                    # Convert to a 4D tensor for resize operation.
                    data_tensor = data_tensor.unsqueeze(0).unsqueeze(0)

                    # Resize image.
                    data_tensor = transforms.Resize(
                        (self.img_size, self.img_size))(data_tensor)

                    # Remove extra dimensions.
                    data_tensor = data_tensor.squeeze(0).squeeze(0)

                    group.append(data_tensor)

            # Find the corresponding bounding box data for image.
            bbox_row = bbox_data.loc[bbox_data[
                'image_id'] == out_filename]

            # Extract the bounding box coordinates.
            if not bbox_row.empty:
                xmin = bbox_row['xmin'].values[0]
                ymin = bbox_row['ymin'].values[0]
                xmax = bbox_row['xmax'].values[0]
                ymax = bbox_row['ymax'].values[0]

                # Create a dictionary to store the target data
                target = {
                    'boxes': torch.tensor([[xmin, ymin, xmax, ymax]],
                                          device='cpu'),
                    'labels': torch.tensor([1], device='cpu'),
                }

                # Create RGB image tensor with the 3 images as channels.
                data = torch.stack(group, dim=0)
                data = torch.cat([data[0:1], data[1:2], data[2:3]], dim=0)

                # Create tuple of data and label and append to list.
                label = (data, target)
                labels.append(label)

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


# Define the Faster R-CNN loss function
class FasterRCNNLoss(torch.nn.Module):
    """ DOCSTRING """

    def __init__(self):
        """ DOCSTRING """
        super(FasterRCNNLoss, self).__init__()

    def forward(self, classification_output, regression_output,
                anchors, targets):
        """ DOCSTRING """
        # Calculate the classification loss
        classification_loss = torch.nn.MultiLabelSoftMarginLoss()(
            classification_output, targets[:, :, :-4])

        # Calculate the localization loss
        regression_loss = torchvision.ops.SmoothL1Loss()(regression_output,
                                                         targets[:, :, -4:])

        # Combine the classification and localization losses
        loss = classification_loss + regression_loss

        return loss


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


def validate(net, criterion, validation_loader, device):
    """ DOCSTRING """
    net.eval()
    val_loss = 0
    num_correct = 0
    total_examples = 0
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs = inputs.to(device)
            labels = [lab.to(device) for lab in labels]
            outputs = net(inputs)
            loss_dict = criterion(outputs, labels)
            loss = sum(loss for loss in loss_dict.values())
            val_loss += loss.item()
            num_correct += get_num_correct(net, validation_loader)
            total_examples += len(labels)
    val_accuracy = num_correct / total_examples
    return val_loss, val_accuracy


def test(net, criterion, test_loader, device):
    """ DOCSTRING """
    net.eval()
    test_loss = 0
    num_correct = 0
    total_examples = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = [lab.to(device) for lab in labels]
            outputs = net(inputs)
            loss_dict = criterion(outputs, labels)
            loss = sum(loss for loss in loss_dict.values())
            test_loss += loss.item()
            num_correct += get_num_correct(net, test_loader)
            total_examples += len(labels)
    test_accuracy = num_correct / total_examples
    return test_loss, test_accuracy


def box_iou(box1, box2):
    """
    Calculates IoU of two bounding boxes.

    Args:
        box1 (torch.Tensor): bounding box of shape (4,) in format
        (x1, y1, x2, y2)
        box2 (torch.Tensor): bounding box of shape (4,) in format
        (x1, y1, x2, y2)

    Returns:
        float: IoU of the two bounding boxes
    """
    x1, y1, x2, y2 = box1
    xx1, yy1, xx2, yy2 = box2

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (xx2 - xx1) * (yy2 - yy1)

    inter_x1 = max(x1, xx1)
    inter_y1 = max(y1, yy1)
    inter_x2 = min(x2, xx2)
    inter_y2 = min(y2, yy2)

    inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)

    iou = inter_area / (area1 + area2 - inter_area)

    return iou


def get_num_correct(model, data_loader, iou_threshold=0.5):
    """
    Counts the number of correct predictions for Faster R-CNN.

    Args:
        model (torch.nn.Module): trained Faster R-CNN model
        data_loader (torch.utils.data.DataLoader): data loader for the dataset
        iou_threshold (float, optional): IoU threshold for considering a detection as correct.
            Default is 0.5.

    Returns:
        int: number of correct predictions
    """
    num_correct = 0

    for images, targets in data_loader:
        # feed images to the model and obtain the output dict
        model.eval()
        with torch.no_grad():
            outputs = model(images)

        # extract predicted class scores, predicted box regression offsets, 
        # and ground-truth target labels and target box regression offsets
        scores = outputs['classifier']
        regressions = outputs['regressor']
        labels = [t['labels'] for t in targets]
        gt_boxes = [t['boxes'] for t in targets]

        # decode predicted box regression offsets using model's anchor boxes
        anchors = model.anchor_generator(images)
        pred_boxes = model.box_coder.decode(regressions, anchors)

        # compute IoU between predicted and ground-truth boxes
        batch_size = len(images)
        for i in range(batch_size):
            gt_boxes_i = gt_boxes[i]
            scores_i = scores[i]
            pred_boxes_i = pred_boxes[i]

            if len(gt_boxes_i) == 0:
                continue

            iou = box_iou(pred_boxes_i, gt_boxes_i)

            # classify a predicted box as correct if its predicted class 
            # is the same as the ground-truth class and its
            # IoU is above the threshold
            max_iou, max_idx = iou.max(dim=1)
            correct_mask = (scores_i.argmax(dim=1) == labels[i][max_idx]) & (
                max_iou >= iou_threshold)
            num_correct += correct_mask.sum().item()

    return num_correct



def create_data_loader(data, batch_size):
    """Creates a data loader for the dataset.

    Args:
        data: the dataset
        batch_size: the size of the batch

    Returns:
        A DataLoader for the dataset.
    """
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn  # Add collate_fn argument
    )
    return loader


def collate_fn(batch):
    """Collates a batch of data.

    Args:
        batch: a list of data samples

    Returns:
        A tuple containing the inputs and the labels.
    """
    inputs = torch.stack([sample[0] for sample in batch])
    labels = []
    for sample in batch:
        labels.append(sample[1])
    labels = tuple(labels)
    return inputs, labels


def main():
    """ Runs the bulk of the CNN code.
        Implements Faster R-CNN with 3-channel input.
        """

    # Directory information.
    data_dir = 'E:\\data\\output\\bmp_out_scantype_localise\\pos'
    results_path = "E:\\data\\output\\results\\faster_scantype"
    save_file = "E:\\data\\output\\nets\\faster_scantype.pth"
    file_name = "faster_scantype.txt"
    folder = "faster_scantype"

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

    # Sets up the datasets with a random split, all pos for ROIs.
    train_dataset, validation_dataset, test_dataset = \
        torch.utils.data.random_split(dataset,
                                      [num_train, num_validation,
                                          num_test])

    # Makes sure CNN training runs on GPU, if available.
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda") if torch.cuda.is_available() \
    #    else
    device = "cpu"
    torch.device("cpu")
    print(f"Running on {device}\n")

    # Defines batch sizes.
    train_batchsize = 8  # Depends on computation hardware.
    eval_batchsize = 4  # Can be small due to small dataset size.

    # Loads images for training in a random order.
    train_loader = create_data_loader(train_dataset, train_batchsize)
    validation_loader = create_data_loader(validation_dataset, eval_batchsize)
    test_loader = create_data_loader(test_dataset, eval_batchsize)

    # Set random seeds for reproducibility.
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Define the convoluted neural network.
    net = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

    # This network takes a 3 channel input.
    net.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7),
                          stride=(2, 2), padding=(3, 3), bias=False)

    # Casts CNN to run on device.
    net = net.to(device)

    # Define criterion to compute the Fast R-CNN loss.
    criterion = FasterRCNNLoss()
    # criterion = nn.CrossEntropyLoss()

    # Cast criterion to run on device.
    criterion = criterion.to(device)

    # Define the error minimizer with a learning rate of 0.001.
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
        """for batch_index, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            outputs = net(inputs, labels)
            print(outputs)
            loss_dict = criterion(outputs, labels)
            loss = sum(loss for loss in loss_dict.values())
            loss_total += loss.item()
            error_minimizer.zero_grad()
            loss.backward()
            error_minimizer.step()"""

        for batch_index, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            outputs = net(inputs, labels)
            loss_classifier = outputs['loss_classifier'].item()
            loss_box_reg = outputs['loss_box_reg'].item()
            loss_objectness = outputs['loss_objectness'].item()
            loss_rpn_box_reg = outputs['loss_rpn_box_reg'].item()

            loss = torch.tensor(0.5 * loss_classifier +
                                0.5 * loss_box_reg +
                                0.5 * loss_objectness +
                                0.5 * loss_rpn_box_reg, requires_grad=True)

            loss_total += loss.item()

            error_minimizer.zero_grad()
            loss.backward()
            error_minimizer.step()
            # Calculate accuracy statistics.
            total_train_examples += len(labels)
            num_correct_train += get_num_correct(net, train_loader)

            # Print loss statistics.
            if batch_index % 100 == 0:
                print(f"Batch {batch_index}, Loss: {loss_total/counter:.4f}")
            counter += 1

        # Calculate training accuracy.
        train_accuracy = num_correct_train / total_train_examples
        train_accs.append(train_accuracy)

        val_losses.append(val_loss)

        # Get results:
        # Loss:
        average_loss = loss_total / counter
        losses.append(average_loss)

        # Total prediction accuracy of network on training set.
        train_acc = num_correct_train / total_train_examples
        print(f"Training accuracy: {train_acc}")
        train_accs.append(train_acc)

        # Predict on validation set (similar to training set):
        total_val_examples = 0
        num_correct_val = 0
        val_loss_total = 0
        val_counter = 0

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
                net = net.to(device)
                targets = targets.to(device)
                predictions = net(inputs)
                _, predicted_class = predictions.max(1)
                total_val_examples += predicted_class.size(0)
                num_correct_val += predicted_class.eq(targets).sum().item()
                val_loss = criterion(predictions, targets)
                val_loss = val_loss.to(device)
                val_loss_total = val_loss_total + val_loss.item()
                val_counter = val_counter + 1

        # Get results:
        # Calculate validation loss and accuracy.
        val_loss, val_accuracy = validate(net, criterion,
                                          validation_loader, device)
        # Total prediction accuracy of network on validation set.
        print(f"Validation accuracy: {val_accuracy}")
        val_accs.append(val_accuracy)

        # Save validation loss.
        val_average_loss = val_loss_total / val_counter
        val_losses.append(val_average_loss)

        # Save model if the validation accuracy is the best so far.
        if val_accuracy > best_validation_accuracy:
            best_validation_accuracy = val_accuracy
            print("Validation accuracy improved; saving model.")
            net_final = deepcopy(net)
            net_final = net_final.to(device)

            # Save final CNN in the specified filepath.
            torch.save(net_final.state_dict(), save_file)

        # For graph calculations later if early stopping happens.
        epoch_counter = epoch_counter + 1

        # Check via early stopping if the CNN is overfitting.
        if early_stopper.early_stop(val_average_loss):
            print("Early stopping...")
            break

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

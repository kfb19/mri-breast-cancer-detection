""" DOCSTRING HERE """

import random
import os
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.utils import make_grid
from skimage.io import imread
import matplotlib.pyplot as plt
from eval_metrics import Evaluation


# pylint: disable=E1101
class ScanDataset(Dataset):
    """ DOCSTRING HERE """

    def __init__(self, data_dir, img_size):
        """ DOCSTRING HERE """
        self.data_dir = data_dir
        self.img_size = img_size

        # assign labels to data within this Dataset
        self.labels = None
        self.create_labels()

    def create_labels(self):
        """ DOCSTRING HERE """
        # create and store a label (positive/1 or negative/0 for each image)
        # each label is the tuple: (img filename, label number (0 or 1))
        labels = []
        print('building DBC dataset labels.')
        # iterate over each class
        for target, target_label in enumerate(['neg', 'pos']):
            case_dir = os.path.join(self.data_dir, target_label)
            # iterate over all images in the class/case type
            for fname in os.listdir(case_dir):
                if '.bmp' in fname:
                    fpath = os.path.join(case_dir, fname)
                    labels.append((fpath, target))

        self.labels = labels

    def normalise(self, img):
        """ DOCSTRING HERE """
        # normalize image pixel values to range [0, 255]
        # img expected to be array

        # convert uint16 -> float
        img = img.astype(float) * 255. / img.max()
        # convert float -> unit8
        img = img.astype(np.uint8)

        return img

    def __getitem__(self, idx):
        """ DOCSTRING HERE """
        # required method for accessing data samples
        # returns data with its label
        fpath, target = self.labels[idx]

        # load img from file (bmp)
        img_arr = imread(fpath, as_gray=True)

        # normalize image
        img_arr = self.normalise(img_arr)

        # convert to tensor (PyTorch matrix)
        data = torch.from_numpy(img_arr)
        data = data.type(torch.FloatTensor)

        # add image channel dimension (to work with neural network)
        data = torch.unsqueeze(data, 0)

        # resize image
        data = transforms.Resize((self.img_size, self.img_size))(data)

        return data, target

    def __len__(self):
        """ DOCSTRING HERE """
        # required method for getting size of dataset
        return len(self.labels)


def main():
    """ DOCSTRING HERE """
    # directory where our .png data is (created in the previous post)
    data_dir = 'E:\\data\\output\\bmp_out_consec_classify'
    # change to read the above from the other file?
    # length in pixels of size of image once resized for the network
    img_size = 128
    dataset = ScanDataset(data_dir, img_size)
    print(f"Dataset length {len(dataset)}")

    train_fraction = 0.8
    validation_fraction = 0.1
    test_fraction = 0.1
    dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size}")

    num_train = int(train_fraction * dataset_size)
    num_validation = int(validation_fraction * dataset_size)
    num_test = int(test_fraction * dataset_size)
    print(f"Training: {num_train}\nValidation: {num_validation}\nTesting: " +
          "{num_test}")

    train_dataset, validation_dataset, test_dataset = \
        torch.utils.data.random_split(dataset,
                                      [num_train, num_validation, num_test])

    # GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}")

    train_batchsize = 16  # depends on your computation hardware
    eval_batchsize = 8  # can be small due to small dataset size
    train_loader = DataLoader(train_dataset,
                              batch_size=train_batchsize,
                              shuffle=True
                              # images are loaded in random order
                              )

    validation_loader = DataLoader(validation_dataset,
                                   batch_size=eval_batchsize)

    test_loader = DataLoader(test_dataset,
                             batch_size=eval_batchsize)

    # set random seeds for reproducibility CHECKME WHY?
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    net = resnet50()

    # 3 channel input
    net.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7),
                          stride=(2, 2), padding=(3, 3), bias=False)

    net = net.to(device)

    criterion = nn.CrossEntropyLoss()

    error_minimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    # learning rate??

    epochs = 5

    net_final = deepcopy(net)

    best_validation_accuracy = 0.
    # used to pick the best-performing model on the validation set

    # for training visualization later
    train_accs = []
    val_accs = []

    # training loop
    for epoch in range(epochs):
        # set network to training mode, so that its parameters can be changed
        net.train()

        # print training info
        print(f"### Epoch {epoch}:")

        # statistics needed to compute classification accuracy:
        # the total number of image examples trained on
        total_train_examples = 0

        # the number of examples classified correctly
        num_correct_train = 0

        # iterate over the training set once
        for batch_index, (inputs, targets) in tqdm(enumerate(train_loader),
                                                   total=len(train_dataset) //
                                                   train_batchsize):
            # load the data onto the computation device.
            # inputs are a tensor of shape:
            # (batch size, number of channels, image height, image width).
            # targets are a tensor of one-hot-encoded class
            # labels for the inputs,
            # of shape (batch size, number of classes)
            # in other words,
            inputs = inputs.to(device)
            targets = targets.to(device)

            # reset changes (gradients) to parameters
            error_minimizer.zero_grad()

            # get the network's predictions on the training set batch
            net: Module = resnet50()
            predictions = net(inputs)

            # evaluate the error, and estimate
            #   how much to change the network parameters
            loss = criterion(predictions, targets)
            loss.backward()

            # change parameters
            error_minimizer.step()

            # calculate predicted class label
            # the .max() method returns the maximum entries, and their indices;
            # we just need the index with the highest probability,
            #   not the probability itself.
            _, predicted_class = predictions.max(1)
            total_train_examples += predicted_class.size(0)
            num_correct_train += predicted_class.eq(targets).sum().item()

        # get results
        # total prediction accuracy of network on training set
        train_acc = num_correct_train / total_train_examples
        print(f"Training accuracy: {train_acc}")
        train_accs.append(train_acc)

        # predict on validation set (similar to training set):
        total_val_examples = 0
        num_correct_val = 0

        # switch network from training mode (parameters can be trained)
        #   to evaluation mode (parameters can't be trained)
        net.eval()

        with torch.no_grad():  # don't save parameter changes
            #                      since this is not for training
            for batch_index, (inputs, targets) in \
                tqdm(enumerate(validation_loader),
                     total=len(validation_dataset)//eval_batchsize):
                inputs = inputs.to(device)
                targets = targets.to(device)
                net: Module = resnet50()
                predictions = net(inputs)
                _, predicted_class = predictions.max(1)
                total_val_examples += predicted_class.size(0)
                num_correct_val += predicted_class.eq(targets).sum().item()

        # get results
        # total prediction accuracy of network on validation set
        val_acc = num_correct_val / total_val_examples
        print(f"Validation accuracy: {val_acc}")
        val_accs.append(val_acc)

        # Finally, save model if the validation accuracy is the best so far
        if val_acc > best_validation_accuracy:
            best_validation_accuracy = val_acc
            print("Validation accuracy improved; saving model.")
            net_final = deepcopy(net)

    epochs_list = list(range(epochs))

    plt.figure()
    plt.plot(epochs_list, train_accs, 'b-', label='training set accuracy')
    plt.plot(epochs_list, val_accs, 'r-', label='validation set accuracy')
    plt.xlabel('epoch')
    plt.ylabel('prediction accuracy')
    plt.ylim(0.5, 1)
    plt.title('Classifier training evolution:\nprediction accuracy over time')
    plt.legend()
    plt.show()

    # helper function for plotting a batch of images

    def plot_imgbatch(imgs):
        imgs = imgs.cpu()
        imgs = imgs.type(torch.IntTensor)
        plt.figure(figsize=(15, 3*(imgs.shape[0])))
        grid_img = make_grid(imgs, nrow=5)
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()

    total_test_examples = 0
    num_correct_test = 0

    # true and false positive counts
    false_pos_count = 0
    true_pos_count = 0
    false_neg_count = 0
    true_neg_count = 0

    # visualize a random batch of data with examples
    num_viz = 10
    viz_index = random.randint(0, len(test_dataset)//eval_batchsize)

    # see how well the final trained model does on the test set
    with torch.no_grad():
        # don't save parameter gradients/changes since this is
        # not for model training
        for batch_index, (inputs, targets) in enumerate(test_loader):
            # make predictions
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = net_final(inputs)

            # compute prediction statistics
            _, predicted_class = predictions.max(1)
            total_test_examples += predicted_class.size(0)
            num_correct_test += predicted_class.eq(targets).sum().item()

            # thanks to
            #   https://gist.github.com/the-bass/cae9f3976866776dea17a5049013258d
            confusion_vector = predicted_class / targets
            num_true_pos = torch.sum(confusion_vector == 1).item()
            num_false_pos = torch.sum(confusion_vector == float('inf')).item()
            num_false_neg = torch.sum(torch.isnan(confusion_vector)).item()
            num_true_neg = torch.sum(confusion_vector == 0).item()

            true_pos_count += num_true_pos
            false_pos_count += num_false_pos
            false_neg_count += num_false_neg
            true_neg_count += num_true_neg

            # plot predictions
            if batch_index == viz_index:
                print('Example Images:')
                plot_imgbatch(inputs[:num_viz])
                print('Target labels:')
                print(targets[:num_viz].tolist())
                print('Classifier predictions:')
                print(predicted_class[:num_viz].tolist())

    # get total results
    # total prediction accuracy of network on test set
    file_name = "resnet50_iteration_1.txt"

    evaluation = Evaluation(false_pos_count, false_neg_count,
                            true_pos_count, true_neg_count, file_name)
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

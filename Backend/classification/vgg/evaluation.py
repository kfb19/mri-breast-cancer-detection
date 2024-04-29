""" This module calculates the evaluation metrics
for a CNN's performance, and saves them to a file
for analysis.
"""

import os


class Evaluation():
    """ This class evaluated a CNN based off a selection of
    metrics.

        Attributes:
            accuracy: the ratio of correctly predicted
            classifications (true positives and true negatives)
            to the total number of classifications.
            npv: the result of dividing the number of true negative
            classifications by the number of true negative plus the
            number of false negatives.
            ppv: aka precision, calculated by dividing the number of
            true positive classifications by the number of true positives
            plus the number of false positives.
            sensitivity: calculated by dividing the number of true positive
            classifications divided by the total number of positive scans
            (true positives and false negatives).
            specificity: the result of dividing the number of true negative
            classifications by the total number of negative scans (true
            negatives and false positives)
        """

    # Set attributes to a not-possible value (-1).
    accuracy = -1
    npv = -1
    ppv = -1
    sensitivity = -1
    specificity = -1

    def __init__(self, false_p, false_n, true_p, true_n, filename, folder):
        """ Initialises the object for evaluating a CNN.

        Args:
            false_p: the number of false positives.
            false_n: the number of false negatives.
            true_p: the number of true positives.
            true_n: the number of true negatives.
            file_name: the name of the file to save to.
            folder: the name of the folder to save to.
        """

        # Sets required variables.
        false_positives = false_p
        false_negatives = false_n
        true_positives = true_p
        true_negatives = true_n
        file_name = filename
        folder_name = folder

        # Calls functions to generate metrics then save to file.
        self.generate_metrics(false_positives, false_negatives,
                              true_positives, true_negatives)
        self.save_to_file(file_name, folder_name, false_positives,
                          false_negatives, true_positives, true_negatives)

    def generate_metrics(self, false_positives, false_negatives,
                         true_positives, true_negatives):
        """ Generates evaluation metrics from the number
        of false positives, true positives, false negatives
        and true negatives, and save these to the object.

        Args:
            false_positives: the number of false positives.
            false_negatives: the number of false negatives.
            true_positives: the number of true positives.
            true_negatives: the number of true negatives.
        """

        # Calulcate accuracy.
        self.accuracy = (true_positives + true_negatives) / \
            (true_positives + true_negatives + false_positives +
             false_negatives)
        # Calculate negative predicted value.
        if true_negatives + false_negatives == 0:
            self.npv = "No negative predictions. NPV not possible.\n"
        else:
            self.npv = true_negatives / (true_negatives + false_negatives)
        # Caluclate positive predicted value.
        if true_positives + false_positives == 0:
            self.ppv = "No positive predictions. PPV not possible.\n"
        else:
            self.ppv = true_positives / (true_positives + false_positives)
        # Calculate sensitivity.
        self.sensitivity = true_positives / (true_positives + false_negatives)
        # Calculate specificity.
        self.specificity = true_negatives / (true_negatives + false_positives)

    def save_to_file(self, file_name, folder, false_p, false_n, true_p,
                     true_n):
        """ Saves the calculated metrics to a file.

        Args:
            file_name: the name of the file to save to.
            folder: the name of the folder to save to.
            false_p: the number of false positives.
            false_n: the number of false negatives.
            true_p: the number of true positives.
            true_n: the number of true negatives.
        """

        file_path = 'PATH\\results'

        # Creating required file paths.
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        file_path = os.path.join(file_path, folder)

        if not os.path.exists(file_path):
            os.makedirs(file_path)

        file_path = os.path.join(file_path, file_name)

        # Writes metrics to file, providing it doesn't already exist.
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(f"Test set accuracy: {self.accuracy}\n")
                file.write(f"True positive classifications: {true_p}\n")
                file.write(f"False positive classifications: {false_p}\n")
                file.write(f"True negative classifications: {true_n}\n")
                file.write(f"False negative classifications: {false_n}\n")
                file.write(f"Negative predictive value: {self.npv}\n")
                file.write(f"Positive predictive value: {self.ppv}\n")
                file.write(f"Sensitivity: {self.sensitivity}\n")
                file.write(f"Specificity: {self.specificity}\n")
            file.close()
        else:
            print("File already exists, need new file name.\n")

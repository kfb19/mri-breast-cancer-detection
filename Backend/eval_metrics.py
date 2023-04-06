""" DOC STRING """

import os


class Evaluation():
    """ DOCSTRING HERE """
    accuracy = -1
    npv = -1
    ppv = -1
    sensitivity = -1
    specificity = -1

    def __init__(self, false_p, false_n, true_p, true_n, file_name):
        """ DOCSTRING HERE """
        false_positives = false_p
        false_negatives = false_n
        true_positives = true_p
        true_negatives = true_n

        self.generate_metrics(false_positives, false_negatives,
                              true_positives, true_negatives)
        self.save_to_file(file_name, false_positives, false_negatives,
                          true_positives, true_negatives)

    def generate_metrics(self, false_positives, false_negatives,
                         true_positives, true_negatives):
        """ INSERT DOCSTRING """
        self.accuracy = (true_positives + true_negatives) / \
            (true_positives + true_negatives + false_positives +
             false_negatives)
        self.npv = true_negatives / (true_negatives + false_negatives)
        self.ppv = true_positives / (true_positives + false_positives)
        self.sensitivity = true_positives / (true_positives + false_negatives)
        self.specificity = true_negatives / (true_negatives + false_positives)

    def save_to_file(self, file_name, false_p, false_n, true_p, true_n):
        """ INSERT DOCSTRING """
        file_path = 'E:\\data\\output'
        folder_name = '\\results'
        path = file_path + folder_name
        if not os.path.exists(path):
            os.makedirs(path)

        path = path + "\\" + file_name

        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as file:
                file.write(f"Test set accuracy: {self.accuracy}")
                file.write(f"{true_p} true positive classifications\n")
                file.write(f"{false_p} false positive classifications\n")
                file.write(f"{true_n} true negative classifications\n")
                file.write(f"{false_n} false negative classifications\n")
                file.write(f"Negative predictive value: {self.npv}\n")
                file.write(f"Positive predictive value: {self.ppv}\n")
                file.write(f"Sensitivity: {self.sensitivity}\n")
                file.write(f"Specificity: {self.specificity}\n")
            file.close()
        else:
            print("File already exists, need new file name.\n")

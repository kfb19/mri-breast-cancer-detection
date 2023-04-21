""" The EarlyStopper class is used to implement
early stopping while training the model, so that
it does not overfit.
"""

import numpy as np


class EarlyStopper:
    """ Implements early stopping during the training of CNN.

        Attributes:
            patience: the number of epochs the model will wait before stopping
            the training if the validation loss doesn't improve
            min_delta: the minimum change in validation loss to be considered
            as improvement
            counter: the number of epochs with no improvement in
            validation loss
            min_validation_loss: the best validation loss seen so far
    """

    def __init__(self, patience=1, min_delta=0):
        """ Initalises a member of the EarlyStopper
        class and sets basic variables.

        Args:
            patience: the number of epochs the model will wait before
            stopping the training if the validation loss doesn't improve,
            default value = 1
            min_delta: the minimum change in validation loss to be
            considered as improvement, default value = 0
        """
        # Set variables based on input.
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        # Set to positive infinity.
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        """ Performs a check for early stopping while the CNN is running.

        Args:
            validation_loss: the validation loss of the epoch running
        Returns:
            True if early stopping required
            False if early stopping not required
        """
        # Compares current validation loss with the best seen so far.
        if validation_loss < self.min_validation_loss:
            # New best - set counter to 0.
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            # Not better by enough - add 1 to counter.
            self.counter += 1
            # If over "patience" (num epochs) with no improvement, stop.
            if self.counter >= self.patience:
                return True
        return False

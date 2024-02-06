"""
A module containing useful utilities for deep learning
"""
import torch


class HyperCaller():
    """
    Class to call hyperparameters for deep-learning
    """
    def __init__(self):
        self.op_dict = {
            'SGD': torch.optim.SGD,
            'ADAM': torch.optim.Adam,
            'RMSP': torch.optim.RMSprop,
        }
        self.fn_dict = {
            'BCE': torch.nn.BCELoss,
            'BCEL': torch.nn.BCEWithLogitsLoss,
            'CEL': torch.nn.CrossEntropyLoss,
        }

    def __call__(self, config: dict) -> dict:
        """
        hyperparamter calling for single values
        """
        hypers = dict()
        for key, value in config.items():
            if key == 'OPTIMIZER':
                hypers[key] = self.op_dict[value]
            elif key == 'LOSS':
                hypers[key] = self.fn_dict[value]
            else:
                hypers[key] = value

        return hypers


class EarlyStopper():
    """ Early stops the training
    
    Notes:
        stop if validation loss doesn't improve after a given patience.

    Attributes:
        self.patience (int): 
            How long to wait after last time validation loss improved.
        self.counter (int):
            Current count after last time validation loss improved.
        self.best_score (float): Current best score
        self.early_stop (bool): Whether to stop the process after this
        self.val_loss_min (float): minimum validation loss so far
        self.delta (float): Minimum change in the monitored quantity
    """
    def __init__(self, patience=5, delta=0):
        """
        Args:
            patience (int):
                How long to wait after last time validation loss improved.
            delta (float):
                Minimum change in the monitored quantity
                to qualify as an improvement.
        """
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss: float) -> str:
        """
        Args:
            val_loss (float): new validation loss to assess

        Returns:
            message (str): the message to send out
        """
        message = ""

        if self.best_loss is None:
            self.best_loss = val_loss

        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            message = f"EarlyStopping counter: {self.counter} out of {self.patience}"
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            message = f"Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f})"
            self.best_loss = val_loss
            self.counter = 0

        return message

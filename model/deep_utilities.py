'''
A module containing useful utilities for deep learning
'''

import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F


def print_and_log(message: str, log_file=None):
    '''
    function to print and log the message simultaneously
    '''
    print(message)
    if log_file:
        log_file.write(message+'\n')


class HyperCaller():
    '''
    class about calling hyperparameters
    '''
    def __init__(self, config_hyper: dict):
        self.config_hyper = config_hyper

    def hyper_calling(self) -> dict:
        '''
        hyperparamter calling for single values
        '''
        op_dict = {
            'SGD': opt.SGD,
            'ADAM': opt.Adam,
            'rmsprop': opt.RMSprop,
        }

        fn_dict = {
            'BCE': F.binary_cross_entropy,
            'BCEL': F.binary_cross_entropy_with_logits,
            'val_loss': F.binary_cross_entropy,
        }

        hypers = {}
        for key, value in self.config_hyper.items():
            param = value[0] # single value
            if key == 'OPTIMIZER':
                hypers[key.lower()] = op_dict[param]
            elif key == 'VALUE':
                hypers[key.lower()] = fn_dict[param]
            else:
                hypers[key.lower()] = param
        return hypers


class EarlyStopper():
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, log_file=None, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved. (Default: 7)
            verbose (bool): If True, prints a message for each validation loss improvement.
                            (Default: False)
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.log_file = log_file
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss: float, model: object, save_dir='.'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_dir)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print_and_log(f'EarlyStopping counter: {self.counter} out of {self.patience}',
                          self.log_file)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_dir)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: object, save_dir: str):
        '''Saves model when validation loss decrease.'''
        print_and_log(
            f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). ' \
            + '...Saving model ...', self.log_file)
        torch.save(model.state_dict(), f'{save_dir}/checkpoint.pt')
        self.val_loss_min = val_loss

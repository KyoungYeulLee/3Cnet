"""
Pytorch networks are defined to be used for 3Cnet
"""
import torch
import numpy

from typing import Tuple, List, Dict
from torch.utils.data import DataLoader
from torch import tensor


class FeatureExtractor(torch.nn.Module):
    """
    Network to combine sequences and MSA pattern into flat features using LSTM
    """
    def __init__(
        self,
        num_aa: int,
        msa_aa_size: int,
        embed_dim: int,
        hidden_dim: int,
        output_dim: int,
        p_dropout: float,
    ):
        super().__init__()

        self.embed = torch.nn.Embedding(
            num_embeddings = num_aa,
            embedding_dim = embed_dim,
            padding_idx=0,
        )
        self.ref_bi_lstm = torch.nn.LSTM(
            input_size = embed_dim,
            hidden_size = int(hidden_dim/2),  # bi-directional
            batch_first = True,
            bidirectional = True,
        )
        self.alt_bi_lstm = torch.nn.LSTM(
            input_size = embed_dim,
            hidden_size = int(hidden_dim/2),  # bi-directional
            batch_first = True,
            bidirectional = True,
        )
        self.msa_bi_lstm = torch.nn.LSTM(
            input_size = msa_aa_size,
            hidden_size = int(hidden_dim/2),  # bi-directional
            batch_first = True,
            bidirectional = True,
        )
        self.ref_msa_lstm = torch.nn.LSTM(
            input_size = hidden_dim * 2,
            hidden_size = output_dim,
            batch_first = True
        )
        self.alt_msa_lstm = torch.nn.LSTM(
            input_size = hidden_dim * 2,
            hidden_size = output_dim,
            batch_first = True
        )
        self.dropout = torch.nn.Dropout(p=p_dropout)

    def forward(
        self,
        ref_tensor: tensor,
        alt_tensor: tensor,
        msa_tensor: tensor,
    ) -> tensor:
        """
        Args:
            ref_tensor (
                tensor(dtype=int, size=(batch_size, n_seq))
            ): reference sequence tensor
            alt_tensor (
                tensor(dtype=int, size=(batch_size, n_seq))
            ): alternative sequence tensor
            msa_tensor (
                tensor(dtype=float, size=(batch_size, n_seq, n_aa))
            ): MSA feature tensor

        Returns:
            tensor(dtype=float, size=(batch_size, ouput_dim * 2))
        """
        x_ref = self.embed(ref_tensor)  # (batch_size, n_seq, embed_dim)
        x_alt = self.embed(alt_tensor)  # (batch_size, n_seq, embed_dim)
        x_ref, _ = self.ref_bi_lstm(x_ref)  # (batch_size, n_seq, hidden_dim)
        x_alt, _ = self.alt_bi_lstm(x_alt)  # (batch_size, n_seq, hidden_dim)
        x_msa, _ = self.msa_bi_lstm(msa_tensor)
            # (batch_size, n_seq, hidden_dim)
        x_ref = torch.cat((x_ref, x_msa), dim=2)
            # (batch_size, n_seq, hidden_dim * 2)
        x_alt = torch.cat((x_alt, x_msa), dim=2)
            # (batch_size, n_seq, hidden_dim * 2)
        x_ref, _ = self.ref_msa_lstm(x_ref)  # (batch_size, n_seq, output_dim)
        x_alt, _ = self.alt_msa_lstm(x_alt)  # (batch_size, n_seq, output_dim)
        x_ref = x_ref[:, -1, :]  # shape = (batch_size, output_dim)
        x_alt = x_alt[:, -1, :]  # shape = (batch_size, output_dim)
        x_ref = self.dropout(x_ref)
        x_alt = self.dropout(x_alt)
        x_tensor = torch.cat((x_ref, x_alt), dim=1)
            # shape = (batch_size, output_dim * 2)

        return x_tensor


class SingleTask(torch.nn.Module):
    """Network for 3Cnet trained by single task.
    
    Notes:
        The object of this class will be the model to train and evaluate.
    """
    def __init__(
        self,
        num_aa: int,
        msa_aa_size: int,
        embed_dim: int,
        hidden_dim: int,
        output_dim: int,
        fc_size: int,
        n_classes: int,
        p_dropout: float,
    ):
        super().__init__()
        self.common_net = FeatureExtractor(
            num_aa = num_aa,
            msa_aa_size = msa_aa_size,
            embed_dim = embed_dim,
            hidden_dim = hidden_dim,
            output_dim = output_dim,
            p_dropout = p_dropout,
        )
        self.full_connected = torch.nn.Linear(output_dim * 2, fc_size)
        self.classifier = torch.nn.Linear(fc_size, n_classes)
        self.dropout = torch.nn.Dropout(p=p_dropout)

    def forward(
        self,
        ref_tensor: tensor,
        alt_tensor: tensor,
        msa_tensor: tensor,
    ) -> tensor:
        """
        Args:
            ref_tensor (
                tensor(dtype=int, size=(batch_size, n_seq))
            ): reference sequence tensor
            alt_tensor (
                tensor(dtype=int, size=(batch_size, n_seq))
            ): alternative sequence tensor
            msa_tensor (
                tensor(dtype=float, size=(batch_size, n_seq, n_aa))
            ): MSA feature tensor

        Returns:
            tensor(dtype=float, size=(batch_size, n_classes))
        """
        x_tensor = self.common_net(ref_tensor, alt_tensor, msa_tensor)
        x_tensor = self.full_connected(x_tensor)  # (batch_size, fc_size)
        x_tensor = self.dropout(x_tensor)
        x_tensor = self.classifier(x_tensor)  # (batch_size, n_classes)

        return x_tensor

    def optimize_epoch(
        self,
        train_loader: DataLoader,
        criterion: torch.nn,
        optimizer: torch.optim,
    ) -> float:
        """ Train the model for an epoch for the given dataset
        
        Args:
            train_loader (DataLoader): training dataset loader
            criterion (loss_function): loss function to back-propagate
            optimizer (optimizer): optimizer to train the network

        Returns:
            (float): average loss for the train dataset
        """
        average_loss = 0.0
        self.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out_tensor = self.forward(batch['ref'], batch['alt'], batch['msa'])
            train_loss = criterion(out_tensor, batch['label'])
            train_loss.backward()
            optimizer.step()
            average_loss += train_loss.item()
        average_loss /= len(train_loader.dataset)

        return average_loss

    def evaluate_epoch(
        self,
        test_loader: DataLoader,
        criterion: torch.nn,
    ) -> Tuple[float, float]:
        """ Evaluate the model for an epoch for the given dataset
        
        Args:
            test_loader (DataLoader): test dataset loader
            criterion (loss_function): loss function to evaluate the loss

        Returns:
            Tuple[float, float]: average test loss and accuracy
        """
        self.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for batch in test_loader:
                out_tensor = self.forward(
                    batch['ref'],
                    batch['alt'],
                    batch['msa'],
                )
                test_loss += criterion(out_tensor, batch['label']).item()
                pred = out_tensor.max(dim=1, keepdim=False)[1]
                correct += pred.eq(batch['label']).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)

        return test_loss, accuracy

class MultiTask(torch.nn.Module):
    """Network for 3Cnet trained by multiple tasks.
    For example, both clinical and conservation data can be trained simultaneously.
    
    Notes:
        The object of this class will be the model to train and evaluate.
    """
    def __init__(
        self,
        num_aa: int,
        msa_aa_size: int,
        embed_dim: int,
        hidden_dim: int,
        output_dim: int,
        fc_size: int,
        n_classes: int,
        p_dropout: float,
    ):
        super().__init__()
        self.common_net = FeatureExtractor(
            num_aa = num_aa,
            msa_aa_size = msa_aa_size,
            embed_dim = embed_dim,
            hidden_dim = hidden_dim,
            output_dim = output_dim,
            p_dropout = p_dropout,
        )
        self.clinical_fc = torch.nn.Linear(output_dim * 2, fc_size)
        self.conservation_fc = torch.nn.Linear(output_dim * 2, fc_size)
        self.clinical_cls = torch.nn.Linear(fc_size, n_classes)
        self.conservation_cls = torch.nn.Linear(fc_size, n_classes)
        self.dropout = torch.nn.Dropout(p=p_dropout)
        self.iterator = None

    def forward(
        self,
        ref_tensor: tensor,
        alt_tensor: tensor,
        msa_tensor: tensor,
        is_clinical: bool,
    ) -> tensor:
        """
        Notes:
            Two separate networks are called respectively
            according to the given 'is_clinical'

        Args:
            ref_tensor (
                tensor(dtype=int, size=(batch_size, n_seq))
            ): reference sequence tensor
            alt_tensor (
                tensor(dtype=int, size=(batch_size, n_seq))
            ): alternative sequence tensor
            msa_tensor (
                tensor(dtype=float, size=(batch_size, n_seq, n_aa))
            ): MSA feature tensor
            is_clinical (bool): whether to use the network for clinical data

        Returns:
            tensor(dtype=float, size=(batch_size, n_classes))
        """
        x_tensor = self.common_net(ref_tensor, alt_tensor, msa_tensor)
        if is_clinical:
            x_tensor = self.clinical_fc(x_tensor)  # (batch_size, fc_size)
            x_tensor = self.dropout(x_tensor)
            x_tensor = self.clinical_cls(x_tensor)  # (batch_size, n_classes)
        else:
            x_tensor = self.conservation_fc(x_tensor)  # (batch_size, fc_size)
            x_tensor = self.dropout(x_tensor)
            x_tensor = self.conservation_cls(x_tensor)
                # (batch_size, n_classes)

        return x_tensor

    def set_iterator(self, ensemble_loader: DataLoader) -> bool:
        """
        initialize ensemble iterator
        
        Args:
            ensemble_loader (DataLoader): Ensemble dataset loader

        Returns:
            (bool): True if succeed
        """
        self.iterator = iter(ensemble_loader)

        return True

    def optimize_step(
        self,
        batch: tensor,
        criterion: torch.nn,
        optimizer: torch.optim,
        is_clinical: bool,
    ) -> float:
        """
        Train the model for the given batch
        
        Args:
            batch (tensor): training batch
            criterion (loss_function): loss function to back-propagate
            optimizer (optimizer): optimizer to train the network
            is_clinical (bool): whether to use the network for clinical data

        Returns:
            (float): summation of the loss for the batch
        """
        self.train()
        optimizer.zero_grad()
        out_tensor = self.forward(
            batch['ref'],
            batch['alt'],
            batch['msa'],
            is_clinical,
        )
        train_loss = criterion(out_tensor, batch['label'])
        train_loss.backward()
        optimizer.step()

        return train_loss.item()

    def optimize_epoch(
        self,
        train_loader: DataLoader,
        ensemble_loader: DataLoader,
        criterion: torch.nn,
        optimizer: torch.optim,
    ) -> float:
        """ Train the model for an epoch for the given dataset
        
        Args:
            train_loader (DataLoader): training dataset loader
            ensemble_loader (DataLoader): Ensemble dataset loader
            criterion (loss_function): loss function to back-propagate
            optimizer (optimizer): optimizer to train the network

        Returns:
            (float): average loss for the train dataset
        """
        if self.iterator is None:
            print("Iterator should be set before optimization")
            return None

        train_loss = 0.0
        for train_batch in train_loader:
            try:
                ensemble_batch = next(self.iterator)
            except StopIteration:
                self.iterator = iter(ensemble_loader)
                ensemble_batch = next(self.iterator)
            self.optimize_step(
                ensemble_batch,
                criterion,
                optimizer,
                is_clinical=False,
            )
            train_loss += self.optimize_step(
                train_batch,
                criterion,
                optimizer,
                is_clinical=True,
            )
        train_loss /= len(train_loader.dataset)

        return train_loss

    def evaluate_epoch(
        self,
        test_loader: DataLoader,
        criterion: torch.nn,
    ):
        """ Evaluate the model for an epoch for the given dataset

        Args:
            test_loader (DataLoader): test dataset loader
            criterion (loss_function): loss function to evaluate the loss

        Returns:
            Tuple[float, float]: average test loss and accuracy
        """
        self.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for batch in test_loader:
                out_tensor = self.forward(
                    batch['ref'],
                    batch['alt'],
                    batch['msa'],
                    is_clinical=True
                )
                test_loss += criterion(out_tensor, batch['label']).item()
                pred = out_tensor.max(dim=1, keepdim=False)[1]
                correct += pred.eq(batch['label']).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)

        return test_loss, accuracy

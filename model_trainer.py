"""
The module for training the 3Cnet model
"""
import os
import sys
import argparse
import numpy as np
import pickle as pk

from datetime import datetime
from logging import Logger
from typing import Tuple, List, Dict
from pathlib import Path
from copy import deepcopy
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split

from cccnet.utils import get_config, get_logger, save_config
from cccnet.deep_utils import HyperCaller, EarlyStopper
from cccnet.dataset_builder import Builder
from cccnet.torch_dataset import ThreeCnetDataset
from cccnet.torch_network import SingleTask, MultiTask

from neuron.constants import HGVSp, ProteinMutationTypes
from neuron.seq_collection import SeqCollection

import torch
from torch import tensor
from torch.utils.data import DataLoader

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def receive_args() -> argparse.Namespace:
    """ Receive arguments given by a user

    Returns (argparse.Namespace): parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--save_name",
        type=str,
        default="",
        help="the name to save the config and model",
    )

    return parser.parse_args()

class Trainer:
    """ Train the model using given trining dataset

    Attributes:
        self.config (dict): configuration dictionary
        self.model_dir (str): the directory to save the model
        self.train_dir (str): the directory to save the training status
        self.model (torch.nn.Module): the model object. initially None.
    """
    def __init__(
        self,
        config: dict,
        save_name="",
        logger=Logger(__name__),
    ):
        self.logger = logger
        self.model_dir = config.MODEL_DIR
        if save_name:
            self.model_dir = os.path.join(CURRENT_DIR, self.model_dir, save_name)
        if os.path.exists(self.model_dir):
            logger.info("#Error: Model directory alreadly exists")
            sys.exit(1)

        os.makedirs(self.model_dir, exist_ok=True)
        self.train_dir = os.path.join(self.model_dir, 'TRAINING')
        os.makedirs(self.train_dir, exist_ok=True)
        save_config(config, save_dir=self.train_dir)
        win_size = config.ARCHITECTURES.WIN_SIZE
        self.msa_aa_size = config.ARCHITECTURES.MSA_AA_SIZE
        self.data_builder = Builder(
            win_size,
            self.msa_aa_size,
            logger,
        )
        self.model_type = config.ARCHITECTURES.MODEL_TYPE
        if self.model_type == 'SingleTask':
            self.model_config = config.ARCHITECTURES.SINGLE_TASK
        elif self.model_type == 'MultiTask':
            self.model_config = config.ARCHITECTURES.MULTI_TASK
        else:
            logger.info("#ERROR: Unexpected model name in configuration")
            sys.exit(1)

        self.train_data = config.HGVSPS.TRAIN
        self.train_hyper = config.TRAIN.HYPERPARAMS
        self.missense_only = config.TRAIN.MISSENSE_ONLY
        self.save_each_epoch = config.TRAIN.SAVE_EACH_EPOCH
        self.valid_ratio = config.TRAIN.VALIDATION_RATIO
        self.model = None

    def _split_train_valid(
        self,
        seq_collection: SeqCollection,
        hgvsp_list: list,
        valid_ratio: float,
    ) -> Tuple[SeqCollection, SeqCollection]:
        """ Split sequence collection into training set and validation set.

        Notes:
            Sequence objects for output collections
            are called by reference from the given collection (shared memory).

        Args:
            seq_collection (SeqCollection): sequence collection to split
            hgvsp_list (list): the list of HGVSp to be used as dataset
                hgvsp_list <= seq_collection.seqs.keys()
            valid_ratio (float): the ratio of validation set from dataset

        Returns:
            Tuple[SeqCollection, SeqCollection]
                train_seqs: sequence collection to train the model
                valid_seqs: sequence collection to validate the model
        """
        train_seqs = SeqCollection(
            hgvs_to_reference_seq_mapping=dict(),
            unusable_hgvses=[],
            sequence_type=seq_collection.sequence_type,
        )
        valid_seqs = SeqCollection(
            hgvs_to_reference_seq_mapping=dict(),
            unusable_hgvses=[],
            sequence_type=seq_collection.sequence_type,
        )
        train_list, valid_list = train_test_split(
            hgvsp_list,
            test_size=valid_ratio,
            shuffle=True,
        )

        try:
            for hgvsp in train_list:
                train_seqs.seqs[hgvsp] = seq_collection.seqs[hgvsp]
            for hgvsp in valid_list:
                valid_seqs.seqs[hgvsp] = seq_collection.seqs[hgvsp]
        except KeyError:
            self.logger.info("#ERROR: HGVSp {hgvsp} not in seq_collection")
            return

        self.logger.info(
            (
                f"the number of training set = {len(train_seqs.seqs)}, "
                f"the number of validation set = {len(valid_seqs.seqs)}"
            )
        )

        return train_seqs, valid_seqs

    def get_train_valid_dataset(
        self,
        dataset_dict: Dict[str, ThreeCnetDataset],
        train_seqs: SeqCollection,
     ) -> None:
        """get torch dataset of training and validation dataset using given dataset.

        Notes:
            validation dataset is retrieved only when validation ratio > 0

        Args:
            dataset_dict (Dict[str, ThreeCnetDataset]): pytorch dataset dictionary
            train_seqs (SeqCollection): training dataset

        """
        if self.valid_ratio > 0:
            train_hgvsp = list(train_seqs.seqs.keys())
            train_set, valid_set = self._split_train_valid(
                seq_collection=train_seqs,
                hgvsp_list=train_hgvsp,
                valid_ratio=self.valid_ratio,
            )
            self.logger.info("...Training dataset parsing...")
            dataset_dict['train'] = self.data_builder.get_torch_dataset(
                train_set,
                do_balance_label=True,
            )
            self.logger.info("...Valid dataset is from training set...")
            dataset_dict['valid'] = self.data_builder.get_torch_dataset(
                valid_set,
                do_balance_label=False,
            )
        else:
            dataset_dict['train'] = self.data_builder.get_torch_dataset(
                train_seqs,
                do_balance_label=True,
            )

        return

    def get_dataset(self, dataset_dict: Dict[str, ThreeCnetDataset]) -> None:
        """retrive torch dataset from training data and assign them to dataset_dict
        
        Notes:
            keys of the dataset_dict can include
            1) 'train'
            2) 'valid' (if validation ratio > 0)
            3) 'ensemble' (if conservation path is given)

        Args:
            dataset_dict (Dict[str, ThreeCnetDataset]): pytorch dataset dictionary
        """
        self.logger.info("...Retriving ClinVar...")
        clinvar_path = self.train_data['CLINICAL']
        clinical_seqs = self.data_builder.parse_HGVSPs(
            clinvar_path,
            self.missense_only,
        )
        
        common_path = self.train_data['COMMON']
        if common_path:
            self.logger.info("...Retriving GnomAD...")
            common_seqs = self.data_builder.parse_HGVSPs(
                common_path,
                self.missense_only,
            )
            clinical_seqs = clinical_seqs.merge(common_seqs)

        self.get_train_valid_dataset(dataset_dict, clinical_seqs)
            # dataset_dict['train'] and/or dataset_dict['valid'] is retrieved

        if self.model_type == 'MultiTask':
            conservation_path = self.train_data['CONSERVATION']
            self.logger.info("...Retriving Converation...")
            conservation_seqs = self.data_builder.parse_HGVSPs(
                conservation_path,
                self.missense_only,
            )
            
            conservation_dataset = self.data_builder.get_torch_dataset(
                conservation_seqs,
                do_balance_label=False,
            )
            dataset_dict['ensemble'] = conservation_dataset

        return

    def build_model(self) -> bool:
        """ Build the model object based on the configuration.

        Notes:
            self.model is None after initialization.
            This function builds the model and annotates it to self.model.

        Returns:
            (bool): True if succeed to build the model else False
        """
        self.model = None
        if self.model_type == 'SingleTask':
            self.model = SingleTask(
                num_aa = self.model_config['NUM_AA'],
                msa_aa_size = self.msa_aa_size,
                embed_dim = self.model_config['EMBED_SIZE'],
                hidden_dim = self.model_config['HIDDEN_SIZE'],
                output_dim = self.model_config['OUTPUT_SIZE'],
                fc_size = self.model_config['FC_SIZE'],
                n_classes = 2,
                p_dropout = self.train_hyper['DROPOUT'],
            ).to(DEVICE)
        elif self.model_type == 'MultiTask':
            self.model = MultiTask(
                num_aa = self.model_config['NUM_AA'],
                msa_aa_size = self.msa_aa_size,
                embed_dim = self.model_config['EMBED_SIZE'],
                hidden_dim = self.model_config['HIDDEN_SIZE'],
                output_dim = self.model_config['OUTPUT_SIZE'],
                fc_size = self.model_config['FC_SIZE'],
                n_classes = 2,
                p_dropout = self.train_hyper['DROPOUT'],
            ).to(DEVICE)

        if self.model is None:
            self.logger.info("#ERROR: Model fails to be created.")
            return False

        return True

    def train_network(
        self,
        dataset_dict: Dict[str, ThreeCnetDataset],
    ) -> bool:
        """ Train the model using given datasets

        Args:
            dataset_dict (dict): dictionary containing datasets
                (key): ID for each dataset. 'train', 'valid', 'test', ...
                (value): pytorch dataset (ThreeCnetDataset)

        Returns:
            (bool): True if training is correctly finished
        """
        # model loading (if not defined yet)
        if self.model is None:
            self.logger.info("...Building model network...")
            if not self.build_model(): # building model
                self.logger.info("#Error: failed to build model")
                return False

        hyper_caller = HyperCaller()
        hypers = hyper_caller(self.train_hyper)
        early_stopper = EarlyStopper(patience=hypers['PATIENCE'])
        optimizer = hypers['OPTIMIZER'](
            self.model.parameters(),
            lr=hypers['LEARNING_RATE'],
        )
        criterion = hypers['LOSS'](reduction='sum')
        start_time = datetime.now().isoformat(timespec="seconds")
        train_log_path = os.path.join(self.train_dir, f"{start_time}.log")

        train_loader = DataLoader(
            dataset_dict['train'],
            batch_size=hypers['BATCH_SIZE'],
            shuffle=True,
        )
        if 'valid' in dataset_dict:
            valid_loader = DataLoader(
                dataset_dict['valid'],
                batch_size=hypers['BATCH_SIZE'],
                shuffle=False,
            )
        if 'ensemble' in dataset_dict:
            ensemble_loader = DataLoader(
                dataset_dict['ensemble'],
                batch_size=hypers['BATCH_SIZE'],
                shuffle=True,
            )
        if self.model_type == 'MultiTask':
            self.model.set_iterator(ensemble_loader)

        with open(train_log_path, 'w', 1) as fout:
            fout.write("#Epoch\tTrain_loss\tTest_loss\tAccuracy\tImproved\n")
            for epoch in range(1, hypers['EPOCH'] + 1):
                if self.model_type == 'SingleTask':
                    train_loss = self.model.optimize_epoch(
                        train_loader,
                        criterion,
                        optimizer,
                    )
                elif self.model_type == 'MultiTask':
                    train_loss = self.model.optimize_epoch(
                        train_loader,
                        ensemble_loader,
                        criterion,
                        optimizer,
                    )

                valid_loss, accuracy = self.model.evaluate_epoch(
                    valid_loader,
                    criterion,
                )

                self.logger.info(
                    (
                        f"epoch = {epoch}, train loss = {train_loss}, "
                        f"valid loss = {valid_loss}, accuracy = {accuracy}"
                    )
                )
                msg = early_stopper(valid_loss)
                if msg:
                    self.logger.info(msg)
                is_improved = (early_stopper.counter == 0)  # best renewal
                fout.write(
                    (
                        f"{epoch}\t{train_loss}\t{valid_loss}\t"
                        f"{accuracy}\t{is_improved}\n"
                    )
                )

                if self.save_each_epoch:
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.model_dir, f"{epoch}.pt"),
                    )
                elif is_improved:
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.model_dir, f"{epoch}.pt"),
                    )

                if early_stopper.early_stop:
                    self.logger.info("...Early stopping...")
                    break

        return True

    def train(self) -> bool:
        """ Dataset loading and model training

        Returns:
            (bool): True when run successfully
        """
        dataset_dict = dict()
        self.get_dataset(dataset_dict)

        # Train network using dataset
        self.logger.info("...Training model...")
        if not self.train_network(dataset_dict):
            self.logger.info("#Error: Model fails to be trained.")
            return False

        self.logger.info("## Training done")

        return True


if __name__ == "__main__":
    ARGS = receive_args()
    CONFIG = get_config()
    LOGGER = get_logger(
        module_name = Path(__file__).stem,
        data_dir=CONFIG.MODEL_DIR,
        file_name=ARGS.save_name,
    )
    LOGGER.info(ARGS)

    model_trainer = Trainer(CONFIG, ARGS.save_name, LOGGER)
    LOGGER.info('...Training start...')

    if model_trainer.train():
        LOGGER.info("### Correctly finished")
    else:
        LOGGER.info("### Unexpected exit")

    LOGGER.info('...Training end...')

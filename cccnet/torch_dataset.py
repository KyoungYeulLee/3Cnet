"""
Pytorch datasets are defined to be used for 3Cnet
"""
import os
import sys
import numpy as np
from typing import Dict

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

from neuron.seq_collection import SeqCollection
from neuron.featurizers import ThreeCNetProteinSeqFeaturizer

import torch
from torch import tensor
from torch.utils.data import Dataset

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


def to_tensor(sample: dict) -> dict:
    """
    A transformer to change numpy sample to tensor sample
    """
    sample_tensor = {
        key: torch.from_numpy(value).to(device=DEVICE)
        for key, value in sample.items()
    }

    return sample_tensor


class ThreeCnetDataset(Dataset):
    """ Pytorch variant dataset for 3Cnet

    Notes:
        This class includes both hgvsp2rap and seq_collection
        to use pre-built sequence feature data (rap) for iteration
        while building MSA features instantly when samples are called.

    Attributes:
        self.hgvsp_list (list): list of variant HGVSp (indexed accordingly)
        self.hgvsp2rap (dict): dictionary with sequence data for each HGVSp
            key (str): HGVSp
            value (tuple): (reference seq, alternative seq, pathogenicity) 
        self.auto_window_size (int): the size of window
            seq length = 2 * auto_window_size + 1
        self.msa_aa_size (int):
            the size of AA used in the Mutiple Sequence Alignments
        self.seq_collection (SeqCollection): sequence collection of variants
        self.used_featurizer (ThreeCNetProteinSeqFeaturizer):
            featurizer to build MSA feature matrix for each varinant
    """
    def __init__(
        self,
        hgvsp_list: list,
        hgvsp2rap: dict,
        seq_collection: SeqCollection,
        auto_window_size: int,
        msa_aa_size: int,
        used_featurizer: ThreeCNetProteinSeqFeaturizer,
    ):
        super().__init__()
        self.hgvsp_list = hgvsp_list
        self.hgvsp2rap = hgvsp2rap
        self.auto_window_size = auto_window_size
        self.msa_aa_size = msa_aa_size
        self.seq_collection = seq_collection
        self.used_featurizer = used_featurizer

    def __len__(self) -> int:
        """
        get length of the dataset
        """
        return len(self.hgvsp_list)

    def __getitem__(self, index: int):
        """
        get sample (variant) for the index
        """
        if torch.is_tensor(index):
            index = index.tolist()

        hgvsp = self.hgvsp_list[index]
        ref, alt, label = self.hgvsp2rap[hgvsp]
        seq_obj = self.seq_collection.seqs[hgvsp]
        msa = seq_obj.featurize(
            auto_window_size=self.auto_window_size,
            featurizer=self.used_featurizer,
            featurize_fields=["msa"]
        )["msa"]  # get only msa among the features

        if msa is None:
            msa = np.zeros(
                (2 * self.auto_window_size + 1, self.msa_aa_size),
                dtype=np.float32,
            )  # zero_padding

        data_dict = {
            'ref': np.array(ref, dtype=np.int64),
            'alt': np.array(alt, dtype=np.int64),
            'msa': np.array(msa, dtype=np.float32),
            'label': np.array(label, dtype=np.int64),
        }

        data_dict = to_tensor(data_dict)

        return data_dict

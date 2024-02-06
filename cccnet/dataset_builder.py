"""
The module for the building dataset for 3Cnet
from parsing DB to build pytorch dataset object
"""
import os
import sys
import numpy as np
import pickle as pk
from collections import namedtuple
from typing import Dict, List, Literal

from logging import Logger

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(CURRENT_DIR)
sys.path.append(PARENT_DIR)

from neuron.constants import HGVSp, ProteinMutationTypes
from neuron.seq_database import SeqDatabase
from neuron.seq_collection import SeqCollection
from neuron.sequences import ProteinSeqObj
from neuron.featurizers import ThreeCNetProteinSeqFeaturizer
from torch_dataset import ThreeCnetDataset


RAP = namedtuple('RAP', ['ref', 'mut', 'label'])


class Builder:
    """ Parse DB and build the dataset of variants

    Attributes:
        self.logger (Logger): process logger
        self.win_size (int): the size of amino acid window
            when win_size == 100, the length of seq == 201.
        self.msa_aa_size (int): the number of amino-acids for MSA
        self.used_featurizer (ThreeCNetProteinSeqFeaturizer):
            amino-acid featurizer to turn string into List[int]
    """
    def __init__(
        self,
        win_size: int,
        msa_aa_size: int,
        logger=Logger(__name__),
    ):
        self.logger = logger
        self.sequence_db = SeqDatabase()
        self.win_size = win_size
        self.msa_aa_size = msa_aa_size
        self.used_featurizer = ThreeCNetProteinSeqFeaturizer(
            token_length=1, gap_token=ProteinSeqObj.gap_tok
        )

    def build_dataset(
        self,
        hgvsps: list,
        missense_only: bool,
        remove_synonym: bool=True,
        label_dict: dict=None,
        label_value: np.array=None,
    ) -> SeqCollection:
        """ Build sequence collection object for given HGVSp
        
        Notes:
            Some HGVSp can be omitted during the process. see log files.

        Args:
            hgvsp (List[str]): the list of HGVSp to get sequence objects.
            missense_only (bool): Whether to allow only missense variants
            remove_synonym (bool): Whether to remove synonymous variants
            label_dict (Dict[HGVSp, np.array]): 
                When given, impose metadata "label" to the sequence collection
                so that it contains pathogenicity (np.array) for given HGVSp
            label_value (np.array):
                When given, impose metadata "label" to the sequence collection
                so that all the HGVSp have the given pathogenicity (np.array)

        Returns:
            seq_collection (SeqCollection): the collection of sequence objects
        """
        seq_collection = self.sequence_db.select(hgvsps)
        self.logger.info(
            f"{len(seq_collection.seqs)} are left after sequence selection"
        )

        for reason in seq_collection.memory:
            target_hgvsps = seq_collection.memory[reason]
            if reason == 'selection_pass':
                continue

            self.logger.info(
                f"{len(target_hgvsps)} are removed because of {reason}"
            )

        if remove_synonym:
            seq_collection = seq_collection.filter_by_mutation_types(
                filter_mutation_types=[ProteinMutationTypes.SYNONYMOUS],
                discard_mode = True,
            ) # remove synonymous variants
            self.logger.info(
                (
                    f"{len(seq_collection.seqs)} are left "
                    f"after synonymous filtering"
                )
            )

        if missense_only:
            seq_collection = seq_collection.filter_by_mutation_types(
                filter_mutation_types=[ProteinMutationTypes.MISSENSE],
                discard_mode = False,
            )  # retain missense variants only
            self.logger.info(
                (
                    f"{len(seq_collection.seqs)} are left "
                    f"after missense_only filtering"
                )
            )

        if label_dict is not None:
            seq_collection.assign_metadata(
                metadata_key="label",
                mapping_dict=label_dict,
            )
        elif label_value is not None:
            seq_collection.assign_metadata(
                metadata_key="label",
                metadata_value=label_value,
            )

        return seq_collection

    def parse_HGVSPs(
        self,
        hgvsp_path: str,
        missense_only: bool,
    ) -> SeqCollection:
        """ Parse conservation data from given file path
        
        Args:
            hgvsp_path (str): the file path for HGVSp (comma separated)
            missense_only (bool): Whether to allow only missense variants

        Returns:
            seq_collection (SeqCollection): the collection of sequence objects
        """
        hgvsp2label = dict()
        with open(hgvsp_path) as fin:
            for line in fin:
                words = line.rstrip('\n').split(',')
                if len(words) == 1:
                    hgvsp = words[0]
                    label = -1 # no label
                else:
                    hgvsp, label = words[0], words[1]
                hgvsp2label[hgvsp] = np.array(int(label))

        hgvsps_list = list(
            hgvsp2label.keys()
        )

        self.logger.info(
            f"the number of parsed HGVSp = {len(hgvsps_list)}"
        )
        
        seq_collection = self.build_dataset(
            hgvsps=hgvsps_list,
            missense_only=missense_only,
            label_dict=hgvsp2label,
        )
        
        return seq_collection

    def get_torch_dataset(
        self,
        seq_collection: SeqCollection,
        do_balance_label: bool,
    ) -> ThreeCnetDataset:
        """ Get pytorch dataset from given sequence collection

        Notes:
            Some HGVSp can be omitted during the process. see log files.

        Args:
            seq_collection (SeqCollection): the collection of sequence objects
            do_balance_label (bool):
                Whether to solve the imbalance between pathogenicity
                by multiple sampling for smaller set

        Returns:
            (ThreeCnetDataset): the pytorch dataset for the sequences
        """
        count = {
            'total': 0,
            'with_seq': 0,
            'with_msa': 0,
            'pathogenic': 0,
            'benign': 0,
            'unknown': 0,
            'error': 0,
        }
        hgvsp2rap = dict()
        hgvsp_list = list(seq_collection.seqs.keys())

        for hgvsp in hgvsp_list:
            seq_obj = seq_collection.seqs[hgvsp]
            label = seq_obj.metadata["label"]

            count['total'] += 1
            try:
                seq_features = seq_obj.featurize(
                    auto_window_size=self.win_size,
                    featurizer=self.used_featurizer,
                )

                if (
                    seq_features['ref'] is not None
                    and seq_features['mut'] is not None
                ):
                    hgvsp2rap[hgvsp] = RAP(
                        seq_features['ref'],
                        seq_features['mut'],
                        label,
                    )
                    count['with_seq'] += 1
                    if label == 0:
                        count['benign'] += 1
                    elif label == 1:
                        count['pathogenic'] += 1
                    else:
                        count['unknown'] += 1

                    if seq_features['msa'] is not None:
                        count['with_msa'] += 1

                else:
                    count['error'] += 1

            except (RuntimeError, TypeError, KeyError) as e:
                count['error'] += 1

        self.logger.info(
            f"HGVSp with encoded sequence (ref, alt) = {count['with_seq']}"
        )
        self.logger.info(
            f"HGVSp with pathogenic label = {count['pathogenic']}"
        )
        self.logger.info(f"HGVSp with benign label = {count['benign']}")
        self.logger.info(f"HGVSp with encoded MSA = {count['with_msa']}")
        self.logger.info(f"HGVSp failed to be featurized = {count['error']}")

        hgvsp_list = list(hgvsp2rap.keys())
            # allow hgvsp with featurized ref, alt sequences

        self.logger.info(
            (
                f"Pathogenic HGVSp = {count['pathogenic']}, "
                f"benign HGVSp = {count['benign']}"
            )
        )

        if do_balance_label:
            benign_fold = int(count['benign'] / count['pathogenic'])
            pathogenic_fold = int(count['pathogenic'] / count['benign'])

            if benign_fold >= 2:
                for hgvsp in hgvsp2rap:
                    if hgvsp2rap[hgvsp].label == 1:  # label is pathogenic
                        hgvsp_list += [hgvsp] * (benign_fold - 1)

            elif pathogenic_fold >= 2:
                for hgvsp in hgvsp2rap:
                    if hgvsp2rap[hgvsp].label == 0:  # label is benign
                        hgvsp_list += [hgvsp] * (pathogenic_fold - 1)

            ### recount pathogenic, benign HGVSp
            count['benign'], count['pathogenic'] = (0, 0)
            for hgvsp in hgvsp_list:
                label = hgvsp2rap[hgvsp].label
                if label == 0:
                    count['benign'] += 1
                elif label == 1:
                    count['pathogenic'] += 1

            self.logger.info(
                (
                    f"Pathogenic HGVSp = {count['pathogenic']}, "
                    f"benign HGVSp = {count['benign']} after balancing labels"
                )
            )

        dataset = ThreeCnetDataset(
            hgvsp_list=hgvsp_list,
            hgvsp2rap=hgvsp2rap,
            seq_collection=seq_collection,
            auto_window_size=self.win_size,
            msa_aa_size=self.msa_aa_size,
            used_featurizer=self.used_featurizer,
        )

        return dataset

    def save_dataset(self, dataset: ThreeCnetDataset, save_path: str) -> None:
        """
        save the pytorch dataset into the the path
        """
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        self.logger.info(f"...Saving dataset to {save_path}...")
        with open(save_path, 'wb') as fout:
            pk.dump(dataset, fout)

        return

"""
The module for evaluating the scores of variant
and assessing the performance if labels are given.
"""

import os
import sys
import argparse
import numpy as np
import pickle as pk

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
from logging import Logger
from typing import Tuple, Dict, List, Union
from pathlib import Path
from copy import deepcopy
from omegaconf import OmegaConf
from cccnet.utils import get_config, get_logger, save_config
from cccnet.dataset_builder import Builder
from cccnet.torch_dataset import ThreeCnetDataset
from cccnet.torch_network import SingleTask, MultiTask

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def receive_args() -> argparse.Namespace:
    """receive arguments given by a user

    Returns (argparse.Namespace): parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--epoch_num",
        type=int,
        help="the epoch nuber of the model to be used",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="",
        help="the model name to use for inference (the dir name within model_dir)",
    )
    parser.add_argument(
        "-s",
        "--save_name",
        type=str,
        default="",
        help="the name to save the test result",
    )

    return parser.parse_args()


class Evaluator:
    """Class to evaluate variants to get scores
        and performance of the assessment
    Notes:
        Some HGVSp failed to be featurized can be omitted
    Attributes:
        self.config (dict): configuration dictionary
        self.model_dir (str): the directory where models are saved
        self.eval_dir (str): the directory to save the results
        self.data_builder (Builder): builder for pytorch dataset
        self.model (torch.nn.Module): the model object. initially None.
    """

    def __init__(
        self,
        config: dict,
        model_name: str,
        epoch_num: int,
        save_name: str = "",
        logger=Logger(__name__),
    ):
        self.logger = logger
        if model_name:
            self.model_dir = os.path.join(CURRENT_DIR, config.MODEL_DIR, model_name)
        else:
            self.model_dir = os.path.join(CURRENT_DIR, config.MODEL_DIR)

        if not os.path.exists(self.model_dir):
            print(self.model_dir)
            logger.info("#Error: Model directory not found")
            sys.exit(1)

        self.eval_dir = os.path.join(self.model_dir, "EVAL")
        if save_name:
            self.eval_dir = os.path.join(self.eval_dir, save_name)
        if os.path.exists(self.eval_dir):
            logger.info("#Error: Evaluation directory alreadly exists")
            sys.exit(1)
        os.makedirs(self.eval_dir, exist_ok=True)

        save_config(config, save_dir=self.eval_dir)
        win_size = config.ARCHITECTURES.WIN_SIZE
        self.msa_aa_size = config.ARCHITECTURES.MSA_AA_SIZE
        self.data_builder = Builder(
            win_size,
            self.msa_aa_size,
            logger,
        )
        self.model_type = config.ARCHITECTURES.MODEL_TYPE
        if self.model_type == "SingleTask":
            self.model_config = config.ARCHITECTURES.SINGLE_TASK
        elif self.model_type == "MultiTask":
            self.model_config = config.ARCHITECTURES.MULTI_TASK
        else:
            logger.info("#ERROR: Unexpected model name in configuration")
            sys.exit(1)

        self.model_epoch = epoch_num
        self.test_data = config.HGVSPS.TEST
        self.test_name = config.TEST.TEST_NAME
        self.batch_size = config.TEST.BATCH_SIZE
        self.model = None

    def load_test_dataset(
        self,
        test_hgvsp_path: str = "",
    ) -> Tuple[ThreeCnetDataset, Dict]:
        """Load test dataset
        Notes:
            Some HGVSp can be omitted during the process. See log files.

        Args:
            test_hgvsp_path (str):
                the file path having HGVSp to evaluate.
                if empty string is given, try to get HGVSp from saved dataset.
        Returns:
            Tuple[ThreeCnetDataset, Dict]:
                ThreeCnetDataset: pytorch dataset to evaluate
                Dict: mapping dictionary.
                    (key) HGVSp from the file, (value) standardized HGVSp
        """
        test_dataset = None
        mapping_dict = dict()

        self.logger.info("...Building dataset from the hgvsp files...")
        test_seqs = self.data_builder.parse_HGVSPs(test_hgvsp_path, missense_only=False)

        # Make pytorch dataset
        test_dataset = self.data_builder.get_torch_dataset(
            test_seqs,
            do_balance_label=False,
        )
        mapping_dict = test_dataset.seq_collection.aliases

        if test_dataset is None:
            self.logger.info("#Error: dataset has not been created")
            return

        return test_dataset, mapping_dict

    def build_model(self) -> bool:
        """Build the model object based on the configuration.
        Notes:
            self.model is None after initialization.
            This function builds the model and annotates it to self.model.
        Returns:
            (bool): True if succeed to build the model else False
        """
        self.model = None
        if self.model_type == "SingleTask":
            self.model = SingleTask(
                num_aa=self.model_config["NUM_AA"],
                msa_aa_size=self.msa_aa_size,
                embed_dim=self.model_config["EMBED_SIZE"],
                hidden_dim=self.model_config["HIDDEN_SIZE"],
                output_dim=self.model_config["OUTPUT_SIZE"],
                fc_size=self.model_config["FC_SIZE"],
                n_classes=2,
                p_dropout=0.0,  # no dropout
            ).to(DEVICE)

        elif self.model_type == "MultiTask":
            self.model = MultiTask(
                num_aa=self.model_config["NUM_AA"],
                msa_aa_size=self.msa_aa_size,
                embed_dim=self.model_config["EMBED_SIZE"],
                hidden_dim=self.model_config["HIDDEN_SIZE"],
                output_dim=self.model_config["OUTPUT_SIZE"],
                fc_size=self.model_config["FC_SIZE"],
                n_classes=2,
                p_dropout=0.0,  # no dropout
            ).to(DEVICE)

        if self.model is None:
            self.logger.info("#ERROR: Model fails to be created.")
            return False

        return True

    def test_model(
        self,
        test_dataset: ThreeCnetDataset,
        mapping_dict: dict = dict(),
        save_dir: str = "",
    ) -> Dict[str, Tuple[float, int]]:
        """test dataset by the model and return the result
        Notes:
            If any label of the dataset has negative value (indicating no label),
            the performance metrix cannot be calculated (written as None).
            The result will be saved to a file only when save_dir is not empty.
        Args:
            test_dataset (ThreeCnetDataset): dataset to be evaluated.
            mapping_dict (dict): (key) ID, (value) standard HGVSp.
                The values are the keys of sequence objects of the dataset.
                if the value is not found, the score will be given as None.
            save_dir (str): the path to save the performance and scores.
        Returns:
            Dict[str, Tuple[float, int]]: the dictionary containing the result.
                (key) ID
                (value) a tuple of the 3Cnet score (float) and label (int)
        """
        if self.model is None:
            self.logger.info("#ERROR: Model should be created first.")
            return dict()

        key2scores = dict()
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.model.eval()
        pred_stack = []
        ans_stack = []
        for batch in test_loader:
            if self.model_type == "SingleTask":
                out_tensor = self.model(
                    batch["ref"],
                    batch["alt"],
                    batch["msa"],
                )  # (batch, n_classes) <- float(0 ~ 1)
            elif self.model_type == "MultiTask":
                out_tensor = self.model(
                    batch["ref"], batch["alt"], batch["msa"], is_clinical=True
                )  # shape = (batch, n_classes) <- float(0 ~ 1)
            out_tensor = softmax(out_tensor, dim=1)  # softmax activation
            pred = out_tensor.detach().cpu().numpy()[:, 1]
            # get pathogenic probability
            pred_stack.append(pred)
            ans = batch["label"].detach().cpu().numpy()
            # shape = (batch,), int(0, 1, -1), -1 means no label
            ans_stack.append(ans)
        pred_stack = np.concatenate(pred_stack, axis=0)
        ans_stack = np.concatenate(ans_stack, axis=0)

        # calculate metrics using pred & ans
        if np.any(ans_stack < 0):
            acc, roc_auc, pr_auc = None, None, None
        else:
            pred_bin = np.array(pred_stack > 0.5, dtype=int)
            acc = accuracy_score(ans_stack, pred_bin)
            roc_auc = roc_auc_score(ans_stack, pred_stack)
            precision, recall, _ = precision_recall_curve(
                ans_stack,
                pred_stack,
            )
            pr_auc = auc(recall, precision)

        hgvsp2scores = dict()
        for hgvsp, pred, ans in zip(
            test_dataset.hgvsp_list,
            pred_stack,
            ans_stack,
        ):
            hgvsp2scores[hgvsp] = (pred, ans)

        if not mapping_dict:
            self.logger.info(
                (
                    "# Mapping dictionary is not given. "
                    "Scoring result cannot be saved or returned."
                )
            )
            return dict()

        key2scores = dict()
        for key in mapping_dict:
            hgvsp = mapping_dict[key]
            if hgvsp is not None and hgvsp in hgvsp2scores:
                pred, ans = hgvsp2scores[hgvsp]
            else:
                pred, ans = 0.0, None
            key2scores[key] = (pred, ans)

        if save_dir:
            with open(save_dir, "w") as fout:
                fout.write(f"##ACC={acc},ROCAUC={roc_auc},PRAUC={pr_auc}\n")
                fout.write("#ID\tHGVSp\tPRED\tANS\n")
                for key, (pred, ans) in key2scores.items():
                    hgvsp = mapping_dict[key]
                    fout.write(f"{key}\t{hgvsp}\t{pred}\t{ans}\n")

        return key2scores

    def evaluate(self, do_save: bool) -> dict:
        """Dataset & model loading and testing
        Args:
            do_save (bool): if True, save the result to a file.
        Returns:
            Dict[str, Tuple[float, int]]: the dictionary containing the result
                (key) ID
                (value) a tuple of the 3Cnet score (float) and label (int)
        """
        # model loading (if not defined yet)
        if self.model is None:
            # Build network
            self.logger.info("...Building model network...")
            if not self.build_model():
                return dict()  # failed to build model

            # Load model parameter
            self.logger.info("...Loading model...")
            self.model.load_state_dict(
                torch.load(os.path.join(self.model_dir, f"{self.model_epoch}.pt"))
            )

        # Get test dataset and ID mapping dictionary
        if self.test_name not in self.test_data:
            self.logger.info(
                f"#ERROR: no test dataset named {self.test_name} (see config)"
            )
            return dict()

        test_hgvsp_path = self.test_data[self.test_name]
        test_dataset, mapping_dict = self.load_test_dataset(
            test_hgvsp_path=test_hgvsp_path,
        )
        if test_dataset is None:
            self.logger.info("#ERROR: test dataset failed to load")
            return dict()

        # get the test result
        self.logger.info("...Testing model...")
        save_dir = ""
        if do_save:
            save_dir = os.path.join(self.eval_dir, "pred.tsv")

        key2scores = self.test_model(
            test_dataset,
            mapping_dict=mapping_dict,
            save_dir=save_dir,
        )

        return key2scores


if __name__ == "__main__":
    ARGS = receive_args()
    CONFIG = get_config()
    LOGGER = get_logger(
        module_name=Path(__file__).stem,
        data_dir=CONFIG.MODEL_DIR,
        file_name=ARGS.save_name,
    )
    LOGGER.info(ARGS)

    model_evaluator = Evaluator(
        CONFIG, ARGS.model_name, ARGS.epoch_num, ARGS.save_name, LOGGER
    )
    LOGGER.info("...Evaluation start...")

    if model_evaluator.evaluate(do_save=True):
        LOGGER.info("### Correctly finished")
    else:
        LOGGER.info("### Unexpected exit")

    LOGGER.info("...Evaluation end...")

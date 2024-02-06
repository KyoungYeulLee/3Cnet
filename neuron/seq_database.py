from collections import OrderedDict
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    MutableSet,
    Optional,
    Union,
)

from neuron.constants import HGVSp, seq_id
from neuron.errors import *
from neuron.seq_collection import SeqCollection
from neuron.utils import get_config


class SeqDatabase:
    # keys for dictionary monitoring steps where sequence processing fail
    outcome_criteria = [
        "seq_id_not_found",
        "mutation_code_err",
        "residue_err",
        "other_err",
        "selection_pass",
    ]

    outcome_category_to_order = {
        category: step_num + 1 for step_num, category in enumerate(outcome_criteria)
    }

    order_to_outcome_category = {val: key for key, val in outcome_category_to_order.items()}

    def __init__(self) -> None:
        self._clear_and_init_memory()
        self.config = get_config()
        self.data_root = Path(__file__).parent.parent.resolve()
        self.sequences = self.load_sequences(
            seq_path=self.data_root / self.config.SEQUENCES
        )

    def load_sequences(self, seq_path: Path, delim: str = "\t") -> Dict[str, str]:
        """
        Parse sequence identifiers and their amino acid sequences from file.

        Args:
            seq_path (Path): Filepath containing sequence id to AA mappings.
            delim (str): Delimiter separating sequence ids from AA sequences. Defaults to "\t".

        Note:
            Sequence identifiers must be formatted as they will be used.

            (Identifier in file)     (Identifiers/HGVSps in select())
            NP_0000412.1          -> NP_0000412.1                  (OK)
            NP_0000412.1          -> NP_0000412.1:p.A54Q           (OK)
            NP_0000412            -> NP_0000412                    (OK)
            NP_0000412            -> NP_0000412:p.A54Q             (OK)
            NP_0000412.1          -> NP_0000412                    (ERROR)
            NP_0000412.1          -> NP_0000412:p.A54Q             (ERROR)
            NP_0000412            -> NP_0000412.1:p.A54Q           (ERROR)
        """

        seq_dict = dict()

        with seq_path.open("r") as fh:
            for row in fh:
                if not row:
                    continue

                sequence_id, sequence_aa = row.strip().split(delim)

                seq_dict[sequence_id] = sequence_aa

        return seq_dict

    def _clear_and_init_memory(self) -> None:
        """Init the memory object to its initial state.

        Note:
            Memory is cleared before each new `select()` run.
        """

        if not hasattr(self, "memory"):
            self.memory: OrderedDict[str, MutableSet[str]] = OrderedDict()
        else:
            self.memory.clear()

        for memory_category in SeqDatabase.outcome_criteria:
            self.memory[memory_category] = set()

    def select(self, seq_identifiers: Iterable[Union[HGVSp, seq_id]]) -> SeqCollection:
        """
        Retrieve a SeqCollection from DB

        Args:
            seq_identifiers (Iterable[Union[HGVSp, seq_id]]):
                Reference IDs with or without HGVS-compliant mutation terms.

        Returns:
            SeqCollection

        Examples:
            # Sequences
            >>> collection = self.select(
                    seq_identifiers = ['NR_024540'],
                )
            >>> collection.seqs
            {'NP_009231.2': ProteinSeq(std_id=NP_009231.2, seq_len=1884, ...)}
        """
        self._clear_and_init_memory()

        hgvsp_to_reference_seqs = dict()

        iterator = seq_identifiers if seq_identifiers else self.sequences

        for hgvs in iterator:
            if ":" in hgvs:
                seq_id, mut_code = hgvs.split(":")
            else:
                seq_id, mut_code = hgvs, None

            if seq_id not in self.sequences:
                self.memory["seq_id_not_found"].add(hgvs)
                continue

            seq_aa = self.sequences[seq_id]

            hgvsp_to_reference_seqs[hgvs] = seq_aa

        return SeqCollection(
            hgvs_to_reference_seq_mapping=hgvsp_to_reference_seqs,
            unusable_hgvses=self.memory,
            sequence_type="protein",
            check_seq_integrity=False,
        )

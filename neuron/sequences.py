from abc import ABC, abstractmethod, abstractclassmethod
from pathlib import Path
import numpy
import numpy.typing as npt
import re
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

from omegaconf import OmegaConf

from Bio.Data.CodonTable import IUPACData

from neuron.constants import HGVSp, ProteinMutationTypes as ProtMut, seq_aa
from neuron.errors import (
    MSAArrayFileNotFound,
    UnexpectedMutationError,
    UnexpectedResidueError,
)
from neuron.featurizers import Featurizer
from neuron.utils import get_config


class SequenceObj(ABC):
    allow_mid_seq_gaps_ref: bool = False
    allow_mid_seq_gaps_mut: bool = True
    max_stoploss_extension_len: int = 100
    msa_arrays = dict()

    @property
    @abstractclassmethod
    def dtype(self) -> str:
        raise NotImplementedError

    @property
    @abstractclassmethod
    def gap_tok(self) -> str:
        raise NotImplementedError

    @property
    @abstractclassmethod
    def unknown_tok(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def featurize(self):
        raise NotImplementedError

    @classmethod
    def free_msa_cache(
        cls,
        sequence_ids=None,
    ) -> None:
        """
        Class method for manually clearing loaded MSA arrays.

        Note:
            Providing no arguments clears *ALL* MSA arrays.
            Multiple SequenceObj instances may share the same cached MSA array.

        Args:
            sequence_ids (
                Optional[Iterable[Union[ProteinID, TranscriptID]]], optional
            ):
                If specified, clears only that specific MSA array.
                If specified sequence ID does not exist in cache, do nothing.
        """
        if not sequence_ids:
            cls.msa_arrays.clear()
            return

        for seq_id in sequence_ids:
            if seq_id in cls.msa_arrays:
                del cls.msa_arrays[seq_id]


class ProteinSeqObj(SequenceObj):
    gap_tok = "-"  # token representing zero padding and gaps created by indels
    unknown_tok = "X"  # token representing AAs that exist but are unknown
    __operator_tokens = ["ins", "del", "fs", "ext", "dup"]
    dtype = "sequence_protein"

    config = get_config()

    msa_root_dir = Path(config.MSA_DIR)

    def __init__(
        self,
        reference_seq: seq_aa,
        hgvs: Optional[HGVSp] = None,
        featurizer: Optional[Featurizer] = None,
        load_msa_at_init: bool = False,
        require_msa: bool = False,
        check_seq_integrity: bool = False,
    ):
        """
        General-purpose class for protein sequences and their variants.
        This class does not verify whether the sequence actually belongs to the protein id provided.

        ARGS
        `aa_seqs` {List[str]}: Amino acid sequences. 1 char per residue.
        `mut_codes` {List[str]}: The part after the colon in HGVSp notations.
            Used to generate variants of `aa_seqs`.
        `prot_ids` {List[str]}: Protein ids.
        `load_msa_at_init` {bool}:
            Whether to attempt loading of MSA array at init time.
            If False, defer operation until needed at featurization.
        `require_msa` {bool}:
            Is this instance required to have an MSA array?
            (if True, raises error if MSA array .npy is missing)
        `check_seq_integrity` {bool}:
            Verify whether sequence contains only valid AA characters.
            Set to True if unexpected (non-AA) characters may be found in the sequence.
            WARNING: This is a heavy operation and will reduce processing throughput by >= 50%


        USEFUL ATTRIBUTES / METHODS
        ids: protein ids
        mut_codes_parsed: mutation codes after tokenization
        mut_seqs: sequences after changes specified by mut_codes is applied to
        orig_seqs: reference sequences

        [USAGE EXAMPLE]
        import prot_mut_utils as utils

        # fake sequences and hgvsps in example
        aa_seqs = ['QQQQQQQQQQQQQQQQQQ', 'RRRRRRRRRRRR']
        hgvsps = ['NP_00001.1:p.Gly6Glu', 'NP_00002.4:p.Glu10Gly']

        colon_idxes = [hgvsp.index(':') for hgvsp in hgvsps]
        prot_ids = [
            hgvsps[hgvsp_idx][:colon_idx]
            for hgvsp_idx, colon_idx in enumerate(colon_idxes)
        ]
        mut_codes = [
            hgvsps[hgvsp_idx][colon_idx+1:]
            for hgvsp_idx, colon_idx in enumerate(colon_idxes)
        ]
        proteins = utils.ProteinSeqs(
            aa_seqs=aa_seqs,
            prot_ids=prot_ids,
            mut_codes=mut_codes
        )
        """

        self.featurizer: Optional[Featurizer] = featurizer
        self.orig_seq: seq_aa = reference_seq.upper()
        self.mut_seq: Optional[seq_aa]
        self.raw_hgvs: HGVSp = hgvs
        self.std_hgvs: HGVSp
        self.prot_id: HGVSp
        self._mut_code: Optional[List[List[str]]]
        self.mutation_types: Set[ProtMut] = set()
        self.metadata: Dict[str, Any] = dict()

        # MSA에 어디에 얼만큼 padding이 필요한지 기록하는 변수들
        self.reference_offsets: Dict[int, int] = dict()
        self._current_offset_start_idx: int = 0
        self._current_offset_length: int = 0

        if hgvs and ":" in hgvs:
            _prot_id, _raw_mut_code = hgvs.split(":")
            self.prot_id = HGVSp(_prot_id)
        else:
            self.prot_id = hgvs
            _raw_mut_code = None
            self.mutation_types.add(ProtMut.NONE)

        if check_seq_integrity:
            for residue in self.orig_seq:
                if residue not in IUPACData.extended_protein_letters:
                    raise UnexpectedResidueError(
                        f"Unexpected residue in aa_seq argument: {self.orig_seq}"
                    )

        if _raw_mut_code:
            self._mut_code = self.parse_mut_code(_raw_mut_code)
            self.orig_seq, self.mut_seq = self.mutate()
            hgvs_suffix = ";".join(
                ["".join([str(tok) for tok in mut_code]) for mut_code in self._mut_code]
            )
            self.std_hgvs = HGVSp(f"{self.prot_id}:p.{hgvs_suffix}")
        else:
            self._mut_code = None
            self.mut_seq = None
            self.std_hgvs = self.prot_id

        if load_msa_at_init:
            self.load_msa_arr(protein_id=self.prot_id, missing_is_error=require_msa)

    def __eq__(self, __o: Any) -> bool:
        if not isinstance(__o, ProteinSeqObj):
            raise TypeError(
                f"object being compared should be an ProteinSeqObj instance, not {type(__o)}"
            )

        if (
            self.raw_hgvs == __o.raw_hgvs
            and self.orig_seq == __o.orig_seq
            and self.mut_seq == __o.mut_seq
            and self.mutation_types == __o.mutation_types
            and self.metadata == __o.metadata
        ):
            return True

        return False

    def __len__(self) -> int:
        return len(self.orig_seq)

    def __repr__(self) -> str:
        seq_len = self.__len__()
        return f"ProteinSeq(std_id={self.std_hgvs}, seq_len={seq_len}, ...)"


    @classmethod
    def load_msa_arr(cls, protein_id: str, missing_is_error: bool) -> None:
        """
        Load MSA .npy array, if available.
        If it is already loaded, do nothing.
        This was converted from a private instance method to a class method as part of #102.

        Note:
            This method does not check the integrity of the numpy array itself.

        Args:
            protein_id (str): Name of protein whose MSA to load (including version)
            missing_is_error (bool): Forces an error if MSA .npy is not found

        Raises:
            MSAArrayFileNotFound: When MSA .npy file is not found
        """
        if protein_id in ProteinSeqObj.msa_arrays:
            return

        expected_msa_path = cls.msa_root_dir / f"{protein_id}.npy"

        if expected_msa_path.exists():
            ProteinSeqObj.msa_arrays[protein_id] = numpy.load(expected_msa_path)
            return

        if missing_is_error:
            raise MSAArrayFileNotFound(
                f"MSA .npy file for {protein_id} required at {expected_msa_path}, but not found"
            )

    @classmethod
    def raw_to_std_hgvs(cls, hgvs_raw: HGVSp) -> HGVSp:
        """
        Standardize an HGVS mutation without creating a ProteinSeqObj instance.

        Note:
            Amino Acids are converted to 1-char uppercase.

        Args:
            hgvs_raw (HGVSp): HGVSp term that needs standardizing

        Returns:
            HGVSp: standardized HGVSp

        Examples:
            >>> hgvs_std = ProteinSeqObj.raw_to_std_hgvs(
                    hgvs_raw='NP_10001.1:p.Ser34Thr'
                )
            >>> hgvs_std
            'NP_10001.1:p.S34T'
        """
        seq_id, raw_mut_code = hgvs_raw.split(":")

        tokenized_mut_codes = cls.parse_mut_code(mut_codes=raw_mut_code)
        str_casted_mut_code = [[str(tok) for tok in mut_code] for mut_code in tokenized_mut_codes]

        hgvs_std = ";".join(["".join(mut_code) for mut_code in str_casted_mut_code])
        return f"{seq_id}:p.{hgvs_std}"

    def parse_mut_code(self, mut_codes: str) -> List[List[str]]:
        """
        Process and standardize HGVSp mutation codes.

        Note:
            ";" is the seperator for multiple mutation codes.
            Each mutation code is stardardized in formatting and separated into tokens by `_standardize_mut_code()`.

        Args:
            mut_codes (str): HGVSp style mutation code.

        Returns:
            List[List[str]]: Each sublist is one mutation code. The sublists contain tokens of the mutation code's amino acids(in 1-char form), position indices(1-based), and mutation type descriptors(del, ins, ?, fs, etc.).

        Examples:
            >>> self.parse_mut_code(mut_codes="p.Arg52_Gly54delinsGluGluTer")
            [["R", 52, "_", "G", 54, "del", "ins", "E", "E", "*"]]
        """
        if mut_codes.startswith("p."):
            mut_codes = mut_codes[2:]
        elif not mut_codes:
            mut_codes = ""

        parsed = [
            self._standardize_mut_code(mut_code=code, as_tokens=True)
            for code in mut_codes.split(";")
        ]

        return parsed

    def mutate(self) -> Tuple[seq_aa, seq_aa]:
        """
        Apply mutations using tokenized mutation codes.

        Note:
            It is possible to specify multiple mutations in one HGVSp.
            If the HGVSp has multiple mutations, make sure they are separated using semicolons(;).
            The mutations are applied cumulatively in a left -> right order.
            Beware, the consistency of the sequence is not examined between the application of
            one mutation and the next.
            (Be especially careful with length or position-altering mutations)

            The original sequence may also be modified(padded) for net-insertion mutations.
            (depends on the value of `self.allow_mid_seq_gaps_ref`)


        Returns:
            Tuple[List[str], List[str]]: (
                List[possibly modified original amino acid sequence],
                List[mutated amino acid sequence]
            )

        Examples:
            >>>
            >>> self.mutate( # p.C3_E5delinsX*
                aa_seq="ABCDEFGHI",
                mut_code=['C', 3, '_', 'E', 5, 'del', 'ins', 'X', '*']
            )
            ['ABCDEFGHI', 'ABX------']
        """

        reference_seq = self.orig_seq

        mut_code = self._mut_code

        seq_to_mutate = reference_seq

        for code in mut_code:
            self._add_mutation_tags(mut_code=code)
            reference_seq, seq_to_mutate = self._apply_prot_mut(reference_seq, seq_to_mutate, code)
            if self._current_offset_length > 0:
                self.reference_offsets[self._current_offset_start_idx] = self._current_offset_length

            # 값이 negative인 경우도 reset해 주기 위해 if 문 밖에 있음.
            self._current_offset_start_idx = 0
            self._current_offset_length = 0

        # Not clearing the offset dictionary when gaps are not allowed in the reference sequence
        # causes index errors when featurizing MSA arrays in featurizers.py::featurize_msa.
        # Related issue: https://github.com/3billion/neuron/issues/90
        if not self.allow_mid_seq_gaps_ref:
            self.reference_offsets.clear()

        modified_reference = (
            reference_seq.replace(ProteinSeqObj.gap_tok, "")
            if not self.allow_mid_seq_gaps_ref
            else reference_seq
        )

        mutated_seq = (
            seq_to_mutate.replace(ProteinSeqObj.gap_tok, "")
            if not self.allow_mid_seq_gaps_mut
            else seq_to_mutate
        )

        return modified_reference, mutated_seq

    def _add_mutation_tags(self, mut_code: List[Union[str, int]]) -> None:
        """
        Annotate mutation types according to the HGVSp's mutation code.

        Note:
            This method assumes the existence of a mutation code.
            (don't call this in __init__ if it's a no-mutation situation)

        Args:
            mut_code (List[Union[str, int]]):
                Post-regularization and post-tokenization mutation code.
                One change should be handled at one time for cases with multiple mutations
                (ex: p.R29M;Q43*).
        """

        if not isinstance(mut_code, list):
            raise ValueError(
                "Only regularized and tokenized mutation codes are accepted. (use `self._standardize_mut_code())"
            )

        if "=" in mut_code:  # p.*8=
            self.mutation_types.add(ProtMut.SYNONYMOUS)
            return

        if len(mut_code) == 3 and mut_code[-1] in IUPACData.extended_protein_letters:
            self.mutation_types.add(ProtMut.MISSENSE)
            return

        if "fs" in mut_code:
            self.mutation_types.add(ProtMut.FRAMESHIFT)

        if "dup" in mut_code:
            self.mutation_types.add(ProtMut.DUPLICATION)

        is_ext = "ext" in mut_code
        if is_ext:
            if isinstance(mut_code[3], int) and mut_code[3] < 0:
                self.mutation_types.add(ProtMut.EXTENSION_5PRIME)
            else:
                self.mutation_types.add(ProtMut.EXTENSION_3PRIME)

        if mut_code[0] == "*" or mut_code[-1] == "?":
            self.mutation_types.add(ProtMut.EXTENSION_3PRIME)

        if mut_code[-1] == "*":
            self.mutation_types.add(ProtMut.STOP_GAIN)
        elif mut_code[-2] == "*":
            if mut_code[-1] == "?" or mut_code[0] == "*":
                self.mutation_types.add(ProtMut.EXTENSION_3PRIME)
            else:
                self.mutation_types.add(ProtMut.STOP_AMBIGUOUS)

        is_del = "del" in mut_code
        is_ins = "ins" in mut_code
        if is_del and is_ins:
            self.mutation_types.add(ProtMut.DELETION_INSERTION)
        elif is_del:
            self.mutation_types.add(ProtMut.DELETION)
        elif is_ins:
            self.mutation_types.add(ProtMut.INSERTION)

        is_start_loss = (
            mut_code[0] == "M" and mut_code[1] == 1 and not (is_ins or is_ext)
        ) or mut_code[0] == 0

        if is_start_loss:
            self.mutation_types.add(ProtMut.START_LOSS)

    def _apply_prot_mut(
        self,
        reference_seq: str,
        seq_to_mutate: str,
        mut_code: List[Union[str, int]],
    ) -> Tuple[str, str]:
        """
        Inner helper method for `mutate()`.
        Applies exactly one mutation code to the amino acid sequence.

        Note:
            If the mutation is a net-insertion, gap tokens may be introduceed to the original
            sequence to maintain equal length (depending on the value of `allow_mid_seq_gaps_ref`).
            For cases where stop-loss recovery or the length of extension is unknown,
            `self.stoploss_extension_len` number of unknown tokens are appended to the sequence.

        Args:
            reference_seq (str):
                Reference sequence of the protein. This sequence may also be modified depending of the mutation type (i.e. adding gaps when mutations are insertions)
            seq_to_mutate (str): Sequence to apply the mutation code to.
            mut_code (List[Union[str, int]]):
                Exactly one HGVS-compatible variant term describing how to change the protein sequence `aa_seq`. Residues are expected to be in all uppercase while variant descriptors (ie. fs, ter, del, etc.) are expected to be in all lowercase.
        Raises:
            UnexpectedMutationError:
                - When any part of mutation code formatting is incomplete or invalid
                - When mutation code does not match the sequence
        Returns:
            Tuple[str, str]:
                (
                    updated_reference:
                        reference sequence updated to match length of modified sequence,
                    mutated_sequence:
                        sequence after `mut_code` has been applied
                )
        Examples:
            >>> self._apply_prot_mut(
                reference_seq = "ABCDEFGHIJ",
                seq_to_mutate = "ABCDEFGHIJ",
                mut_code = "C3_D4insXX"
            )
            ("ABC--DEFGHIJ", "ABCXXDEFGHIJ")
        """

        mut_code_len = len(mut_code)

        if not mut_code or "=" in mut_code:  # p.*8=
            return reference_seq, reference_seq

        if "?" == mut_code[0]:  # p.?
            raise UnexpectedMutationError(
                f"Splicing variants are not supported because their amino acid sequences are uncertain and their HGVSps are not actionable: {mut_code}"
            )

        is_frameshift = "fs" in mut_code
        is_del = "del" in mut_code
        is_ins = "ins" in mut_code
        is_dup = "dup" in mut_code
        is_ext = "ext" in mut_code
        is_start_loss = (
            mut_code[0] == "M" and mut_code[1] == 1 and not (is_ins or is_ext)
        ) or mut_code[0] == 0

        is_stop_gain = mut_code[-1] == "*"

        if is_start_loss:
            return reference_seq, self.gap_tok * len(seq_to_mutate)

        # Check if no operator or "_" within first 3 tokens
        op_idx = None
        for idx, tok in enumerate(mut_code[:3]):
            if tok == "_" or tok in self.__operator_tokens:
                op_idx = idx
                break

        if not op_idx:  # p.D3X, p.*46Lfs*5, P5Tfs*?, p.Asp3Ter
            if mut_code[0] == "*":  # covers fs-ter and ext-ter
                if is_ext:
                    ter_token_idx = mut_code[1:].index("*") + 1
                    ext_len_token = ter_token_idx + 1
                    total_ext_len = (
                        self.max_stoploss_extension_len
                        if mut_code[ext_len_token] == "?"
                        else mut_code[ext_len_token]
                    )
                    reference_seq, mutated = self.__extend_3prime(
                        orig_seq=reference_seq,
                        mut_target=reference_seq,
                        known_aas=mut_code[2],
                        total_extension_length=total_ext_len,
                    )
                if is_frameshift:
                    reference_seq, mutated = self.__frameshift(
                        orig_seq=reference_seq,
                        mut_target=reference_seq,
                        mut_code=mut_code,
                    )

                return reference_seq, mutated

            elif mut_code[2] == "*":
                # implement stop-gain by deleting everything after given position
                mutated = self.__delete(
                    orig_seq=reference_seq,
                    mut_target=reference_seq,
                    mut_code=mut_code,
                )
            else:  # simple missense and 3' extensions
                mutated = self.__substitute(
                    mut_target=reference_seq,
                    pos=mut_code[1],
                    alt=mut_code[2],
                    ref=mut_code[0],
                )
                if "ext" in mut_code:
                    ter_token_idx = mut_code.index("*")
                    ext_len_token = ter_token_idx + 1
                    total_ext_len = (
                        self.max_stoploss_extension_len
                        if mut_code[ext_len_token] == "?"
                        else mut_code[ext_len_token]
                    )
                    reference_seq, mutated = self.__extend_3prime(
                        orig_seq=reference_seq,
                        mut_target=mutated,
                        known_aas="",
                        total_extension_length=total_ext_len,
                    )
                elif is_frameshift:
                    reference_seq, mutated = self.__frameshift(
                        orig_seq=reference_seq,
                        mut_target=mutated,
                        mut_code=mut_code,
                    )

                return reference_seq, mutated

        elif is_ext and not is_del:
            if isinstance(mut_code[3], int) and mut_code[3] < 0:
                reference_seq, mutated = self.__extend_5prime(
                    orig_seq=reference_seq,
                    mut_target=reference_seq,
                    total_extension_len=abs(mut_code[3]),
                )
            else:
                pass

        elif is_dup:  # p.S2_K4dup
            return self.__duplicate(
                orig_seq=reference_seq,
                mut_target=seq_to_mutate,
                mut_code=mut_code,
            )

        elif is_del:  # p.Y530del, p.Y530_P534del
            mutated = self.__delete(
                orig_seq=reference_seq,
                mut_target=reference_seq,
                mut_code=mut_code,
            )
            if is_ins:  # p.W1461_Y1462delinsLPI
                ins_token_idx = mut_code.index("ins")
                fs_token_idx = mut_code.index("fs") if is_frameshift else None

                # p.W1461_Y1462delinsLPIIR* or p.W1461_Y1462delinsLPI
                to_insert = (
                    mut_code[ins_token_idx + 1 :]
                    if not fs_token_idx
                    else mut_code[ins_token_idx + 1 : fs_token_idx]
                )

                reference_seq, mutated = self.__insert(
                    orig_seq=reference_seq,
                    mut_target=mutated,
                    pos=mut_code[1] - 1,
                    to_insert=to_insert,
                )

                if is_frameshift:  # p.K654delinsKVfsTer
                    reference_seq, mutated = self.__frameshift(
                        orig_seq=reference_seq,
                        mut_target=mutated,
                        mut_code=mut_code,
                    )

            elif is_ext:  # p.G208_W383delext*?
                ext_token_idx = mut_code.index("ext")
                if "*" in mut_code[ext_token_idx:]:
                    ter_token_idx = mut_code[ext_token_idx:].index("*") + ext_token_idx
                    ext_len_token = ter_token_idx + 1
                    known_aas = mut_code[ext_token_idx + 1 : ter_token_idx]
                    total_ext_len = (
                        self.max_stoploss_extension_len
                        if mut_code[ext_len_token] == "?"
                        else mut_code[ext_len_token] + 1
                    )

                else:
                    known_aas = mut_code[ext_token_idx + 1 :]
                    total_ext_len = len(known_aas)

                reference_seq, mutated = self.__extend_3prime(
                    orig_seq=reference_seq,
                    mut_target=mutated,
                    known_aas=known_aas,
                    total_extension_length=total_ext_len,
                )
            return reference_seq, mutated

        elif is_ins and not is_del:  # p.V721_C722insAH or p.V741_Y742ins*
            pos = mut_code[1]

            if not isinstance(pos, int):
                raise UnexpectedMutationError(f"pos expected to be int, but was {type(pos)}")

            to_insert = mut_code[mut_code.index("ins") + 1 :]

            if "*" not in mut_code or mut_code.index("*") == mut_code_len - 1:
                reference_seq, mutated = self.__insert(
                    orig_seq=reference_seq,
                    mut_target=reference_seq,
                    pos=pos,
                    to_insert=to_insert,
                )
            else:
                raise UnexpectedMutationError("Termination token found at unexpected index")
        elif is_frameshift:
            reference_seq, mutated = self.__frameshift(
                orig_seq=reference_seq,
                mut_target=reference_seq,
                mut_code=mut_code,
            )

        return reference_seq, mutated

    def __substitute(self, mut_target: str, pos: int, alt: str, ref: Optional[str] = None) -> str:
        """Simple single amino acid substitution.
        Note:
            Does not include stop-loss, stop-gain, or start-loss types.
        Args:
            mut_target (str): sequence to mutate
            pos (int): index to modify
            alt (str): new amino acid
            ref (str): original amino acid at index pos
        Raises:
            UnexpectedResidueError: raised when sequence does not match mut_code
            UnexpectedMutationError: when mutated pos >= seq length
        Returns:
            str: sequence after mutation is applied
        Examples:
            >>> self.__substitute(mut_target="ABCDE", mut_code=["B",2,"X"])
            "AXCDE"
        """
        pos -= 1
        if pos > len(mut_target):
            raise UnexpectedMutationError(
                f"missense index ({pos}) > sequence length ({len(mut_target)})"
            )
        if ref and mut_target[pos] != ref:
            raise UnexpectedResidueError(
                f"Reference residue in mutation code ({ref}) does not match residue in sequence ({mut_target[pos]}) at index {pos}"
            )
        return mut_target[:pos] + alt + mut_target[pos + 1 :]

    def __delete(
        self,
        orig_seq: str,
        mut_target: str,
        mut_code: List[Union[str, int]],
        refs: Optional[Tuple[str, str]] = None,
    ) -> str:
        """Apply a sequence deletion operation.
        Note:
            Assumes gaps don't already exist in mut_target
            Indices in pos are INCLUSIVE
        Args:
            mut_target (str): sequence to mutate
            pos (List[int, int]): indicies to modify
            refs (Optional[List[str, str]], optional): original amino acids at index pos. Defaults to None.
        Raises:
            UnexpectedMutationError: Invalid deletion index range
            UnexpectedResidueError: If provided `refs` do not match with sequence
        Returns:
            str: Sequence after deletion
        Examples:
            >>> self.__delete(
                    mut_target="ABCDE",
                    pos=[2, 4],
                    refs=["B", "D"]
                )
            "A---E"
        """

        # In case we get something like p.*578delext*?
        # Don't delete anything, return as-is and
        # let __extend_3prime handle it downstream
        if mut_code[0] == "*":
            return orig_seq, mut_code

        if mut_code[2] == "*":
            pos = [mut_code[1] - 1, len(orig_seq) - 1]
        else:
            pos = (
                [mut_code[1] - 1, mut_code[4] - 1]
                if mut_code[2] == "_"
                else [mut_code[1] - 1, mut_code[1] - 1]
            )

        if pos[0] < 1 or pos[1] > len(mut_target):
            raise UnexpectedMutationError(
                f"deletion indices [{pos}] is outside of sequence range [1, {len(mut_target)}]"
            )
        if refs and (mut_target[pos[0]] != refs[0] or mut_target[pos[1]] != refs[1]):
            raise UnexpectedResidueError(
                f"One or more reference residue in mutation code ({refs}) does not match residues in sequence ({mut_target[pos[0]]}, {mut_target[pos[1]]}) at indices {pos[0]} and {pos[1]}"
            )

        len_deletion = pos[1] - pos[0] + 1
        self._current_offset_start_idx = pos[0]
        self._current_offset_length -= len_deletion

        return mut_target[: pos[0]] + self.gap_tok * len_deletion + mut_target[pos[1] + 1 :]

    def __insert(self, orig_seq: str, mut_target: str, pos: int, to_insert: str) -> Tuple[str, str]:
        """Apply a sequence insertion operation
        Note:
            Items in `to_insert` are INSERTED BEHIND the `pos` index.
            `to_insert` argument may contain a termination token. (AAs only,AAs + "*", "*" only combinations are all possible.) If termination token is found in mut_code, treat as a stop-gain after the insertion of AAs, if any.
        Args:
            orig_seq (str): original sequence
            mut_target (str): sequence to mutate
            pos (int): position of insertion
            to_insert (str): what to insert
        Raises:
            UnexpectedMutationError: if insertion idx is out of bounds
        Returns:
            Tuple[str, str]:
                [
                    updated original sequence,
                    sequence after insertion
                ]
        Examples:
            >>> self.__insert(
                orig_seq="ABCDE",
                mut_target="ABCDE",
                pos=2,
                to_insert="XXXX"
            )
            ["AB----CDE", "ABXXXXCDE"]
            >>> self.__insert(
                orig_seq="ABCDE",
                mut_target="AB--E",
                pos=2,
                to_insert="ZZ*"
            )
            ["ABCDE", "ABZZ-"]
        """
        if pos > len(mut_target):
            raise UnexpectedMutationError(
                f"missense index ({pos}) >= sequence length ({len(mut_target)})"
            )

        has_ter_token = False
        len_insertion = len(to_insert)
        if to_insert != ["*"]:
            self._current_offset_length += len_insertion
            if not self._current_offset_start_idx:
                self._current_offset_start_idx = pos + 1
            else:
                self._current_offset_start_idx += self._current_offset_length + 1

        if "*" in to_insert:
            has_ter_token = True
            if to_insert.index("*") != len_insertion - 1:
                raise UnexpectedMutationError(
                    "Termination token present, but not final token when there should be nothing after it"
                )

            to_insert = to_insert[:-1]
            len_insertion -= 1

        to_insert = "".join(to_insert)

        existing = mut_target[pos : pos + len_insertion]
        num_mut_target_fillable_gaps = sum([1 for residue in existing if residue == self.gap_tok])
        # num_orig_seq_gaps = len_insertion - num_mut_target_fillable_gaps

        replacement_range = [pos, pos + num_mut_target_fillable_gaps]

        mut_target = (
            mut_target[: replacement_range[0]] + to_insert + mut_target[replacement_range[1] :]
        )

        if num_mut_target_fillable_gaps < len_insertion:
            num_new_gaps_in_orig = len_insertion - num_mut_target_fillable_gaps
            gap_insert_pos = pos + num_mut_target_fillable_gaps
            orig_seq = (
                orig_seq[:gap_insert_pos]
                + (self.gap_tok * num_new_gaps_in_orig)
                + orig_seq[gap_insert_pos:]
            )

        if has_ter_token:
            # terminate protein after inserting specified aas, if any
            ter_token_insert_pos = replacement_range[1]
            num_new_gaps_in_mut = len(mut_target) - ter_token_insert_pos
            mut_target = mut_target[:ter_token_insert_pos] + self.gap_tok * num_new_gaps_in_mut

        return (orig_seq, mut_target)

    def __duplicate(self, orig_seq: str, mut_target: str, mut_code: List[str]) -> Tuple[str, str]:
        to_duplicate = (
            orig_seq[mut_code[1] - 1 : mut_code[4]] if mut_code[2] == "_" else mut_code[0]
        )

        pos = mut_code[4] if mut_code[2] == "_" else mut_code[1]

        orig_seq, mutated = self.__insert(
            orig_seq=orig_seq,
            mut_target=mut_target,
            pos=pos,
            to_insert=to_duplicate,
        )

        return orig_seq, mutated

    def __frameshift(
        self, orig_seq: str, mut_target: str, mut_code: List[Union[str, int]]
    ) -> Tuple[str, str]:
        """Apply a sequence frameshift operation.
        Note:
            Uses the latest position token when applying the frameshift.
            The frameshift can be a combination of known and unknown AAs.
        Args:
            orig_seq (str): original sequence
            mut_target (str): sequence to mutate
            mut_code (List[Union[str, int]]): tokenized list containing mutation code
        Raises:
            UnexpectedMutationError: mut_code does not specify where to begin applying the frameshift
        Returns:
            Tuple[str, str]: [
                original sequence with gaps added, if needed.,
                sequence after frameshift
            ]
        Examples:
            >>> self.__frameshift(
                orig_seq='ABCDE',
                mut_target='ABCDE',
                mut_code=['D', 4, 'Z', 'fs', '*', '4']
            )
            ['ABCDE--','ABCZXXX']
        """

        fs_token_idx = mut_code.index("fs")

        if str(mut_code[fs_token_idx - 1]) not in IUPACData.extended_protein_letters:
            mut_code.insert(fs_token_idx, self.unknown_tok)
            fs_token_idx += 1

        fs_continuation_len = (
            self.max_stoploss_extension_len if mut_code[-1] == "?" else mut_code[-1]
        )

        # a list of one or two ints that should be in ascending order
        pos_digits = [tok for tok in mut_code[:fs_token_idx] if isinstance(tok, int)]

        # fs_start_pos is used to determine how much of the original sequence to preserve when applying the frameshift
        fs_start_pos = len(orig_seq)
        toks_added = []
        is_del = "del" in mut_code
        is_ins = "ins" in mut_code
        # extra padding for missense -> frameshift case.
        # stop-loss frameshift cases add a new residue after termination, but mid-sequence(missense) frameshifts replace an existing residue, so a 1-position padding is needed
        additional_padding = 0

        if not is_del and not is_ins:
            fs_start_pos = pos_digits[0]
            token_before_fs = mut_code[fs_token_idx - 1]
            toks_added = [
                token_before_fs
                if token_before_fs in IUPACData.extended_protein_letters
                else self.unknown_tok
            ]

            if mut_code[0] != "*":
                fs_start_pos -= 1
                additional_padding = (
                    1  # special 1-position gap padding for missense -> frameshift case
                )

        if is_del:
            fs_start_pos = pos_digits[0] - 1
        if is_ins:
            ins_token_idx = mut_code.index("ins")
            toks_added = mut_code[ins_token_idx + 1 : fs_token_idx]
            if "*" in toks_added:
                raise UnexpectedResidueError(
                    "Unexpected insertion of termination token followed by frameshift"
                )

            fs_start_pos = fs_start_pos if is_del else pos_digits[-1] - 1 + len(toks_added)

        preserved_seq = mut_target[:fs_start_pos] + "".join(toks_added)

        mutated = preserved_seq + self.unknown_tok * (fs_continuation_len - len(toks_added))

        mutated_with_gaps = mutated + self.gap_tok * (
            len(orig_seq) - (len(preserved_seq) + fs_continuation_len) + additional_padding
        )

        # remove any extra gaps in positions affected by frameshift
        orig_seq = orig_seq[:fs_start_pos] + orig_seq[fs_start_pos:].rstrip("-")

        modified_orig = orig_seq + self.gap_tok * (len(mutated) - len(orig_seq))

        return modified_orig, mutated_with_gaps

    def __extend_3prime(
        self,
        orig_seq: str,
        mut_target: str,
        known_aas: str,
        total_extension_length: int,
    ) -> Tuple[str, str]:
        """Apply 3' extension to sequence.
        Note:
            Called when key tokens `fs` or `ext` are explicitly in mut_code.
            If there are no known AAs, pass empty string to `known_aas`
            Gap tokens are appended to the reference sequence to match new length of mutated sequence.
        Args:
            orig_seq (str): original sequence
            mut_target (str): sequence to mutate
            known_aas (str): part of the extension whose AAs are known, if any
            total_extension_length (int): extension length. equal to len(known_aas) + len(unknown_aas).
        Returns:
            Tuple[str, str]: [
                mut_target after gap addition,
                mut_target after extension is applied
            ]
        Examples:
            >>> self.__apply_3prime_extension(
                orig_seq="ABCDEFG",
                mut_target="ABCDEFG",
                known_aas="QQQ",
                total_extension_length=10
            )
            ("ABCDEFG----------", "ABCDEFGQQQXXXXXXX")
        """
        num_known_aas = len(known_aas)

        if num_known_aas > total_extension_length:
            raise UnexpectedMutationError("extension length is less than length of known AAs")

        num_unknown_aas = total_extension_length - num_known_aas

        mut_target_result = mut_target + known_aas + self.unknown_tok * num_unknown_aas
        gap_inserted_result = orig_seq + self.gap_tok * total_extension_length

        return (gap_inserted_result, mut_target_result)

    def __extend_5prime(
        self, orig_seq: str, mut_target: str, total_extension_len: int
    ) -> Tuple[str, str]:
        """Apply 5' extension to sequence
        Note:
            For now, functionally identical to inserting `unknown_toks` right after the first AA residue. Applies sequence change but does not track that the position of the first AA is now negative.
        Args:
            orig_seq (str): original sequence
            mut_target (str): sequence to mutate
            total_extension_length (int): length of extension
        Returns:
            Tuple[str, str]: [
                orig_seq after gap addition,
                mut_target after extension is applied
            ]
        Examples:
            >>> self.__apply_5prime_extension(
                orig_seq="ABCDE",
                mut_target="ABCDE",
                total_extension_len=3
            )
            ("A---BCDE", "AXXABCDE")
        """
        self._current_offset_start_idx = 1
        self._current_offset_length = total_extension_len

        mut_target_result = (
            mut_target[0] + self.unknown_tok * (total_extension_len - 1) + mut_target
        )
        gap_inserted_result = orig_seq[0] + self.gap_tok * total_extension_len + orig_seq[1:]

        return (gap_inserted_result, mut_target_result)

    @classmethod
    def _standardize_mut_code(
        cls, mut_code: str, as_tokens: bool = False
    ) -> Union[List[Union[int, str]], str]:
        """
        Method for mutation code standardization.

        Given a HGVS-compliant protein mutation code, converts the formatting so that all amino acids are in 1-char uppercase form and variant descriptors (del, ins, fs, ter, dup, ext, etc.) are all lowercase.

        All residues in `mut_code` must be capitalized, and all variant type descriptors (except for Ter) must be lowercase!

        `mut_code` {str}: HGVS-compliant protein variant term.
            Only pass the part after the colon(:). No protein IDs!
        `as_tokens` {bool}:
            Parse numbers as int and return in list-of-tokens form without joining?
        """

        mut_code = mut_code.replace("%3D", "=")  # Handle some unusual cases in ClinVar
        mut_code = mut_code.replace("%3d", "=")  # Handle some unusual cases in ClinVar
        mut_code = mut_code.replace("Ter", "*")
        mut_code = mut_code.replace("ter", "*")

        needs_unknown_tok_insertion: bool = False
        fs_tok_idx: Optional[int] = None

        # Handle shortened frameshift HGVSp versions
        if "fs" in mut_code:
            # for abbreviated hgvsps without termination info
            # (ex: p.*34Rfs -> p.*34Rfs*?)
            if mut_code.endswith("fs"):
                mut_code += "*?"
            # for frameshift cases without initial altered AA info
            # #(ex: p.R23fs -> p.R23Xfs)
            fs_tok_idx = mut_code.index("fs")
            if mut_code[fs_tok_idx - 1].isdigit():
                needs_unknown_tok_insertion = True

        mut_code = re.sub(pattern=r"[\[\]\(\)]+", repl="", string=mut_code)

        # Determine if mut_code is 1-char or 3-char form by looking at the code from the beginning to the first appearance of a number.
        match = re.search(pattern=r"([A-Z][a-z]{2})(?![\*$])", string=mut_code, flags=re.ASCII)

        # mut_code is in 1-char fmt only when all chars before the first number are uppercase letters
        if match and match[0] in IUPACData.protein_letters_3to1_extended:
            if needs_unknown_tok_insertion:
                mut_code = mut_code[:fs_tok_idx] + "Xaa" + mut_code[fs_tok_idx:]
            tokens = cls._tokenize_mut_code(mut_code=mut_code, aa_char_len=3)
        else:
            if needs_unknown_tok_insertion:
                mut_code = mut_code[:fs_tok_idx] + "X" + mut_code[fs_tok_idx:]
            tokens = cls._tokenize_mut_code(mut_code=mut_code, aa_char_len=1)

        return tokens if as_tokens else "".join([str(tok) for tok in tokens])

    @classmethod
    def _tokenize_mut_code(
        cls, mut_code: str, aa_char_len: Literal[1, 3] = 1
    ) -> List[Union[str, int]]:
        """
        The word "token" here is the most basic unit of meaning
        according to the HGVS nomenclature system.

        If using aa_char_len=3, first letter of each amino acid is expected to be capitalized.

        `mut_code` {str}: mutation code to parse
        `aa_char_len` {int}: Number of characters representing one amino acid

        Some naive examples:
            p.T24C -> ('T', 24, 'C')
            p.C76_G79del -> ('C', 76, '_', 'G', 79, 'del')
            p.T24Ter -> ('T', 24, '*')
            p.K23_L24insRSG -> ('K', 23, '_', 'L', 24, 'ins', 'R', 'S', 'G')
            p.A76_C77delinsST -> ('A', 76, '_', 'C', 77, 'del', 'ins', 'S', 'T')
        """

        # If char is upper but is not part of Ter, treat as separate token
        tokens = list()
        current_tok = ""

        for idx, char in enumerate(mut_code):
            # token is either AA or a mutation type identifier (ter, ins, del, etc.)
            if char.isalpha():
                if aa_char_len == 1:
                    if char.isupper():
                        tokens.append(char)
                        current_tok = ""
                    else:  # is lowercase
                        current_tok += char
                        if current_tok == "fs" or len(current_tok) >= 3:
                            tokens.append(current_tok)
                            current_tok = ""
                        else:
                            continue
                elif aa_char_len == 3:
                    current_tok += char
                    if current_tok == "fs" or len(current_tok) >= 3:
                        tokens.append(current_tok)
                        current_tok = ""
                    else:
                        continue
                else:
                    # if mutation code's AAs are neither 1-char nor 3-char
                    raise ValueError(
                        f"Unexpected aa_char_len value:{aa_char_len}. Only 1 or 3 are valid values."
                    )

            # token is either a residue index or a symbol ('_', '?', '*', etc.)
            else:
                if char.isdigit():
                    current_tok += char

                    if idx + 1 == len(mut_code):
                        break
                    else:
                        if mut_code[idx + 1].isdigit():
                            continue
                        else:
                            tokens.append(current_tok)
                            current_tok = ""
                elif char in ["_", "?", "*", "=", "/"]:
                    tokens.append(char)
                    current_tok = ""
                    continue
                elif char == "-" and not current_tok:
                    current_tok = char
                    continue
                else:
                    raise UnexpectedMutationError(
                        f"Unexpected nonalphabet token {char} at index {idx} in {mut_code}"
                    )

        if current_tok:
            tokens.append(current_tok)

        # convert numeric tokens to ints. lstrip() is for handling negative ints
        tokens = [int(tok) if tok.lstrip("-").isdigit() else tok for tok in tokens]

        if aa_char_len == 3:
            tokens = [
                IUPACData.protein_letters_3to1_extended[tok]
                if str(tok) in IUPACData.protein_letters_3to1_extended
                else tok
                for tok in tokens
            ]

        valid_aa_set = set(IUPACData.protein_letters_1to3_extended.keys())
        valid_str_token_set = valid_aa_set.union(set(cls.__operator_tokens), {"*", "=", "?", "_"})

        for tok in tokens:
            if tok not in valid_str_token_set and not isinstance(tok, int):
                raise UnexpectedMutationError(
                    f"Found an unexpected token ({tok}) while processing mutation code: ({mut_code})"
                )

        return tokens

    def featurize(
        self,
        window: Optional[Tuple[int, int]] = None,
        auto_window_size: Optional[int] = None,
        featurizer: Optional[Featurizer] = None,
        require_msa: bool = False,
        featurize_fields: Iterable[Literal["ref", "mut", "msa"]] = {
            "ref",
            "mut",
            "msa",
        },
        return_metadata: bool = False,
    ) -> Dict[str, Union[List[int], Optional[npt.ArrayLike], Tuple[int, int], Dict[int, int]]]:
        """
        Method that processed data into evenly-sized, near-trainable numeric form.

        Note:
            An instance that extends `Featurizer` must be pre-assigned or passed here.
            If `require_msa` is `False`, `None` is returned for the MSA component if MSA array file is not available.

        Args:
            window (Optional[Tuple[int, int]], optional):
                Define explicit window range. This argument takes precedence
                over `auto_window_size`.
            auto_window_size (Optional[int]):
                Automatically generate window of size
                `2 * auto_window_size + 1`, centered on the first mutated AA.
            featurizer (Optional[Featurizer], optional):
                Define featurizer to use. Defaults to None.
            require_msa (bool): Raise error if MSA array is not available?
            featurize_fields (Iterable[Literal["ref", "alt", "msa"]]):
                Use this to control which features to obtain.
                `{"ref", "mut"}` returns `None` for MSA while
                `{"msa"}` returns `None` for keys `"ref"` and `"mut"`.
            return_metadata (bool):
                Option for the remote retrieval and featurization of MSA arrays.
                This option should be left False unless there is a specific use case in mind.

        Raises:
            MSAArrayFileNotFound: When MSA .npy file is required but not found
            RuntimeError: No featurizer available
            TypeError: Object passed is not a Featurizer
            TypeError: Passed Featurizer class is not compatible

        Returns:
            Dict[str, Union[List[int], Optional[npt.ArrayLike], Optional[Dict[int, int]]]]:
                "ref": a flat list of ints encoding the reference sequence
                "mut": a flat list of ints encoding the mutated sequence
                "msa": a 2D array of floats encoding the multi-species MSA
                "_reference_offsets": optional key added when return_metadata=True
                "_window": optional key added when return_metadata=True

        """
        if window is not None:
            auto_window_size = None

        if auto_window_size is not None:
            window = self._generate_featurization_window(auto_window_size=auto_window_size)

        used_featurizer = featurizer if featurizer else self.featurizer

        if not used_featurizer:
            raise RuntimeError("Cannot featurize: No featurizer class was provided")
        if not isinstance(used_featurizer, Featurizer):
            raise TypeError(
                f"featurize() requires a Featurizer instance, but {type(used_featurizer)} was provided instead"
            )
        if self.dtype not in used_featurizer.compatible_dtypes:
            raise TypeError(
                f"Featurizer ({type(used_featurizer)}) is not compatible with {type(self)}"
            )

        encode_ref = "ref" in featurize_fields
        encode_mut = "mut" in featurize_fields
        encode_msa = "msa" in featurize_fields

        if encode_ref:
            ref_encoding = (
                used_featurizer.featurize_seq(seq=self.orig_seq, window=window)
                if window is not None
                else used_featurizer.featurize_seq(seq=self.orig_seq)
            )
        else:
            ref_encoding = None

        if encode_mut:
            if hasattr(self, "mut_seq") and self.mut_seq:
                mut_encoding = (
                    used_featurizer.featurize_seq(seq=self.mut_seq, window=window)
                    if window is not None
                    else used_featurizer.featurize_seq(seq=self.mut_seq)
                )
            else:
                mut_encoding = list()
        else:
            mut_encoding = None

        if encode_msa:
            if self.prot_id not in ProteinSeqObj.msa_arrays:
                self.load_msa_arr(protein_id=self.prot_id, missing_is_error=require_msa)

            featurized_msa = (
                used_featurizer.featurize_msa(
                    msa_arr=ProteinSeqObj.msa_arrays[self.prot_id],
                    window=window,
                    reference_offsets=self.reference_offsets,
                )
                if self.prot_id in ProteinSeqObj.msa_arrays
                else None
            )
        else:
            featurized_msa = None

        features = {
            "ref": ref_encoding,
            "mut": mut_encoding,
            "msa": featurized_msa,
        }

        if return_metadata:
            features["_window"]: Tuple[int, int] = window
            features["_reference_offsets"]: Dict[int, int] = self.reference_offsets

        return features

    def _generate_featurization_window(self, auto_window_size: int) -> Tuple[int, int]:
        """
        Automatically generate featurization window based on window size and
        mutation code.

        Note:
            A ProteinSeqObj may have more than one mutation code, but current
            implementation only considers the first one when generating
            featurization window.

        Args:
            auto_window_size (int):
                Number of AAs to include in the featurization window on either side of the mutation site.
                Final window is size (2* `auto_window_size` + 1), and is centered on the earliest AA

        Raises:
            RuntimeError: Unsupported mutation type

        Returns:
            Tuple[int, int]: [inclusive starting idx, exclusive ending idx]

        Examples:
            >>> self._mut_code
            [['Glu', 10, 'Arg']]
            >>> self._generate_featurization_window(auto_window_size=4)
            [5, 14]
        """
        if ProtMut.NONE in self.mutation_types:
            return [0, 2 * auto_window_size + 1]

        for mut_code in self._mut_code:
            mut_start_pos = int(mut_code[1])

            if ProtMut.SYNONYMOUS in self.mutation_types:
                return [0, 2 * auto_window_size + 1]
            elif ProtMut.INSERTION in self.mutation_types:
                mut_start_idx = mut_start_pos
            elif any(
                mut_type in self.mutation_types
                for mut_type in [
                    ProtMut.MISSENSE,
                    ProtMut.DELETION,  # Recovery 있는 start-loss도 여기 포함
                    ProtMut.DELETION_INSERTION,
                    ProtMut.EXTENSION_3PRIME,
                    ProtMut.STOP_GAIN,
                    ProtMut.STOP_AMBIGUOUS,
                ]
            ):
                mut_start_idx = mut_start_pos - 1
            elif ProtMut.EXTENSION_5PRIME in self.mutation_types:
                mut_start_pos = 2
                mut_start_idx = mut_start_pos - 1

            elif ProtMut.START_LOSS in self.mutation_types:
                mut_start_pos = 1
                mut_start_idx = mut_start_pos - 1

            elif ProtMut.DUPLICATION in self.mutation_types:
                if "_" in mut_code:
                    mut_start_pos = mut_code[mut_code.index("_") + 2]
                mut_start_idx = mut_start_pos

            else:
                raise RuntimeError(f"Unsupported mutation type: {self.mutation_types}")

            window = [
                mut_start_idx - auto_window_size,
                mut_start_idx + auto_window_size + 1,
            ]
        return window

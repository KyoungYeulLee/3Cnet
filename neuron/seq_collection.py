from collections import defaultdict
from copy import deepcopy
from typing import (
    Any,
    DefaultDict,
    Dict,
    Hashable,
    Iterable,
    List,
    Literal,
    MutableSet,
    Optional,
    Set,
    Union,
)
import warnings

from tqdm import tqdm

from neuron.constants import HGVSp, ProteinMutationTypes, seq_aa, seq_id
from neuron.errors import UnexpectedMutationError, UnexpectedResidueError, NoActionPerformedWarning
from neuron.sequences import ProteinSeqObj, SequenceObj


class SeqCollection:
    """
    A wrapper class that holds ProteinSeqObj instances.

    Notable instance-level attributes:
        seqs:
            This attribute is a dictionary that maps standardized HGVSps to their corresponding
            ProteinSeqObj instances.
        aliases:
            This attribute is a dictionary that maps the user-provided, "raw" HGVSps to standardized hgvsps.
            Users can, but are not required to, be aware of standardized HGVSps
            when handling SeqCollection instances.
            The ProteinSeqObj of a raw HGVSp can be queried using dictionary-like syntax.
            (refer to SeqCollection.__getitem__())
        memory:
            SeqCollection receives a multi-item mapping of HGVSps and their reference sequences.
            HGVSps that are malformed and therefore unprocessable are stored in this attribute.
    """

    def __init__(
        self,
        hgvs_to_reference_seq_mapping: Dict[Union[HGVSp, seq_id], seq_aa],
        unusable_hgvses: Dict[str, MutableSet[HGVSp]],
        sequence_type: Union[Literal["protein"], Literal["transcript"]],
        check_seq_integrity: bool = False,
    ) -> None:
        # Instance variable declarations
        self.seqs: Dict[HGVSp, ProteinSeqObj] = dict()
        # Dict[HGVS_RAW, HGVS_STD]
        self.aliases: Dict[HGVSp, HGVSp] = dict()
        self.memory: DefaultDict[str, MutableSet[HGVSp]] = defaultdict(set, unusable_hgvses)
        self.memory_all: property[List[HGVSp]]
        self.sequence_type: str = sequence_type

        # Official init begins
        if sequence_type.lower() == "protein":
            seq_obj_type = ProteinSeqObj
        else:
            raise ValueError(f"Invalid sequence type: {sequence_type}")

        if not hgvs_to_reference_seq_mapping:
            return

        for hgvs, sequence in tqdm(
            hgvs_to_reference_seq_mapping.items(),
            desc=">>> Generating sequence objects...",
        ):
            try:
                seq_obj = seq_obj_type(
                    hgvs=hgvs, reference_seq=sequence, check_seq_integrity=check_seq_integrity
                )
                self.seqs[seq_obj.std_hgvs] = seq_obj
            except UnexpectedMutationError:
                self.memory["mutation_code_err"].add(hgvs)
                continue
            except UnexpectedResidueError:
                self.memory["residue_err"].add(hgvs)
                continue
            except Exception as e:
                self.memory["other_err"].add(hgvs)
                continue

            self.aliases[seq_obj.raw_hgvs] = seq_obj.std_hgvs

    def __len__(self) -> int:
        return len(self.aliases)

    def __getitem__(self, key: HGVSp) -> ProteinSeqObj:
        # Don't worry about the raw -> standardized conversion.
        # Just use it like a dictionary; the conversion is handled internally.
        return self.seqs[self.aliases[key]]

    @property
    def memory_all(self) -> Set[HGVSp]:
        nested_set = self.memory.values()
        return set([item for subset in nested_set for item in subset])

    @memory_all.setter
    def memory_all(self, arg: Any):
        raise ValueError("This property should not be manually assigned")

    def assign_metadata(
        self,
        metadata_key: Hashable,
        metadata_value: Optional[Any] = None,
        target_keys: Optional[Iterable[HGVSp]] = None,
        default_value: Optional[Any] = None,
        mapping_dict: Optional[Dict[HGVSp, Any]] = None,
        strict: bool = False,
        allow_overwrite: bool = False,
    ) -> None:
        """
        Add user-defined metadata to ProteinSeqObj instances.

        Note:
            Metadata are assigned to the `.metadata` attribute
            of each ProteinSeqObj instance in the form of key: value pairs.

            If `target_keys` and `mapping_dict` are both provided,
            `mapping_dict` takes precedence.

        Args:
            metadata_key (Hashable): Key of new metadata
            metadata_value (Optional[Any]):
                Value of new metadata. Defaults to None.
            target_keys (Optional[Iterable[Union[HGVSp, HGVSc]]]):
                Use this to assign metadata to a specific subset of sequence objects.
                Specify HGVSps in RAW (pre-standardization) form.
            default_value (Optional[Any]):
                Value to assign for all HGVSps not specified when using
                `mapping_dict` or `target_keys` arguments.
                If None, no action is performed on HGVSps not specified.
                Defaults to None.
            mapping_dict (Optional[Dict[Union[HGVSp], Any]]):
                Use this to assign potentially unique metadata on a per-HGVSp basis.
            strict (bool):
                If True, an error is raised if metadata assignment is attempted for an HGVSp
                not found within the SeqCollection object.
            allow_overwrite (bool):
                If True, allows the overwrite of existing metadata key-value pairs.

        Raises:
            TypeError:
                Wrong data type for target_keys or mapping_dict arguments
            KeyError:
                When strict=True and a mentioned HGVSp is not in the collection
            RuntimeError:
                When allow_overwrite=False and an overwrite is attempted

        Examples:
        (Assume SETUP is performed for all scenarios,
        scenarios are independent of one another otherwise)

            ### SETUP
            >>> example_key = "label"
            >>> example_collection = SeqCollection(...)
            >>> example_hgvsp = (some generic HGVSp found within example_collection)

            ### Scenario 1: Assign the same value to all HGVSps
            >>> example_collection.assign_metadata(
                metadata_key=example_key,
                metadata_value=0
            )
            >>> example_collection[example_hgvsp].metadata[example_key]
            0

            ### Scenario 2: Assign 0 to specific HGVSps; assign 1 to all other HGVSps
            >>> special_hgvsps = ['NP_00001.1', 'NP_000002.2']
            >>> example_collection.assign_metadata(
                metadata_key=example_key,
                metadata_value=0,
                target_keys=special_hgvsps,
                default_value=1
            )
            >>> example_collection['NP_00001.1'].metadata[example_key]
            0
            >>> example_collection[example_hgvsp].metadata[example_key]
            1

            ### Scenario 3: Assign varying values to specific HGVSps, assign -1 to all others
            >>> special_mapping = {
                'NP_00001.1:p.Thr21Arg': 99,
                'NP_000002.2:p.Arg42Met': 'ADEFS'
            }
            >>> example_collection.assign_metadata(
                metadata_key=example_key,
                mapping_dict=special_mapping,
                default_value=-1
            )
            >>> example_collection['NP_00001.1:p.Thr21Arg'].metadata[example_key]
            99
            >>> example_collection['NP_000002.2:p.Arg42Met'].metadata[example_key]
            'ADEFS'
            >>> example_collection[example_hgvsp].metadata[example_key]
            -1
        """

        if metadata_value is None and target_keys is None and mapping_dict is None:
            raise ValueError(
                "Either (metadata_value) or (metadata_value and target_keys) or (mapping_dict) must be defined"
            )

        specified_hgvs = set()
        if mapping_dict is not None:
            if not isinstance(mapping_dict, dict):
                raise TypeError(f"Expected a dict for mapping_dict but got: {type(mapping_dict)}")
            if not mapping_dict:
                if strict:
                    raise ValueError("Mapping dict must not be empty")
                warnings.warn(
                    message="Empty mapping_dict provided; returning without any changes",
                    category=NoActionPerformedWarning,
                )
                return
            specified_hgvs = set(mapping_dict.keys())
        elif target_keys is not None:
            if not issubclass(type(target_keys), Iterable):
                raise TypeError(
                    f"Expected an iterable for target_keys but got: {type(target_keys)}"
                )
            specified_hgvs = set(target_keys)

        if (mapping_dict is not None or target_keys is not None) and strict:
            hgvs_not_found = specified_hgvs.difference(set(self.aliases.keys()))
            if hgvs_not_found:
                raise KeyError(f"Raw HGVS keys {hgvs_not_found} were not found in the collection.")

        for raw_hgvs, std_hgvs in self.aliases.items():
            seq_obj = self.seqs[std_hgvs]
            if mapping_dict:
                if raw_hgvs in mapping_dict:
                    to_assign = mapping_dict[raw_hgvs]
                else:
                    if default_value is not None:
                        to_assign = default_value
                    else:
                        continue
            elif target_keys:
                if raw_hgvs in target_keys:
                    to_assign = metadata_value
                else:
                    if default_value is not None:
                        to_assign = default_value
                    else:
                        continue
            else:
                to_assign = metadata_value

            if (
                not allow_overwrite
                and metadata_key in seq_obj.metadata
                and seq_obj.metadata[metadata_key] != to_assign
            ):
                raise RuntimeError(
                    f"Attempted value-changing overwrite of existing metadata tag {metadata_key} for {raw_hgvs}"
                )

            seq_obj.metadata[metadata_key] = to_assign

    def __create_empty(self, sequence_type: str) -> "SeqCollection":
        return SeqCollection(
            hgvs_to_reference_seq_mapping=dict(),
            unusable_hgvses=[],
            sequence_type=sequence_type,
        )

    def filter_by_msa_availability(self, discard_mode: bool = True) -> "SeqCollection":
        """
        Filter ProteinSeqObjs by MSA numpy array availability.

        Note:
            Availability is determined by the existence
            of the expected .npy path.
            No correctness or sanity checks of the files are performed.
            Users are recommended to manually check whether expected paths are up-to-date.

        Args:
            discard_mode (bool, optional):
                If True: discard cases without MSAs.
                If False: discard cases with MSAs.
                Defaults to True.

        Returns:
            SeqCollection: Copy of this SeqCollection instance after filtering

        Examples:
            # NP_001215.2.npy exists, NP_000010.1 does not
            >>> self.aliases
            {
                "NP_001215.2:p.Arg3Gln": "NP_001215.2:p.R3Q",
                "NP_001215.2:p.R3_R4insKTer": "NP_001215.2:p.R3_R4insK*",
                "NP_000010.1": "NP_000010.1",
            }
            >>> filtered = self.filter_by_msa_availability(
                discard_mode=True  # True means discard if no MSA array file
            )
            >>> filtered.aliases
            # "NP_000010.1" is deleted in both `filtered.aliases` and `filtered.seqs` and is recorded in `filtered.unusables_dict`
            {
                "NP_001215.2:p.Arg3Gln": "NP_001215.2:p.R3Q",
                "NP_001215.2:p.R3_R4insKTer": "NP_001215.2:p.R3_R4insK*"
            }
        """
        excluded_reason = "filtered by MSA availability"

        aliases_new = dict()
        to_preserve_seqs = dict()
        to_discard_raw = set()

        new_copy = self.__create_empty(sequence_type=self.sequence_type)

        for raw_hgvs, std_hgvs in self.aliases.items():
            seq_obj = self.seqs[std_hgvs]
            msa_path = ProteinSeqObj.msa_root_dir / f"{seq_obj.prot_id}.npy"

            if (not msa_path.exists() and discard_mode) or (
                msa_path.is_file() and not discard_mode
            ):
                to_discard_raw.add(raw_hgvs)
            else:
                aliases_new[raw_hgvs] = std_hgvs
                to_preserve_seqs[std_hgvs] = self.seqs[std_hgvs]

        new_copy.seqs = deepcopy(to_preserve_seqs)
        new_copy.memory = deepcopy(self.memory)
        if to_discard_raw:
            new_copy.memory[excluded_reason] = to_discard_raw
        new_copy.aliases = aliases_new

        return new_copy

    def filter_by_sequence_id(
        self,
        filter_ids: Iterable[seq_id],
        discard_mode: bool = False,
    ) -> "SeqCollection":
        """
        Given an iterable of sequence IDs, return a copy of this instance
        that has been filtered by the sequence IDs.

        Note:
            Sequence IDs in `filter_ids` can be without version info.
            Sequence IDs without version info will match all versions.

        Args:
            filter_ids (Union[Iterable[ProteinID], Iterable[TranscriptID]]):
                Iterable of sequence IDs to use as a filter
            discard_mode (bool, optional):
                If True, discard sequences that match filter.
                If False, discard sequences that don't match filter.
                Defaults to False.

        Raises:
            ValueError: `filter_ids` argument is empty or False-y

        Returns:
            SeqCollection: Copy of this SeqCollection instance after filtering

        Examples:
            >>> self.aliases
            {
                "NP_001215.2:p.Arg3Gln": "NP_001215.2:p.R3Q",
                "NP_001215.2:p.R3_R4insKTer": "NP_001215.2:p.R3_R4insK*",
                "NP_000010.1": "NP_000010.1",
            }
            >>> filtered = self.filter_by_sequence_id(
                filter_ids=["NP_000010"],
                discard_mode=True
            )
            >>> filtered.aliases
            # "NP_000010.1" is deleted in both `filtered.aliases` and `filtered.seqs` and is recorded in `filtered.unusables_dict`
            {
                "NP_001215.2:p.Arg3Gln": "NP_001215.2:p.R3Q",
                "NP_001215.2:p.R3_R4insKTer": "NP_001215.2:p.R3_R4insK*"
            }
        """

        if not filter_ids:
            raise ValueError("At least one sequence ID is required.")

        excluded_reason = "filtered by sequence ID"

        aliases_new = dict()
        to_preserve_seqs = dict()
        to_discard_raw = set()

        new_copy = self.__create_empty(sequence_type=self.sequence_type)

        for raw_hgvs, std_hgvs in self.aliases.items():
            is_matched = any(
                raw_hgvs.split(":", maxsplit=1)[0] == filter_seq_id
                if "." in filter_seq_id
                else raw_hgvs.split(".", maxsplit=1)[0] == filter_seq_id
                for filter_seq_id in filter_ids
            )

            if (is_matched and discard_mode) or ((not is_matched) and (not discard_mode)):
                to_discard_raw.add(raw_hgvs)
            else:
                aliases_new[raw_hgvs] = std_hgvs
                to_preserve_seqs[std_hgvs] = self.seqs[std_hgvs]

        new_copy.seqs = deepcopy(to_preserve_seqs)
        new_copy.memory = deepcopy(self.memory)
        if to_discard_raw:
            new_copy.memory[excluded_reason] = to_discard_raw
        new_copy.aliases = aliases_new

        return new_copy

    def filter_by_mutation_types(
        self,
        filter_mutation_types: List[ProteinMutationTypes],
        discard_mode: bool = False,
    ) -> "SeqCollection":
        """
        Filter sequences within the collection by one or more sequence types.


        Note:
            - discard_mode=True: discards all ProteinSeqObjs that are tagged with at least one
              of the specified sequence types
            - discard_mode=False: keep ProteinSeqObjs that are tagged with at least one
              of the specified sequence types

            Please refer to the `neuron.constants.ProteinMutationTypes` Enum class
            for the definition of sequence types.

            Discarded sequences are recorded in the collection's memory by their raw HGVSp values.

        Args:
            filter_mutation_types (List[ProteinMutationTypes]):
                Mutation types to filter by
            discard_mode (bool):
                If True, discard specified types. If False, keep specifiec types.
                Default False.

        Returns:
            SeqCollection: A new copy of the filtered collection.
            (original collection remains unchanged.)

        Examples:
            >>> from neuron.constants import ProteinMutationTypes as ProtMut
            >>> my_seq_collection = SeqCollection(
                    hgvs_to_reference_seq_mapping={
                        "NP_000001.1": "RRRRRRRRRR",
                        "NP_000002.2:p.R3Ter": "RRRRRRRRRR"
                    },
                    unusable_hgvses = []
                )
            >>> new_collection = my_seq_collection.filter_by_mutation_types(
                    filter_mutation_types=[ProtMut.STOP_GAIN], discard_mode=True
                )
            >>> new_collection.seqs.keys()
            dict_keys(['NP_000001.1'])
            >>> new_collection.unusables_dict
            {'contains_excluded_type': 'NP_000002.2:p.R3Ter'}
        """

        if not filter_mutation_types:
            raise ValueError("One or more mutation types must be provided.")

        excluded_reason = "missing_required_type" if not discard_mode else "contains_excluded_type"

        to_preserve = dict()
        to_discard_raw = set()
        to_discard_std = set()
        for std_hgvs, seq_obj in self.seqs.items():
            if seq_obj.mutation_types.intersection(set(filter_mutation_types)):
                if not discard_mode:
                    to_preserve[std_hgvs] = seq_obj
                else:
                    to_discard_std.add(std_hgvs)
            else:
                if not discard_mode:
                    to_discard_std.add(std_hgvs)
                else:
                    to_preserve[std_hgvs] = seq_obj

        aliases_new = dict()
        for self_raw_hgvs, self_std_hgvs in self.aliases.items():
            if self_std_hgvs in to_discard_std:
                to_discard_raw.add(self_raw_hgvs)
            else:
                aliases_new[self_raw_hgvs] = self_std_hgvs

        new_copy = SeqCollection(
            hgvs_to_reference_seq_mapping=dict(),
            unusable_hgvses=[],
            sequence_type=self.sequence_type,
        )
        new_copy.seqs = deepcopy(to_preserve)
        new_copy.aliases = aliases_new
        new_copy.memory = deepcopy(self.memory)

        if to_discard_raw:
            new_copy.memory[excluded_reason] = to_discard_raw
        return new_copy

    def __merge_memory(
        self, other_collection: "SeqCollection", new_collection: "SeqCollection"
    ) -> "SeqCollection":
        for (
            unusable_reason,
            unusable_raw_hgvses,
        ) in other_collection.memory.items():
            existing_unusables = new_collection.memory[unusable_reason]
            new_collection.memory[unusable_reason] = existing_unusables.union(unusable_raw_hgvses)

        return new_collection

    def intersection(self, other: "SeqCollection") -> "SeqCollection":
        """
        Keep sequences found in both collections.

        Note:
            Keeps keys found in both `self.seqs` and `other.seqs`.
            Keys not found in both collections are recorded in the new collection's memory.

            Assuming the existence of two SeqCollection instances A and B,
            A.intersection(B) is equal to B.intersection(A).

        Args:
            other (SeqCollection): Another SeqCollection instance

        Returns:
            SeqCollection: A new SeqCollection instance reflecting the results of the intersection

        Examples:
            >>> A = SeqCollection(...)
            >>> B = SeqCollection(...)
            >>> A_intersect_B = A.intersection(B)
        """
        self_std_keys = set(self.seqs.keys())
        other_std_keys = set(other.seqs.keys())

        new_collection = self.__create_empty(sequence_type=self.sequence_type)

        new_seqs: Dict[HGVSp, SequenceObj] = dict()
        new_aliases: Dict[HGVSp, HGVSp] = dict()
        new_self_unusables: Set[HGVSp] = set()

        candidate_std_keys = self_std_keys.intersection(other_std_keys)

        for self_raw_hgvs, self_std_hgvs in self.aliases.items():
            if self_std_hgvs in candidate_std_keys:
                self_seq_obj = deepcopy(self.seqs[self_std_hgvs])
                new_seqs[self_std_hgvs] = self_seq_obj
                new_aliases[self_raw_hgvs] = self_std_hgvs

            else:
                new_self_unusables.add(self_raw_hgvs)

        for other_raw_hgvs, other_std_hgvs in other.aliases.items():
            if other_std_hgvs not in candidate_std_keys:
                new_self_unusables.add(other_raw_hgvs)

        new_collection.memory = deepcopy(self.memory)
        new_collection.memory["excluded_by_intersection"] = new_collection.memory[
            "excluded_by_intersection"
        ].union(list(new_self_unusables))
        new_collection = self.__merge_memory(other_collection=other, new_collection=new_collection)

        new_collection.seqs = new_seqs
        new_collection.aliases = new_aliases

        return new_collection

    def exclude(self, other: "SeqCollection", strict: bool = False) -> "SeqCollection":
        """
        Exclude HGVSps found in `other.seqs` from `self.seqs`.

        Note:
            Affected keys in self.aliases are also removed.
            Keys removed from `self.seqs` as the result of this operation
            are recorded in the new collection's memory.

        Args:
            other (SeqCollection):
                Another SeqCollection instance specifying HGVSps to remove
            strict (bool):
                If True, keys in `other` but missing in `self` raise a ValueError.

        Returns:
            SeqCollection: A new SeqCollection instance reflecting the results of the exclusion


        Examples:
            >>> A = SeqCollection(...)
            >>> B = SeqCollection(...)
            >>> a_minus_b = A.exclude(B)
        """
        to_discard_std: Set[HGVSp] = set(other.seqs.keys())

        if strict:
            self_missing_std_keys = set(self.seqs.keys()).difference(to_discard_std)
            if self_missing_std_keys:
                raise ValueError(
                    f"{len(self_missing_std_keys)} standardized HGVS keys are found in other but not in self: {self_missing_std_keys}"
                )

        new_collection: SeqCollection = self.__create_empty(sequence_type=self.sequence_type)

        new_seqs: Dict[HGVSp, SequenceObj] = dict()
        new_aliases: Dict[HGVSp, HGVSp] = dict()

        to_discard_raw: Set[HGVSp] = set()
        new_unusables: Dict[str, Set[HGVSp]] = deepcopy(self.memory)

        for self_raw_hgvs, self_std_hgvs in self.aliases.items():
            if self_std_hgvs in to_discard_std:
                to_discard_raw.add(self_raw_hgvs)
            else:
                new_seqs[self_std_hgvs] = deepcopy(self.seqs[self_std_hgvs])
                new_aliases[self_raw_hgvs] = self_std_hgvs

        new_unusables["subtracted"] = new_unusables["subtracted"].union(to_discard_raw)

        new_collection.seqs = new_seqs
        new_collection.aliases = new_aliases
        new_collection.memory = new_unusables

        return new_collection

    def merge(self, other: "SeqCollection") -> "SeqCollection":
        """
        Merge the content of `self` with the content of `other`.

        Note:
            When a key is found in both `self` and `other`, `self` takes precedence.

        Args:
            other (SeqCollection): SeqCollection instance to merge with

        Returns:
            SeqCollection: A new SeqCollection instance reflecting the results of the merge

        Examples:
            >>> A = SeqCollection(...)
            >>> B = SeqCollection(...)
            >>> a_plus_b = A.merge(B)
        """
        new_collection = deepcopy(self)
        for other_raw_hgvs, other_std_hgvs in other.aliases.items():
            if other_raw_hgvs not in new_collection.aliases:
                new_collection.aliases[other_raw_hgvs] = other_std_hgvs
                if other_std_hgvs not in new_collection.seqs:
                    new_collection.seqs[other_std_hgvs] = deepcopy(other.seqs[other_std_hgvs])

        new_collection = self.__merge_memory(other_collection=other, new_collection=new_collection)

        return new_collection

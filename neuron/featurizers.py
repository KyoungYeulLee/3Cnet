from abc import ABC, abstractmethod
from typing import Dict, List, NewType, Set, Tuple, Union

import numpy as np
import numpy.typing as npt
from neuron.aa_to_int_mappings import ThreeCNetAAMappings

ReferenceSeqIntEmbedding = NewType(
    name="Integer embedding of reference sequence", tp=List[List[int]]
)
MutatedSeqIntEmbedding = NewType(
    name="Integer embedding of mutated sequence", tp=List[List[int]]
)


class Featurizer(ABC):
    @property
    @abstractmethod
    def compatible_dtypes(self) -> Union[Set[str], Set[type]]:
        """Property defining data types/classes with which the Featurizer subclass is compatible. Define as a set of one or more types.
        """
        raise NotImplementedError

    @abstractmethod
    def featurize_seq(self, *args, **kwargs):
        raise NotImplementedError


class ThreeCNetProteinSeqFeaturizer(Featurizer):
    """Encode protein sequences as specified in 3Cnet.

    Note:
        The reference and mutated versions of a sequence are featurized as a pair, only at the segment specified by the `window` argument.

        When `token_length = N`, the featurized number at index I represents a segment of N amino acids centered at I. Stride is fixed at 1.
        Any index that goes out-of-bounds is zero-padded before featurization.

        3Cnet uses a exponential-sum one-hot encoding scheme, where each AA position is given a number from 0 to 22.

        If we assume that `token_length=3` and -, R, H, and K encode to 0, 1, 2, and 3 respectively, RRHHKK is padded to -RRHHKK- and each AA in the token is raised to a power higher than the last.

    Raises:
        ValueError: If `token_length` is not odd
        ValueError: If reference sequences, mutated sequences, and windows are not equal in length
        ValueError: If window sizes are inconsistent
        IndexError: If starting index of a window is later than its ending index

    Examples:
        >>> featurizer = ThreeCNetProteinSeqFeaturizer(
            token_length=3,
            gap_token=ProteinSeq.gap_aa_token
        )
        >>> featurizer.featurize(
            ref_seq: ['RHK'],
            mut_seq: ['R-K'],
            window: [[0, 2]]
        )
        ([
            [ # encoded reference sequence
                1801, # (-) 0*23^0 + (R) 1*23^1 + (H) 2*23^2
                1634, # (R) 1*23^0 + (H) 2*23^1 + (K) 3*23^2
                71    # (H) 2*23^0 + (K) 3*23^1 + (-) 0*23^2
            ]
        ],
        [
            [ # encoded mutated sequence
                23  , # (-) 0*23^0 + (R) 1*23^1 + (-) 0*23^2
                1588, # (R) 1*23^0 + (-) 0*23^1 + (K) 3*23^2
                69    # (-) 0*23^0 + (K) 3*23^1 + (-) 0*23^2
            ]
        ])
    """

    compatible_dtypes = {"sequence_protein"}
    aa_mapper = ThreeCNetAAMappings.AA2INT

    def __init__(self, token_length: int, gap_token: str):

        if token_length % 2 != 1:
            # token당 짝수 길이의 featurization은 미지원.
            raise ValueError(
                f"Token length must always be odd. Invalid value: {token_length}"
            )

        self.token_length = token_length
        self.gap_token = gap_token

    def featurize_msa(
        self,
        msa_arr: npt.ArrayLike,
        window: Tuple[int, int],
        reference_offsets: Dict[int, int],
    ) -> npt.ArrayLike:
        """
        Adds padding to msa array using `reference_offsets`, then applies
        slicing according to window.

        Info:
            Padding is inserted as a zero array.

        Args:
            msa_arr (npt.ArrayLike):
                A 2D matrix of shape [seq_len, num_features]
            window (Tuple[int, int]):
                Region of msa feature array to extract after padding
            reference_offsets (Dict[int, int]):
                Dictionary representing starting idxes of msa paddings and their lengths

        Raises:
            ValueError: If msa array is not a 2D matrix

        Returns:
            npt.ArrayLike:
                subset of msa array extracted after padding is applied

        Examples:
            >>> ex_msa_arr = <np.ndarray of shape [500, 2]>
            >>> featurized_msa = self.featurize_msa(
                msa_arr=ex_msa_arr,
                window=[0,8], # [inclusive, exclusive)
                reference_offsets={3: 2}
            )
            >>> featurized_msa
            [[0,0], [1,1], [2,2], [0,0], [0,0], [3,3], [4,4], [5,5]]
        """
        if len(msa_arr.shape) != 2:
            raise ValueError(
                f"MSA featurization logic is intended for 2D matrices, but provided array's shape is {msa_arr.shape}"
            )

        len_seq, num_dims = msa_arr.shape
        padding_dtype = msa_arr.dtype

        if window[1] > len_seq:
            msa_arr = np.append(
                arr=msa_arr,
                values=np.zeros(
                    shape=(window[1] - len_seq + 1, num_dims),
                    dtype=padding_dtype,
                ),
                axis=0,
            )

        upstream_padding = 0
        if window[0] < 0:
            upstream_padding = abs(window[0])
            msa_arr = np.append(
                arr=np.zeros(
                    shape=(upstream_padding, num_dims), dtype=padding_dtype
                ),
                values=msa_arr,
                axis=0,
            )
            window[0] += upstream_padding
            window[1] += upstream_padding

        if not reference_offsets:
            return msa_arr[window[0] : window[1]]

        piece_starting_offset = 0

        for pad_start_idx, pad_len in reference_offsets.items():

            piece_starting_offset += pad_start_idx
            pad_arr = np.zeros(shape=(pad_len, num_dims), dtype=padding_dtype)
            msa_arr = np.insert(
                arr=msa_arr,
                obj=piece_starting_offset + upstream_padding,
                values=pad_arr,
                axis=0,
            )

        return msa_arr[window[0] : window[1]]

    def featurize_seq(self, seq: str, window: Tuple[int, int],) -> List[int]:
        """Method called by the class that needs featurizing.

        Note:
            The lengths of all arguments must match and window size must remain consistent (though location may change).
            window indices are inclusive at start and exclusive at end.

         Args:
            seq (str): amino acid sequence to featurize
            window (Tuple[int, int]): region of sequences to featurize

        Raises:
            IndexError: If starting index is later than ending index

        Returns:
            List[int]: sequence featurized into list of ints

        Examples:
            >>> feat = ThreeCNetProteinSeqFeaturizer(
                token_length=1, gap_token='-'
            )
            >>> feat.featurize(seq='ABCDEFG', window=[3, 5])
            [4, 5, 6]
        """

        if not seq:
            return []

        start_idx, end_idx = window
        len_ref = len(seq)

        if start_idx > end_idx:
            raise IndexError(
                f"Starting idx ({start_idx}) must be smaller than last index ({end_idx})"
            )
        if self.token_length > 1:
            start_idx -= (self.token_length - 1) // 2
            end_idx += (self.token_length - 1) // 2

        offset = abs(start_idx)
        start_idx += offset
        end_idx += offset

        zero_padded_seq = (
            (self.gap_token * offset)
            + seq
            + (self.gap_token * (end_idx - len_ref))
        )
        encoding = list()
        for idx in range(start_idx, end_idx - (self.token_length - 1)):
            val = 0
            for exponent, residue_pos in enumerate(
                range(idx, idx + self.token_length)
            ):
                val += self.aa_mapper[zero_padded_seq[residue_pos]] * (
                    len(self.aa_mapper.keys()) ** exponent
                )
            encoding.append(val)

        return encoding

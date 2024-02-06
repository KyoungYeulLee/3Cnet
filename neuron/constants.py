from enum import auto, Enum

# String identifiers that uniquely identifies amino acid sequences within its set.
# ex: NP_0013155.1 or ENSP15432412
seq_id = str

# All-uppercase, one-character-per-amino-acid repersentation of amino acid sequences.
seq_aa = str

# The combination of a seq_id and its suffix in HGVS syntax
# describing the sequence's deviation from its reference.
# ex: NP_0013155.1:p.Q65del or ENSP15432412:p.D23*
HGVSp = str

class ProteinMutationTypes(Enum):
    # Mutation types according to
    # https://varnomen.hgvs.org/recommendations/protein/
    DELETION = auto()
    DELETION_INSERTION = auto()
    DUPLICATION = auto()
    EXTENSION_5PRIME = auto()  # upstream extension

    # downstream extension or stop-loss
    EXTENSION_3PRIME = auto()
    """
    In order to qualify as EXTENSION_3PRIME, the variant must satisfy one or more of below criteria:
    - Sequence changes begin at the termination (*) token (p.*46Lfs*5, p.*46Pheext*?)
    - HGVSp contains an extension (ext) token but is not a 3-prime extension (p.*46Pheext*?)
    - Total length of the extension cannot be determined (p.K4delinsFFfs, p.*46Pheext*?)
    """

    FRAMESHIFT = auto()
    INSERTION = auto()
    MISSENSE = auto()  # aka single substitution
    NONE = auto()  # when protein does not have any mutations (reference state)
    REPEAT = auto()  # not yet supported
    START_LOSS = auto()
    STOP_GAIN = auto()

    STOP_AMBIGUOUS = auto()
    """
    In order to qualify as STOP_AMBIGUOUS, the variant must satisfy thr below criteria:
    - Sequence changes begin mid-sequence, and the new total length of the sequence is known,
    but whether the change is a truncation or an extension relative to the original sequence
    cannot be determined (p.P5Tfs*3)
    """

    SYNONYMOUS = auto()

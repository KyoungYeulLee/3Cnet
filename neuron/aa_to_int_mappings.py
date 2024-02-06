from Bio.Data.IUPACData import extended_protein_letters


class ThreeCNetAAMappings:
    # AA <-> int mapping used by 3cnet in Oct 2021
    AA2INT = {
        "-": 0,
        "R": 1,
        "H": 2,
        "K": 3,
        "D": 4,
        "E": 5,
        "S": 6,
        "T": 7,
        "N": 8,
        "Q": 9,
        "C": 10,
        "U": 11,
        "G": 12,
        "P": 13,
        "A": 14,
        "I": 15,
        "L": 16,
        "M": 17,
        "F": 18,
        "W": 19,
        "Y": 20,
        "V": 21,
        "X": 22,  # unknown or any
    }
    INT2AA = {val: key for key, val in AA2INT.items()}


class IUPACAAExtendedMappings:
    # "X" is already included in extended set
    AA2INT = {aa: idx for idx, aa in enumerate(extended_protein_letters)}
    AA2INT["-"] = len(AA2INT)

    INT2AA = {val: key for key, val in AA2INT.items()}

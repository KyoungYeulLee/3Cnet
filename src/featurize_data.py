'''
modules to featurize data from sequence data and mutation code
'''
import sys
import re
import numpy as np


def change_t2s(three_char: str) -> str:
    '''
    Change three character string representing amino acids into single character representation
    :three_char {string} : three character string for each amino acid
    :return {char} : single character for corresponding amino acid
    '''
    three_char = three_char.lower()
    t2s_table = {
        'arg': 'r', 'his': 'h', 'lys': 'k', 'asp': 'd',
        'glu': 'e', 'ser': 's', 'thr': 't', 'asn': 'n',
        'gln': 'q', 'cys': 'c', 'sec': 'u', 'gly': 'g',
        'pro': 'p', 'ala': 'a', 'ile': 'i', 'leu': 'l',
        'met': 'm', 'phe': 'f', 'trp': 'w', 'tyr': 'y',
        'val': 'v', 'any': 'x'}
    return t2s_table.get(three_char)


def change_s2t(single_char: str) -> str:
    '''
    Change three character string representing amino acids into single character representation
    :single_char {string} : single_char string for each amino acid
    :return {char} : three character for corresponding amino acid
    '''
    single_char = single_char.lower()
    s2t_table = {
        'r': 'arg', 'h': 'his', 'k': 'lys', 'd': 'asp',
        'e': 'glu', 's': 'ser', 't': 'thr', 'n': 'asn',
        'q': 'gln', 'c': 'cys', 'u': 'sec', 'g': 'gly',
        'p': 'pro', 'a': 'ala', 'i': 'ile', 'l': 'leu',
        'm': 'met', 'f': 'phe', 'w': 'trp', 'y': 'tyr',
        'v': 'val', 'x': 'any'}
    return s2t_table.get(single_char)


def change_s2i(single_char: str) -> int:
    '''
    Change single character representation of amino acids into specific numbers
    :parameter {type} : single character for each amino acid
    :return {int} : integer representation for corresponding amino acid
    '''
    single_char = single_char.lower()
    s2i_table = {
        'r': 1, 'h': 2, 'k': 3, 'd': 4, 'e': 5,
        's': 6, 't': 7, 'n': 8, 'q': 9, 'c': 10,
        'u': 11, 'g': 12, 'p': 13, 'a': 14, 'i': 15,
        'l': 16, 'm': 17, 'f': 18, 'w': 19, 'y': 20,
        'v': 21, 'x': 0}
    return s2i_table.get(single_char)


def mut2tup(mut_code: str) -> tuple:
    '''
    Change the string representing a mutation into a tuple containing ...
    mutation residue, original amino acid, mutated amino acid ...
    where amino acids are represented by three character string.
    :mut_code {string} : string representation of mutations. Ex) 'p.his39arg'
    :return {tup} : (mutation residue, original amino acid, mutated amino acid).
        empty tuple for error.
    '''
    if mut_code[0:2] != 'p.':
        mut_code = 'p.' + mut_code

    try:
        mut_tup = ()
        # start_lost all removed
        if re.search(pattern="p.Met1[A-Za-z]+", string=mut_code) is not None:
            mut_tup = (0, 'start_lost_all')
        # inframe deletion & start_lost
        elif mut_code[-3:] == 'del':
            mut_tup = deletion_tup(mut_code)
        # stop_gain
        elif mut_code[-3:] == 'Ter' or mut_code[-1] == '*':
            mut_tup = termination_tup(mut_code)
        # frameshift
        elif 'fs' in mut_code:
            mut_tup = frameshift_tup(mut_code)
        # missense
        else:
            mut_tup = missense_tup(mut_code)

    except (ValueError, AttributeError) as e:
        print(mut_code)
        mut_tup = ()

    return mut_tup


def missense_tup(mut_code: str) -> tuple:
    '''
    get mutation tuple for missense mutation
    '''
    if mut_code[3].isnumeric(): # single char
        mut_ori = change_s2t(mut_code[2].lower()) # change it to the three-char format
        mut_aft = change_s2t(mut_code[-1].lower())
        mut_res = int(mut_code[3:-1])
    else: # three char
        mut_ori = mut_code[2:5].lower()
        mut_aft = mut_code[-3:].lower()
        mut_res = int(mut_code[5:-3])

    change_s2i(change_t2s(mut_ori))
    change_s2i(change_t2s(mut_aft))

    mut_tup = (mut_res, mut_ori, mut_aft, 'missense')

    return mut_tup


def deletion_tup(mut_code: str) -> tuple:
    '''
    get mutation tuple for missense mutation
    '''
    mut_split = mut_code[2:-3].split('_') # split without front 'p.' & back 'del'

    mut_tup = ()
    if len(mut_split) == 1: # one amino acid deletion
        mut_init = mut_split[0]
        init_aa = mut_init[0:3]
        init_res = int(mut_init[3:])

        mut_tup = (init_res, init_aa, init_res, init_aa, 'inframe_del')

    else: # several amino acids deletion
        mut_init, mut_tail = mut_split
        init_aa = mut_init[0:3].lower()
        init_res = int(mut_init[3:])
        tail_aa = mut_tail[0:3].lower()
        tail_res = int(mut_tail[3:])

        # start-lost
        if (init_res == 2 and tail_aa == 'met') or (init_res == 1 and init_aa == 'met'):
            mut_tup = (tail_res, tail_aa, init_res, init_aa, 'start_lost')
        # inframe indel
        else:
            mut_tup = (tail_res, tail_aa, init_res, init_aa, 'inframe_del')

    return mut_tup


def termination_tup(mut_code: str) -> tuple:
    '''
    get mutation tuple for termination variant
    '''
    mut_init = ''
    if mut_code[-3:] == 'Ter':
        mut_init = mut_code[2:-3] # without front 'p.' & back 'Ter'
    elif mut_code[-1] == '*':
        mut_init = mut_code[2:-1] # without front 'p.' & back '*'

    init_aa = mut_init[0:3].lower()
    init_res = int(mut_init[3:])

    mut_tup = (init_res, init_aa, 'stop_gain')

    return mut_tup


def frameshift_tup(mut_code: str) -> tuple:
    '''
    get mutation tuple for frameshift mutation
    '''
    mut_desc = mut_code[2:].split('fs')[0]
    ref_aa = mut_desc[0:3].lower()
    if mut_desc[-1].isnumeric(): # Glu51
        mut_res = int(mut_desc[3:])
        alt_aa = 'any' # no clue
    else: # Glu51Val
        mut_res = int(mut_desc[3:-3])
        alt_aa = mut_desc[-3:].lower()

    mut_tup = (mut_res, ref_aa, alt_aa, 'frameshift')

    return mut_tup


def check_res(mut_tup: tuple, seq: str) -> bool:
    '''
    check whether mutation code (ex. 'p.his39arg') is valid for the input sequence
    '''
    res_is_correct = True
    try:
        if mut_tup[-1] == 'missense':
            mut_res, mut_ori = mut_tup[0:2]
            mut_idx = mut_res - 1
            res_is_correct = bool(seq[mut_idx].lower() == change_t2s(mut_ori))

        elif mut_tup[-1] in ['start_lost', 'inframe_del']:
            tail_res, tail_aa, init_res, init_aa = mut_tup[0:4]
            tail_idx = tail_res - 1
            init_idx = init_res - 1
            res_is_coorect = (
                bool(seq[tail_idx].lower() == change_t2s(tail_aa)) and
                bool(seq[init_idx].lower() == change_t2s(init_aa))
            )

        elif mut_tup[-1] == 'stop_gain':
            init_res, init_aa = mut_tup[0:2]
            init_idx = init_res - 1
            res_is_correct = bool(seq[init_idx].lower() == change_t2s(init_aa))

        elif mut_tup[-1] == 'frameshift':
            mut_res, mut_ori = mut_tup[0:2]
            mut_idx = mut_res - 1
            res_is_correct = bool(seq[mut_idx].lower() == change_t2s(mut_ori))

    except IndexError:
        res_is_correct = False

    return res_is_correct


def seq2input(seq: str, seq_len: int, mut: tuple) -> np.array:
    '''
    Change protein sequence into numpy array consists of the integer representation of amino-acids
    :seq {string} : protein sequence represented by single character amino acids
    :seq_len {int} : the length of sequences to be represented into an array
    :mut {int or tuple} : (mutation residue, original amino acid, mutated amino acid)
        When mut is integer, the sequence does not changed.
        On the other hand, if mut is a tuple the sequence is mutated as it indicates
    :return {array} : 1 dimenstional array containing integer reprentation of amino-acids
    '''
    if isinstance(mut, int):
        mut_idx = mut - 1
        mut_tag = 'ref'
        # mutation index is the index of sequence string for a specific residue. 
        # seq[mut_idx] == 'h' if the (mut_res)th residue is histidine
    else:
        mut_idx = mut[0] - 1
        mut_tag = mut[-1]

    seq_np = np.zeros(seq_len, dtype=int)
    if mut_idx == -1: # start_lost_all
        mut_idx = int(seq_len/2)

    start_idx = mut_idx - int(seq_len/2)
    end_idx = mut_idx + int(seq_len/2)

    for np_idx, aa_idx in enumerate(range(start_idx, end_idx+1)):
        if aa_idx < 0: # before start
            continue
        if aa_idx >= len(seq): # after end
            break

        token_int = 0
        if mut_tag == 'ref':
            token_int = _get_token_number(seq, aa_idx, mut_idx)

        elif mut_tag == 'missense':
            # mut = (mut_res, mut_ori, mut_aft, 'missense')
            mut_aft = mut[2]
            token_int = _get_token_number(seq, aa_idx, mut_idx, mut_aft)

        elif mut_tag == 'start_lost_all':
            # use zero-padding
            break

        elif mut_tag in ['start_lost', 'inframe_del']:
            # mut = (tail_res, tail_aa, init_res, init_aa, 'start_lost')
            init_idx = mut[2] - 1 # init_idx == init_res - 1
            if init_idx <= aa_idx <= mut_idx: # mut_idx == tail_res - 1
                continue
            token_int = _get_token_number(seq, aa_idx, mut_idx)

        elif mut_tag == 'stop_gain':
            # mut = (init_res, init_aa, 'stop_gain')
            if aa_idx >= mut_idx: # termination after mutation index
                break
            token_int = _get_token_number(seq, aa_idx, mut_idx)

        elif mut_tag == 'frameshift':
            # mut = (mut_res, ref_aa, alt_aa, 'frameshift')
            mut_aft = mut[2]
            if aa_idx > mut_idx:
                break
            token_int = _get_token_number(seq, aa_idx, mut_idx, mut_aft)

        else:
            print("unexpected mutation tag")
            sys.exit(1)

        seq_np[np_idx] = token_int

    return seq_np


def _get_token_number(seq: str,
                      aa_idx: int,
                      mut_idx: int,
                      mut_aft='') -> int:
    '''
    get the unique number for a multi-amino-acid token
    :seq {string} : protein sequence represented by single character amino acids
    :aa_idx {int} : the residue position (starting from 0) of transcript sequence
    :mut_idx {int} : the position where the mutation has occurred
    :mut_aft {str} : three character amino-acid representation for the mutated residue.
        an empty string should be assigned for the reference sequence.
    :skip_region {list} : the empty region of the sequence to skip.
        for example, skip_region == [11, 42] -> aa_idx = 11~42 (residue = 12~43) empty
        default is an empty list which means there is no skip region
    '''
    token_int = 0
    if aa_idx >= len(seq): # when the residue position is out of sequence
        token_int = 0
    elif mut_aft and aa_idx == mut_idx: # for the mutated residue
        token_int = change_s2i(change_t2s(mut_aft)) # integer value for the alternative amino-acid
    else:
        token_int = change_s2i(seq[aa_idx])

    return token_int

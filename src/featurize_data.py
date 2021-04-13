'''
modules to featurize data from sequence data and mutation code
'''
import sys
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
        'val': 'v'}
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
        'v': 'val'}
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
    try:
        if mut_code[0:2] != 'p.':
            mut_code = 'p.' + mut_code

        if mut_code[3].isnumeric(): # single char
            mut_ori = change_s2t(mut_code[2].lower())
            mut_aft = change_s2t(mut_code[-1].lower())
            mut_res = int(mut_code[3:-1])
        else: # three char
            mut_ori = mut_code[2:5].lower()
            mut_aft = mut_code[-3:].lower()
            mut_res = int(mut_code[5:-3])

        change_s2i(change_t2s(mut_ori))
        change_s2i(change_t2s(mut_aft))

        mut_tup = (mut_res, mut_ori, mut_aft)

    except (ValueError, AttributeError) as e:
        mut_tup = ()

    return mut_tup


def rebuild_mut(mut_tup: tuple) -> str:
    '''
    mut_tup to standard hgvs format
    '''
    mut_res, mut_ori, mut_aft = mut_tup
    mut_code = f"p.{mut_ori.capitalize()}{mut_res}{mut_aft.capitalize()}"
    return mut_code


def get_snvbox_format(mut_id: str) -> str:
    out_id = ''
    np_id, mut_code = mut_id.split(':')
    mut_tup = mut2tup(mut_code)

    if not mut_tup:
        out_id = ''
    else:
        mut_res, mut_ori, mut_aft = mut_tup
        new_ori = change_t2s(mut_ori).upper()
        new_aft = change_t2s(mut_aft).upper()
        new_code = f"{new_ori}{mut_res}{new_aft}"
        out_id = np_id + ':' + new_code

    return out_id


def check_res(mut_code: str, seq: str) -> bool:
    '''
    check whether mutation code (ex. 'p.his39arg') is valid for the input sequence
    '''
    mut_res, mut_ori, _ = mut2tup(mut_code)
    mut_idx = mut_res - 1
    try:
        res_is_correct = bool(seq[mut_idx].lower() == change_t2s(mut_ori))
    except IndexError:
        res_is_correct = False
    return res_is_correct


def seq2input(seq: str, seq_len: int, use_seq_block: bool, mut=1) -> np.array:
    '''
    Change protein sequence into numpy array consists of the integer representation of amino-acids
    :seq {string} : protein sequence represented by single character amino acids
    :seq_len {int} : the length of sequences to be represented into an array
    :mut {int or tuple} : (mutation residue, original amino acid, mutated amino acid)
        When mut is integer, the sequence does not changed.
        On the other hand, if mut is a tuple the sequence is mutated as it indicates
    :use_seq_block {bool} : whether to use sequences around mutation residue or not
    :return {array} : 1 dimenstional array containing integer reprentation of amino-acids
    '''
    if isinstance(mut, int):
        mut_idx = mut - 1
    else:
        mut_res, _, mut_aft = mut
        mut_idx = mut_res - 1
    seq_np = np.zeros(seq_len, dtype=int)

    if use_seq_block:
        for i, aa_idx in enumerate(range(mut_idx-int(seq_len/2), mut_idx+int(seq_len/2)+1)):
            if aa_idx < 0:
                continue
            if aa_idx >= len(seq):
                break
            if not isinstance(mut, int) and aa_idx == mut_idx:
                idx = change_s2i(change_t2s(mut_aft))
            else:
                idx = change_s2i(seq[aa_idx])
            seq_np[i] = idx

    else:
        for i, aa_res in enumerate(seq):
            if not isinstance(mut, int) and mut_idx == i:
                idx = change_s2i(change_t2s(mut_aft))
            else:
                idx = change_s2i(aa_res)
            seq_np[i] = idx
                
    return seq_np

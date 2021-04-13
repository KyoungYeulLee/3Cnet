'''
A module to build dataset to be used for predictor
'''
import sys
import pickle as pk
import logging
import numpy as np
import yaml
import featurize_data as fd


def load_mut2act(mut_path: str, default_act=0.0) -> dict:
    '''
    load a list of mutations in the file.
    The file can have multiple column only if the header is given after #.
    If not, it should have a single column of mutation IDs given in the HGVSp format.
    :mut_path {string}: the file path for mutation file.
    :default_act {float}: the default activity for mutation when the value is not given
    :return {dict}: a dictionary having keys of mutation id and values of their activity
    '''
    with open(mut_path) as file:
        mut2act = dict()
        count_dup = 0
        col = ['HGVSp']
        for file_line in file:
            if file_line[0] == '#':
                col = file_line[1:].strip('\n').split('\t')
                continue
            words = file_line.strip('\n').split('\t')
            row = dict(zip(col, words))

            mut = row['HGVSp']
            if 'pathogenicity' in col:
                activity = float(row['pathogenicity'])
            else:
                activity = default_act
            if mut in mut2act:
                count_dup += 1
            mut2act[mut] = activity
    logging.info("The number of duplication of the same mutations in the data = %i", count_dup)
    return mut2act


def load_np2seq(seq_csv_path: str, np_ids=None) -> dict:
    '''
    retrive a dictionary {np_id:seq} from csv_file
    :seq_csv_file {str} : the file path for sequence csv file : line = "np_id,seq"
    :np_ids {list} : a list of ids if selection of specific ids is necessary, else none
    :return {dict} : {np_id:seq}
    '''
    with open(seq_csv_path) as seq_file:
        np2seq = dict()
        for file_line in seq_file:
            if file_line[0] == '#':
                headers = file_line[1:].strip('\n').split(',')
                continue
            words = file_line.strip('\n').split(',')
            data = dict(zip(headers, words))
            if np_ids is not None and data['transcipt_id'] not in np_ids:
                continue
            np2seq[data['transcipt_id']] = data['sequence']
        logging.info("the number of allowed transcripts = %i", len(np2seq))
    return np2seq


def get_seq_input(np2seq: dict, mut_list: list, seq_len: int, use_seq_block: bool) -> tuple:
    '''
    build a numpy array encoding seqence data for the mutation list
    :np2seq {dict} : {np:seq}
    :mut_list {list} : list of mutation in HGVSp terms
    :seq_len {int} : the length for numpy array
    :use_seq_block {bool} : whether to use sequences around mutation residue or not
    :return {array} : 2D array for integer representation of seqences
    '''
    count = {
        'np_not_found': 0,
        'unexpected_mut': 0,
        'featurize_failure': 0,
        'incorrect residue': 0,
        'length_limit': 0,
    }
    ref_data = []
    seq_data = []
    mut_ids = []
    for mut_idx, mut in enumerate(mut_list):
        if mut_idx % 10000 == 0:
            print(mut_idx)
        np_id, mut_code = mut.split(':')
        if np_id not in np2seq:  # np_id not in sequence dictionary
            count['np_not_found'] += 1
            continue
        seq = np2seq[np_id]
        # mut_tup = (mut_res, mut_ori, mut_aft)
        mut_tup = fd.mut2tup(mut_code)
        if not mut_tup:  # unexpected mutation format
            count['unexpected_mut'] += 1
            continue
        if not fd.check_res(mut_code, seq):
            count['incorrect residue'] += 1
            continue
        if not use_seq_block and seq_len < len(seq):
            count['length_limit'] += 1
            continue

        # calling the 4th argument as integer means reference sequence (amino acid not changed)
        ref_data.append(fd.seq2input(seq, seq_len, use_seq_block, mut_tup[0]))
        seq_data.append(fd.seq2input(seq, seq_len, use_seq_block, mut_tup))
        mut_ids.append(mut)
    logging.info("the number of mutation np_id not found = %i", count['np_not_found'])
    logging.info(
        "the number of unexpectedly formatted mutations = %i", count['unexpected_mut'])
    logging.info(
        "the number of mutations failed to be featurized = %i", count['featurize_failure'])
    logging.info(
        "the number of mutations with incorrect residues = %i", count['incorrect residue'])
    logging.info(
        "the number of mutations exceed the maximum length = %i", count['length_limit'])
    logging.info(
        "the number of mutations transformed into input data = %i", len(mut_ids))
    input_data = (np.array(ref_data), np.array(seq_data))
    return input_data, np.array(mut_ids)


def get_patho_input(mut2act: dict, mut_ids: list, threshold: float, reverse=False) -> np.array:
    '''
    get labels for pathogenicity from mut2act dictionary and the list of target mutations
    :mut2act {dict}: a mut2act dictonary generated from load_mut2act function
    :mut_ids {list}: a list of mutations of which pathogenicity will be examined
    :threshold {float}: the threshold of activity to determine pathogenicity of the mutation
    :rev {bool}: True if positive set has larger values than theshold
    '''
    count = {
        'pos': 0,
        'neg': 0,
    }
    patho_np = []
    for mut in mut_ids:
        activity = mut2act[mut]
        pos_flag = activity > threshold
        if reverse:
            pos_flag = not pos_flag

        if pos_flag:
            patho_np.append([1, 0])  # pathogenic
            count['pos'] += 1
        else:
            patho_np.append([0, 1])  # benign
            count['neg'] += 1
    logging.info("the number of positive data = %i", count['pos'])
    logging.info("the number of negative data = %i", count['neg'])
    return np.array(patho_np)


def get_site_np(mut_ids: list, seq_len: int, use_seq_block: bool) -> np.array:
    '''
    return site_np from mutation id
    :mut_ids {list}: list of mutation codes="NP_000143.2:p.Phe181Leu"
    :seq_len {int}: length of sequence array
    :use_seq_block {bool}: whether to use sequence block (True) or whole sequence (False)
    :return {np.array}: N * 2 array containging start residue and end residue of data
    '''
    site_np = []
    for mut in mut_ids:
        mut_code = mut.split(':')[1]
        mut_tup = fd.mut2tup(mut_code)
        mut_res = mut_tup[0]
        if use_seq_block:
            start_res = mut_res - int(seq_len/2)
            end_res = mut_res + int(seq_len/2)
        else:
            start_res = 1
            end_res = seq_len
        res_range = np.array([start_res, end_res])
        site_np.append(res_range)
    return np.array(site_np)


def save_dataset(dataset_path: str, input_seq: tuple, patho_np=None, site_np=None):
    '''
    save dataset file to the path concatenating i
    nput_seq and pathonp
    :dataset_path {str}: the path to save the data set
    :input_seq {tup}: a tuple contaning ref_np and seq_np
    :patho_np {numpy array}: a numpy array for pathogenicity. None if not known
    '''
    ref_np, seq_np = input_seq
    if patho_np is None:  # pathogenicity is unknown
        patho_np = np.array([[0, 0]] * len(ref_np), dtype=int)
    dataset = (ref_np, seq_np, patho_np)
    if site_np is not None:
        dataset = (ref_np, seq_np, patho_np, site_np)
    with open(dataset_path, 'wb') as set_file:
        pk.dump(dataset, set_file, protocol=4)


def main():
    '''
    main function
    '''
    logging.basicConfig(filename='build_dataset.log',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    logging.info('=== dataset building ===')

    # dataset options
    seq_len = 201
    use_seq_block = True
    mut_threshold = 0.5

    # input files
    mut_path = "../data/variant_data/common_variants.txt" # path of data file having variants.
    refseq_path = "../data/transcript_seq.csv" # path for sequences of Refseq transcripts

    # output files
    dataset_path = "../data/sequence_data/common_dataset.bin" # path to save dataset file
    mut_id_path = "../data/sequence_data/common_mut.npy" # path to save the variant ID file

    mut2act = load_mut2act(mut_path)
    np2seq = load_np2seq(refseq_path)
    input_data, mut_ids = get_seq_input(np2seq, list(mut2act.keys()), seq_len, use_seq_block)
    input_patho = get_patho_input(mut2act, mut_ids, mut_threshold)
    site_np = get_site_np(mut_ids, seq_len, use_seq_block)
    save_dataset(dataset_path, input_data, patho_np=input_patho, site_np=site_np)
    np.save(mut_id_path, mut_ids)


if __name__ == '__main__':
    main()

'''
A module training deep learning model to predict pathogenicity from given samples
'''
import argparse
import logging
import pickle as pk
import os
import sys
import numpy as np
import yaml
import LSTM_models as pm
import deep_utilities as du


### Data processors

def data_subset(dataset: tuple, sample_idx: np.array) -> tuple:
    '''
    retrieve subset of dataset from sample indexes
    :dataset {tuple}: (refer_seq.npy, input_seq.npy, output_path.npy)
    :sample_idx {numpy array}: indexes for dataset to be retrieved
                              if indexes are duplicated, the data are included multiple times
    :return {tuple}: subset data of dataset (same format with dataset)
    '''
    sub_dataset = []
    for data in dataset:
        sub_dataset.append(data[sample_idx])
    return tuple(sub_dataset)


def attach_data(dataset: tuple, novel_set: tuple) -> tuple:
    '''
    attach novel data to the original data set
    :dataset {tuple}: the original data set
    :novel_set {tuple}: the data set to be attached
    :return {tuple}: the updated data set
    '''
    union_dataset = []
    for ori_data, new_data in zip(dataset, novel_set):
        uni_data = np.concatenate((ori_data, new_data), axis=0)
        union_dataset.append(uni_data)
    return tuple(union_dataset)


def get_dataset(data_path: str, id_path: str) -> tuple:
    '''
    get data set and mutation id from data file
    :data_path {string}: the path for data file
    :id_path {string}: the path for mutation id (numpy array) file
    :return {tuple, numpy array}: (dataset, mut_ids)
      dataset: (ref_seq, alt_seq, pathogenicity). Each element is a numpy array
      mut_ids: mutation ids corresponding to the samples in the data (same order)
    '''
    mut_ids = np.load(id_path)
    np_ids = np.array([mut.split(':')[0] for mut in mut_ids])
    with open(data_path, 'rb') as data_file:
        dataset = pk.load(data_file)
        dataset = (dataset[0], dataset[1], dataset[2], dataset[3], np_ids)
    logging.info("the number of samples in the dataset = %i", len(mut_ids))
    return dataset, mut_ids


def attach_snvbox(dataset, snvbox_path):
    '''
    attach snvbox features at the end of dataset tuple
    for example, rapsi format -> rapsix format where 'x' stands for snvbox features
    '''
    if snvbox_path[-4:] == 'none':
        sample_len = len(dataset[0])
        snvbox_feature = np.zeros((sample_len, 85))
    else:
        snvbox_feature = np.load(snvbox_path)

    if len(dataset[0]) != len(snvbox_feature):
        print("the length of snvbox feature is different with dataset")
        sys.exit()
    extended_dataset = tuple(list(dataset) + [snvbox_feature])
    return extended_dataset


def remove_vus(dataset, mut_ids):
    '''
    remove the samples having ambiguous pathogenicity
    if mut_ids == [], do not select mutation ids
    '''
    patho_np = dataset[2]
    non_vus_idx = np.array([idx for idx, patho in enumerate(patho_np) if np.sum(patho) == 1])
    # patho_np = [[1, 0], [0, 1], ...]. np.sum(patho)==1 means benign or pathogenic sample
    dataset = data_subset(dataset, non_vus_idx)
    mut_ids = mut_ids[non_vus_idx]
    logging.info("the number of samples after removing VUS = %i", len(mut_ids))
    return dataset, mut_ids


def split_sample(sample_array: np.array, mut_ids=np.array([]), test_mut=np.array([]), ratio=0.8):
    '''
    get sample representations (usually indexes for dataset) of train set and test set.
    if test_mut is not offered (empty), the test set is selected randomly according to ratio
    :sample_array {numpy array}: an array representing samples (ex. indexes for dataset)
    :mut_ids {np.array}: the list of mutations ordered as same as sample_array
    :test_mut {np.array}: list of mutations to be used as test set.
      the rest of data become train set.
    :ratio {float}: the fraction of samples to be selected as test set.
      used only if test_mut is empty.
    return {numpy array, numpy array}: sample arrays for train and test set
    '''
    if test_mut.size != 0:
        # external test mutations are offered
        test_idx = np.array([idx for idx, mut in enumerate(mut_ids) if mut in test_mut])
        train_array = np.delete(sample_array, test_idx)
        test_array = sample_array[test_idx]
    else:
        # for random sampling
        data_size = len(sample_array)
        boundary = int(data_size * ratio)
        sample_idx = np.array(range(data_size))
        np.random.shuffle(sample_idx)
        train_idx, test_idx = sample_idx[0:boundary], sample_idx[boundary:]
        train_array, test_array = sample_array[train_idx], sample_array[test_idx]
    logging.info("the number of samples designated as training set = %i", len(train_array))
    logging.info("the number of samples designated as test set = %i", len(test_array))
    return (train_array, test_array)


def get_pathogenicity(dataset: tuple, sample_idx: np.array) -> np.array:
    '''
    get a representation  for pathogenicity in the form of numpy array
    where 1 indicates pathogenic and 0 indicates benign
    :dataset {tuple}: dataset to retrieve pathogenicity
    :sample_idx {numpy array}: the sample indexes of the dataset to retrieve the pathogenicity
    :return {numpy array}: a numpy array having 1 for pathogenic samples and 0 for benign samples
    '''
    patho_array = dataset[2][sample_idx]
    # patho_array[:,0] = the 1st elements of patho array. ex. [[1, 0], [0, 1], ...] -> [1, 0, ...]
    return patho_array[:, 0]


def amplify_pos(sample_array: np.array, pos_array: np.array, neg_add=0):
    '''
    amplify positive data for balancing (cannot be applied when pos > neg)
    :sample_array {numpy array}: the array representing samples before amplifying
    :pos_array {numpy array}: the array representing positiveness of sample_array
      (1: positive, 0: negative)
      the order of sample_array and pos_array should be same
    :neg_add {int}: the number of negative data to be added if they exist
    :return {numpy array}: the sample array of which positive data are amplified
    '''
    pos_sample = np.array([samp for samp, patho in zip(sample_array, pos_array) if patho == 1])
    pos_count, neg_count = len(pos_sample), len(sample_array)-len(pos_sample)+neg_add
    n_fold = int(neg_count / pos_count)
    if n_fold > 0:
        add_pos = np.repeat(pos_sample, n_fold-1)
        sample_array = np.concatenate((sample_array, add_pos))
    log_str = f"The number of pos_data = {pos_count}, neg_data = {neg_count}, fold = {n_fold}"
    logging.info(log_str)
    return sample_array


### Constructors
def build_directory(log_dir: str):
    '''
    build the directory to save model and other variables
    exit if the same model name already exists
    '''
    try:
        if os.path.exists(log_dir): # prevent overwriting previous model
            print(f"{log_dir} is already exists. Stopping the process")
            sys.exit()
        os.makedirs(log_dir)
    except OSError:
        print(f"Error: Creating directory: {log_dir}")


def save_config(config_path: str, log_dir: str):
    '''
    save config file into the model directory with usage manual
    '''
    with open(config_path) as conf_file:
        contents = conf_file.read()
    with open(f"{log_dir}/config.yaml", 'w') as save_file:
        save_file.write(f"# python train_model.py -c {log_dir}/config.yaml\n")
        save_file.write(f"# python test_model.py -c {log_dir}/config.yaml\n")
        save_file.write(contents)


### model trainers
class SingleTrainer():
    '''
    model trainer for a single Mark model
    '''
    def __init__(self, config: dict):
        self.config = config
        self.data_dir = config['DATA_DIR']
        self.model_type = config['MODEL']['MODEL_TYPE']
        self.log_dir = os.path.join(self.data_dir,
                                    config['MODEL']['MODEL_DIR'],
                                    config['MODEL']['MODEL_TYPE'],
                                    config['MODEL']['MODEL_NAME'])
        self.dataset = tuple()
        self.data_mut = np.array([])
        self.add_set = tuple()
        self.add_mut = np.array([])

        self.dataset, self.data_mut = get_dataset(
            os.path.join(self.data_dir, config['MODEL']['DATA_PATH']),
            os.path.join(self.data_dir, config['MODEL']['DATA_ID_PATH']),
        )

        if get_data_format(self.model_type) == 'rapsix':
            self.dataset = attach_snvbox(
                self.dataset,
                os.path.join(self.data_dir, config['MODEL']['DATA_SNVBOX']),
            )

        self.dataset, self.data_mut = remove_vus(self.dataset, self.data_mut)

        if config['MODEL']['ADD_PATH'] != 'none':
            self.add_set, self.add_mut = get_dataset(
                os.path.join(self.data_dir, config['MODEL']['ADD_PATH']),
                os.path.join(self.data_dir, config['MODEL']['ADD_ID_PATH']),
            )

            if get_data_format(self.model_type) == 'rapsix':
                self.add_set = attach_snvbox(
                    self.add_set,
                    os.path.join(self.data_dir, config['MODEL']['ADD_SNVBOX']),
                )

            self.add_set, self.add_mut = remove_vus(self.add_set, self.add_mut)


    def set_train_test(self) -> tuple:
        '''
        get index for train set and test set
        '''
        config = self.config
        data_idx = np.array(range(len(self.data_mut))) # index array for dataset
        test_ids = np.array([])

        if 'TEST_ID_PATH' in config['MODEL']:
            test_ids = np.load(
                os.path.join(self.data_dir, config['MODEL']['TEST_ID_PATH'])
            )

        if test_ids.size != 0: # use given test_ids as test set and the other become train set
            train_idx, test_idx = split_sample(data_idx, mut_ids=self.data_mut,
                                               test_mut=test_ids)
        else: # random sampling
            if 'TEST_RATIO' in config['MODEL']:
                test_ratio = config['MODEL']['TEST_RATIO'][0]
            else:
                test_ratio = 0.2
            train_idx, test_idx = split_sample(data_idx, ratio=1-test_ratio)
            test_ids = self.data_mut[test_idx]

        np.save(
            os.path.join(self.log_dir, 'test_ids.npy'),
            test_ids
        )

        train_idx, test_idx = split_sample(train_idx, ratio = 0.99)
        train_idx = amplify_pos(train_idx, get_pathogenicity(self.dataset, train_idx),
                                neg_add=len(self.add_mut))

        return train_idx, test_idx

    def train_model(self):
        '''
        build and test model
        '''
        train_idx, test_idx = self.set_train_test()
        train_set = data_subset(self.dataset, train_idx)
        test_set = data_subset(self.dataset, test_idx)
        if self.add_mut.size > 0:
            train_set = attach_data(train_set, self.add_set)

        self.config['log_dir'] = self.log_dir
        classifier = get_model(self.model_type)(self.config)
        classifier.build()
        classifier.train(train_set, test_set)


class EnsembleTrainer(SingleTrainer):
    '''
    model trainer for Mark models using both training data and ensemble data
    '''
    def __init__(self, config: dict):
        super().__init__(config)
        self.ensemble_set, _ = get_dataset(
            os.path.join(self.data_dir, config['MODEL']['ENSEMBLE_PATH']),
            os.path.join(self.data_dir, config['MODEL']['ENSEMBLE_ID_PATH']),
        )

        if get_data_format(self.model_type) == 'rapsix':
            self.ensemble_set = attach_snvbox(
                self.ensemble_set,
                os.path.join(self.data_dir, config['MODEL']['ENSEMBLE_SNVBOX']),
            )

    def train_model(self):
        train_idx, test_idx = self.set_train_test()
        train_set = data_subset(self.dataset, train_idx)
        test_set = data_subset(self.dataset, test_idx)
        if self.add_mut.size != 0:
            train_set = attach_data(train_set, self.add_set)

        self.config['log_dir'] = self.log_dir
        classifier = get_model(self.model_type)(self.config)
        classifier.build()
        classifier.ensemble_train(train_set, test_set, self.ensemble_set)


### Controllers
def get_model(model_type: str) -> 'class':
    '''
    get a corresponding model object according to the given model type
    '''
    model_dict = {
        'SingleTask': pm.SingleTask,
        'MultiTask': pm.MultiTask,
        '3Cnet': pm.ThreeCnet,
    }
    return model_dict[model_type]


def get_data_format(model_type: str) -> str:
    '''
    get a corresponding data format according to the given model type
    '''
    data_dict = {
        'SingleTask': 'rapsi',
        'MultiTask': 'rapsi',
        '3Cnet': 'rapsix',
    }
    return data_dict[model_type]


def get_train_type(model_type: str) -> 'class':
    '''
    get a corresponding model trainers for the given model type
    '''
    train_dict = {
        'SingleTask': SingleTrainer,
        'MultiTask': EnsembleTrainer,
        '3Cnet': EnsembleTrainer,
    }
    return train_dict[model_type]


### main function

def main():
    '''
    main function
    '''
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--config', default='config.yaml', help='path for config file')
    parser.add_argument('-t', '--title', default='code', help='title for this process (optional)')
    args = parser.parse_args()
    logging.basicConfig(filename='train_model.py.log',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

    with open(args.config) as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)
    logging.info(f"=== start {args.title} ===")

    ### build necessary directories for the model
    log_dir = os.path.join(config['DATA_DIR'],
                           config['MODEL']['MODEL_DIR'],
                           config['MODEL']['MODEL_TYPE'],
                           config['MODEL']['MODEL_NAME'])
    build_directory(log_dir)
    save_config(args.config, log_dir)

    ### train the model
    trainer = get_train_type(config['MODEL']['MODEL_TYPE'])(config)
    trainer.train_model()

    logging.info(f"... end {args.title} ...")


if __name__ == '__main__':
    main()

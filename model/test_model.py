'''
A module to test the model and get performances
python test_model.py -n (model_num) (-c config.yaml)
'''

import argparse
import logging
import yaml
import numpy as np
import train_model as tr
import deep_utilities as du


class SingleTester():
    '''
    Validation for a single model
    '''
    def __init__(self, config: dict):
        self.config = config
        self.model_num = config['VALID']['MODEL_NUM'][0]
        self.test_metrics = config['VALID']['VALID_METRICS']
        self.log_dir = f"{config['MODEL']['MODEL_DIR']}/{config['MODEL']['MODEL_TYPE']}/" +\
                    config['MODEL']['MODEL_NAME']
        self.model = None

        self.ext_dataset, self.ext_mut = \
            tr.get_dataset(config['VALID']['VALID_DATA_PATH'], config['VALID']['VALID_ID_PATH'])

        if tr.get_data_format(config['MODEL']['MODEL_TYPE']) == 'rapsix':
            self.ext_dataset = tr.attach_snvbox(self.ext_dataset, config['VALID']['VALID_SNVBOX'])

        self.ext_dataset, self.ext_mut = tr.remove_vus(self.ext_dataset, self.ext_mut)

    def model_build(self):
        '''
        build model
        '''
        self.config['log_dir'] = self.log_dir
        self.model = tr.get_model(self.config['MODEL']['MODEL_TYPE'])(self.config)
        self.model.build()

    def get_test_set(self):
        '''
        return test set and corresponding ids for the test set
        '''
        config = self.config
        if config['VALID']['USE_TEST_ID'][0]: # when using samples of test_ids among the dataset
            test_ids = np.load(f"{config['log_dir']}/test_ids.npy")
            ext_idx = np.array(range(len(self.ext_mut)))
            _, test_idx = tr.split_sample(ext_idx, mut_ids=self.ext_mut, test_mut=test_ids)
            test_set = tr.data_subset(self.ext_dataset, test_idx)
            test_ids = self.ext_mut[test_idx]
        else: # using the whole dataset
            test_set = self.ext_dataset
            test_ids = self.ext_mut
        return test_set, test_ids

    def get_test_pred(self, model_idx: int, test_set: tuple, test_ids: np.array) -> np.array:
        '''
        get prediction result for test set
        :model_idx {int}: epoch number of the model to be used
        '''
        config = self.config
        valid_name = config['VALID']['VALID_NAME']
        self.model.load(model_idx)
        test_pred = self.model.predict(test_set)
        np.save(f"{config['log_dir']}/{valid_name}_{model_idx}.npy", test_pred)
        with open(f"{config['log_dir']}/{valid_name}_{model_idx}_ours.txt", 'w') as score_file:
            for mut, ans, pred in zip(test_ids, test_set[2][:, 0], test_pred[:, 0]):
                score_file.write(f"{mut},{ans},{pred}\n")
        return test_pred

    def save_performance(self, test_set: np.array, test_pred: np.array, test_metrics: list, \
                         model_idx: int):
        '''
        save validation performance to a file
        '''
        result_list = []
        valid_name = self.config['VALID']['VALID_NAME']
        if test_metrics: # skip when test_metrics == []
            result_list = self.model.evaluate(test_set[2], test_pred, metrics=test_metrics, \
                out_path=f"{self.config['log_dir']}/{valid_name}_{model_idx}.txt")
        return result_list

    def test_model(self):
        '''
        test model
        '''
        self.model_build()
        test_set, test_ids = self.get_test_set()
        test_pred = self.get_test_pred(self.model_num, test_set, test_ids)
        self.save_performance(test_set, test_pred, self.test_metrics, self.model_num)


def get_test_type(test_type: str) -> 'class':
    '''
    get a corresponding model trainers for the given model type
    '''
    test_dict = {
        'single': SingleTester,
    }
    return test_dict[test_type]


def main():
    '''
    main function
    '''
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--config', default='config.yaml', help='path for config file')
    parser.add_argument('-t', '--title', default='code', help='title of this job')
    args = parser.parse_args()
    logging.basicConfig(filename='test_model.log',
                       format='%(asctime)s %(levelname)-8s %(message)s',
                       level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    with open(args.config) as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)

    logging.info(f"=== start {args.title} ===")

    tester = get_test_type(config['VALID']['VALID_TYPE'])(config)
    tester.test_model()

    logging.info(f"... end {args.title} ...")


if __name__ == '__main__':
    main()

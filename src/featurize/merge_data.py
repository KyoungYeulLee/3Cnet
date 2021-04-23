import os
import argparse
import logging
import yaml
import numpy as np
import pickle as pk


def main(config):
    '''
    main function
    '''
    data_dir = config['DATA_DIR']
    merged_data_path = os.path.join(data_dir, config['MERGE']['MERGED_DATA'])
    merged_mut_path = os.path.join(data_dir, config['MERGE']['MERGED_MUT'])
    merged_snv_path = os.path.join(data_dir, config['MERGE']['MERGED_SNV'])

    mut_np = np.array([])
    snv_np = np.array([])
    dataset = ()
    for data_path, mut_path, snv_path in zip(config['MERGE']['DATA_PATH_LIST'],
                                             config['MERGE']['MUT_PATH_LIST'],
                                             config['MERGE']['SNV_PATH_LIST']):
        # merged mut id
        new_mut = np.load(os.path.join(data_dir, mut_path))
        mut_np = np.concatenate((mut_np, new_mut))

        # merge snvbox features
        if snv_path != 'none':
            new_snv = np.load(os.path.join(data_dir, snv_path))
        elif snv_path == 'none':
            sample_len = len(new_mut)
            new_snv = np.zeros((sample_len, 85))

        if len(snv_np) == 0:
            snv_np = new_snv
        else:
            snv_np = np.concatenate((snv_np, new_snv))

        # merge dataset
        with open(os.path.join(data_dir, data_path), 'rb') as data_file:
            new_dataset = pk.load(data_file)

        if not dataset:
            dataset = new_dataset
        else:
            merged_dataset = []
            for pre_data, new_data in zip(dataset, new_dataset):
                merged_data = np.concatenate((pre_data, new_data))
                merged_dataset.append(merged_data)
            dataset = tuple(merged_dataset)
            del merged_dataset

    with open(merged_data_path, 'wb') as merged_file:
        pk.dump(dataset, merged_file)
    np.save(merged_mut_path, mut_np)
    np.save(merged_snv_path, snv_np)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--config', default='config.yaml', type=str, help='path for config file')
    parser.add_argument('-t', '--title', default='code', type=str, help='title of this job')
    args = parser.parse_args()
    logging.basicConfig(filename='merge_data.py.log',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    with open(args.config) as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)

    logging.info(f"=== start {args.title} ===")

    main(config)

    logging.info(f"... end {args.title} ...")

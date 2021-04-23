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
    dataset_path = os.path.join(data_dir, config['VARIANT']['DATASET_PATH'])
    mut_path = os.path.join(data_dir, config['VARIANT']['MUT_PATH'])
    variant_path = os.path.join(data_dir, config['VARIANT']['VARIANT_PATH'])

    with open(dataset_path, 'rb') as data_file:
        dataset = pk.load(data_file)

    patho_np = dataset[2]
    mut_np = np.load(mut_path)

    with open(variant_path, 'w') as var_file:
        var_file.write("#HGVSp\tpathogenicity\n")
        for mut_id, patho in zip(mut_np, patho_np):
            if patho[0] + patho[1] != 1:
                continue
            var_file.write(f"{mut_id}\t{patho[0]}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--config', default='config.yaml', type=str, help='path for config file')
    parser.add_argument('-t', '--title', default='code', type=str, help='title of this job')
    args = parser.parse_args()
    logging.basicConfig(filename='get_variants.py.log',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    with open(args.config) as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)

    logging.info(f"=== start {args.title} ===")

    main(config)

    logging.info(f"... end {args.title} ...")

# Installation
---
3Cnet was trained uses the following versions of software:
- Python 3.7
- CUDA 10.0
- PyTorch 1.4
- nvidia driver version 410.48
- Ubuntu 18.04

## Preprocess: Docker
Follow the steps in this section if you prefer a ready-to-go environment.
If you prefer to set up the environment on your own, skip directly to "Clone the 3Cnet repository."

We recommend you have at least 40GB of free storage.

### Install Docker and nvidia-docker2
**Docker Engine** (we use Docker 20.10.5)
https://docs.docker.com/engine/install/

**nvidia-docker2**

`sudo apt-get update`

`sudo apt-get install -y nvidia-docker2`

### Pull the 3Cnet Docker image from Docker Hub
The Docker image for 3Cnet is based on one of NVIDIA NGC's offerings.

`sudo docker pull 3cnet-docker`

### Run docker image interactively

`sudo docker run --gpus all -it -v </absolute/path/to/mount>:/workspace 3cnet-docker`

See https://ngc.nvidia.com/catalog/containers/nvidia:pytorch for other execution examples.

## Clone the 3Cnet repository

`git clone https://github.com/KyoungYeulLee/3Cnet.git`

## Run `download_data.py` to retrieve necessary files from Zenodo

`cd 3Cnet`

`python download_data.py`

# Code excecution (continuing from data download)

1. To train 3Cnet

`cd model`

`python train_model.py`

1. To evaluate 3Cnet performance

`python test_model.py`

1. To re-create the training/evaluation datasets

`cd ../src`

`python build_dataset.py`

# Data and files deep-dive

- 3Cnet/src: 
- 3Cnet/data
- 3Cnet/model

1. File List (data): 
        A. transcript_ids.txt: transcript ids of RefSeq human genome data
        B. transcript_seq.csv: transcript sequences of RefSeq human genome data
        C. msa_data.tar.gz: NP_*.npy files representing each residues of conservative proportion of 20-amino acids
        D. variant_data
                i. clinvar_data.txt: labeled variants of pathogenic and benign from ClinVar
                ii. common_variants.txt: labeled variants of benign from gnomAD
                iii. conservation_data.txt: labeled variants of pathogenic-like and benign-like derived from conservative
                        information
        E. validataion_result
                i. external_clinvar_data.txt: external clinvar variants data for validation in which 9 insilico predictive scores
                        are (3Cnet, REVEL, VEST4, SIFT, PolyPhen2, PrimateAI, CADD, FATHMM, DAN)
                ii. patient_data.txt: inhouse patients variants data for validation in which 3 insilico predictive scores are
                        (3Cnet, REVEL, PrimateAI)
        F. SNVBOX_features
                i. clinvar_features.npy: tabular data set for features from SNVBox for each ClinVar variants
                ii. common_features.npy: tabular data set for features from SNVBox for each gnomAD variants
                iii. conservation_features.npy: tabular data set for features from SNVBox for each variants from conservation
                        information


2. File List (src)
        A. build_dataset.py: pre-processing raw data
        B. featurize_data.py: featurize data to use as deep learning model input


3. File List
A. config.yaml: configuration file
B. deep_utilities.py: utilities for model training
c. evaluate_metrics.py: evaluation metrics from scikit-learn
D. train_model.py: main function for training process
E. pt_models.py: pytorch models to train
F. my_networks.py: pytorch networks to train
G. my_datasets.py: pytorch datasets for training data
H. test_model.py : main function for evaluation process

#
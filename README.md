﻿# Installation

__3Cnet was trained uses the following versions of software:__
- Python 3.7
- CUDA 10.0
- PyTorch 1.4
- nvidia driver version 410.48
- Ubuntu 18.04

<br>

## Preprocess: Docker
<br>

Follow the steps in this section if you prefer a ready-to-go environment.
If you prefer to set up the environment on your own, skip directly to "Clone the 3Cnet repository."

We recommend you have at least 40GB of free storage.

<br>

### __Install Docker and nvidia-docker2__

<ins>Docker Engine</ins> (we use Docker 20.10.5)

https://docs.docker.com/engine/install/


<ins>nvidia-docker2</ins>

```bash
$ sudo apt-get update
$ sudo apt-get install -y nvidia-docker2
```
<br>

### __Pull the 3Cnet Docker image from Docker Hub__
The Docker image for 3Cnet is based on one of NVIDIA NGC's offerings.

```bash
$ sudo docker pull 3cnet-docker
```

### __Run docker image interactively__

```bash
$ sudo docker run --gpus all -it -v </absolute/path/to/mount>:/workspace 3cnet-docker
```

See https://ngc.nvidia.com/catalog/containers/nvidia:pytorch for other execution examples.

<br>

## Clone the 3Cnet repository

```bash
$ git clone https://github.com/KyoungYeulLee/3Cnet.git
```

## Run `download_data.py` to retrieve necessary files from Zenodo

```bash
$ cd 3Cnet
$ python download_data.py
$ tar -xvf data.tar.gz
```

# Code excecution (continuing from data download)

1. To train 3Cnet

```bash
$ cd ./src/model
$ python train_model.py
```

2. To evaluate 3Cnet performance

```bash
$ python test_model.py
```

3. To re-create the training/evaluation datasets

```bash
$ cd ../featurize
$ python build_dataset.py
```

# Data and files deep-dive
<ins>Underlined</ins> files are the top-level scripts intended to be directly modified or executed by the user.


1. src/featurize
   - <ins>build_dataset.py</ins>: Run this to parse and process raw data into pytorch-compatible inputs.
   - featurize_data.py: A dependency used by `build_dataset.py` that converts HGVSp nomenclature to amino acid sequences.
  
2. src/model
   - <ins>config.yaml</ins>: A file specifying paths and hyperparameters used for model training or evaluation. Alter this to modify file paths and/or settings.
   - deep_utilities.py: A generic collection of utility functions and classes used broadly by other files in src/model.
   - LSTM_datasets.py: Dataset class definition for 3Cnet.
   - LSTM_models.py: A wrapper class for 3Cnet defining low-level training routines.
   - LSTM_networks.py: The 3Cnet architecture is defined here (nn.Module).
   - evaluate_metrics.py: Definition of metrics used for model evaluation.
   - <ins>train_model.py</ins>: Top-level script for 3Cnet training. Outcomes are saved in `data/model/(MODEL_TYPE)/(MODEL_NAME)`.
   - <ins>test_model.py</ins>: Evaluate model using values under the `VALID` key in `config.yaml`. The `MODEL_NUM` key represents the epoch # to use and must be defined for this script to run as intended.

3. data/ 
   
   - msa_data/: NP_*.npy files representing each residues of conservative proportion of 20-amino acids
   - sequence_data/: Contains output generated by <ins>`build_dataset.py`</ins>. (required to train 3Cnet)
     - *.bin files: files containing amino acid sequences
     - *.npy files: files containing HGVSp identifiers
   - SNVBOX_features/: SNVBOX-generated feature vectors. Files in this folder correspond to those in `sequence_data/`.
     - clinvar_features.npy: tabular data set for features from SNVBox for each ClinVar variants
     - common_features.npy: tabular data set for features from SNVBox for each gnomAD variants
     - conservation_features.npy: tabular data set for features from SNVBox for each variants from conservation information
   - validation_result/: Contains data pertaining to the external clinvar test set and patient data test results.
     - external_clinvar_data.txt: external clinvar variants data for validation in which 9 insilico predictive scores are listed (3Cnet, REVEL, VEST4, SIFT, PolyPhen2, PrimateAI, CADD, FATHMM, DANN)
     - patient_data.txt: inhouse patients variants data for validation in which 3 insilico predictive scores are (3Cnet, REVEL, PrimateAI)
   - variant_data/
     - clinvar_data.txt: pathogenic-or-benign-labeled  variants from ClinVar
     - common_variants.txt: benign-labeled variants from gnomAD
     - conservation_data.txt: pathogenic-like and benign-like variants inferred from conservation data
   - transcript_ids.txt: transcript ids of RefSeq human genome data
   - transcript_seq.csv: transcript sequences of RefSeq human genome data

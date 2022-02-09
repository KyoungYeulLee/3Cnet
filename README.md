# Update Log
## Feb 7, 2022
- Corrected an error in SNVBOX feature files that led to decreased performance
  - Please see: https://zenodo.org/record/6016720

# Installation

__3Cnet was trained using the following versions of software:__
- [Python v3.7](https://www.python.org)
- [CUDA v10.0](https://developer.nvidia.com/cuda-toolkit)
- [PyTorch v1.4](https://pytorch.org)
- [NVIDIA driver version v410.48](https://www.nvidia.com/Download/index.aspx)
- [Ubuntu v18.04](https://ubuntu.com)

<br>

### STEP 1: Preprocess: Docker
- Follow the steps in this section if you prefer a ready-to-go environment.
- If you prefer to set up the environment on your own, skip directly to "Clone the 3Cnet repository."
- We recommend you have at least 40GB of free storage.
- See [NVIDIA NGS pytorch container docs](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) for other execution examples.
#### __Install Docker and nvidia-docker2__
<ins>Docker Engine</ins> (we use Docker 20.10.5)
> https://docs.docker.com/engine/install/
<ins>NVIDIA/nvidia-docker2</ins>
```bash
$ sudo apt-get update
$ sudo apt-get install -y nvidia-docker2
```
#### __Pull the 3Cnet Docker image from Docker Hub__
- The Docker image for 3Cnet is based on one of NVIDIA NGC's offerings.
```bash
$ sudo docker pull 3billion/3cnet:0.0.1
```
#### __Run docker image interactively__
```bash
$ sudo docker run --gpus all -it -v </absolute/path/to/mount>:/workspace 3billion/3cnet:0.0.1
$ cd workspace
```

<br>

### STEP 2: Clone the 3Cnet repository

```bash
$ git clone https://github.com/KyoungYeulLee/3Cnet.git
```

<br>

### STEP 3: Run `download_data.py` to retrieve necessary files from [Zenodo](https://zenodo.org)

```bash
$ cd 3Cnet
$ python download_data.py
$ tar -xvf data.tar.gz
```

<br>

# Code execution (continuing from data download)

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

<br>

# Data and file structures
<ins>Underlined</ins> files are the top-level files or scripts intended to be directly modified or executed by the user.

- <ins>download_data.py</ins>: Retrieves `data/` directory from Zenodo.
- 3cnet.yaml: Anaconda-compatible environment yaml. (deprecated, also contains dependencies not directly used by 3cnet)

### **src/featurize**
   - <ins>build_dataset.py</ins>: Run this to parse and process raw data into pytorch-compatible inputs.
   - <ins>config.yaml</ins>: A file specifying paths used by scripts in `src/featurize`
   - featurize_data.py: A dependency used by `build_dataset.py` that converts HGVSp nomenclature to amino acid sequences.
   - get_variants.py: Used to map sequences in `/data/variant_data` to back HGVSp identifiers.
   - merge_data.py: Script that bundles the two different sequence data with SNVbox features. (generates `*_dataset.bin`, `*_mut.npy`, and `*_snvbox.npy` files)
  
### **src/model**
   - <ins>config.yaml</ins>: A file specifying paths and hyperparameters used for model training or evaluation. Alter this to modify file paths and/or settings.
   - deep_utilities.py: A generic collection of utility functions and classes used broadly by other files in src/model.
   - LSTM_datasets.py: Dataset class definition for 3Cnet.
   - LSTM_models.py: A wrapper class for 3Cnet defining low-level training routines.
   - LSTM_networks.py: The 3Cnet architecture is defined here (nn.Module).
   - evaluate_metrics.py: Definition of metrics used for model evaluation.
   - <ins>train_model.py</ins>: Top-level script for 3Cnet training. Outcomes are saved in `data/model/(MODEL_TYPE)/(MODEL_NAME)`.
   - <ins>test_model.py</ins>: Evaluate model using values under the `VALID` key in `config.yaml`. The `MODEL_NUM` key in `config.yaml` represents the epoch # to load and must be defined for this script to run as intended.

### **data/**
   - msa_data/: NP_*.npy files representing each residues of conservative proportion of 21-amino acids

   - variant_data/
     - clinvar_data.txt: pathogenic-or-benign-labeled  variants from ClinVar
     - common_variants.txt: benign-labeled variants from gnomAD
     - conservation_data.txt: pathogenic-like and benign-like variants inferred from conservation data
     - truncated_variants.txt: ClinVar(ver. 2020.04) variants of the following consequences - `start lost, stop gained, deletion, frameshift`
     - nonsynonymous_variants.txt: Entries from `truncated_variants.txt` plus missense variants (also ver. 2020.04)
     - external_missense_variants.txt: Missense variants found in ClinVar 2020.08 but not in ClinVar 2020.04
     - external_truncated_variants.txt: Start lost, stop gained, deletion, and frameshift variants found in ClinVar 2020.08 but not in ClinVar 2020.04
     - external_nonsyn_variants.txt: File that combines `external_missense_variants.txt` and `external_truncated_variants.txt`
     - patient_variants.txt: Collection of disease-causing and non-causal variants from 111 patients (variant duplicates removed)

   - sequence_data/: Contains output generated by <ins>`build_dataset.py`</ins>. (required to train 3Cnet)
     - *_dataset.bin files: files containing amino acid sequences
     - *_mut.npy files: files containing HGVSp identifiers

   - SNVBOX_features/: SNVBOX-generated feature vectors. Files in this folder correspond to those in `sequence_data/`.
     - *_snvbox.npy: Tabular features of variants generated by SNVBox

   - validation_result/: Contains data pertaining to the external clinvar test set and patient data test results.
     - external_clinvar_missense.tsv: Variants from `external_missense_variants.txt` and their scores generated by various algorithms (3Cnet, REVEL, VEST4, SIFT, PolyPhen2, PrimateAI, CADD, FATHMM, DANN)
     - external_clinvar_nonsynonymous.tsv: Variants from `external_nonsyn_variants.txt` and their scores generated by various algorithms
     - external_clinvar_truncated.tsv: Variants from `external_truncated_variants.txt` and their scores generated by various algorithms
     - patient_all_scores.tsv: Variants from `patient_variants.txt` and their scores generated by various algorithms
     - patient_3scores.tsv: inhouse patients variants data for validation in which 3 insilico predictive scores are (3Cnet, REVEL, PrimateAI)

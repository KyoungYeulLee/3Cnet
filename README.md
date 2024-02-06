# Update Log

## Feb 7, 2024
- Released 3Cnet version 2.0
  - Please see: https://zenodo.org/records/10212255
  - Major changes
    - 3Cnet v2 is no longer dependent to SNVBOX features.
    - Almost all types of in-exon variants can be inferred (see neuron/constants.py).
    - Better performance compared to 3Cnet v1 (ROC-AUC = 91% -> 93% for external clinvar).
## Feb 7, 2022
- Corrected an error in SNVBOX feature files that led to decreased performance
  - Please see: https://zenodo.org/record/6016720
## May 7, 2021
- Initial release of 3Cnet

# Installation

__3Cnet ver.2 was trained using the following versions of software:__
- [Python v3.8](https://www.python.org)
- [CUDA v11.1](https://developer.nvidia.com/cuda-toolkit)
- [PyTorch v1.9.1](https://pytorch.org)
- [NVIDIA driver version v525.60.13](https://www.nvidia.com/Download/index.aspx)
- [Ubuntu18.04](https://ubuntu.com)

We recommend you have at least 40GB of free storage.
<br>

## STEP 1: Clone the 3Cnet repository

```bash
$ git clone https://github.com/KyoungYeulLee/3Cnet.git
```
<br>

## STEP 2: Set up environment
We assume that you are running our model on one or more NVIDIA GPUs.

### Option 1: Use Docker (recommended)

#### __Install Docker and nvidia-container-toolkit__
<ins>Docker Engine</ins> (we use Docker 20.10.9)
> https://docs.docker.com/engine/install/

<ins>NVIDIA/container-toolkit (to use NVIDIA GPUs)</ins>
> https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

<ins>If you don't usually have access to root/sudo, consider Docker Rootless</ins>
> https://docs.docker.com/engine/security/rootless/

#### __Build the 3Cnet Docker image__
```bash
$ sudo docker build -t 3billion/3cnet:v2.0.0 .
```
#### __Run docker image interactively__
```bash
$ sudo docker run --gpus all -it -v $(pwd):/workspace 3billion/3cnet:v2.0.0 bash
$ cd workspace
```

### Option 2: Install using pip

```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```
<br>

## STEP 3: Run `download_data.py` to retrieve necessary files from [Zenodo](https://zenodo.org)

(uses [requests](https://requests.readthedocs.io/en/latest/) and [tqdm](https://tqdm.github.io/))
```bash
$ python download_data.py
```
<br>

# TODO


# Code execution (continuing from data download)

1. To train 3Cnet

```bash
$ python model_trainer.py -s model_name
```

2. To evaluate 3Cnet performance

```bash
$ python model_evaluator.py -m model_name -e 30 -s test_result
```
Note that you need to select a proper epoch number (30 in the example)


<br>

# Data and file structures

- download_data.py: Retrieves `data/` directory from Zenodo.
- model_trainer.py: Top-level script for 3Cnet training. Outcome includes model parameters, training log, config backup
- model_evaluator.py: Evaluate model using trained model parameters. The test result will be saved in the model dir (pred.tsv)
- omegaconf.yaml: Anaconda-compatible environment yaml.

### **neuron**
   - aa_to_int_mappings.py: mapping between amino-acid string to integer representation.
   - constants.py: definition of variants used in this project.
   - errors.py: definition of errors.
   - seq_database.py: Script that parse sequence information from the data.
   - seq_collection.py: Script that define the collection of sequence objects.
   - sequences.py: Script that define the sequence object.
   - featurizer.py: Script that featurize sequence object into trainable features.
   - utils.py: Utility script.
  
### **cccnet**
   - dataset_builder.py: Class that build pytorch dataset from HGVSp written files
   - torch_dataset.py: Dataset class definition for 3Cnet.
   - torch_network.py: The 3Cnet architecture is defined here (nn.Module).
   - deep_utils.py: Utility script for deep learning.
   - utils.py: Utility script.

### **data**
   - reference_sequences.tsv: the file containing sequence ID and its amino-acid sequence.
   - msa_arrays/: NP_*.npy files representing each residues of conservative proportion of 21-amino acids

   - train_hgvsps/
     - train_clinvar_hgvsps.tsv: pathogenic-or-benign-labeled  variants from ClinVar
     - train_gnomad_hgvsps.tsv: benign-labeled variants from gnomAD
     - train_conservation_hgvsps.tsv: pathogenic-like and benign-like variants inferred from conservation data

   - test_hgvsps/: Contains data pertaining to the external clinvar test set and patient data test results.
     - test_clinvar_missense_hgvsps.tsv: Variants from external clinvar (missense variants)
     - test_clinvar_non-missense_hgvsps.tsv: Variants from external clinvar (non-missense variants)
     - test_inhouse_hgvsps.tsv: inhouse patients variants (missense variants)

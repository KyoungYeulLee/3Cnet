DATA & FILE OVERVIEW


3cnet.yaml: conda environment file


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
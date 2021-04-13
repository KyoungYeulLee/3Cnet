'''
A module containing pytorch dataset (torch.utils.data inherited)
'''

import numpy as np
import torch
import torch.utils.data as td


USE_CUDA = torch.cuda.is_available
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


class ToTensor():
    '''
    A transformer to change numpy sample to tensor sample
    '''
    def __call__(self, sample: dict) -> dict:
        '''
        Return tensor transformed sample.
        '''
        sample_tensor = {}
        for key, value in sample.items():
            if isinstance(value[0], str):
                sample_tensor[key] = value
                continue
            sample_tensor[key] = torch.from_numpy(value).to(DEVICE)
        return sample_tensor


class MutSpecDataset(td.Dataset):
    '''
    A dataset for pathogen/benign data with rapsi format
    '''
    def __init__(self, dataset: tuple, transform=ToTensor()):
        '''
        initialization
        # mut_id: 1 dim np.array having NP_ID
        # res_range: 2 dim(N*2) np.array indicating start and end point
        '''
        super().__init__()
        self.ref_array = dataset[0].astype(np.int64)
        self.alt_array = dataset[1].astype(np.int64)
        self.patho_array = dataset[2].astype(np.float32)
        self.res_range = dataset[3].astype(np.int64)
        self.mut_ids = dataset[4]

        self.transform = transform

    def __len__(self) -> int:
        '''
        get length of sample
        '''
        return len(self.ref_array)

    def __getitem__(self, idx) -> dict:
        '''
        get a sample at a specific index
        :idx {integer | list | tensor}
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            'ref_seq': self.ref_array[idx], # shape = (batch_size, n_seq)
            'alt_seq': self.alt_array[idx], # shape = (batch_size, n_seq)
            'patho': self.patho_array[idx], # shape = (batch_size, n_classes)
            'res_range': self.res_range[idx], # shape = (batch_size, 2)
            'mut_ids': self.mut_ids[idx], # shape = (batch_size,), dtype='str'
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

class MutSNVDataset(td.Dataset):
    '''
    A dataset for pathogen/benign data with rapsix format
    '''
    def __init__(self, dataset: tuple, transform=ToTensor()):
        '''
        initialization
        '''
        super().__init__()
        self.ref_array = dataset[0].astype(np.int64)
        self.alt_array = dataset[1].astype(np.int64)
        self.patho_array = dataset[2].astype(np.float32)
        self.res_range = dataset[3].astype(np.int64)
        self.mut_ids = dataset[4]
        self.snv_feature = dataset[5].astype(np.float32)

        self.transform = transform

    def __len__(self) -> int:
        '''
        get length of sample
        '''
        return len(self.ref_array)

    def __getitem__(self, idx) -> dict:
        '''
        get a sample at a specific index
        :idx {integer | list | tensor}
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            'ref_seq': self.ref_array[idx], # shape = (batch_size, n_seq)
            'alt_seq': self.alt_array[idx], # shape = (batch_size, n_seq)
            'patho': self.patho_array[idx], # shape = (batch_size, n_classes)
            'res_range': self.res_range[idx], # shape = (batch_size, 2)
            'mut_ids': self.mut_ids[idx], # shape = (batch_size,), dtype='str'
            'snv_feature': self.snv_feature[idx], # shape = (barch_size, snv_feat_num)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class MSADataset(td.Dataset):
    '''
    A dataset having MSA matrix
    '''
    def __init__(self, msa_mat: np.array, transform=ToTensor()):
        '''
        initialization
        '''
        super().__init__()
        self.msa_mat = msa_mat.astype(np.float32) # shape = (seq_length, n_aa)
        self.transform = transform

    def __len__(self) -> int:
        '''
        get length of sample (seq_length)
        '''
        return len(self.msa_mat)

    def __getitem__(self, idx) -> dict:
        '''
        get a sample at a specific index
        :idx {integer | list | tensor}
        CAUTION: This dataset returns a single sliced MSA for a sample
        idx represents the residual positions of Amino-acids, not sample index
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'msa_slice': self.msa_mat[idx]} # shape = (n_seq, n_aa)

        if self.transform:
            sample = self.transform(sample)

        return sample

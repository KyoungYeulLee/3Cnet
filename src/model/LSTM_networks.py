'''
A module containing pytorch networks (torch.nn.module inherited)
'''

import torch
import torch.nn.functional as F
tensor = torch.tensor


class EmblemNet(torch.nn.Module):
    '''
    network to combine sequence and MSA pattern into flat features using LSTM
    '''
    # pylint: disable=too-many-instance-attributes
    def __init__(self, *args):
        '''
        initialization
        '''
        super().__init__()
        n_vocab, embed_dim, hidden_dim, output_dim, n_aa, dropout_p = args
        hidden_dim = int(hidden_dim / 2) # to compensate double sized output dimension of bi-LSTM
        self.embed = torch.nn.Embedding(n_vocab, embed_dim, padding_idx=0)
        self.ref_bi_lstm = \
            torch.nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.alt_bi_lstm = \
            torch.nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.msa_bi_lstm = torch.nn.LSTM(n_aa, hidden_dim, batch_first=True, bidirectional=True)
        self.ref_lstm = torch.nn.LSTM(hidden_dim * 4, output_dim, batch_first=True)
        self.alt_lstm = torch.nn.LSTM(hidden_dim * 4, output_dim, batch_first=True)
        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, ref_tensor: tensor, alt_tensor: tensor, msa_tensor: tensor) -> tensor:
        '''
        ref_tensor: shape=(batch_size, n_seq)
        alt_tensor: shape=(batch_size, n_seq)
        msa_tensor: shpae=(batch_size, n_seq, n_aa)
        return: shape=(batch_size, ouput_dim * 2)
        '''
        x_ref = self.embed(ref_tensor) # (batch_size, n_seq, embed_dim)
        x_alt = self.embed(alt_tensor) # (batch_size, n_seq, embed_dim)
        x_ref, _ = self.ref_bi_lstm(x_ref) # (batch_size, n_seq, hidden_dim)
        x_alt, _ = self.alt_bi_lstm(x_alt) # (batch_size, n_seq, hidden_dim)
        x_msa, _ = self.msa_bi_lstm(msa_tensor) # (batch_size, n_seq, hidden_dim)
        x_ref = torch.cat((x_ref, x_msa), dim=2) # (batch_size, n_seq, hidden_dim * 2)
        x_alt = torch.cat((x_alt, x_msa), dim=2) # (batch_size, n_seq, hidden_dim * 2)
        x_ref, _ = self.ref_lstm(x_ref) # (batch_size, n_seq, output_dim)
        x_alt, _ = self.alt_lstm(x_alt) # (batch_size, n_seq, output_dim)
        x_ref = x_ref[:, -1, :] # shape = (batch_size, output_dim)
        x_alt = x_alt[:, -1, :] # shape = (batch_size, output_dim)
        x_ref = self.dropout(x_ref)
        x_alt = self.dropout(x_alt)
        x_tensor = torch.cat((x_ref, x_alt), dim=1) # shape = (batch_size, output_dim * 2)
        return x_tensor


class SingleTask(torch.nn.Module):
    '''
    network architecture for single-task learning
    '''
    def __init__(self, *args):
        '''
        initialization
        '''
        super().__init__()
        n_vocab, embed_dim, hidden_dim, output_dim, fc_size, n_classes, n_aa, dropout_p = args
        self.common_net = EmblemNet(n_vocab, embed_dim, hidden_dim, output_dim, n_aa, dropout_p)
        self.fc_layer = torch.nn.Linear(output_dim * 2, fc_size)
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.classifier = torch.nn.Linear(fc_size, n_classes)

    def forward(self, ref_tensor: tensor, alt_tensor: tensor, msa_tensor: tensor) -> tensor:
        '''
        ref_tensor: shape=(batch_size, n_seq)
        alt_tensor: shape=(batch_size, n_seq)
        msa_tensor: shpae=(batch_size, n_seq, n_aa)
        return: shape=(batch_size, n_classes)
        '''
        x_tensor = self.common_net(ref_tensor, alt_tensor, msa_tensor)
        x_tensor = self.fc_layer(x_tensor) # shape = (batch_size, fc_size)
        x_tensor = self.dropout(x_tensor)
        x_tensor = self.classifier(x_tensor) # shape = (batch_size, n_classes)
        output_tensor = F.softmax(x_tensor, dim=1)
        return output_tensor


class MultiTask(torch.nn.Module):
    '''
    network architecture for multi-task learning
    '''
    def __init__(self, *args):
        '''
        initialization
        '''
        super().__init__()
        n_vocab, embed_dim, hidden_dim, output_dim, fc_size, n_classes, n_aa, dropout_p = args
        self.common_net = EmblemNet(n_vocab, embed_dim, hidden_dim, output_dim, n_aa, dropout_p)
        self.clinvar_fc = torch.nn.Linear(output_dim * 2, fc_size)
        self.toxicity_fc = torch.nn.Linear(output_dim * 2, fc_size)
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.clinvar_cls = torch.nn.Linear(fc_size, n_classes)
        self.toxicity_cls = torch.nn.Linear(fc_size, n_classes)

    def forward(self, ref_tensor: tensor, alt_tensor: tensor, msa_tensor: tensor) -> tuple:
        '''
        ref_tensor: shape=(batch_size, n_seq)
        alt_tensor: shape=(batch_size, n_seq)
        msa_tensor: shpae=(batch_size, n_seq, n_aa)
        return {tuple}: (clinvar_tensor, toxicity_tensor), shape=(batch_size, n_classes)
        '''
        x_tensor = self.common_net(ref_tensor, alt_tensor, msa_tensor)
        # shape = (batch_size, output_dim * 2)
        clinv_tensor = self.clinvar_fc(x_tensor) # shape = (batch_size, fc_size)
        clinv_tensor = self.dropout(clinv_tensor)
        clinv_tensor = self.clinvar_cls(clinv_tensor) # shape = (batch_size, n_classes)
        clinv_tensor = F.softmax(clinv_tensor, dim=1)
        tox_tensor = self.toxicity_fc(x_tensor) # shape = (batch_size, fc_size)
        tox_tensor = self.dropout(tox_tensor)
        tox_tensor = self.toxicity_cls(tox_tensor) # shape = (batch_size, n_classes)
        tox_tensor = F.softmax(tox_tensor, dim=1)
        return clinv_tensor, tox_tensor


class ThreeCnet(torch.nn.Module):
    '''
    network architecture for 3Cnet (multi-task learning + SNVBox features)
    '''
    def __init__(self, *args):
        '''
        initialization
        '''
        super().__init__()
        n_vocab, embed_dim, hidden_dim, output_dim, fc_size, n_classes, n_aa, dropout_p, \
            snv_size, snv_fc = args
        self.common_net = EmblemNet(n_vocab, embed_dim, hidden_dim, output_dim, n_aa, dropout_p)
        self.clinvar_fc = torch.nn.Linear(output_dim * 2, fc_size)
        self.toxicity_fc = torch.nn.Linear(output_dim * 2, fc_size)
        self.snv_fc = torch.nn.Linear(snv_size, snv_fc)
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.clinvar_cls = torch.nn.Linear(fc_size + snv_fc, n_classes)
        self.toxicity_cls = torch.nn.Linear(fc_size + snv_fc, n_classes)

    def forward(self, ref_tensor: tensor, alt_tensor: tensor, msa_tensor: tensor, \
                snv_tensor: tensor) -> tuple:
        '''
        ref_tensor: shape=(batch_size, n_seq)
        alt_tensor: shape=(batch_size, n_seq)
        msa_tensor: shpae=(batch_size, n_seq, n_aa)
        snv_tensor: shpae=(batch_size, snz_size)
        return {tuple}: (clinvar_tensor, toxicity_tensor), shape=(batch_size, n_classes)
        '''
        x_tensor = self.common_net(ref_tensor, alt_tensor, msa_tensor)
        # shape = (batch_size, output_dim * 2)
        snv_tensor = self.snv_fc(snv_tensor) # shape = (batch_size, snv_fc)
        snv_tensor = self.sigmoid(snv_tensor)
        snv_tensor = self.dropout(snv_tensor) # shape = (batch_size, snv_fc)

        clinv_tensor = self.clinvar_fc(x_tensor) # shape = (batch_size, fc_size)
        clinv_tensor = self.sigmoid(clinv_tensor)
        clinv_tensor = self.dropout(clinv_tensor) # shape = (batch_size, fc_size)
        clinv_tensor = torch.cat((clinv_tensor, snv_tensor), dim=1)
        # (batch_size, fc_size + snv_fc)
        clinv_tensor = self.clinvar_cls(clinv_tensor) # shape = (batch_size, n_classes)
        clinv_tensor = F.softmax(clinv_tensor, dim=1) # shape = (batch_size, n_classes)

        tox_tensor = self.toxicity_fc(x_tensor) # shape = (batch_size, fc_size)
        tox_tensor = self.sigmoid(tox_tensor)
        tox_tensor = self.dropout(tox_tensor) # shape = (batch_size, fc_size)
        tox_tensor = torch.cat((tox_tensor, snv_tensor), dim=1) # (batch_size, fc_size + snv_fc)
        tox_tensor = self.toxicity_cls(tox_tensor) # shape = (batch_size, n_classes)
        tox_tensor = F.softmax(tox_tensor, dim=1) # shape = (batch_size, n_classes)

        return clinv_tensor, tox_tensor

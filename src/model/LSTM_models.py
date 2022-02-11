'''
A module containing pytorch models for pathogenicity predictor
'''
import os
import sys
import timeit
import numpy as np
import torch
import torch.utils.data as td
import evaluate_metrics as em
import deep_utilities as du
import LSTM_datasets as md
import LSTM_networks as mn


USE_CUDA = torch.cuda.is_available
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
tensor = torch.tensor


class SingleTask():
    '''
    the pytorch version of Mark0 (result should be same)
    '''
    def __init__(self, config: dict):
        '''
        initialization
        '''
        hyper_caller = du.HyperCaller(config['HYPERS'])
        hypers = hyper_caller.hyper_calling()
        hypers['log_dir'] = config['log_dir']

        self.data_dir = config['DATA_DIR']
        self.n_vocab = 22
        self.n_aa = self.n_vocab - 1
        self.n_classes = 2
        self.hypers = hypers
        self.model = None
        self.save_each_epoch = config['MODEL']['SAVE_EACH_EPOCH'][0]
        self.msa_dict = {}

        with open(
            os.path.join(
                self.data_dir,
                config['MODEL']['TID_PATH'],
            )
        ) as np_file:
            for file_line in np_file:
                words = file_line.strip('\n').split('\t')
                np_id = words[0]
                msa_mat = np.load(
                    os.path.join(
                        self.data_dir,
                        config['MODEL']['MSA_DIR'],
                        f"{np_id}.npy",
                    )
                )
                self.msa_dict[np_id] = md.MSADataset(msa_mat)

    def check_model(self):
        '''
        check whether model is built or not. if not, the process stops
        '''
        if self.model is None:
            print("build the model first using build()")
            sys.exit()

    def build(self):
        '''
        built the model object
        '''
        hypers = self.hypers
        args = (self.n_vocab, hypers['emb_size'], hypers['rnn_size'], hypers['rnn_size'],
                hypers['fc_size'], self.n_classes, self.n_aa, hypers['dropout'])
        self.model = mn.SingleTask(*args).to(DEVICE)

    def get_msa_tensor(self, np_ids: np.array, res_range: np.array) -> tensor:
        '''
        retrieve a tensor for the MSA batch given transcript ids and residue ranges
        '''
        msa_stack = []
        for np_id, (start_res, end_res) in zip(np_ids, res_range):
            msa_dataset = self.msa_dict[np_id]
            front_pad = tensor([], dtype=torch.float32, device=DEVICE)
            end_pad = tensor([], dtype=torch.float32, device=DEVICE)
            msa_slice = tensor([], dtype=torch.float32, device=DEVICE)

            if start_res < 1:
                front_pad = tensor([[0.0] * self.n_aa] * (1-start_res),
                                   dtype=torch.float32, device=DEVICE)
                start_res = 1

            if end_res > len(msa_dataset):
                if start_res > len(msa_dataset):
                    pad_length = end_res - start_res + 1
                    end_res = start_res - 1
                else:
                    pad_length = end_res - len(msa_dataset) # len(msa_dataset) == final_res + 1
                    end_res = len(msa_dataset)
                end_pad = tensor([[0.0] * self.n_aa] * pad_length,
                                 dtype=torch.float32, device=DEVICE)

            if start_res - 1 < end_res:
                msa_slice = msa_dataset[start_res-1:end_res]['msa_slice']

            msa_slice = torch.cat((front_pad, msa_slice, end_pad), dim=0) # shape = (n_seq, n_aa)
            msa_stack.append(msa_slice)
        msa_stack = torch.stack(msa_stack, dim=0) # shape = (batch, n_seq, n_aa)
        return msa_stack

    def train_epoch(self, train_loader: object, criterion: 'func', optimizer: 'func'):
        '''
        model training for an epoch
        '''
        self.check_model()
        self.model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            ref_seq, alt_seq, patho_ans = batch['ref_seq'], batch['alt_seq'], batch['patho']
            mut_ids, res_range = batch['mut_ids'], batch['res_range']
            msa_tensor = self.get_msa_tensor(mut_ids, res_range)
            patho_train = self.model(ref_seq, alt_seq, msa_tensor)
            train_loss = criterion(patho_train, patho_ans)
            train_loss.backward()
            optimizer.step()

    def evaluate_epoch(self, test_loader: object, criterion: 'func') -> tuple:
        '''
        model testing for a test set
        '''
        self.check_model()
        self.model.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for batch in test_loader:
                ref_seq, alt_seq, patho_ans = batch['ref_seq'], batch['alt_seq'], batch['patho']
                msa_tensor = self.get_msa_tensor(batch['mut_ids'], batch['res_range'])
                patho_test = self.model(ref_seq, alt_seq, msa_tensor)
                test_loss += criterion(patho_test, patho_ans, reduction='sum').item()
                pred = patho_test.max(dim=1, keepdim=False)[1] # shape = (batch_size,)
                ans = patho_ans.max(dim=1, keepdim=False)[1] # shape = (batch_size,)
                correct += pred.eq(ans).sum().item()
        test_loss /= len(test_loader.dataset) * self.n_classes # N dimensional answer tensor
        accuracy = correct / len(test_loader.dataset)
        return test_loss, accuracy

    def train(self, train_set: tuple, test_set: tuple):
        '''
        The model training process
        '''
        self.check_model()
        hypers = self.hypers
        log_file = open(hypers['log_dir']+'/log.txt', 'w', 1)
        start = timeit.default_timer()
        early_stopping = du.EarlyStopper(patience=hypers['patience'], log_file=log_file)
        optimizer = hypers['optimizer'](self.model.parameters(), lr=hypers['learning_rate'])
        train_loader = td.DataLoader(md.MutSpecDataset(train_set), batch_size=hypers['batch_size'],
                                     shuffle=True)
        test_loader = td.DataLoader(md.MutSpecDataset(test_set), batch_size=hypers['batch_size'],
                                    shuffle=False)
        for epoch in range(hypers['epoch']):
            self.train_epoch(train_loader, hypers['value'], optimizer)
            train_loss = self.evaluate_epoch(train_loader, hypers['value'])
            test_loss = self.evaluate_epoch(test_loader, hypers['value'])
            result = f"epoch = {epoch}, train loss = {train_loss}, test loss = {test_loss}"
            du.print_and_log(result, log_file)
            if self.save_each_epoch:
                torch.save(self.model.state_dict(), f"{hypers['log_dir']}/model_{epoch}.pt")
            early_stopping(test_loss[0], self.model, save_dir=hypers['log_dir'])
            if early_stopping.early_stop:
                du.print_and_log("...Early stopping...", log_file)
                break
        message = f"The time taken for a iterative training was {timeit.default_timer() - start} s"
        du.print_and_log(message, log_file)
        log_file.close()

    def load(self, model_num):
        '''
        Load model parameters from a file
        the file placed in the log_dir should have the name in the format of "model_(model_num).pt"
        '''
        self.check_model()
        model_path = f"{self.hypers['log_dir']}/model_{model_num}.pt"
        self.model.load_state_dict(torch.load(model_path))

    def predict(self, test_set):
        self.check_model()
        self.model.eval()
        hypers = self.hypers
        test_loader = td.DataLoader(md.MutSpecDataset(test_set), batch_size=hypers['batch_size'],
                                    shuffle=False)
        pred_stack = []
        for batch in test_loader:
            ref_seq, alt_seq = batch['ref_seq'], batch['alt_seq']
            msa_tensor = self.get_msa_tensor(batch['mut_ids'], batch['res_range'])

            with torch.no_grad():
                pred = self.model(ref_seq, alt_seq, msa_tensor).detach().cpu().numpy()
            pred_stack.append(pred)

        return np.concatenate(pred_stack, axis=0)

    def evaluate(self, y_test, y_pred, metrics=None, out_path=''):
        '''
        evaluate the performance of the model comparing the label and prediciton
        :y_test {numpy array}: an array containing labels of the test set which is 1 or 0
        :y_pred {numpy array}: an array containing predicion of the test set ranging from 0 to 1
        :metrics {list}: performance metrics to evaluate. if None, only accuray is calculate
        :out_path {str}: the path for the file to write down the performances. if empty not saved
        :return {list}: the performance list for metrics
        '''
        self.check_model()
        if metrics is None:
            metrics = ['accuracy']
        answer, score = y_test[:, 0], y_pred[:, 0]
        met_dict = {
            'accuracy': em.accuracy,
            'roc_auc': em.roc_auc,
            'pr_auc': em.pr_auc,
        }
        result_list = []
        for met in metrics:
            result = met_dict[met](answer, score)
            result_list.append(result)
        if out_path:
            with open(out_path, 'w') as result_file:
                for met, result in zip(metrics, result_list):
                    result_file.write(f"{met} = {result}\n")
        return result_list


class MultiTask(SingleTask):
    '''
    epoch defined for training set not ensemble set
    '''
    def build(self):
        '''
        built the model object
        '''
        hypers = self.hypers
        args = (self.n_vocab, hypers['emb_size'], hypers['rnn_size'], hypers['rnn_size'],
                hypers['fc_size'], self.n_classes, self.n_aa, hypers['dropout'])
        self.model = mn.MultiTask(*args).to(DEVICE)

    def evaluate_epoch(self, test_loader: object, criterion: 'func') -> tuple:
        '''
        model testing for a test set
        '''
        self.check_model()
        self.model.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for batch in test_loader:
                ref_seq, alt_seq, patho_ans = batch['ref_seq'], batch['alt_seq'], batch['patho']
                msa_tensor = self.get_msa_tensor(batch['mut_ids'], batch['res_range'])
                patho_test, _ = self.model(ref_seq, alt_seq, msa_tensor)
                test_loss += criterion(patho_test, patho_ans, reduction='sum').item()
                pred = patho_test.max(dim=1, keepdim=False)[1] # shape = (batch_size,)
                ans = patho_ans.max(dim=1, keepdim=False)[1] # shape = (batch_size,)
                correct += pred.eq(ans).sum().item()
        test_loss /= len(test_loader.dataset) * self.n_classes # N dimensional answer tensor
        accuracy = correct / len(test_loader.dataset)
        return test_loss, accuracy

    def optimize_step(self, batch, criterion, optimizer, is_train=True):
        optimizer.zero_grad()
        ref_seq, alt_seq, patho_ans = batch['ref_seq'], batch['alt_seq'], batch['patho']
        mut_ids, res_range = batch['mut_ids'], batch['res_range']
        msa_tensor = self.get_msa_tensor(mut_ids, res_range)
        patho_train, patho_ensemble = self.model(ref_seq, alt_seq, msa_tensor)
        if is_train:
            train_loss = criterion(patho_train, patho_ans)
        else:
            train_loss = criterion(patho_ensemble, patho_ans)
        train_loss.backward()
        optimizer.step()

    def ensemble_train(self, train_set: tuple, test_set: tuple, ensemble_set: tuple):
        '''
        ensemble training
        '''
        self.check_model()
        hypers = self.hypers
        log_file = open(hypers['log_dir']+'/log.txt', 'w', 1)
        start = timeit.default_timer()
        early_stopping = du.EarlyStopper(patience=hypers['patience'], log_file=log_file)
        optimizer = hypers['optimizer'](self.model.parameters(), lr=hypers['learning_rate'])
        train_loader = td.DataLoader(md.MutSpecDataset(train_set), batch_size=hypers['batch_size'],
                                     shuffle=True)
        test_loader = td.DataLoader(md.MutSpecDataset(test_set), batch_size=hypers['batch_size'],
                                    shuffle=False)
        ensemble_loader = td.DataLoader(md.MutSpecDataset(ensemble_set),
                                        batch_size=hypers['batch_size'], shuffle=True)
        ensemble_iterator = iter(ensemble_loader) # first shuffling

        for epoch in range(hypers['epoch']):
            self.model.train()
            for train_batch in train_loader:
                try: # get the next ensemble batch
                    ensemble_batch = next(ensemble_iterator)
                except StopIteration: # shuffle only when all the ensemble data were used
                    ensemble_iterator = iter(ensemble_loader)
                    ensemble_batch = next(ensemble_iterator)
                self.optimize_step(ensemble_batch, hypers['value'], optimizer, is_train=False)
                self.optimize_step(train_batch, hypers['value'], optimizer, is_train=True)

            train_loss = self.evaluate_epoch(train_loader, hypers['value'])
            test_loss = self.evaluate_epoch(test_loader, hypers['value'])
            result = f"epoch = {epoch}, train loss = {train_loss}, test loss = {test_loss}"
            du.print_and_log(result, log_file)
            if self.save_each_epoch:
                torch.save(self.model.state_dict(), f"{hypers['log_dir']}/model_{epoch}.pt")
            early_stopping(test_loss[0], self.model, save_dir=hypers['log_dir'])
            if early_stopping.early_stop:
                du.print_and_log("...Early stopping...", log_file)
                break
        message = f"The time taken for a iterative training was {timeit.default_timer() - start} s"
        du.print_and_log(message, log_file)
        log_file.close()

    def get_pred(self, pr_tr, pr_en, eval_type):
        '''
        get prediction for a sample using different types of evaluation criteria
        '''
        self.check_model() # to avoid pylint R0201 (no-self-use) error
        if eval_type == 'train':
            pred = pr_tr

        elif eval_type == 'ensem':
            pred = pr_en

        elif eval_type == 'max':
            if pr_tr[0] > pr_en[0]:
                pred = pr_tr
            else:
                pred = pr_en

        elif eval_type == 'average':
            pred = (pr_tr + pr_en) / 2

        return pred

    def predict(self, test_set):
        self.check_model()
        self.model.eval()
        hypers = self.hypers
        test_loader = td.DataLoader(md.MutSpecDataset(test_set), batch_size=hypers['batch_size'],
                                    shuffle=False)
        pred_stack = []
        for batch in test_loader:
            ref_seq, alt_seq = batch['ref_seq'], batch['alt_seq']
            msa_tensor = self.get_msa_tensor(batch['mut_ids'], batch['res_range'])
            with torch.no_grad():
                pred_train, pred_ensemble = \
                    self.model(ref_seq, alt_seq, msa_tensor) # (batch_size, n_classes)
    
            pred_train = pred_train.detach().cpu().numpy()
            pred_ensemble = pred_ensemble.detach().cpu().numpy()
            for pr_tr, pr_en in zip(pred_train, pred_ensemble):
                pred = self.get_pred(pr_tr, pr_en, eval_type='train')
                pred_stack.append(pred)
        return np.array(pred_stack, dtype=np.float32)


class ThreeCnet(MultiTask):
    '''
    3Cnet
    '''
    def __init__(self, config: dict):
        '''
        Initialization
        '''
        super().__init__(config)
        self.snv_size = 85

    def build(self):
        '''
        built the model object
        '''
        hypers = self.hypers
        args = (self.n_vocab, hypers['emb_size'], hypers['rnn_size'], hypers['rnn_size'], \
                hypers['fc_size'], self.n_classes, self.n_aa, hypers['dropout'], \
                self.snv_size, hypers['snv_fc_size'])
        self.model = mn.ThreeCnet(*args).to(DEVICE)

    def optimize_step(self, batch, criterion, optimizer, is_train=True):
        optimizer.zero_grad()
        ref_seq, alt_seq, patho_ans = batch['ref_seq'], batch['alt_seq'], batch['patho']
        mut_ids, res_range = batch['mut_ids'], batch['res_range']
        snv_feature = batch['snv_feature']
        msa_tensor = self.get_msa_tensor(mut_ids, res_range)
        patho_train, patho_ensemble = self.model(ref_seq, alt_seq, msa_tensor, snv_feature)
        if is_train:
            train_loss = criterion(patho_train, patho_ans)
        else:
            train_loss = criterion(patho_ensemble, patho_ans)
        train_loss.backward()
        optimizer.step()

    def evaluate_epoch(self, test_loader: object, criterion: 'func') -> tuple:
        '''
        model testing for a test set
        '''
        self.check_model()
        self.model.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for batch in test_loader:
                ref_seq, alt_seq, patho_ans = batch['ref_seq'], batch['alt_seq'], batch['patho']
                msa_tensor = self.get_msa_tensor(batch['mut_ids'], batch['res_range'])
                snv_feature = batch['snv_feature']
                patho_test, _ = self.model(ref_seq, alt_seq, msa_tensor, snv_feature)
                test_loss += criterion(patho_test, patho_ans, reduction='sum').item()
                pred = patho_test.max(dim=1, keepdim=False)[1] # shape = (batch_size,)
                ans = patho_ans.max(dim=1, keepdim=False)[1] # shape = (batch_size,)
                correct += pred.eq(ans).sum().item()
        test_loss /= len(test_loader.dataset) * self.n_classes # N dimensional answer tensor
        accuracy = correct / len(test_loader.dataset)
        return test_loss, accuracy

    def ensemble_train(self, train_set: tuple, test_set: tuple, ensemble_set: tuple):
        '''
        ensemble training
        '''
        self.check_model()
        hypers = self.hypers
        log_file = open(hypers['log_dir']+'/log.txt', 'w', 1)
        start = timeit.default_timer()
        early_stopping = du.EarlyStopper(patience=hypers['patience'], log_file=log_file)
        optimizer = hypers['optimizer'](self.model.parameters(), lr=hypers['learning_rate'])
        train_loader = td.DataLoader(md.MutSNVDataset(train_set), batch_size=hypers['batch_size'],
                                     shuffle=True)
        test_loader = td.DataLoader(md.MutSNVDataset(test_set), batch_size=hypers['batch_size'],
                                    shuffle=False)
        ensemble_loader = td.DataLoader(md.MutSNVDataset(ensemble_set),
                                        batch_size=hypers['batch_size'], shuffle=True)
        ensemble_iterator = iter(ensemble_loader) # first shuffling

        for epoch in range(hypers['epoch']):
            self.model.train()
            for train_batch in train_loader:
                try: # get the next ensemble batch
                    ensemble_batch = next(ensemble_iterator)
                except StopIteration: # shuffle only when all the ensemble data were used
                    ensemble_iterator = iter(ensemble_loader)
                    ensemble_batch = next(ensemble_iterator)
                self.optimize_step(ensemble_batch, hypers['value'], optimizer, is_train=False)
                self.optimize_step(train_batch, hypers['value'], optimizer, is_train=True)

            train_loss = self.evaluate_epoch(train_loader, hypers['value'])
            test_loss = self.evaluate_epoch(test_loader, hypers['value'])
            result = f"epoch = {epoch}, train loss = {train_loss}, test loss = {test_loss}"
            du.print_and_log(result, log_file)
            if self.save_each_epoch:
                torch.save(self.model.state_dict(), f"{hypers['log_dir']}/model_{epoch}.pt")
            early_stopping(test_loss[0], self.model, save_dir=hypers['log_dir'])
            if early_stopping.early_stop:
                du.print_and_log("...Early stopping...", log_file)
                break
        message = f"The time taken for a iterative training was {timeit.default_timer() - start} s"
        du.print_and_log(message, log_file)
        log_file.close()

    def predict(self, test_set):
        self.check_model()
        self.model.eval()
        hypers = self.hypers
        test_loader = td.DataLoader(md.MutSNVDataset(test_set), batch_size=hypers['batch_size'],
                                    shuffle=False)
        pred_stack = []
        for batch in test_loader:
            ref_seq, alt_seq = batch['ref_seq'], batch['alt_seq']
            msa_tensor = self.get_msa_tensor(batch['mut_ids'], batch['res_range'])
            snv_feature = batch['snv_feature']

            with torch.no_grad():
                pred_train, pred_ensemble = \
                    self.model(ref_seq, alt_seq, msa_tensor, snv_feature) # (batch_size, n_classes)

            pred_train = pred_train.detach().cpu().numpy()
            pred_ensemble = pred_ensemble.detach().cpu().numpy()
            for pr_tr, pr_en in zip(pred_train, pred_ensemble):
                pred = self.get_pred(pr_tr, pr_en, eval_type='train') # eval_type can be changed
                pred_stack.append(pred)
        return np.array(pred_stack, dtype=np.float32)

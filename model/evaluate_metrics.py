'''
A module for evaluation metrics for model evaluation
'''
import numpy as np
import sklearn.metrics as sk


def accuracy(answer: np.array, score: np.array) -> float:
    '''
    calculate accuracy
    '''
    score_bin = np.array(score >= 0.5, dtype=int)
    acc = sk.accuracy_score(answer, score_bin)
    return acc


def roc_auc(answer: np.array, score: np.array) -> float:
    '''
    calculate ROC AUC
    '''
    auc_value = sk.roc_auc_score(answer, score)
    return auc_value


def pr_auc(answer: np.array, score: np.array) -> float:
    '''
    calculate PR AUC
    '''
    precision, recall, _ = sk.precision_recall_curve(answer, score)
    auc_value = sk.auc(recall, precision)
    return auc_value

import numpy as np
import pickle
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix, matthews_corrcoef, hamming_loss
from sklearn.metrics import accuracy_score, average_precision_score,precision_score,f1_score,recall_score

import matplotlib.pyplot as plt
import pandas as pd

def compute_auroc(labels, outputs):
    fpr, tpr, _ = roc_curve(labels, outputs)
    auc_val = auc(fpr, tpr)
    return auc_val, fpr, tpr

def compute_auprc(labels, outputs):
    precision, recall, _ = precision_recall_curve(labels, outputs)
    auc_val = auc(recall, precision)
    return auc_val, precision, recall

def compute_metric(labels, outputs, thres = 0.5):
    labels, outputs = np.array(labels), np.array(outputs)
    tn, fp, fn, tp = confusion_matrix(labels, outputs > thres).ravel()
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    sensitivity = tp / (tp + fp)
    specificity = tn / (tn + fp)
    precision = sensitivity
    recall = tp / (tp+fn)
    f1score = 2 * (precision * recall) / (precision + recall)
    mcc = matthews_corrcoef(labels, outputs > thres)
    fpr, tpr, _ = roc_curve(labels, outputs)
    auc_val = auc(fpr, tpr)
    print(f'{round(accuracy, 4)}\t{round(sensitivity, 4)}\t{round(specificity, 4)}\t{round(mcc, 4)}\t{round(auc_val, 4)}')
    return auc_val, fpr, tpr

all_labels = ['Colon','Breast','Cervix','Lung','Skin','Prostate']
def compute_metric_labelwise(labels, outputs, thres = 0.5, show = True):
#    all_labels = ['Colon']
    for i in range(len(all_labels)):
        label = all_labels[i]
#        print(label)
        labels_i, outputs_i = labels[:, i], outputs[:, i]
        tn, fp, fn, tp = confusion_matrix(labels_i, outputs_i > thres).ravel()
        accuracy = (tp + tn) / (tn + fp + fn + tp + 1e-20)
        sensitivity = tp / (tp + fp + 1e-20)
        specificity = tn / (tn + fp + 1e-20)
        precision = sensitivity
        recall = tp / (tp+fn + + 1e-20)
        f1score = 2 * (precision * recall) / (precision + recall + 1e-20)
        mcc = matthews_corrcoef(labels_i, outputs_i > thres)

        '''
        print(
            'accuracy', round(accuracy*100, 2), 
            'sensitivity', round(sensitivity*100, 2), 
            'specificity', round(specificity*100, 2), 
            'precision', round(precision*100, 2),
            'recall', round(recall*100, 2),
            'f1_score', round(f1score*100, 2)
        )
        '''
        '''
        print(
            '&',
            round(accuracy*100, 2), '&',
            round(sensitivity*100, 2),  '&',
            round(specificity*100, 2),  '&',
            round(f1score*100, 2),
        )
        '''
        auc_val, precision, recall = compute_auroc(labels_i, outputs_i)
        print(f'{round(auc_val, 3)}\t{round(accuracy,3)}\t{round(sensitivity,3)}\t{round(specificity,3)}\t{round(f1score,3)}\t{round(mcc, 3)}')
    
def compute_metric_score(labels, outputs, thres = 0.5, show = True):
    y_true, y_pred = labels, outputs > thres
    '''
    print('------Weighted------')
    print('Weighted precision', precision_score(y_true, y_pred, average='weighted'))
    print('Weighted recall', recall_score(y_true, y_pred, average='weighted'))
    print('Weighted f1-score', f1_score(y_true, y_pred, average='weighted'))
    '''
    accuracy_per_class = [accuracy_score(y_true[:, i], y_pred[:, i]) for i in range(len(all_labels))]
#    print(accuracy_per_class)
    if show:
        print('------Macro------')
    #    print(accuracy_per_class)
        print('Macro accuracy', np.mean(accuracy_per_class))
        print('Macro precision', precision_score(y_true, y_pred, average='macro'))
        print('Macro recall', recall_score(y_true, y_pred, average='macro'))
        print('Macro f1-score', f1_score(y_true, y_pred, average='macro'))
        print('------Micro------')
        print('Micro accuracy', accuracy_score(y_true.ravel(), y_pred.ravel()))
        print('Micro precision', precision_score(y_true, y_pred, average='micro'))
        print('Micro recall', recall_score(y_true, y_pred, average='micro'))
        print('Micro f1-score', f1_score(y_true, y_pred, average='micro'))
        print('------Hamming loss------')
        print('Hamming loss', hamming_loss(y_true, y_pred))

    return np.mean(accuracy_per_class)
import os
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix, matthews_corrcoef, hamming_loss
from sklearn.metrics import accuracy_score, average_precision_score, precision_score,f1_score,recall_score

from util import *

def compute_auroc(labels, outputs):
    fpr, tpr, _ = roc_curve(labels, outputs)
    auc_val = auc(fpr, tpr)
    return auc_val, fpr, tpr

def compute_auprc(labels, outputs):
    precision, recall, _ = precision_recall_curve(labels, outputs)
    auc_val = auc(recall, precision)
    return auc_val, precision, recall

def compute_metric(labels, outputs, thres = 0.5, show_detail = True):
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
    if show_detail:
        print(f'{round(accuracy, 4)}\t{round(sensitivity, 4)}\t{round(specificity, 4)}\t{round(mcc, 4)}\t{round(auc_val, 4)}')
    return auc_val, fpr, tpr

def compute_metric_labelwise(labels, outputs, thres = 0.5, show_detail = True):
    results_all = []
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
        auc_val, precision, recall = compute_auroc(labels_i, outputs_i)
        results_all.append([auc_val, accuracy, sensitivity, specificity, f1score, mcc])
        if show_detail:
            print(f'{round(auc_val, 3)}\t{round(accuracy,3)}\t{round(sensitivity,3)}\t{round(specificity,3)}\t{round(f1score,3)}\t{round(mcc, 3)}')
    results_all = np.array(results_all)
    metric_all = np.mean(results_all, axis = 0)
    auc_val, accuracy, sensitivity, specificity, f1score, mcc = metric_all
    print('Average')
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

def check_one_set(data_dir, pkl_path, show_iter = True, show_length = True, thres=0.5):
    results = sorted(os.listdir(pkl_path))
    num_split = len(results)
    dataset = pd.read_csv(data_dir, sep='\t')
    dataset['seqlen'] = dataset['text'].apply(lambda x: len(x))
    for i in range(num_split):
        pkl_dir = f'{pkl_path}/{results[i]}'
        if not os.path.exists(pkl_dir):
            print('Not existing', pkl_dir)
            continue
        with open(pkl_dir, 'rb') as fr:
            pkl = pickle.load(fr)
            outputs, labels = pkl
            if show_iter:
                compute_metric(labels[:, 1], outputs[:, 1], thres=thres)
        dataset[i] = outputs[:, 1]

    column_names = range(num_split)
    averages = dataset[column_names].mean(axis=1)
    print('Mean')
    auroc_val, fpr, tpr = compute_metric(labels[:, 1], averages, thres=thres)
    
    d_list = [dataset[(i*20 < dataset['seqlen']) & (dataset['seqlen'] <= (i + 1) * 20)] for i in range(3)]

    for idx, d in enumerate(d_list):
        if show_length:
            print(f'Sequence Length from {idx * 20} to {(idx + 1) * 20}')
        if (len(d) == 0):
            continue
        for i in range(num_split):
            labels, outputs = d['label'].tolist(), d[i].tolist()
            if show_iter and show_length:
                compute_metric(labels, outputs)
        averages = d[column_names].mean(axis=1)
        if show_length:
            print('Mean')
            compute_metric(labels, averages)
    return auroc_val, fpr, tpr
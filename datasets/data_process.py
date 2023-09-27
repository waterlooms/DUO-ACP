import os
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import KFold
from pyteomics import fasta

def split_fasta():
    data = []
    dataset = fasta.read(f'datasets/ACP-MLC/cancerppd_0.9.fasta')
    for x in dataset:
        labelstr = x.description
        words = re.findall(r'[A-Z][a-z]*', labelstr)
        data.append([','.join(words), x.sequence])
    data = np.array(data)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    idx = 0
    os.makedirs(f'datasets/ACP-MLC-10fold/', exist_ok=True)
    for train_index, test_index in kf.split(data):
        idx += 1
        with open(f'MLC_datasets/ACP-MLC-10fold/train_{idx}.fasta', 'w') as fw_train, open(f'MLC_datasets/ACP-MLC-10fold/test_{idx}.fasta', 'w') as fw_test:
            train, test = data[train_index], data[test_index]
            for x in train:
                fw_train.write(f'>{x[0]}\n')
                fw_train.write(f'{x[1]}\n')
            for x in test:
                fw_test.write(f'>{x[0]}\n')
                fw_test.write(f'{x[1]}\n')

def create2():
    train = pd.read_csv('datasets/ACP-Mixed-80/ACP-Mixed-80-train.tsv', sep='\t')
    test = pd.read_csv('datasets/ACP-Mixed-80/ACP-Mixed-80-test.tsv', sep='\t')
    df = pd.concat([train, test], axis=0)
    df.index = np.arange(len(df))
    df['index'] = np.arange(len(df))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folder = f'datasets/ACP-Mixed-80-5fold'
    os.makedirs(folder, exist_ok=True)
    idx = 0
    for train_index, test_index in kf.split(df):
        train, test = df.iloc[train_index], df.iloc[test_index]
        idx += 1
        train.to_csv(f'{folder}/train_{idx}.tsv', index=False, sep='\t')
        test.to_csv(f'{folder}/test_{idx}.tsv', index=False, sep='\t')

def create3():
    df = pd.read_csv('datasets/Case-study/original.csv')
    s1, s2 = df['SEQUENCE'].tolist(), df['CANCER TYPE'].tolist()
    tissue_dict = {}
    for x, y in zip(s1, s2):
        if y == 'Cervical':
            y = 'Cervix'
        if x not in tissue_dict:
            tissue_dict[x] = [y]
        else:
            tissue_dict[x].append(y)
    with open('datasets/Case-study/multiclass.fasta', 'w') as fw1, open('datasets/Case-study/binary.tsv', 'w') as fw2:
        fw2.write('index\tlabel\ttext\n')
        for idx, x in enumerate(tissue_dict):
            title = ','.join(tissue_dict[x])
            fw1.write(f'>{title}\n{x}\n')
            fw2.write(f'{idx}\t{1}\t{x}\n')

def create4():
    dataset = fasta.read(f'datasets/Case-study/muticlass2.fasta')
    with open('datasets/Case-study/binary2.tsv', 'w') as fw2:
        fw2.write('index\tlabel\ttext\n')
        for idx, x in enumerate(dataset):
            fw2.write(f'{idx}\t{1}\t{x.sequence}\n')

if __name__ == '__main__':
    create4()
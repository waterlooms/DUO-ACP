import os
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import KFold
from pyteomics import fasta

def split_fasta():
    data = []
    dataset = fasta.read(f'MLC_datasets/cancerppd_0.9.fasta')
    for x in dataset:
        labelstr = x.description
        words = re.findall(r'[A-Z][a-z]*', labelstr)
        data.append([','.join(words), x.sequence])
    data = np.array(data)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    idx = 0
    os.makedirs(f'MLC_datasets/10fold/', exist_ok=True)
    for train_index, test_index in kf.split(data):
        idx += 1
        with open(f'MLC_datasets/10fold/train_{idx}.fasta', 'w') as fw_train, open(f'MLC_datasets/10fold/test_{idx}.fasta', 'w') as fw_test:
            train, test = data[train_index], data[test_index]
            for x in train:
                fw_train.write(f'>{x[0]}\n')
                fw_train.write(f'{x[1]}\n')
            for x in test:
                fw_test.write(f'>{x[0]}\n')
                fw_test.write(f'{x[1]}\n')

def create2():
    train = pd.read_csv('ACP_datasets/ACP-Mixed-80-train.tsv', sep='\t')
    test = pd.read_csv('ACP_datasets/ACP-Mixed-80-test.tsv', sep='\t')
    df = pd.concat([train], axis=0)
    df.index = np.arange(len(df))
    df['index'] = np.arange(len(df))

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    folder = f'ACP_datasets/Mixed-10fold'
    os.makedirs(folder, exist_ok=True)
    idx = 0
    for train_index, test_index in kf.split(df):
        train, test = df.iloc[train_index], df.iloc[test_index]
        idx += 1
        train.to_csv(f'{folder}/train_{idx}.tsv', index=False, sep='\t')
        test.to_csv(f'{folder}/test_{idx}.tsv', index=False, sep='\t')
        
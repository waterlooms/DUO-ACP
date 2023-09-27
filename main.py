from torch.utils.data import random_split
from data import MyDataset
from runner import train_model, predict_model

def test_binary():
    data_type='binary'
    train_dataset = MyDataset(
        data_dir = 'datasets/ACP-Mixed-80/ACP-Mixed-80-train.tsv', 
        data_type=data_type
    )
    test_dataset = MyDataset(
        data_dir = 'datasets/ACP-Mixed-80/ACP-Mixed-80-test.tsv', 
        data_type=data_type
    )

    for idx in range(5):
        split_train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2])
        train_model(split_train_dataset, val_dataset, data_type, idx = idx, base_lr=2e-6)
        predict_model(test_dataset, data_type, idx = idx)

    
def test_multiclass():
    data_type='multiclass'
    for idx in range(1, 11):
        train_dataset = MyDataset(
            data_dir = f'datasets/ACP-MLC-10fold/train_{idx}.fasta', 
            data_type=data_type
        )
        test_dataset = MyDataset(
            data_dir = f'datasets/ACP-MLC-10fold/test_{idx}.fasta', 
            data_type=data_type
        )
        split_train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2])
        train_model(split_train_dataset, val_dataset, data_type, idx = idx, base_lr=1e-5)
        predict_model(test_dataset, data_type, idx = idx)


def test_binary_cv():
    data_type='binary'
    for idx in range(1, 6):
        train_dataset = MyDataset(
            data_dir = f'datasets/ACP-Mixed-80-5fold/train_{idx}.tsv', 
            data_type=data_type
        )
        test_dataset = MyDataset(
            data_dir = f'datasets/ACP-Mixed-80-5fold/test_{idx}.tsv', 
            data_type=data_type
        )
        split_train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2])
        train_model(split_train_dataset, val_dataset, data_type, idx = idx, base_lr=2e-6)
        predict_model(test_dataset, data_type, idx = idx)


def test_case_study_binary():
    data_type='binary'
    test_dataset = MyDataset(
        data_dir = f'datasets/Case-study/binary2.tsv', 
        data_type=data_type
    )
    for idx in range(1, 6):
        predict_model(test_dataset, data_type, idx = idx)
    
def test_case_study_multiclass():
    data_type='multiclass'
    for idx in range(1, 11):
        test_dataset = MyDataset(
            data_dir = f'datasets/Case-study/multiclass2.fasta', 
            data_type=data_type
        )
        predict_model(test_dataset, data_type, idx = idx)

if __name__ == '__main__':
    test_case_study_binary()
    test_case_study_multiclass()
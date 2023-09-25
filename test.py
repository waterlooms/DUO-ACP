from data import MyDataset
from model import ESM_Transformer_1, ESM_Transformer_3
from util import tmp_dir, show_PCA1, compute_auc

import torch
from torch.utils.data import DataLoader, random_split

def test():
    data_type='binary'
    train_dataset = MyDataset(
        data_dir = 'ACP_datasets/ACP-Mixed-80-train.tsv', 
        data_type=data_type
    )
    split_train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2])
    train_loader = DataLoader(split_train_dataset, batch_size=32, shuffle=True, drop_last=True)

    save_dir1 = f'{tmp_dir}/model1_{0}'
    model1 = ESM_Transformer_1(data_type)
    model1.load_state_dict(torch.load(save_dir1))

    model3 = ESM_Transformer_3(data_type)
    model3.load_model1(save_dir1)

    data_list, result_list, label_list = [], [], []
    for item in train_loader:
        x1, x2, seqlen, mask, labels = item['embed_features'], item['local_features'],\
                item['seqlen'],item['mask'], item['y']
        outputs, internal = model3(x1, x2, seqlen, mask)
        data_list.append(internal)
        label_list.append(labels)
        result_list.append(outputs)
    data_list = torch.vstack(data_list).detach().numpy()
    label_list = torch.vstack(label_list).detach().numpy()
    result_list = torch.vstack(result_list).detach().numpy()
    show_PCA1(data_list, label_list[:, 1], 0, f'{tmp_dir}/PCA.png')

    res = compute_auc(label_list, result_list, data_type)
    print(res)
if __name__ == '__main__':
    test()
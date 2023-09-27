import os
import pickle
import datetime
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import ESM_Transformer_1, ESM_Transformer_2, ESM_Transformer_3, ESM_Transformer_4
from util import *

str_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


class Runner:
    def __init__(self, model, lr=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Run on device: %s' % self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr = lr)


    def train(self, train_loader):
        train_loss = 0
        internal_list = []
        outputs_list, labels_list = [], []
        for batch in train_loader:
            x1, x2, seqlen, mask, labels = batch['embed_features'].to(self.device), batch['local_features'].to(self.device),\
                batch['seqlen'].to(self.device),\
                batch['mask'].to(self.device), batch['y'].to(self.device)
            self.optimizer.zero_grad()
            outputs, internal = self.model(x1, x2, seqlen, mask)
            internal_list.append(internal.detach().cpu())
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            outputs_list.append(outputs.detach().cpu().numpy())
            labels_list.append(labels.detach().cpu().numpy())
        train_loss /= len(train_loader)
        outputs_list = np.vstack(outputs_list)
        labels_list = np.vstack(labels_list)
        return train_loss, outputs_list, labels_list, internal_list


    def val(self, val_loader):
        self.model.eval()
        val_loss = 0
        outputs_list, labels_list = [], []
        internal_list = []
        with torch.no_grad():
            for batch in val_loader:
                x1, x2, seqlen, mask, labels = batch['embed_features'].to(self.device), batch['local_features'].to(self.device),\
                batch['seqlen'].to(self.device),\
                batch['mask'].to(self.device), batch['y'].to(self.device)
                outputs, internal = self.model(x1, x2, seqlen, mask)
                internal_list.append(internal.cpu())
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                outputs_list.append(outputs.detach().cpu().numpy())
                labels_list.append(labels.detach().cpu().numpy())
        val_loss /= len(val_loader)
        outputs_list = np.vstack(outputs_list)
        labels_list = np.vstack(labels_list)
        return val_loss, outputs_list, labels_list, internal_list

def train(train_dataset, val_dataset, model, lr, save_dir, data_type, stop_sign = 'train'):
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)
    
    best_loss, best_idx = 100, -1
    train_loss_list, val_loss_list, acc_list, auc_list = [], [], [], []
    total_epoch, patience = 300, 40

    runner = Runner(model, lr)
    with tqdm(total=total_epoch) as pbar:
        for epoch in range(total_epoch):
            train_loss, train_outputs, train_labels, train_internal = runner.train(train_loader)
            train_loss_list.append(train_loss)

            val_loss, test_outputs, test_labels, test_internal = runner.val(val_loader)
            val_loss_list.append(val_loss)
            
            acc = compute_acc(test_labels, test_outputs, data_type)
            acc_list.append(acc)
            auc = compute_auc(test_labels, test_outputs, data_type)
            auc_list.append(auc)


            # 保存最佳模型
            if stop_sign == 'train':
                loss = train_loss
            elif stop_sign == 'valid':
                loss = val_loss
                
            if loss < best_loss:
                best_loss = loss
                best_idx = epoch
                torch.save(runner.model.state_dict(), save_dir)
                counter = 0
            else:
                counter += 1
                if counter > patience:
                    break

            pbar.set_postfix(train_loss=train_loss, val_loss=val_loss)
            pbar.update()
            if epoch % 5 == 4:
                draw_loss([train_loss_list, val_loss_list], ['train_loss', 'val_loss'])
                draw_auc_epoch([auc_list, acc_list], ['AUC', 'ACC'])
                draw_PCA([train_internal, test_internal], [train_labels[:, 1], test_labels[:, 1]], epoch + 1)
    print(f"Best model is from epoch {best_idx}: {round(best_loss,3)}")

def predict(test_dataset, model, data_type, name, idx):
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model_path = f'{tmp_dir}/{data_type}/{name}_{idx}' 
    print(f'Loaded model from {model_path}')
    model.load_state_dict(torch.load(model_path))

    runner = Runner(model)
    val_loss, outputs, labels, internal_list = runner.val(test_loader)
    os.makedirs(f'{result_dir}/{data_type}/{str_time}/{name}', exist_ok=True)
    pkl_dir = f'{result_dir}/{data_type}/{str_time}/{name}/{idx}.pkl'
    with open(pkl_dir, 'wb') as fw:
        print(f'Result saved to {pkl_dir}')
        pickle.dump([outputs, labels], fw)


def train_model(train_dataset, val_dataset, data_type, idx, base_lr):
    model1 = ESM_Transformer_1(data_type)
    save_dir1 = f'{tmp_dir}/{data_type}/model1_{idx}'
    train(
        train_dataset = train_dataset, 
        val_dataset = val_dataset, 
        model = model1, 
        lr = 5 * base_lr, 
        save_dir = save_dir1, 
        data_type = data_type,
        stop_sign = 'valid',
    )

    model2 = ESM_Transformer_2(data_type)
    save_dir2 = f'{tmp_dir}/{data_type}/model2_{idx}'
    train(
        train_dataset = train_dataset, 
        val_dataset = val_dataset, 
        model = model2, 
        lr = 5 * base_lr, 
        save_dir = save_dir2, 
        data_type = data_type,
        stop_sign = 'valid',
    )
    
    model3 = ESM_Transformer_3(data_type)
    model3.load_model1(save_dir1)
    model3.load_model2(save_dir2)
    save_dir3 = f'{tmp_dir}/{data_type}/model3_{idx}'
    train(
        train_dataset = train_dataset, 
        val_dataset = val_dataset, 
        model = model3, 
        lr = base_lr, 
        save_dir = save_dir3, 
        data_type = data_type,
        stop_sign = 'valid',
    )
    
    model4 = ESM_Transformer_4(data_type)
    model4.load_model1(save_dir1)
    model4.load_model2(save_dir2)
    save_dir4 = f'{tmp_dir}/{data_type}/model4_{idx}'
    train(
        train_dataset = train_dataset, 
        val_dataset = val_dataset, 
        model = model4, 
        lr = base_lr, 
        save_dir = save_dir4, 
        data_type = data_type,
        stop_sign = 'valid',
    )
    
def predict_model(test_dataset, data_type, idx = 0):
    
    predict(
        test_dataset=test_dataset, 
        model = ESM_Transformer_1(data_type), 
        data_type = data_type,
        name = 'model1', 
        idx = idx
    )
    predict(
        test_dataset=test_dataset, 
        model = ESM_Transformer_2(data_type), 
        data_type = data_type,
        name = 'model2', 
        idx = idx
    )
    predict(
        test_dataset=test_dataset, 
        model = ESM_Transformer_3(data_type), 
        data_type = data_type,
        name = 'model3', 
        idx = idx
    )
    predict(
        test_dataset=test_dataset, 
        model = ESM_Transformer_4(data_type), 
        data_type = data_type,
        name = 'model4', 
        idx = idx
    )

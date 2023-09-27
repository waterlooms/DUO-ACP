import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve, confusion_matrix


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(42)

current_folder = '.'
tmp_dir = f'{current_folder}/tmp'
result_dir = f'{current_folder}/result'
all_labels = ['Colon','Breast','Cervix','Lung','Skin','Prostate','Blood']

def draw_loss(loss_list, labels):
    plt.figure()
    for loss, label in zip(loss_list, labels):
        plt.plot(np.arange(len(loss)), loss, label = label)
    plt.title('Loss')
    plt.legend()
    plt.savefig(f'{tmp_dir}/loss.png')
    plt.close()

def draw_auc_epoch(auc_lists, name_list):
    plt.figure()
    for auc_list, name in zip(auc_lists, name_list):
        plt.plot(np.arange(len(auc_list)), auc_list, label = name)
    plt.title('auc_epoch')
    plt.legend()
    plt.savefig(f'{tmp_dir}/auc.png')
    plt.close()

def compute_auc(labels, outputs, data_type):
    if data_type == 'binary':
        labels, outputs = labels[:, 1], outputs[:, 1]
        fpr, tpr, thresholds = roc_curve(labels, outputs)
        auc_val = auc(fpr, tpr)
    else:
        auc_val = []
        for i in range(len(all_labels)):
            labels_i, outputs_i = labels[:, i], outputs[:, i]
            fpr, tpr, thresholds = roc_curve(labels_i, outputs_i)
            auc_val.append(auc(fpr, tpr))
        auc_val = np.mean(auc_val)
    return auc_val

def compute_acc(labels, outputs, data_type, thres = 0.5):
    if data_type == 'binary':
        labels, outputs = labels[:, 1], outputs[:, 1]
        tn, fp, fn, tp = confusion_matrix(labels, outputs > thres).ravel()
        accuracy = (tp + tn) / (tn + fp + fn + tp)
    else:
        accuracy = []
        for i in range(len(all_labels)):
            labels_i, outputs_i = labels[:, i], outputs[:, i]
            tn, fp, fn, tp = confusion_matrix(labels_i, outputs_i > thres).ravel()
            accuracy.append((tp + tn) / (tn + fp + fn + tp))
        accuracy = np.mean(accuracy)
    return accuracy


def show_PCA1(data, labels, train_epoch, save_dir):
    # 创建PCA模型，将数据降到2维
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    data_true = data_2d[labels==1]
    data_false = data_2d[labels==0]

    # 查看PCA的主成分（特征向量）
    components = pca.components_
#    print("Principal Components:")
#    print(components)

    # 可视化降维后的数据
    plt.figure(figsize=(8, 6))
    plt.title(f'Trained for {train_epoch} epochs')
    plt.scatter(data_true[:, 0], data_true[:, 1], label='True')
    plt.scatter(data_false[:, 0], data_false[:, 1], label='False')
    plt.legend()
    plt.grid()
    plt.savefig(save_dir)
    plt.close()
'''
def show_PCA2(data_list, labels, train_epoch, save_dir):
    # 创建PCA模型，将数据降到2维
    pca = PCA(n_components=1)
    data_2d1 = pca.fit_transform(data_list[0])
    data_2d2 = pca.fit_transform(data_list[1])

    data_true1 = data_2d1[labels==1]
    data_false1 = data_2d1[labels==0]

    data_true2 = data_2d2[labels==1]
    data_false2 = data_2d2[labels==0]

    # 查看PCA的主成分（特征向量）
    components = pca.components_
#    print("Principal Components:")
#    print(components)

    # 可视化降维后的数据
    plt.figure(figsize=(8, 6))
    plt.title(f'Trained for {train_epoch} epochs')
    plt.scatter(data_true1, data_true2, label='True')
    plt.scatter(data_false1, data_false2, label='False')
    plt.legend()
    plt.grid()
    plt.savefig(save_dir)
    plt.close()
'''

def draw_PCA(interal_list, labels, train_epoch):
    x1 = torch.vstack(interal_list[0])
    x2 = torch.vstack(interal_list[1])
    show_PCA1(x1, labels[0], train_epoch, f'{tmp_dir}/PCA1.png')
    show_PCA1(x2, labels[1], train_epoch, f'{tmp_dir}/PCA2.png')

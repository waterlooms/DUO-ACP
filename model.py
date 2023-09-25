import torch
import torch.nn as nn
from util import *


d1, d2 = 320, 256

class ESM_Transformer_1(nn.Module):
    '''
        只使用预训练
    '''
    def __init__(self, data_type):
        super(ESM_Transformer_1, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d1, 
            nhead=8,
            dim_feedforward=1024, 
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        if data_type == 'binary':
            self.output = nn.Softmax(dim=1)
            output_dim = 2
        else:
            self.output = nn.Sigmoid()
            output_dim = len(all_labels)
        self.linear = nn.Linear(d1, output_dim)

    def forward(self, x1, x2, seqlen, mask):
        x = self.transformer_encoder(x1, src_key_padding_mask = mask)
        x = x[:, 0, :].squeeze(1)
        xc = x.clone()
        x = self.linear(x)
        x = self.output(x)
        return x, xc

class ESM_Transformer_1_embed(nn.Module):
    def __init__(self):
        super(ESM_Transformer_1_embed, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d1, 
            nhead=8,
            dim_feedforward=1024, 
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
    
    def forward(self, x1, x2, mask):
        x = self.transformer_encoder(x1, src_key_padding_mask = mask)
        x = x[:, 0, :].squeeze(1)
        return x
    
class ESM_Transformer_2(nn.Module):
    '''
        只使用one hot
    '''
    def __init__(self, data_type):
        super(ESM_Transformer_2, self).__init__()
        self.embedding = nn.Embedding(20, d2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d2, 
            nhead=4,
            dim_feedforward=512, 
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        if data_type == 'binary':
            self.output = nn.Softmax(dim=1)
            output_dim = 2
        else:
            self.output = nn.Sigmoid()
            output_dim = len(all_labels)
        self.linear = nn.Linear(d2, output_dim)
    
    def forward(self, x1, x2, seqlen, mask):
        x = self.embedding(x2)
        x = self.transformer_encoder(x, src_key_padding_mask = mask)
        x = x[:, 0, :].squeeze(1)
        xc = x.clone()
        x = self.linear(x)
        x = self.output(x)
        return x, xc
    
class ESM_Transformer_2_embed(nn.Module):
    def __init__(self):
        super(ESM_Transformer_2_embed, self).__init__()
        self.embedding = nn.Embedding(20, d2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d2, 
            nhead=4,
            dim_feedforward=512, 
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
    
    def forward(self, x1, x2, mask):
        x = self.embedding(x2)
        x = self.transformer_encoder(x, src_key_padding_mask = mask)
        x = x[:, 0, :].squeeze(1)
        return x

class ESM_Transformer_3(nn.Module):
    '''
        使用预训练+one hot
    '''
    def __init__(self, data_type):
        super(ESM_Transformer_3, self).__init__()        
        self.relu = nn.ReLU()
        self.embed1 = ESM_Transformer_1_embed()
        self.embed2 = ESM_Transformer_2_embed()
        self.linear1 = nn.Linear(d1 + d2, d1 + d2)
        self.linear2 = nn.Linear(d1 + d2, d1 + d2)
        
        if data_type == 'binary':
            self.output = nn.Softmax(dim=1)
            output_dim = 2
        else:
            self.output = nn.Sigmoid()
            output_dim = len(all_labels)
        self.linear3 = nn.Linear(d1 + d2, output_dim)
        
    def load_model1(self, pth):
        self.embed1.load_state_dict(torch.load(pth), strict=False)
#        for params in self.embed1.parameters():
#            params.requires_grad = False

    def load_model2(self, pth):
        self.embed2.load_state_dict(torch.load(pth), strict=False)
#        for params in self.embed2.parameters():
#            params.requires_grad = False

    
    def forward(self, x1, x2, seqlen, mask):
        y1 = self.embed1(x1, x2, mask)
        y2 = self.embed2(x1, x2, mask)
        x = torch.cat([y1, y2], dim = 1)
#        x = self.relu(self.linear1(x))
#        x = self.relu(self.linear2(x))
        xc = x.clone()
        x = self.linear3(x)
        x = self.output(x)
        return x, xc
    
class ESM_Transformer_4(nn.Module):
    '''
        使用预训练 + one hot + seqlen
    '''
    def __init__(self, data_type):
        super(ESM_Transformer_4, self).__init__()        
        self.relu = nn.ReLU()
        self.embed1 = ESM_Transformer_1_embed()
        self.embed2 = ESM_Transformer_2_embed()
        self.linear1 = nn.Linear(d1 + d2, d1 + d2)
        self.linear2 = nn.Linear(d1 + d2, d1 + d2)
        
        if data_type == 'binary':
            self.output = nn.Softmax(dim=1)
            output_dim = 2
        else:
            self.output = nn.Sigmoid()
            output_dim = len(all_labels)
        self.linear3 = nn.Linear(d1 + d2 + 1, output_dim)
        
    def load_model1(self, pth):
        self.embed1.load_state_dict(torch.load(pth), strict=False)
#        for params in self.embed1.parameters():
#            params.requires_grad = False

    def load_model2(self, pth):
        self.embed2.load_state_dict(torch.load(pth), strict=False)
#        for params in self.embed2.parameters():
#            params.requires_grad = False

    
    def forward(self, x1, x2, seqlen, mask):
        y1 = self.embed1(x1, x2, mask)
        y2 = self.embed2(x1, x2, mask)
        x = torch.cat([y1, y2], dim = 1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        xc = x.clone()
        x = torch.cat([x, seqlen.view(-1, 1)], dim = 1)
        x = self.linear3(x)
        x = self.output(x)
        return x, xc



def test():
    from data import MyDataset
    from torch.utils.data import DataLoader
    
    data_type='binary'
    train_dataset = MyDataset(
        data_dir = 'ACP_datasets/ACP-Mixed-80-test.tsv', 
        data_type=data_type
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, drop_last=True)
    test2(train_loader, data_type)
    
    data_type='multiclass'
    train_dataset = MyDataset(
		data_dir = 'MLC_datasets/10fold/train_1.fasta', 
		data_type=data_type
	)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, drop_last=True)
    test2(train_loader, data_type)

def test2(train_loader, data_type):
    model1 = ESM_Transformer_1(data_type)
    save_dir1 = f'{current_folder}/tmp/model_main1.pth'
    torch.save(model1.state_dict(), save_dir1)

    model2 = ESM_Transformer_2(data_type)
    save_dir2 = f'{current_folder}/tmp/model_main2.pth'
    torch.save(model2.state_dict(), save_dir2)

    model3 = ESM_Transformer_3(data_type)
    model3.load_model1(save_dir1)
    model3.load_model2(save_dir2)

    model4 = ESM_Transformer_4(data_type)
    model4.load_model1(save_dir1)
    model4.load_model2(save_dir2)

    for batch in train_loader:
        x1, x2, seqlen, mask, y = batch['embed_features'], batch['local_features'], batch['seqlen'], batch['mask'], batch['y'] 
        output, internal = model1(x1, x2, seqlen, mask)
        print(output.shape, internal.shape)
        output, internal = model2(x1, x2, seqlen, mask)
        print(output.shape, internal.shape)
        output, internal = model3(x1, x2, seqlen, mask)
        print(output.shape, internal.shape)
        output, internal = model4(x1, x2, seqlen, mask)
        print(output.shape, internal.shape)

if __name__ == '__main__':
    test()
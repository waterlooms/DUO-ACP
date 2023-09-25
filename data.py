import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

import esm
import pickle
from pyteomics import fasta

from tqdm import tqdm
from util import *

def BINARY(fastas):
	AA = 'ARNDCQEGHILKMFPSTWYV'
	encodings = []
	header = ['#']
	for i in range(1, len(fastas[0][1]) * 20 + 1):
		header.append('BINARY.F'+str(i))
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], i[1]
		code = [name]
		for aa in sequence:
			if aa == '-':
				code = code + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
				continue
			for aa1 in AA:
				tag = 1 if aa == aa1 else 0
				code.append(tag)
		encodings.append(code)
	return encodings

def esm_embeddings(dataset):
	model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
	batch_converter = alphabet.get_batch_converter()
	model.eval()
	device = torch.device("cuda")
	model = model.to(device)
	
	layer = 6
	#dataset = dataset.sort_values(lambda x: len())
	peptide_sequence_list = dataset[['index', 'text']].to_numpy()
	embed = []
	num_sequences = len(peptide_sequence_list)
	for i in range(0, num_sequences, 8):
		batch_labels, batch_strs, batch_tokens = batch_converter(peptide_sequence_list[i: i + 8])
		batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
		batch_tokens = batch_tokens.to(device)
		with torch.no_grad():
			results = model(batch_tokens, repr_layers=[layer], return_contacts=True)
		token_representations = results["representations"][layer].cpu()
		for idx, item in enumerate(token_representations):
			embed.append([item, batch_lens[idx]])
	return embed
	

def handcrafted_features(peptide_sequence_list):
	feature_list = []
	for peptide in tqdm(peptide_sequence_list):
		l = len(peptide)
		fasta_str = [['1', peptide]]
		
		# BINARY
		output = BINARY(fasta_str)[1]
		output.remove(output[0])
		Binary_features = np.array(output).reshape(l, -1)

		feature = {
			'local': np.concatenate([Binary_features], axis=1, dtype=np.float32)
		}
		feature_list.append(feature)
	
	return feature_list

def compute_representations(embeddings, feature_list, label_list, num_types):
	sequence_representations = []
	for i, embedding in enumerate(embeddings):
		x, tokens_len = embedding
		features = feature_list[i]
		x, tokens_len = x[:128], min(128, tokens_len)
		x = np.pad(x, ((0, 128 - len(x)), (0, 0)), 'constant', constant_values=(0))

		local_features = features['local']
		local_features = local_features[:128]
		local_features = local_features.argmax(axis=1)
		local_features = np.pad(local_features, (0, 128 - len(local_features)), 'constant', constant_values=(0))
		mask = [False] * tokens_len + [True] * (128 - tokens_len)
		mask = np.array(mask)
		
		y = np.zeros(num_types)
		y[label_list[i]] = 1

		sequence_representations.append({
			'seqlen': torch.tensor(np.array([tokens_len])),
			'embed_features': x,
			'local_features': local_features,
			'mask': mask,
			'y': y,
		})
	sorted_data = sequence_representations
	return sorted_data

class MyDataset(Dataset):
	def __init__(self, data_dir, data_type, pkl_dir = None):
		if data_type == 'binary':
			data = pd.read_csv(data_dir, sep='\t')
		else:
			self.label2id = {all_labels[i]: i for i in range(len(all_labels))}
			fasta_data = fasta.read(data_dir)
			data = []
			for idx, x in enumerate(fasta_data):
				title, seq = x.description, x.sequence
				if ',' not in title:
					labels = [self.label2id[title]]
				else:
					labels = [self.label2id[x] for x in title.split(',')]
				data.append([idx, seq, labels])
			data = pd.DataFrame(data, columns=['index', 'text', 'label'])
		feature_list = handcrafted_features(data['text'].tolist())
		label_list = data['label'].tolist()
		if pkl_dir != None:
			print(f'Load pickle from {pkl_dir}')
			embeddings = pickle.load(open(pkl_dir, 'rb'))
		else:
			print('Compute feature embedding')
			embeddings = esm_embeddings(data)
		num_types = 2 if data_type == 'binary' else len(all_labels)
		self.data = compute_representations(embeddings, feature_list, label_list, num_types)
		print(f'Dataset: {data_type}, Total {len(self)}')
			
	
	def __getitem__(self, idx):
		return self.data[idx]

	def __len__(self):
		return len(self.data)


def test():
	# Test binary classification
	train_dataset = MyDataset(
		data_dir = 'ACP_datasets/ACP-Mixed-80-train.tsv', 
		data_type = 'binary'
	)
	train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
	for batch in train_loader:
		print(batch['embed_features'].shape, batch['mask'].shape, batch['y'].shape)
		
	# Test multi-label classification	
	train_dataset = MyDataset(
		'MLC_datasets/10fold/train_1.fasta', 
		data_type='multiclass'
	)
	train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
	for batch in train_loader:
		print(batch['embed_features'].shape, batch['mask'].shape, batch['y'].shape)
		
	

if __name__ == '__main__':
	test()
# ## multiwoz

import math, torch, torch.nn as nn, torch.nn.functional as F
import pickle as pkl, random
# from nltk.translate.bleu_score import sentence_bleu

import numpy as np
from torch.autograd import Variable
# import matplotlib.pyplot as plt
import time
import gc
import  os, sys, json
from datetime import datetime
from collections import Counter
import Constants

from tokenizers import ByteLevelBPETokenizer, Tokenizer
import tempfile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)


def tokenize_en(sentence):
#	 return [tok.text for tok in en.tokenizer(sentence)]
	return sentence.split()


def gen_dataset_with_acts(split_name): # [ no of turns , src, tgt, act_vecs, hierarchial_act_vecs]
	file_path = 'hdsa_data/hdsa_data/'
	data_dir = 'data'
	dataset_file = open(file_path+split_name+'.json', 'r')
	dataset = json.load(dataset_file)
	
			
	data = []
	max_sent_len = 48
	responses = []

	for x in dataset:
		dialog_file = x['file']

		src = []
		
		for turn_num, turn in enumerate(x['info']):

			user= 'SOS '+' '.join(turn['user'].lower().strip().split()[:max_sent_len])+' EOS' 
			sys = 'SOS '+' '.join(turn['sys'].lower().strip().split()[:max_sent_len])+' EOS'

			src.append(user)

			hierarchical_act_vecs = np.zeros((Constants.act_len), 'int64')

			if turn['act'] != "None":
				for w in turn['act']:
					d, f, s = w.split('-')
					hierarchical_act_vecs[Constants.domains.index(d)] = 1
					hierarchical_act_vecs[len(Constants.domains) + Constants.functions.index(f)] = 1		 
					hierarchical_act_vecs[len(Constants.domains) + len(Constants.functions) + Constants.arguments.index(s)] = 1

			context = src
			true_response = ('SOS '+' '.join(turn['sys'].lower().strip().split())+' EOS')

			data.append([turn_num, src[:(2*turn_num+1)], [sys], hierarchical_act_vecs, dialog_file, true_response])

			src.append(sys)
			

	print('Length of', split_name,' dataset is', len(data))

	data.sort(key=lambda x:x[0], reverse=True) # first use the longer dialogs to reserve the GPU
	# data = data[:20] # COMMENT THIS IN FINAL RUN
	c=Counter()
	c.update([len(x[1])+len(x[2]) for x in data])
	# print(c)
	
	all_data = [x[1]+x[2] for x in data]
	all_hierarchial_act_vecs = [f[3] for f in data]
	all_dialog_files = [f[4] for f in data]
	true_responses = [f[5] for f in data]
	
	assert(len(all_data)==len(all_hierarchial_act_vecs))
	# print('Loaded and save ', split_name, ' dataset')
	
	return all_data, c, all_hierarchial_act_vecs, all_dialog_files, true_responses


def get_dataset_joint(split): #to do
	data_dir = '../'
	assert split=='train' or split=='val' or split== 'test'
	file_name = data_dir + str(split) + '_dials.json'



def build_vocab(train, load):
	if not load:
		c = Counter()
		idxtoword = {}
		wordtoidx ={}
		idxtoword[0]='PAD'
		idxtoword[1]='UNK'
		idxtoword[2] = 'SOS'
		idxtoword[3]='EOS'
		i=4
		for d in train:
			for s in d: 
				c.update(tokenize_en(s))
		freq_count = 1
		# print('Minimum freq count of words is ', freq_count)
		# print(c)
		for el in list(c):
			if c[el]>= freq_count  and el not in idxtoword.values():
				idxtoword[i]=el
				i+=1
					
		wordtoidx = {v:k for k,v in idxtoword.items()}

		# saving
		with open('data/idxtoword.pkl', 'wb') as file:
			pkl.dump(idxtoword, file)
		with open('data/wordtoidx.pkl', 'wb') as file:
			pkl.dump(wordtoidx, file)
	else:
		# loading	
		idxtoword = pkl.load(open('data/idxtoword.pkl', 'rb'))
		wordtoidx = pkl.load(open('data/wordtoidx.pkl', 'rb'))

	print('build_vocab: ', len(idxtoword), len(wordtoidx))

	return idxtoword, wordtoidx

from tokenizers import ByteLevelBPETokenizer
import json
import tempfile

def build_vocab_freqbased(V_PATH="./data/mwoz-bpe.tokenizer.json", recreate=False): # [ no of turns , src, tgt, act_vecs, hierarchial_act_vecs]
	if not recreate:
		if os.path.exists(V_PATH):
			print("Vocab Exists: ", V_PATH)
			# tokenizer = ByteLevelBPETokenizer()
			tokenizer = Tokenizer.from_file(V_PATH)
			return tokenizer
		
	split_name = 'train'
	file_path = 'hdsa_data/hdsa_data/'
	data_dir = 'data'
	dataset_file = open(file_path+split_name+'.json', 'r')
	dataset = json.load(dataset_file)

	idxtoword = {}
	wordtoidx ={}
	idxtoword[0]='PAD'
	idxtoword[1]='UNK'
	idxtoword[2] = 'SOS'
	idxtoword[3]='EOS'
	i = 4
	
	lines = []
	for x in dataset:
		dialog_file = x['file']
		src = []
		for turn_num, turn in enumerate(x['info']):
			user= turn['user'].lower().strip()
			sys = turn['sys'].lower().strip()
			lines.append(user)
			lines.append(sys)

	print(f"{len(lines)} lines in data.")
	# write to tmp
	fp = tempfile.NamedTemporaryFile("w", delete=False)
	for l in lines:
		fp.write(l+"\n")
	fp.close()
	print(fp.name)
	
	# build tokenizer
	tokenizer = ByteLevelBPETokenizer()
	tokenizer.train([fp.name], 
					vocab_size=30000 , 
					special_tokens=["PAD", "SOS", "EOS", "UNK"]
				   )
	tokenizer.save(V_PATH)
	
	# delete tmp file
	os.system(f"rm {fp.name}")
	return tokenizer


# tokenizer = build_vocab_freqbased()
# vocab_size = len(tokenizer.get_vocab())
# print(vocab_size)




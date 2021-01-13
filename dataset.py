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


def gen_dataset_joint(split_name, non_delex=False): # [ no of turns , src, tgt, act_vecs, hierarchial_act_vecs]
	file_path = 'hdsa_data/hdsa_data/'
	data_dir = 'data'
	dataset_file = open(file_path+split_name+'.json', 'r')
	dataset = json.load(dataset_file)
	
	data = []
	max_sent_len = 48
	MBS=70 # belief state text input len
	MDA=70 # dialog act text input len
	responses = []
	''' 
		# RESULTS - Train, valid, test
		max_bs_triplets:  13 max_act_triplets:  10
		max_bs_triplets:  12 max_act_triplets:  10
		max_bs_triplets:  12 max_act_triplets:  10
	'''

	for x in dataset:
		dialog_file = x['file']

		src = []
		
		for turn_num, turn in enumerate(x['info']):
			# NON-DELEX
			_SYS = 'sys_orig' if non_delex else 'sys'
			_USR = 'user_orig' if non_delex else 'user'
			
			user= 'SOS '+' '.join(turn[_USR].lower().strip().split()[:max_sent_len])+' EOS' 
			sys = 'SOS '+' '.join(turn[_SYS].lower().strip().split()[:max_sent_len])+' EOS'
			sys_delex = 'SOS '+' '.join(turn['sys'].lower().strip().split()[:max_sent_len])+' EOS'
			src.append(user)

			bs_list=[]
			bs = 'SOS '
			for domain, v_all in turn['BS'].items():# v_all is list of [slot_name, value] for that domain
				for v in v_all:
					# domain, slot_name1, domain, slot_name2,..
					bs_list.append([domain, v[0].lower(), v[1].lower()]) #Add v[1] - bs values

			bs_list = sorted(bs_list) # sort in alphabetical according to domain first, then by slot
			for idx, ele in enumerate(bs_list):
				bs += ele[0] + " " + ele[1] + " " + ele[2]
				if idx!=len(bs_list)-1:
					bs += " , " #separate by comma

			bs += ' EOS '
			bs = bs + (MBS -len(bs.split()))*' PAD'
			# In bs.split()- domains - bs[::3], slots-bs[1::3] - won't work with bpe tokenization -> have to change to bs, da metrics. 
			
			# === use turn['KB'] # TODO

			dialog_act_list=[]
			dialog_act = 'SOS '
			if turn['act'] != "None":
				for w in turn['act']:
					d, f, s = w.split('-')
					dialog_act_list.append([d,f,s])

			dialog_act_list = sorted(dialog_act_list) # sort in alphabetical according to domain first, then by action, then by slot

			for idx, da in enumerate(dialog_act_list):
				dialog_act +=  ' '.join(da)
				if idx!=len(dialog_act_list)-1:
					dialog_act += " , " #separate by comma

			dialog_act += ' EOS '
			dialog_act = dialog_act + (MDA-len(dialog_act.split()))*' PAD'

			context = src
			true_response = ('SOS '+' '.join(turn['sys'].lower().strip().split())+' EOS')
			data.append([turn_num, src[:(2*turn_num+1)], [sys_delex], bs, dialog_act, dialog_file, true_response]) # response is delexicalized, history is original

			src.append(sys)

	print('Length of', split_name,' dataset is', len(data))

	data.sort(key=lambda x:x[0]) # Sort by len
	c=Counter()
	c.update([len(x[1])+len(x[2]) for x in data])
	
	all_data = [x[1]+x[2] for x in data]
	all_bs = [f[3] for f in data]
	all_dialog_act = [f[4] for f in data]
	all_dialog_files = [f[5] for f in data]
	true_responses = [f[6] for f in data]
	
	return all_data, c, all_bs, all_dialog_act, all_dialog_files, true_responses


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

def build_vocab_freqbased(V_PATH="./data/mwoz-bpe.tokenizer.json", non_delex=False, recreate=False, vocab_size=2_000): # [ no of turns , src, tgt, act_vecs, hierarchial_act_vecs]
	if os.path.exists(V_PATH) and (not recreate):
		print("Vocab Exists: ", V_PATH)
		# tokenizer = ByteLevelBPETokenizer()
		tokenizer = Tokenizer.from_file(V_PATH)
	else:
		split_name = 'train'
		file_path = 'hdsa_data/hdsa_data/'
		data_dir = 'data'
		
		# NON-DELEX
		_SYS = 'sys_orig' if non_delex else 'sys'
		_USR = 'user_orig' if non_delex else 'user'
		
		dataset_file = open(file_path+split_name+'.json', 'r')
		dataset = json.load(dataset_file)

		lines = []
		for x in dataset:
			dialog_file = x['file']
			src = []
			for turn_num, turn in enumerate(x['info']):
				user= turn[_USR].lower().strip()
				sys = turn[_SYS].lower().strip()
				lines.append(user)
				lines.append(sys)
				
				# Explicitly add the target system responses
				if non_delex:
					sys_delex = turn["sys"].lower().strip()
					lines.append(sys_delex)

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
						vocab_size=vocab_size,
						special_tokens=["PAD", "UNK", "SOS", "EOS"]
					   )
		tokenizer.save(V_PATH)
		print("Vocab Created: ", V_PATH)

		# delete tmp file
		os.system(f"rm {fp.name}")
	
	Constants.PAD = tokenizer.get_vocab()["PAD"]
	Constants.SOS = tokenizer.get_vocab()["SOS"]
	Constants.EOS = tokenizer.get_vocab()["EOS"]
	Constants.UNK = tokenizer.get_vocab()["UNK"]
	print("PAD index is: ", Constants.PAD)
	print("SOS index is: ", Constants.SOS)
	print("EOS index is: ", Constants.EOS)
	print("UNK index is: ", Constants.UNK)
	return tokenizer


# tokenizer = build_vocab_freqbased()
# vocab_size = len(tokenizer.get_vocab())
# print(vocab_size)




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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)


def tokenize_en(sentence):
#     return [tok.text for tok in en.tokenizer(sentence)]
	return sentence.split()


def preprocess_bs(bs_dict, kb_results): # Used for action_pred model
	bs = ""
	no_of_triplets =0 
	for domain, v_all in bs_dict.items(): #v_all is list of [slot_name, value] for that domain
		no_of_triplets += len(v_all)
		bs += domain
		for v in v_all:
			# domain, (slot_name1, value1), (slot_name2, value2),..
			# bs += " " + " ".join(v) + " BS_SEP " 
			bs += " " + v[0].lower() # domain, slot_name1, slot_name2,..
		bs += " BS_SEP "
	bs += " KB_RES " + str(kb_results) + " KB_RES"
	return bs, no_of_triplets


def gen_dataset_action_pred(split_name): # [ no of turns , src, tgt, act_vecs, hierarchial_act_vecs]
	file_path = 'hdsa_data/hdsa_data/'
	data_dir = 'data'
	dataset_file = open(file_path+split_name+'.json', 'r')
	dataset = json.load(dataset_file)
	
	data = []
	max_sent_len = 48
	responses = []
	''' 
		# BS RESULTS - 
		Max Number of triplets - 13, 12, 12 
		Max len with SEP - 39 36 36 -> for domain, slot_name without value.
	'''

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

			bs, _ = preprocess_bs(turn['BS'], turn['KB'])
			# print(bs)

			data.append([turn_num, src[:(2*turn_num+1)], [sys],hierarchical_act_vecs, dialog_file, true_response, bs])

			src.append(sys)

	print('Length of', split_name,' dataset is', len(data))

	data.sort(key=lambda x:x[0]) # Sort by len
	c=Counter()
	c.update([len(x[1])+len(x[2]) for x in data])
	# print(c)
	
	all_data = [x[1]+x[2] for x in data]
	all_hierarchial_act_vecs = [f[3] for f in data]
	all_dialog_files = [f[4] for f in data]
	true_responses = [f[5] for f in data]
	all_bs_kb = [f[6] for f in data]
	
	assert(len(all_data)==len(all_hierarchial_act_vecs))	
	return all_data, c, all_hierarchial_act_vecs, all_dialog_files, true_responses, all_bs_kb



def gen_dataset_joint(split_name): # [ no of turns , src, tgt, act_vecs, hierarchial_act_vecs]
	file_path = 'hdsa_data/hdsa_data/'
	data_dir = 'data'
	dataset_file = open(file_path+split_name+'.json', 'r')
	dataset = json.load(dataset_file)
	
	data = []
	max_sent_len = 48
	MBS=50 # belief state text input len
	MDA=51 # dialog act text input len
	responses = []
	''' 
		# RESULTS - Train, valid, test
		max_bs_triplets:  13 max_act_triplets:  10
		max_bs_triplets:  12 max_act_triplets:  10
		max_bs_triplets:  12 max_act_triplets:  10
	'''
	max_act_triplets =0
	max_bs_triplets =0

	for x in dataset:
		dialog_file = x['file']

		src = []
		
		for turn_num, turn in enumerate(x['info']):
			user= 'SOS '+' '.join(turn['user'].lower().strip().split()[:max_sent_len])+' EOS' 
			sys = 'SOS '+' '.join(turn['sys'].lower().strip().split()[:max_sent_len])+' EOS'

			src.append(user)

			bs_list=[]
			bs = 'SOS SOS '
			for domain, v_all in turn['BS'].items():# v_all is list of [slot_name, value] for that domain
				for v in v_all:
					# domain, slot_name1, domain, slot_name2,..
					bs_list.append([domain, v[0].lower()])

			bs_list = sorted(bs_list) # sort in alphabetical according to domain first, then by slot
			for ele in bs_list:
				bs += ele[0] + " " + ele[1] + " "

			bs += ' EOS EOS '			
			bs = bs + (MBS -len(bs.split()))*' PAD'
			# In bs.split()- domains - bs[::2], slots-bs[1::2]
			
			# === use turn['KB'] # TODO

			dialog_act_list=[]
			dialog_act = 'SOS SOS SOS '
			if turn['act'] != "None":
				for w in turn['act']:
					d, f, s = w.split('-')
					dialog_act_list.append([d,f,s])

			dialog_act_list = sorted(dialog_act_list) # sort in alphabetical according to domain first, then by action, then by slot

			for da in dialog_act_list:
				dialog_act +=  ' '.join(da) + " "
			dialog_act += ' EOS EOS EOS '
			dialog_act = dialog_act + (MDA-len(dialog_act.split()))*' PAD'

			context = src
			true_response = ('SOS '+' '.join(turn['sys'].lower().strip().split())+' EOS')
			data.append([turn_num, src[:(2*turn_num+1)], [sys], bs, dialog_act, dialog_file, true_response])

			src.append(sys)

	print('Length of', split_name,' dataset is', len(data))

	data.sort(key=lambda x:x[0]) # Sort by len
	c=Counter()
	c.update([len(x[1])+len(x[2]) for x in data])
	# print(c)
	
	all_data = [x[1]+x[2] for x in data]
	all_bs = [f[3] for f in data]
	all_dialog_act = [f[4] for f in data]
	all_dialog_files = [f[5] for f in data]
	true_responses = [f[6] for f in data]
	
	return all_data, c, all_bs, all_dialog_act, all_dialog_files, true_responses


def build_vocab_freqbased(load): # [ no of turns , src, tgt, act_vecs, hierarchial_act_vecs]
	split_name = 'val'
	file_path = 'hdsa_data/hdsa_data/'
	data_dir = 'data'
	dataset_file = open(file_path+split_name+'.json', 'r')
	dataset = json.load(dataset_file)

	c = Counter()
	idxtoword = {}
	wordtoidx ={}
	idxtoword[0]='PAD'
	idxtoword[1]='UNK'
	idxtoword[2] = 'SOS'
	idxtoword[3]='EOS'
	idxtoword[4]='BS_SEP'
	idxtoword[5] = 'KB_RES'
	i = 6

	for x in dataset:
		dialog_file = x['file']
		src = []
		for turn_num, turn in enumerate(x['info']):
			user= turn['user'].lower().strip().split()
			sys = turn['sys'].lower().strip().split()
			c.update(user)
			c.update(sys)

	# Used these for only action pred model
	for w in Constants.domains + Constants.functions+Constants.arguments:
		if w not in idxtoword.values():
			idxtoword[i]=w
			i+=1
	# add numbers from 0-4 for kb:
	for n in range(0,5):
		if str(n) not in idxtoword.values():
			idxtoword[i]=str(n)
			i+=1

	# adding delexicalised terms in train
	for k,v in c.items():
		if k[0]=='[' and k[-1]==']' and '[' not in k[1:]:
			idxtoword[i]=k
			i += 1

	for idx, (k,v) in enumerate(c.most_common(1500)):
		if k not in idxtoword.values():
			idxtoword[i] = k
			i += 1
	wordtoidx = {v:k for k,v in idxtoword.items()}
	return idxtoword, wordtoidx







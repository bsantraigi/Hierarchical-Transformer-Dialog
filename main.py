#!/usr/bin/env python
# coding: utf-8

import math, torch, torch.nn as nn, torch.nn.functional as F
import pickle as pkl, random
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from torch.autograd import Variable
import time
import gc
import os, sys
from datetime import datetime
from collections import Counter
import logging
from nltk.util import ngrams
import re, json
from tqdm import tqdm


from dataset import *
from utils import *
from model import *
from joint_model import *
from joint_model_v2 import *
from joint_model_v2_2 import *
from joint_model_v3 import *
from joint_model_v4 import *
from metrics import *
from collections import OrderedDict
from evaluate import evaluateModel
import Constants
import argparse

if not os.path.isdir('running'):
	os.makedirs('running')


def split_to_responses(split): # return original responses, file names
	if split=='train':
		return train_responses, train_dialog_files
	if split=='val':
		return val_responses, val_dialog_files
	if split=='test':
		return test_responses, test_dialog_files
	return ValueError

def get_files_joint(split): # dataset, dataset_counter, dataset_bs, dataset_da
	if split=='train':
		return train, train_counter, train_bs, train_dialog_act
	if split=='val':
		return val, val_counter, val_bs, val_dialog_act
	if split=='test':
		return test, test_counter, test_bs, test_dialog_act
	return ValueError
	
def shuffle(split):
# train,train_counter, train_bs, train_dialog_act, train_dialog_files, train_responses
# shuffle all of these acc of train_counter: #u:#samples
	dataset, dataset_counter, dataset_bs, dataset_da = get_files_joint(split)
	dataset_responses, dataset_dialog_files = split_to_responses(split)
	indices = range(0, len(dataset))
	t =0
	c = list(zip(dataset, dataset_bs, dataset_da, dataset_responses, dataset_dialog_files)) 
	for k,v in dataset_counter.items():# from t,t+v
		random.shuffle(c[t:t+v])
		t += v
	dataset, dataset_bs, dataset_da, dataset_responses, dataset_dialog_files = zip(*c)
	return

def train_epoch(model, epoch, batch_size, criterion, optimizer, scheduler): # losses per batch
	model.train()
	total_loss =0
	total_response_loss =0
	total_bs_loss =0
	total_da_loss =0

	start_time = time.time()
	ntokens = len(idxtoword)
	nbatches = len(train)//batch_size
	
#     if torch.cuda.is_available():
#         stat_cuda('before epoch')
		
	accumulated_steps = 3
	optimizer.zero_grad()

	# response_loss = torch.tensor([0], device=device)
	# da_loss=torch.tensor([0], device=device)

	for i, (data, targets, labels, bs, da) in enumerate(data_loader(train, train_counter, train_bs, train_dialog_act, batch_size, wordtoidx)):

		batch_size_curr = data.shape[1]
		# optimizer.zero_grad()

		output, bs_logits , da_logits = model(data, bs, da, targets)

		response_loss = criterion(output.reshape(-1, ntokens), labels.reshape(-1))
		belief_loss=criterion(bs_logits[:-1].reshape(-1, ntokens), bs[1:].reshape(-1))
		da_loss=criterion(da_logits[:-1].reshape(-1, ntokens), da[1:].reshape(-1))  
		
		cur_loss = response_loss+belief_loss+da_loss

		loss = cur_loss / accumulated_steps
		loss.backward()
	
		torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
		
		if i%accumulated_steps==0:
			optimizer.step()
			optimizer.zero_grad()

		total_loss += cur_loss.item()*batch_size_curr
		total_response_loss += response_loss.item()*batch_size_curr
		total_bs_loss += belief_loss.item()*batch_size_curr
		total_da_loss += da_loss.item()*batch_size_curr

	elapsed = time.time()-start_time

	total_loss /= len(train)
	total_response_loss /= len(train)
	total_bs_loss /=len(train)
	total_da_loss /= len(train)

	logger.debug('==>Epoch {}, Train \tResp Loss: {:0.3f} BS Loss: {:0.3f} DA Loss: {:0.3f}\tTotal loss {:0.3f}\tTime {:0.0f}s'.format(epoch,  total_response_loss, total_bs_loss, total_da_loss, total_loss, elapsed))

	return total_loss


def get_loss_nograd(model, epoch, batch_size,criterion, split): # losses per batch

	model.eval()
	total_loss =0
	total_response_loss =0
	total_bs_loss =0
	total_da_loss =0
	start_time = time.time()
	ntokens = len(idxtoword)
	
	dataset, dataset_counter, dataset_bs, dataset_da = get_files_joint(split)

	# response_loss = torch.tensor([0], device=device)
	# da_loss=torch.tensor([0], device=device)
	
	with torch.no_grad():
		for i, (data, targets, labels, bs, da) in enumerate(data_loader(dataset, dataset_counter, dataset_bs, dataset_da, batch_size, wordtoidx)):

			batch_size_curr = data.shape[1]
			output, bs_logits , da_logits = model(data, bs, da, targets)

			response_loss = criterion(output.reshape(-1, ntokens), labels.reshape(-1))
			belief_loss=criterion(bs_logits[:-1].reshape(-1, ntokens), bs[1:].reshape(-1))
			da_loss=criterion(da_logits[:-1].reshape(-1, ntokens), da[1:].reshape(-1))  

			cur_loss = response_loss+belief_loss+da_loss

			total_loss += cur_loss.item()*batch_size_curr
			total_response_loss += response_loss.item()*batch_size_curr
			total_bs_loss += belief_loss.item()*batch_size_curr
			total_da_loss += da_loss.item()*batch_size_curr

	elapsed = time.time()-start_time

	total_loss /= len(dataset)
	total_response_loss /= len(dataset)
	total_bs_loss /=len(dataset)
	total_da_loss /= len(dataset)

	logger.debug('==>{} \tResp Loss: {:0.3f} BS Loss: {:0.3f} DA Loss: {:0.3f}\tTotal loss {:0.3f}\tTime {:0.0f}s'.format(split, total_response_loss, total_bs_loss, total_da_loss, total_loss, elapsed))

	return total_loss


def evaluate(model, args, dataset, dataset_counter, dataset_bs, dataset_da , batch_size, criterion, split, method='beam', beam_size=None):
	batch_size = args.batch_size
	use_gt = True

	logger.debug('=========== {} search {} ==========='.format(method.upper(), split.upper()))
	if method=='beam':
		logger.debug('Beam size {}'.format(beam_size))
	if use_gt:
		logger.debug('Using ground truth bs while inference')
	model.eval()
	total_loss =0
	total_bs_loss=0
	total_da_loss=0
	total_response_loss=0

	start = time.time()
	nbatches = len(dataset)//batch_size

	response_loss = torch.tensor([0], device=device)
	da_loss=torch.tensor([0], device=device)

	with torch.no_grad():
		for i, (data, targets, labels, bs, da) in enumerate(data_loader(dataset,dataset_counter, dataset_bs, dataset_da , batch_size, wordtoidx)): # , total=len(dataset)//batch_size):

			batch_size_curr = targets.shape[1]

			if method=='beam':
				if isinstance(model, nn.DataParallel):
					# gives list of sentences itself
					bs_output, da_output = model.module.greedy_search_bsda(data, bs, None,  batch_size_curr, use_gt=use_gt)
					output = model.module.translate_batch(data, bs_output, da_output, beam_size, batch_size_curr)
				else:
					bs_output, da_output = model.greedy_search_bsda(data, bs, None, batch_size_curr, use_gt =use_gt)
					output = model.translate_batch(data, bs_output, da_output, beam_size , batch_size_curr)

			elif method=='greedy':
				if isinstance(model, nn.DataParallel):
					output, output_max, bs_logits, bs_output, da_logits, da_output = model.module.greedy_search(data,  batch_size_curr, [d_to_imap, s_to_imap, a_to_imap], bs, da, use_gt = use_gt) # .module. if using dataparallel
				else: # da_output_i in individal vocab indices
					output, output_max, bs_logits, bs_output, da_logits, da_output = model.greedy_search(data, batch_size_curr, [d_to_imap, s_to_imap, a_to_imap], bs, da, use_gt= use_gt)

			if torch.is_tensor(output): # greedy search
				# print(bs_logits.shape, bs.shape) - torch.Size([49, 32, 1515]) torch.Size([50, 32])
				bs_loss = criterion(bs_logits.reshape(-1, ntokens), bs[1:].reshape(-1)) 

				da_loss = criterion(da_logits.reshape(-1, ntokens), da[1:].reshape(-1))
				response_loss = criterion(output.reshape(-1, ntokens), labels.reshape(-1))

				cur_loss = (response_loss+bs_loss+da_loss)
				# cur_loss = bs_loss

				total_response_loss += response_loss.item() * batch_size_curr
				total_bs_loss += bs_loss.item() * batch_size_curr
				total_da_loss += da_loss.item() * batch_size_curr

				total_loss += cur_loss.item() * batch_size_curr

				# output = torch.max(output, dim=2)[1]
				output_max = post_process(output_max.transpose(0,1))
				
				if i==0:
					hyp = output_max
					ref = targets.transpose(0,1)
				else:
					hyp = torch.cat((hyp, output_max), dim=0)
					ref= torch.cat((ref, targets.transpose(0,1)), dim=0)
			else: # beam search
				if i==0:
					hyp = [torch.tensor(l) for l in output]
					ref = targets.transpose(0,1)
				else:
					hyp.extend([torch.tensor(l) for l in output])
					ref= torch.cat((ref, targets.transpose(0,1)), dim=0)

			if i==0: #both greedy,beam
				bs_pred = bs_output.transpose(0,1) # bs, 50 - with sos
				bs_act = bs.transpose(0,1) #bs,50
				da_pred = da_output.transpose(0,1) # bs, 51
				da_act = da.transpose(0,1)
			else:
				bs_pred= torch.cat([bs_pred, bs_output.transpose(0,1)])
				bs_act = torch.cat([bs_act, bs.transpose(0,1)])
				da_pred=torch.cat([da_pred, da_output.transpose(0,1)])# bs, 51
				da_act=torch.cat([da_act, da.transpose(0,1)])


		"""
		bs_pred.shape - bs, 50 - with sos
		bs_act[0].shape, bs_act[1].shape = bs, 25 -> changed to bs,50 below
		in metrics such as accuracy, exlcude sos, eos, pad
		both bs_pred, bs_act in total vocab indices now
		"""

		bs_joint_acc, bs_slot_acc = compute_bs_accuracy(bs_pred, bs_act)
		da_acc, da_hdsa_metrics = compute_da_metrics(da_pred, da_act)

		indices = list(range(0, len(dataset)))
		# indices = list(range(0, args.batch_size)) # uncomment this to run for one batch

		pred_hyp = tensor_to_sents(hyp , wordtoidx)  # hyp[indices]
		pred_ref, all_dialog_files = split_to_responses(split)
		
		bleu_score = BLEU_calc.score(pred_hyp, pred_ref, wordtoidx)*100
		f1_entity = F1_calc.score(pred_hyp, pred_ref, wordtoidx)*100

		total_loss = total_loss/len(dataset)
		total_response_loss = total_response_loss/len(dataset)
		total_bs_loss = total_bs_loss/len(dataset)
		total_da_loss = total_da_loss/len(dataset)

		evaluate_dials = {}
		for i, h in enumerate(pred_hyp):
			if all_dialog_files[i] in evaluate_dials:
				evaluate_dials[all_dialog_files[i]].append(h)
			else:
				evaluate_dials[all_dialog_files[i]]=[h]
		
		# Save model predictions to json also for later evaluation
		if method=='beam':
			model_turns_file = args.log_path+'model_turns_beam_'+str(beam_size)+'_'+split+'.json'
		elif method=='greedy':
			model_turns_file = args.log_path+'model_turns_greedy_'+split+'.json'
		with open(model_turns_file, 'w') as f:
			json.dump(evaluate_dials, f)

		matches, successes = evaluateModel(evaluate_dials) # gives matches(inform), success
		
		data, _, _, _ = get_files_joint(split)

		# decode in individual vocabs
		bs_pred = tensor_to_sents(bs_pred, wordtoidx)
		bs_act = tensor_to_sents(bs_act, wordtoidx)
		da_pred = tensor_to_sents(da_pred, wordtoidx)
		da_act = tensor_to_sents(da_act, wordtoidx)

		if method=='beam':
			pred_file = open(args.log_path+'pred_beam_'+str(beam_size)+'_'+split+'.txt', 'w')
		elif method=='greedy':
			pred_file = open(args.log_path+'pred_greedy_'+split+'.txt', 'w')

		pred_file.write('\n\n***'+split+'***')
		for idx, bs_t, bs_p, da_t, da_p, h, r in zip(indices, bs_act, bs_pred, da_act, da_pred, pred_hyp, pred_ref): 
			pred_file.write('\n\nContext: \n'+str('\n'.join(data[idx][:-1])))
			pred_file.write('\nBS Gold: '+str(bs_t)+'\nBS Pred: '+str(bs_p))
			pred_file.write('\nDA Gold: '+str(da_t)+'\nDA Pred: '+str(da_p))
			pred_file.write('\nGold sentence: '+str(r)+'\nOutput: '+str(h))

	elapsed = time.time()-start	
	if method=='greedy':
		logger.debug('==>{}\tResp Loss: {:0.3f}\tBS Loss: {:0.3f}\tDA Loss: {:0.3f}\tTotal Loss: {:0.3f}\tTime taken: {:0.1f}s'.format(split,  total_response_loss, total_bs_loss, total_da_loss, total_loss, elapsed))

	logger.debug('==>{}\tBelief state Joint acc: {:0.2f}\tSlot acc: {:0.2f}'.format(split,  bs_joint_acc, bs_slot_acc))

	logger.debug('==>{} Dialog Act: Joint acc: {:0.2f}  Slot acc: {:0.2f}'.format(split, da_acc[0], da_acc[1]))
	# logger.debug('==>{} Dialog Act: Joint acc: {:0.2f}  Slot acc: {:0.2f} || HDSA precision: {:0.2f}  recall {:0.2f}  f1_score: {:0.2f}'.format(split, da_acc[0], da_acc[1], da_hdsa_metrics[0], da_hdsa_metrics[1], da_hdsa_metrics[2]))

	criteria = bleu_score+0.5*(matches+successes)
	logger.debug('==>{}\tBleu: {:0.2f}\tF1-Entity {:0.2f}\tInform {:0.2f}\tSuccesses: {:0.2f}\tCriteria: {:0.2f}'.format( split, bleu_score, f1_entity, matches, successes, criteria ))

	return total_loss, bleu_score, f1_entity, matches, successes


# stat_cuda('before training')
def training(model, args, criterion, optimizer, scheduler, optuna_callback=None):
	global best_val_bleu, criteria, best_val_loss_ground
	best_model = None
	train_losses = []
	val_losses = []
	val_bleus = []

	best_val_loss_ground=float("inf")
	best_val_bleu=-float("inf")
	best_criteria=-float("inf")

	logger.debug('At begin of training, Best val loss ground : {:0.7f} Best bleu: {:0.4f}, Best criteria: {:0.4f}'.format(best_val_loss_ground, best_val_bleu, best_criteria))
	logger.debug('====> STARTING TRAINING NOW')

	val_epoch_freq = 3
	for epoch in range(1, args.epochs + 1):
		shuffle('train')
		epoch_start_time = time.time()
		train_loss = train_epoch(model, epoch, args.batch_size, criterion, optimizer, scheduler)

		val_loss_ground = get_loss_nograd(model, epoch, args.batch_size, criterion, 'val')
		
		if val_loss_ground <= best_val_loss_ground:
			best_val_loss_ground=val_loss_ground
			save_model(model, args, 'checkpoint_bestloss.pt',train_loss, val_loss_ground, -1)
		else:
			scheduler.step()

		if epoch < 8:
			save_model(model, args, 'checkpoint.pt',train_loss, val_loss_ground, -1)
			continue

		# for every "val_epoch_freq" epochs, evaluate the metrics
		if epoch%val_epoch_freq!=0:
			save_model(model, args, 'checkpoint.pt', train_loss, val_loss_ground, -1)
			continue

		_, val_bleu, val_f1entity, matches, successes = evaluate(model, args, val, val_counter, val_bs, val_dialog_act , args.batch_size, criterion, 'val', method='greedy')
		val_criteria = val_bleu+0.5*matches+0.5*successes

		if optuna_callback is not None:
			optuna_callback(epoch/val_epoch_freq, val_criteria) # Pass the score metric on validation set here.
		
		if  val_criteria > best_criteria:
			best_criteria = val_criteria
			best_model = model
			logger.debug('==> New optimum found wrt val criteria')
			save_model(model, args, 'checkpoint_criteria.pt',train_loss, val_loss_ground, val_criteria)

		save_model(model, args, 'checkpoint.pt',train_loss, val_loss_ground, val_criteria)
		scheduler.step()
	
	return best_model


def save_model(model, args, name, train_loss, val_loss, val_criteria):
	checkpoint = {
					'model': model.state_dict(),
					'embedding_size': args.embedding_size,
					'nhead':args.nhead,
					'nhid': args.nhid,
					'nlayers_e1': args.nlayers_e1,
					'nlayers_e2': args.nlayers_e2,
					'nlayers_d': args.nlayers_d,
					'dropout': args.dropout
				 }
	if train_loss!=-1:
		checkpoint['train_loss']=train_loss
	if val_loss!=-1:
		checkpoint['val_loss']=val_loss
	if val_criteria!=-1:
		checkpoint['val_criteria']=val_criteria

	# logger.debug('==> Checkpointing everything now...in {}'.format(name))
	torch.save(checkpoint, args.log_path+name)


def load_model(model, checkpoint='checkpoint.pt'):
	load_file = checkpoint
	if os.path.isfile(load_file):
		global best_val_bleu, best_val_loss_ground, criteria
		try:
			print('Reloading previous checkpoint', load_file)

			if not torch.cuda.is_available():
				# load dataparallel model into cpu
				checkpoint = torch.load(load_file,map_location=lambda storage, loc: storage)
				new_state_dict = OrderedDict()
				for k, v in checkpoint['model'].items():
					if k[:6]=="module":
						name = k[7:] # remove `module.`
					else:
						name=k
					new_state_dict[name] = v
				# load params
				model.load_state_dict(new_state_dict)

			else:
				checkpoint = torch.load(load_file)
				model.load_state_dict(checkpoint['model'])
			#	optimizer.load_state_dict(checkpoint['optim'])

			
			if(checkpoint.get('val_loss')):
				best_val_loss_ground = checkpoint['val_loss']
			# else:
				# best_val_loss_ground= get_loss_nograd(model, 0, args.batch_size, criterion, 'val')
				

			if(checkpoint.get('val_criteria')):
				best_val_bleu = checkpoint.get('val_criteria')
				logger.debug('Valid criteria of Loaded model is: {:0.4f}'.format(best_val_bleu))

			logger.debug('Loaded model, Val loss(ground): {:0.8f}'.format(best_val_loss_ground))
		except Exception as e:
			logger.debug('Loading model error')
			logger.debug(e)
	else:
		best_val_loss_ground = float('inf')
		logger.debug('No model to load')
	return best_val_loss_ground



def testing(model, args, criterion, split, method):
	data, dataset_counter, dataset_bs, dataset_da = get_files_joint(split)
	if method =='greedy':
		return evaluate(model, args, data,dataset_counter, dataset_bs, dataset_da , args.batch_size, criterion, split, method='greedy')
	elif method=='beam':
		return evaluate(model, args, data, dataset_counter, dataset_bs, dataset_da  , args.batch_size, criterion, split, method='beam')


def test_split(split, model, args, criterion):
	data, dataset_counter,  dataset_bs, dataset_da = get_files_joint(split)
	# greedy
	evaluate(model, args, data, dataset_counter, dataset_bs, dataset_da, args.batch_size, criterion, split, 'greedy')
	# beam 2
	evaluate(model, args, data, dataset_counter, dataset_bs, dataset_da, args.batch_size, criterion, split, 'beam', 2)
	# beam 3
	evaluate(model, args, data, dataset_counter, dataset_bs, dataset_da, args.batch_size, criterion, split, 'beam', 3)
	# # beam 5
	evaluate(model, args, data, dataset_counter, dataset_bs, dataset_da, args.batch_size, criterion, split, 'beam', 5)
	


# global logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s]:%(message)s")


train,train_counter, train_bs, train_dialog_act, train_dialog_files, train_responses = gen_dataset_joint('train')
val, val_counter, val_bs, val_dialog_act, val_dialog_files, val_responses = gen_dataset_joint('val')
test, test_counter, test_bs, test_dialog_act, test_dialog_files, test_responses =gen_dataset_joint('test')

# # save datasets
# with open('data/train_joint.pkl', 'wb') as f:
# 	pkl.dump([train,train_counter, train_bs, train_dialog_act, train_dialog_files, train_responses], f)
# with open('data/val_joint.pkl', 'wb') as f:
# 	pkl.dump([val, val_counter, val_bs, val_dialog_act, val_dialog_files, val_responses], f)
# with open('data/test_joint.pkl', 'wb') as f:
# 	pkl.dump([test, test_counter, test_bs, test_dialog_act, test_dialog_files, test_responses],f)

# # load datasets
# with open('data/train_joint.pkl', 'rb') as f:
# 	train,train_counter, train_bs, train_dialog_act, train_dialog_files, train_responses = pkl.load(f)
# with open('data/val_joint.pkl', 'rb') as f:
# 	val, val_counter, val_bs, val_dialog_act, val_dialog_files, val_responses =pkl.load(f)
# with open('data/test_joint.pkl', 'rb') as f:
# 	test, test_counter, test_bs, test_dialog_act, test_dialog_files, test_responses=pkl.load(f)


max_sent_len = 50

idxtoword, wordtoidx = build_vocab_freqbased(load=False)
vocab_size = len(idxtoword)

print('length of vocab: ', vocab_size)
# print(idxtoword)

ntokens=len(wordtoidx)

BLEU_calc = BLEUScorer() 
F1_calc = F1Scorer()
# Use these in evaluation - to convert predictions to total vocab
d_to_imap = {}
s_to_imap = {}
a_to_imap = {}
for w, v in Constants.V_domains_wtoi.items():
	d_to_imap[v]=wordtoidx[w]
for w, v in Constants.V_slots_wtoi.items():
	s_to_imap[v]=wordtoidx[w]
for w, v in Constants.V_actions_wtoi.items():
	a_to_imap[v]=wordtoidx[w]


def run(args, optuna_callback=None):
	global logger 

	if args.log_path!="notset":
		log_path = args.log_path
	elif args.model_type=="action_pred":
		log_path ='running/action_pred/'
	elif args.model_type=="joint":
		log_path ='running/joint_simple/'
	elif args.model_type=="joint_v2":
		log_path ='running/joint_v2/'
	elif args.model_type=="joint_v2_2":
		log_path ='running/joint_v2_2/'
	elif args.model_type=="joint_v3":
		log_path ='running/joint_v3/'
	elif args.model_type=="joint_v4":
		log_path ='running/joint_v4/'
	else:
		print('Invalid model type')
		raise ValueError

	if not os.path.isdir(log_path[:-1]):
		os.makedirs(log_path[:-1])

	args.log_path = log_path

	# file logger
	time_stamp = '{:%d-%m-%Y_%H:%M:%S}'.format(datetime.now())
	fh = logging.FileHandler(log_path+'train_'+ time_stamp  +'.log', mode='a')
	fh.setLevel(logging.DEBUG)
	fh.setFormatter(formatter)
	logger.addHandler(fh)

	# console logger - add it when running it on gpu directly to see all sentences
	ch = logging.StreamHandler()
	ch.setLevel(logging.DEBUG)
	ch.setFormatter(formatter)
	logger.addHandler(ch)
	
	logger.debug('===> \n\n' + str(args) + '\n===>\n\n')

	os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

	# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # for single device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(device)
	torch.backends.cudnn.benchmark=True
	
	max_sent_len = 50

	if args.model_type=="action_pred":
		model = Action_predictor(ntokens, args.embedding_size, args.nhead, args.nhid, args.nlayers_e1, args.nlayers_e2, args.dropout)
		criterion = nn.BCELoss()
	elif args.model_type=="joint":
		model = Joint_model(ntokens, args.embedding_size, args.nhead, args.nhid, args.nlayers_e1, args.nlayers_e2, args.nlayers_d, args.dropout).to(device)
		criterion = nn.CrossEntropyLoss(ignore_index=0)
	elif args.model_type=="joint_v2":
		model = Joint_model_v2(ntokens, args.embedding_size, args.nhead, args.nhid, args.nlayers_e1, args.nlayers_e2, args.nlayers_d, args.dropout).to(device)
		criterion = nn.CrossEntropyLoss(ignore_index=0)
	elif args.model_type=="joint_v2_2":
		model = Joint_model_v2_2(ntokens, args.embedding_size, args.nhead, args.nhid, args.nlayers_e1, args.nlayers_e2, args.nlayers_d, args.dropout).to(device)
		criterion = nn.CrossEntropyLoss(ignore_index=0)
	elif args.model_type=="joint_v3":
		model = Joint_model_v3(ntokens, args.embedding_size, args.nhead, args.nhid, args.nlayers_e1, args.nlayers_e2, args.nlayers_d, args.dropout).to(device)
		criterion = nn.CrossEntropyLoss(ignore_index=0)
	elif args.model_type=="joint_v4":
		model = Joint_model_v4(ntokens, args.embedding_size, args.nhead, args.nhid, args.nlayers_e1, args.nlayers_e2, args.nlayers_d, args.dropout).to(device)
		criterion = nn.CrossEntropyLoss(ignore_index=0)

	seed = 123
	torch.manual_seed(seed)

	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.backends.cudnn.benchmark = True
		torch.set_default_tensor_type('torch.cuda.FloatTensor')
		# using data parallel
		model = nn.DataParallel(model, device_ids=[0,1], dim=1)
		print('putting model on cuda')
		model.to(device)
		criterion.to(device)

	print('Total number of trainable parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad)/float(1000000), 'M')
		
	optimizer = torch.optim.Adam(model.parameters(), lr= 0.000125, betas=(0.9, 0.98), eps=1e-9)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=4, gamma=0.9)

	logger.debug('\n\n\n=====>\n')

	if args.model_type=="action_pred":
		# Train only action prediction
		load_model(model, 'checkpoint_ap.pt')
		train_action_pred(model, args, criterion, optimizer, scheduler)
		evaluate_action_pred(model, args, criterion, optimizer, scheduler, 'test')
		return

	# best_val_loss_ground = load_model(model, args.log_path +'checkpoint_criteria.pt')
	#_ = training(model, args, criterion, optimizer, scheduler, optuna_callback)
	best_val_loss_ground = load_model(model, args.log_path + 'checkpoint_criteria.pt') #load model with best criteria


	logger.debug('Testing model\n')
	# _,test_bleu ,test_f1 ,test_matches,test_successes = testing(model, args, criterion, 'test', 'greedy')
	# logger.debug('Test critiera: {:0.3f}'.format(test_bleu+0.5*(test_matches+test_successes)))

	# To get greedy, beam(2,3,5) scores for val, test 
	# test_split('val', model, args, criterion)
	test_split('test', model, args, criterion)

	#_,val_bleu ,_,val_matches,val_successes = testing(model, args, criterion, 'val', 'greedy')
	#val_criteria = val_bleu+0.5*(val_matches+val_successes)
	val_criteria =0
	return val_criteria


if __name__ == '__main__':
	parser = argparse.ArgumentParser() 

	parser.add_argument("-embed", "--embedding_size", default=160, type=int, help = "Give embedding size")
	parser.add_argument("-heads", "--nhead", default=4, type=int,  help = "Give number of heads")
	parser.add_argument("-hid", "--nhid", default=160, type=int,  help = "Give hidden size")

	parser.add_argument("-l_e1", "--nlayers_e1", default=3, type=int,  help = "Give number of layers for Encoder 1")
	parser.add_argument("-l_e2", "--nlayers_e2", default=3, type=int,  help = "Give number of layers for Encoder 2")
	parser.add_argument("-l_d", "--nlayers_d", default=3, type=int,  help = "Give number of layers for Decoder")

	parser.add_argument("-d", "--dropout",default=0.2, type=float, help = "Give dropout")
	parser.add_argument("-bs", "--batch_size", default=32, type=int, help = "Give batch size")
	parser.add_argument("-e", "--epochs", default=30, type=int, help = "Give number of epochs")
	parser.add_argument("-lr", "--learning_rate",default=0.0001, type=float, help = "Give learning rate")
	parser.add_argument("-model", "--model_type", default="joint_v2", help="Give model name one of [joint, joint_v2, joint_v2_2, joint_v3, joint_v4, action_pred]")
	parser.add_argument("-log_path", "--log_path", default="notset", help="Give log path name")

	args = parser.parse_args()
	run(args)


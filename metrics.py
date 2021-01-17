import math, torch, torch.nn as nn, torch.nn.functional as F
from collections import Counter
from nltk.util import ngrams
import json, numpy as np
import Constants
from action_pred import compute_metrics_binary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

max_sent_len = 50


def tensor_to_sents(data, tokenizer): #2d tensor or list of tensors!
	# append eos at end if not generated
	sents=[]
	eos_index = tokenizer.get_vocab()["EOS"] 
	# idxtoword = {v:k for k,v in wordtoidx.items()}
	for line in data:
		l = (line==eos_index).nonzero()[0].item()+1 if (line==eos_index).any() else max_sent_len
		# sents.append(' '.join([idxtoword[e.item()] for e in line[:l]]))
		sents.append(tokenizer.decode(line.long()[:l].tolist()))
	return sents

def compute_bs_metrics(pred, act): #N, 50
# tokenizer.decode doesn't give sos/eos
	joint_matches =0
	joint_total=0
	slot_matches=0
	slot_total=0
	for p,a in zip(pred, act):
		p = [e.strip().split(" ", 2) for e in  p.strip().split(',')]
		a = [e.strip().split(" ", 2) for e in  a.strip().split(',')]
		joint_total += len(a)
		slot_total += 3*len(a)
		for p_e, a_e in zip(p, a):
			joint_flag=1
			for e1, e2 in zip(p_e, a_e):
				if(e1==e2):
					slot_matches+=1
				else:
					joint_flag=0
			if(joint_flag):
				joint_matches+=1

	joint_acc = joint_matches/joint_total * 100
	slot_acc = slot_matches/slot_total * 100
	return joint_acc, slot_acc

def generate_hieract(act): # input - N, triplets, 3
	hieract = np.zeros((44))
	for e in act: # -3 to remove pad,sos,eos
		d = Constants.V_domains_wtoi.get(e[0], -1)
		a = Constants.V_actions_wtoi.get(e[1], -1)
		s = Constants.V_slots_wtoi.get(e[2], -1)
		if d==-1 or a==-1 or s==-1:
			continue
		hieract[d]=1
		hieract[len(Constants.domains)+a]=1
		hieract[len(Constants.domains)+len(Constants.functions)+s]=1
	return hieract

def compute_da_metrics(pred, act): # begin from first triplet(no sos) in both
	joint_matches =0
	joint_total=0
	slot_matches=0
	slot_total=0
	bs = len(pred)
	pred_hieract = np.zeros((bs, 44))
	act_hieract =  np.zeros((bs, 44))

	for idx, (p,a) in enumerate(zip(pred, act)):
		p = [e.strip().split(" ", 2) for e in  p.strip().split(',')]
		a = [e.strip().split(" ", 2) for e in  a.strip().split(',')]
		joint_total += len(a)
		slot_total += 3*len(a)
		for p_e, a_e in zip(p, a):
			joint_flag=1
			for e1, e2 in zip(p_e, a_e):
				if(e1==e2):
					slot_matches+=1
				else:
					joint_flag=0
			if(joint_flag):
				joint_matches+=1
		pred_hieract[idx] = generate_hieract(p)
		act_hieract[idx] = generate_hieract(a)

	precision, recall, f1_score = compute_metrics_binary(pred_hieract.reshape(-1), act_hieract.reshape(-1))
	joint_acc = joint_matches/joint_total * 100
	slot_acc = slot_matches/slot_total * 100
	return (joint_acc, slot_acc), (precision, recall, f1_score)


def obtain_TP_TN_FN_FP(pred, act, TP, TN, FN, FP, elem_wise=False):
	if isinstance(pred, torch.Tensor):
		if elem_wise:
			TP += ((pred.data == 1) & (act.data == 1)).sum(0)
			TN += ((pred.data == 0) & (act.data == 0)).sum(0)
			FN += ((pred.data == 0) & (act.data == 1)).sum(0)
			FP += ((pred.data == 1) & (act.data == 0)).sum(0)
		else:
			TP += ((pred.data == 1) & (act.data == 1)).cpu().sum().item()
			TN += ((pred.data == 0) & (act.data == 0)).cpu().sum().item()
			FN += ((pred.data == 0) & (act.data == 1)).cpu().sum().item()
			FP += ((pred.data == 1) & (act.data == 0)).cpu().sum().item()
		return TP, TN, FN, FP
	else:
		TP += ((pred > 0).astype('long') & (act > 0).astype('long')).sum()
		TN += ((pred == 0).astype('long') & (act == 0).astype('long')).sum()
		FN += ((pred == 0).astype('long') & (act > 0).astype('long')).sum()
		FP += ((pred > 0).astype('long') & (act == 0).astype('long')).sum()
		return TP, TN, FN, FP


class F1Scorer(object):
	## BLEU score calculator via GentScorer interface
	## it calculates the BLEU-4 by taking the entire corpus in
	## Calulate based multiple candidates against multiple references
	def __init__(self):
		pass

	def score(self, old_hypothesis, old_corpus, wordtoidx):
		hypothesis = []
		corpus = []
		if torch.is_tensor(old_hypothesis) or torch.is_tensor(old_hypothesis[0]):
			old_hypothesis = tensor_to_sents(old_hypothesis, wordtoidx)
		if torch.is_tensor(old_corpus):
			old_corpus = tensor_to_sents(old_corpus, wordtoidx)

		for h, c in zip(old_hypothesis, old_corpus):
			# print(h, '\n',  c, '\n\n')
			hypothesis.append([h])
			corpus.append([c])
			
		# containers
		with open('data/placeholder.json') as f:
			placeholder = json.load(f)['placeholder']

		TP, TN, FN, FP = 0, 0, 0, 0
		# accumulate ngram statistics
		for hyps, refs in zip(hypothesis, corpus):
			hyps = [hyp.split() for hyp in hyps]
			refs = [ref.split() for ref in refs]
			
			# Shawn's evaluation
			#refs[0] = [u'GO_'] + refs[0] + [u'EOS_']
			#hyps[0] = [u'GO_'] + hyps[0] + [u'EOS_']
			hyps[0] = ['SOS'] + hyps[0]
			if hyps[0][-1]!='EOS':
				hyps[0] = hyps[0] + ['EOS']
			
			for hyp, ref in zip(hyps, refs):
				pred = np.zeros((len(placeholder), ), 'float32')
				gt = np.zeros((len(placeholder), ), 'float32')
				for h in hyp:
					if h in placeholder:
						pred[placeholder.index(h)] += 1
				for r in ref:
					if r in placeholder:
						gt[placeholder.index(r)] += 1
				TP, TN, FN, FP = obtain_TP_TN_FN_FP(pred, gt, TP, TN, FN, FP)

		precision = TP / (TP + FP + 0.001)
		recall = TP / (TP + FN + 0.001)
		F1 = 2 * precision * recall / (precision + recall + 0.001)
		return F1



class BLEUScorer(object):
	## BLEU score calculator via GentScorer interface
	## it calculates the BLEU-4 by taking the entire corpus in
	## Calulate based multiple candidates against multiple references
	def __init__(self):
		pass

	def score(self, old_hypothesis, old_corpus,wordtoidx,  n=1):
		hypothesis = []
		corpus = []
		if torch.is_tensor(old_hypothesis) or torch.is_tensor(old_hypothesis[0]):
			old_hypothesis = tensor_to_sents(old_hypothesis, wordtoidx)
		if torch.is_tensor(old_corpus):
			old_corpus = tensor_to_sents(old_corpus, wordtoidx)

		for h, c in zip(old_hypothesis, old_corpus):
			# print(h, '\n',  c, '\n\n')
			hypothesis.append([h])
			corpus.append([c])
		
		# print(corpus)
		
		# containers
		count = [0, 0, 0, 0]
		clip_count = [0, 0, 0, 0]
		r = 0
		c = 0
		weights = [0.25, 0.25, 0.25, 0.25]
		# accumulate ngram statistics
		for hyps, refs in zip(hypothesis, corpus):
			hyps = [hyp.split() for hyp in hyps]
			refs = [ref.split() for ref in refs]
			

			# Shawn's evaluation (have SOS and EOS)
			hyps[0] =  ['SOS'] + hyps[0]
			if hyps[0][-1]!='EOS':
				hyps[0] = hyps[0] + ['EOS']
			# print(hyps, '\n',  refs, '\n\n')

			for idx, hyp in enumerate(hyps):
				for i in range(4):
					# accumulate ngram counts
					hypcnts = Counter(ngrams(hyp, i + 1))
					cnt = sum(hypcnts.values())
					count[i] += cnt

					# compute clipped counts
					max_counts = {}
					for ref in refs:
						refcnts = Counter(ngrams(ref, i + 1))
						for ng in hypcnts:
							max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
					clipcnt = dict((ng, min(count, max_counts[ng]))                                    for ng, count in hypcnts.items())
					clip_count[i] += sum(clipcnt.values())

				# accumulate r & c
				bestmatch = [1000, 1000]
				for ref in refs:
					if bestmatch[0] == 0: break
					diff = abs(len(ref) - len(hyp))
					if diff < bestmatch[0]:
						bestmatch[0] = diff
						bestmatch[1] = len(ref)
				r += bestmatch[1]
				c += len(hyp)
				if n == 1:
					break
		# computing bleu score
		p0 = 1e-7
		bp = 1 if c > r else math.exp(1 - float(r) / float(c))
		p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0                 for i in range(4)]
		s = math.fsum(w * math.log(p_n)                       for w, p_n in zip(weights, p_ns) if p_n)
		bleu = bp * math.exp(s)
		return bleu


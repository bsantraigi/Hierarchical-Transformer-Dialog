# ## multiwoz

import math, torch, torch.nn as nn, torch.nn.functional as F
import pickle as pkl, random
# from nltk.translate.bleu_score import sentence_bleu

import numpy as np
from torch.autograd import Variable
# import matplotlib.pyplot as plt
import time
import gc
import  os, sys
from datetime import datetime
from collections import Counter
import Constants

max_sent_len = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

def tokenize_en(sentence):
#     return [tok.text for tok in en.tokenizer(sentence)]
    return sentence.split()

def print_tensors():
    total=0 
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                # print(type(obj),  obj.size())
                total += torch.numel(obj)*4
        except:
            pass
    
    print("{} GB".format(total/((1024**3) )))


def stat_cuda(msg):
    print('--', msg)
    print('allocated: %.2fG, max allocated: %.2fG, cached: %.2fG, max cached: %.2fG' % (
        torch.cuda.memory_allocated() / 1024 / 1024/1024,
        torch.cuda.max_memory_allocated() / 1024 / 1024/1024,
        torch.cuda.memory_cached() / 1024 / 1024/1024,
        torch.cuda.max_memory_cached() / 1024 / 1024/1024
    ))

def check_nan(t, name):
    if (t!=t).any():
        print("found nan, in ", name)


def Rand(start, end, num): 
    res = [] 
    for j in range(num): 
        res.append(random.randint(start, end))   
    return res 


# class batch from annotated transformer with acts
def data_gen_acts(dataset, dataset_bs, dataset_da, batch_size, i, wordtoidx):
    # print(i, len(dataset))
    max_dial_len = len(dataset[i])-1

    upper_bound = i+batch_size
    tokenized_seq = []
    tokenized_bs = []
    tokenized_da = []
    bs_output = []
    da_output = []
    for d, bs, da in zip(dataset[i:upper_bound], dataset_bs[i:upper_bound], dataset_da[i:upper_bound]):
#         print(len(d), end=' ')
        tokenized_seq.append([[wordtoidx.get(word, 1) for word in tokenize_en(sent)] for sent in d])
        tokenized_bs.append([wordtoidx.get(word, 1) for word in tokenize_en(bs)[:-1]])
        tokenized_da.append([wordtoidx.get(word, 1) for word in tokenize_en(da)[:-1]])
        bs_output.extend(bs.split()[1:]) # combine for all samples, split later
        da_output.extend(da.split()[1:])

    bs_output=[[Constants.V_domains_wtoi[w] for w in bs_output[::2]],
               [Constants.V_slots_wtoi[w] for w in bs_output[1::2]]]

    # 2, 25, batch_size
    bs_output = torch.tensor(bs_output, device=device) # 2, bs*triplets
    bs_output = bs_output.reshape(2, batch_size, -1).transpose(1, 2)

    da_output=[[Constants.V_domains_wtoi[w] for w in da_output[::3]], 
               [Constants.V_actions_wtoi[w] for w in da_output[1::3]],
               [Constants.V_slots_wtoi[w] for w in da_output[2::3]] ]

    # 3, 17, batch_size
    da_output = torch.tensor(da_output, device=device) # 3, bs*triplets
    da_output = da_output.reshape(3, batch_size, -1).transpose(1,2)

    seq_lengths = torch.LongTensor([min(len(seq), max_sent_len) for seq in tokenized_seq])
    seq_tensor = torch.zeros(batch_size, max_dial_len, max_sent_len, device=device)

    target_tensor = torch.zeros(batch_size, max_sent_len, device=device)
    label_tensor = torch.zeros(batch_size, max_sent_len, device=device)

    bs_tensor = torch.tensor(tokenized_bs, device=device).transpose(0,1) # 50, batch_size
    da_tensor = torch.tensor(tokenized_da, device=device).transpose(0,1) # 51, batch_size

    for idx,(seq, seqlen) in enumerate(zip(tokenized_seq, seq_lengths)):
        for i in range(seqlen-1):
            seq_tensor[idx, i, :len(seq[i])] = torch.LongTensor(seq[i])
        # last sentence in dialog
        target_tensor[idx, :len(seq[seqlen-1])] = torch.LongTensor(seq[seqlen-1]) 
        # last sentence in dialog from first word, ie without sos
        label_tensor[idx, :len(seq[seqlen-1])-1] = torch.LongTensor(seq[seqlen-1][1:]) 
    
    seq_tensor = seq_tensor.transpose(1,2).reshape(batch_size, -1).transpose(0,1)
    # seq_tensor - (msl*mdl , bs)
    target_tensor = target_tensor.transpose(0,1)
    label_tensor = label_tensor.transpose(0,1)

    # no sos in bs_output and da_output
    return seq_tensor.long(), target_tensor.long(), label_tensor.long(), bs_tensor.long(), da_tensor.long(), bs_output.long(), da_output.long()


def data_loader_acts(dataset, dataset_counter, dataset_bs, dataset_da, batch_size, wordtoidx): 
    # return batches according to dialog len, -> all similar at once
    # do mask also for these
    prev=0
    for dial_len, val in dataset_counter.items():
        for i in range(prev, prev+val, batch_size):
#             print(i, min(batch_size, prev+val-i))
            yield data_gen_acts(dataset, dataset_bs, dataset_da, min(batch_size, prev+val-i), i, wordtoidx)
        #     break # uncomment both break to run 1 batch for SET++,HIER++,joint models
        # break
        prev += val


# Create batches for action predictor
def data_gen_action_pred(dataset, belief_states, act_vecs, batch_size, i, wordtoidx):
    # print(i, len(dataset))

    max_dial_len = len(dataset[i])-1

    upper_bound = i+batch_size
    tokenized_seq = []
    tokenized_bs = []

    for d in dataset[i:upper_bound]:
        tokenized_seq.append([wordtoidx.get(word, 1) for word in tokenize_en(d[-1])])

    for bs in belief_states[i:upper_bound]:
        tokenized_bs.append([wordtoidx.get(word, 1) for word in tokenize_en(bs)])

    batch_actvecs = torch.tensor(act_vecs[i:upper_bound], device=device)
    seq_tensor = torch.zeros(batch_size, max_sent_len, device=device)
    bs_tensor = torch.zeros(batch_size, max_sent_len, device=device)

    for idx, (seq, bs) in enumerate(zip(tokenized_seq, tokenized_bs)):
        seq_tensor[idx, :len(seq)] = torch.LongTensor(seq)
        bs_tensor[idx, :len(bs)] = torch.LongTensor(bs)

    seq_tensor = seq_tensor.transpose(0,1) # seq_tensor - (msl , bs)
    bs_tensor = bs_tensor.transpose(0, 1) # (msl, bs)
    batch_actvecs = batch_actvecs.transpose(0,1) # batch_actvecs - 44, bs
    
    return seq_tensor.long(), bs_tensor.long(), batch_actvecs.float()

# DataLoader for action prediction
def data_loader_action_pred(dataset, dataset_counter, belief_states, act_vecs, batch_size, wordtoidx): 
    # return batches according to dialog len, -> all similar at once
    # do mask also for these
    prev=0
    for dial_len, val in dataset_counter.items():
        for i in range(prev, prev+val, batch_size):
            # print(i, min(batch_size, prev+val-i))
            yield data_gen_action_pred(dataset, belief_states, act_vecs, min(batch_size, prev+val-i), i, wordtoidx)
        #     break # uncomment both break's to run for 1 batch for Action Prediction
        # break
        prev += val


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.norm())
#             print(n, p.grad)
            if p.grad.abs().max()==0.0:
                print('grad became zero: ',n)

    # plt.figure(figsize=(16, 20))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical", fontsize=6)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
#     plt.tight_layout()
#     plt.savefig("temp.png")
    plt.show()

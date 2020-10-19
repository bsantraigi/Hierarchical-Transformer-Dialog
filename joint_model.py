## Joint model

import math, torch, torch.nn as nn, torch.nn.functional as F
from Beam import Beam
import numpy as np
import Constants

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

def _gen_mask_sent(sz):
	mask = ((torch.triu(torch.ones(sz, sz, device=device)) == 1) * 1.0).transpose(0,1)
	mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, 0)    
	return mask

def _gen_mask_hierarchical(src_len, tgt_len):        
    t = torch.zeros(src_len, src_len, device=device)
    f = torch.ones(tgt_len, tgt_len, device=device)
    for i in range(0, src_len, tgt_len):
        t[i:i+tgt_len, i:i+tgt_len]=f
        t[i:i+tgt_len, -tgt_len:] = f
    t = t.float().masked_fill(t==0, float('-inf')).masked_fill(t==1, 0)
    return t

class PositionalEncoding(nn.Module):
	def __init__(self, d_model, dropout, max_len= 5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)#size = (max_len, 1)
		
		if d_model%2==0:
			div_term = torch.exp(torch.arange(0, d_model,2).float() * (-math.log(10000.0)/d_model))
			pe[:, 0::2] = torch.sin(position * div_term)
			pe[:, 1::2] = torch.cos(position*div_term)
		else:
			div_term = torch.exp(torch.arange(0, d_model+1, 2).float() * (-math.log(10000.0)/d_model))
			pe[:, 0::2] = torch.sin(position * div_term)
			pe[:, 1::2] = torch.cos(position * div_term[:-1])

		pe = pe.unsqueeze(1) # size - (max_len, 1, d_model)
		self.register_buffer('pe', pe)
		# print('POS ENC. :', pe.size()) # 5000,1,embed_size
	
	def forward(self, x): # 1760xbsxembed
		x = x+self.pe[:x.size(0), :, :].repeat(1, x.size(1), 1)
		return self.dropout(x)


class Joint_model(nn.Module):
	def __init__(self, ntoken, ninp, nhead, nhid, nlayers_e1, nlayers_e2, nlayers_d, dropout):
		# ninp is embed_size
		super(Joint_model, self).__init__()
		from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
		
		encoder_layers1 = TransformerEncoderLayer(ninp, nhead, nhid, dropout) ## sizes
		self.transformer_encoder=TransformerEncoder(encoder_layers1, nlayers_e1)

		encoder_layers2 = TransformerEncoderLayer(ninp, nhead, nhid, dropout, activation='relu')
		self.transformer_encoder_sent = TransformerEncoder(encoder_layers2, nlayers_e2)

		# Response decoder
		decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout, activation='relu')
		self.response_decoder = TransformerDecoder(decoder_layers, nlayers_d)
		# Belief state decoder
		bs_decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout, activation='relu')
		self.bs_decoder = TransformerDecoder(bs_decoder_layers, nlayers_d)
		# Dialog action decoder
		da_decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout, activation='relu')
		self.da_decoder = TransformerDecoder(da_decoder_layers, nlayers_d)
		
		self.encoder = nn.Embedding(ntoken, ninp)
		self.decoder = nn.Linear(ninp, ntoken)
		self.pos_encoder = PositionalEncoding(ninp, dropout)
		
		self.ninp = ninp

		self.linear_d = nn.Linear(self.ninp, len(Constants.V_domains))
		self.linear_s = nn.Linear(self.ninp, len(Constants.V_slots))
		self.linear_a = nn.Linear(self.ninp, len(Constants.V_actions))

		self.mask_func = _gen_mask_hierarchical
		# self.max_sent_len = tgt_mask.size(0)
		self._reset_parameters()

	def _reset_parameters(self):
		r"""Initiate parameters in the transformer model."""
		for n, p in self.named_parameters():
			if p.dim() > 1:
				# print(n)
				torch.nn.init.xavier_normal_(p)
	
	def init_weights(self):
		initrange = 0.1
		self.encoder.weight.data.uniform_(-initrange, initrange)
		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange, initrange)

	def compute_encoder_output(self, src):
		max_sent_len = 50
		src_mask = torch.zeros(max_sent_len,max_sent_len, device=device)
		batch_size = src.shape[1]
		
		max_dial_len = src.reshape(max_sent_len, -1, batch_size).shape[1]
		
		src_sent = src.reshape(max_sent_len, -1, batch_size).transpose(0,1).reshape(-1, batch_size)
		src_pad_mask_sent= (src_sent==0).transpose(0,1)
		
		# this mask depends on mdl, make dynamically
		src_mask_sent = self.mask_func(max_dial_len*max_sent_len, max_sent_len)

		src = src.reshape(max_sent_len, -1)
		src_pad_mask = (src==0).transpose(0,1)

		src = self.encoder(src) * math.sqrt(self.ninp)
		src = self.pos_encoder(src)

		# encoder 1
		memory_inter = self.transformer_encoder(src, src_mask, src_pad_mask)

		memory_inter = memory_inter.view(max_sent_len, -1, batch_size, self.ninp).transpose(0,1).reshape(-1, batch_size, self.ninp)

		# encoder 2
		memory_inter = self.pos_encoder(memory_inter) # mdl*msl, bs, embed
		memory = self.transformer_encoder_sent(memory_inter, src_mask_sent, src_pad_mask_sent)
		return memory
	
	def forward(self, src, belief, da, tgt):
		max_sent_len = 50
		batch_size = tgt.shape[1]

		tgt_mask = _gen_mask_sent(tgt.shape[0])		
		tgt_pad_mask = (tgt==0).transpose(0,1)

		memory = self.compute_encoder_output(src)

		# Belief state decoder
		# Belief state:  #[belief start, domain slot_name, .., belief end] - msl, bs
		bs_mask = _gen_mask_sent(belief.shape[0])
		bs_pad_mask = (belief==0).transpose(0,1)
		belief = self.encoder(belief)*math.sqrt(self.ninp) # 2*max_triplets, bs, embed
		pred_belief = self.bs_decoder(belief, memory, tgt_mask=bs_mask, tgt_key_padding_mask=bs_pad_mask) # 2*max_triplets, bs, embed
		pred_belief = pred_belief.transpose(0,1).reshape(batch_size, -1, 2*self.ninp).transpose(0,1) # max_triplets, bs, 2*embed
		
		pred_belief_domains = self.linear_d(pred_belief[:,:, :self.ninp]) # max_triplets,bs, Vdomain
		pred_belief_slots = self.linear_s(pred_belief[:,:, self.ninp: ]) # max_triplets, bs, Vslots
		out_belief = [pred_belief_domains, pred_belief_slots]

		# Dialog Act decoder
		da_mask = _gen_mask_sent(da.shape[0])
		da_pad_mask = (da==0).transpose(0,1)
		da = self.encoder(da)*math.sqrt(self.ninp) # 3*max_triplets, bs, embed
		pred_da = self.da_decoder(da, torch.cat([memory,belief]) , tgt_mask=da_mask, tgt_key_padding_mask=da_pad_mask)

		pred_da = pred_da.transpose(0,1).reshape(batch_size, -1, 3*self.ninp).transpose(0,1) # max_triplets, bs, 3
		pred_da_domains = self.linear_d(pred_da[:,:,:self.ninp]) # max_triplets, bs, Vdomain
		pred_da_actions = self.linear_a(pred_da[:,:,self.ninp:2*self.ninp]) #max_triplets, bs Vactions
		pred_da_slots = self.linear_s(pred_da[:,:,2*self.ninp:]) # max_triplets, bs, Vslots
		out_da = [pred_da_domains, pred_da_actions, pred_da_slots]

		# Response decoder - 
		# tgt shape - (msl, batch_size, embed)
		tgt = self.encoder(tgt) * math.sqrt(self.ninp)
		tgt = self.pos_encoder(tgt)
		output = self.response_decoder(tgt, torch.cat([memory, belief, da]), tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask)
		output = self.decoder(output)
		return output, out_belief, out_da

	def decode_belief_state(self, belief, memory): # pass curr_belief - 2*L, bs
		batch_size = belief.shape[1]
		bs_mask = _gen_mask_sent(belief.shape[0])
		bs_pad_mask = (belief==0).transpose(0,1)

		belief = self.encoder(belief)*math.sqrt(self.ninp) # 2*triplets, bs, embed
		pred_belief = self.bs_decoder(belief, memory, tgt_mask=bs_mask, tgt_key_padding_mask=bs_pad_mask) # 2*max_triplets, bs, embed
		pred_belief = pred_belief.transpose(0,1).reshape(batch_size, -1, 2*self.ninp)
		
		pred_belief_domains=self.linear_d(pred_belief[:,-1, :self.ninp]).unsqueeze(0) #1, bs, Vdomain
		pred_belief_slots = self.linear_s(pred_belief[:,-1, self.ninp:]).unsqueeze(0) #1, bs, Vslots

		belief_logits = [pred_belief_domains, pred_belief_slots]
		pred_belief = torch.cat([torch.max(pred_belief_domains,dim=2)[1], torch.max(pred_belief_slots,dim=2)[1]]) # 2, 32
		return pred_belief, belief_logits

	def decode_dialog_act(self, da, belief, memory):
		batch_size = da.shape[1]
		da_mask = _gen_mask_sent(da.shape[0])
		da_pad_mask = (da==0).transpose(0,1)
		da = self.encoder(da)*math.sqrt(self.ninp) # 3*max_triplets, bs, embed
		pred_da = self.da_decoder(da, torch.cat([memory,belief]), tgt_mask=da_mask, tgt_key_padding_mask=da_pad_mask)

		pred_da = pred_da.transpose(0,1).reshape(batch_size, -1, 3*self.ninp)
		pred_da_domains = self.linear_d(pred_da[:,-1,:self.ninp]).unsqueeze(0) # 1,bs, Vdomain
		pred_da_actions = self.linear_a(pred_da[:,-1,self.ninp:2*self.ninp]).unsqueeze(0) # 1,bs, Vactions
		pred_da_slots = self.linear_s(pred_da[:,-1,2*self.ninp:]).unsqueeze(0) # 1,bs, Vslots
		pred_logits = [pred_da_domains, pred_da_actions, pred_da_slots]

		pred_da = torch.cat([torch.max(pred_da_domains,dim=2)[1], torch.max(pred_da_actions, dim=2)[1], torch.max(pred_da_slots,dim=2)[1]]) # 2, 32

		return pred_da, pred_logits

	def decode_response(self, memory, belief, da, tgt):
		# tgt shape - (msl, batch_size, embed)
		tgt_mask = _gen_mask_sent(tgt.shape[0])		
		tgt_pad_mask = (tgt==0).transpose(0,1)
		tgt = self.encoder(tgt) * math.sqrt(self.ninp)
		tgt = self.pos_encoder(tgt)
		output = self.response_decoder(tgt, torch.cat([memory, belief, da]), tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask)
		output = self.decoder(output)[-1,:,:].unsqueeze(0)
		output_max = torch.max(output, dim=2)[1]
		return output, output_max

	def to_vocab_index(self, v, imap): # all v should be for one imap
		# v.apply_(lambda y: imap[y])
		for key,value in imap.items():
			v[v==key]=value
		return v

	def greedy_search(self, src, batch_size, imaps):
		# Index 2 is SOS, index 3 is EOS in all vocabs
		max_sent_len = 50
		max_dial_len = src.reshape(max_sent_len, -1, batch_size).shape[1]

		tgt = 2*torch.ones(1, batch_size , device=device).long()
		eos_tokens = 3*torch.ones(1, batch_size, device=device).long()
		belief = 2*torch.ones(2, batch_size , device=device).long() # 2, bs #used in forward - in terms of vocab index
		belief_out = 2*torch.ones(2, batch_size , device=device).long() # 2, bs
		belief_eos = 3*torch.ones(2, batch_size , device=device).long()
		da = 2*torch.ones(3, batch_size , device=device).long() # 3, bs
		da_out = 2*torch.ones(3, batch_size , device=device).long() # 3, bs

		memory = self.compute_encoder_output(src)

		# Generate belief state
		for i in range(1, max_sent_len//2): # predict 48 words + sos+eos=50
			cur_belief, cur_logits = self.decode_belief_state(belief, memory)
			if i==1:
				belief_logits = [cur_logits[0], cur_logits[1]]
			else:
				belief_logits[0] = torch.cat([belief_logits[0], cur_logits[0]])
				belief_logits[1] = torch.cat([belief_logits[1], cur_logits[1]])
			belief_out = torch.cat([belief_out, cur_belief])
			# cur_belief - 2,32
			cur_belief[0] = self.to_vocab_index(cur_belief[0], imaps[0]) 
			cur_belief[1] = self.to_vocab_index(cur_belief[1], imaps[1]) 
			belief = torch.cat([belief, cur_belief])

		# belief - [50, 32], belief_logits - each ele - ([24, 32, V_d/a/s]) 
		# - while computing bs loss remove SOS in belief targets with logits.

		# belief = torch.cat([belief, belief_eos]) # should I append EOS at end?

		bs_mask = _gen_mask_sent(belief.shape[0])
		bs_pad_mask = (belief==0).transpose(0,1)
		belief_memory = self.encoder(belief)*math.sqrt(self.ninp) # 2*max_triplets, bs, embed

		# Generate dialog act
		for i in range(1, (max_sent_len+1)//3):
			cur_da, cur_logits = self.decode_dialog_act(da, belief_memory, memory)
			if i==1:
				da_logits = [cur_logits[0], cur_logits[1], cur_logits[2]]
			else:
				da_logits[0] = torch.cat([da_logits[0], cur_logits[0]])
				da_logits[1] = torch.cat([da_logits[1], cur_logits[1]])
				da_logits[2] = torch.cat([da_logits[2], cur_logits[2]])
			da_out = torch.cat([da_out, cur_da])
			# cur_da - 3, 32
			cur_da[0] = self.to_vocab_index(cur_da[0], imaps[0])
			cur_da[1] = self.to_vocab_index(cur_da[1], imaps[2])
			cur_da[2] = self.to_vocab_index(cur_da[2], imaps[1])
			da = torch.cat([da, cur_da])

		# da - [51, 32], da_logits - each ele - ([16, 32, V_d/a/s]) 
		# da = torch.cat([da, eos_tokens])

		da_mask = _gen_mask_sent(da.shape[0])
		da_pad_mask = (da==0).transpose(0,1)
		da_memory = self.encoder(da)*math.sqrt(self.ninp) # 3*max_triplets, bs, embed

		# Generate response
		for i in range(1, max_sent_len+1): # predict 48 words + sos+eos=50
			output, output_max = self.decode_response(memory, belief_memory, da_memory, tgt)
			# print('output ', output.shape) # i, bs, vocab
			if i==1:
				logits = output
			else:
				logits = torch.cat([logits, output], dim=0)
			
			tgt = torch.cat([tgt, output_max], dim=0)

		tgt = torch.cat([tgt[:49,:], eos_tokens], dim=0)
		return logits, tgt, belief_logits, belief_out, da_logits, da_out

		
	def translate_batch(self, src, act_vecs, n_bm, batch_size): # , src_pad_mask, tgt_pad_mask
		# adopted from HDSA_Dialog
		device = src.device

		max_sent_len = 50
		max_dial_len = src.reshape(max_sent_len, -1, batch_size).shape[1]

		src = src.transpose(0,1) # src shape changed to (bs*mdl, msl)
		act_vecs = act_vecs.transpose(0, 1) # act_vecs changed to bs,44


		def collate_active_info(src, act_vecs, inst_idx_to_position_map, active_inst_idx_map):
			# Sentences which are still active are collected,
			# so the decoder will not run on completed sentences.
			n_prev_active_inst = len(inst_idx_to_position_map)
			active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
			active_inst_idx = torch.LongTensor(active_inst_idx).to(device)
		
			active_src_seq = collect_active_part(src, active_inst_idx, n_prev_active_inst, n_bm)      
			active_act_vecs = collect_active_part(act_vecs, active_inst_idx, n_prev_active_inst, n_bm)

			active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)       
			return active_src_seq, active_act_vecs, active_inst_idx_to_position_map
			
		def beam_decode_step(inst_dec_beams, len_dec_seq, active_inst_idx_list, src,act_vecs, inst_idx_to_position_map, n_bm):
			''' Decode and update beam status, and then return active beam idx '''
			n_active_inst = len(inst_idx_to_position_map)
				
			dec_partial_seq = [inst_dec_beams[idx].get_current_state() 
							   for idx in active_inst_idx_list if not inst_dec_beams[idx].done]
			dec_partial_seq = torch.stack(dec_partial_seq).to(device)
			dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
			
			# print( src.shape, dec_partial_seq.shape , act_vecs.shape) # src is 50, 150
			logits = self.forward(src.transpose(0,1) , dec_partial_seq.transpose(0,1), act_vecs.transpose(0, 1))[-1, :, :].unsqueeze(0) # error here

			# print(logits.shape)
			word_prob = F.log_softmax(logits, dim=2)
			word_prob = word_prob.view(n_active_inst, n_bm, -1) # active, bms, vocab 
			
			# print(inst_idx_to_position_map) # 0:0, 1:1 map

			# Update the beam with predicted word prob information and collect incomplete instances
			active_inst_idx_list = []
			for inst_idx, inst_position in inst_idx_to_position_map.items():
				is_inst_complete = inst_dec_beams[inst_idx].advance(word_prob[inst_position]) # gotta check advance method here!!
				if not is_inst_complete:
					active_inst_idx_list += [inst_idx]
		
			return active_inst_idx_list
			
		with torch.no_grad():
			# repeat src n_bm times
			# act_vecs shape is bs,44 after T

			src = src.repeat(1, n_bm).reshape(batch_size*n_bm , -1) # bm*batch_size, msl*mdl
			act_vecs = act_vecs.repeat(1, n_bm).reshape(batch_size*n_bm, -1)
			# act_vecs -> bs*n_bm, 44

			
			inst_dec_beams = [Beam(n_bm, device=device) for _ in range(batch_size)]
			active_inst_idx_list = list(range(batch_size))
			inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
			
			for len_dec_seq in range(1, max_sent_len+1):
				active_inst_idx_list = beam_decode_step(inst_dec_beams, len_dec_seq, active_inst_idx_list, src, act_vecs,  inst_idx_to_position_map, n_bm)
				if not active_inst_idx_list:
					break
				src, act_vecs,  inst_idx_to_position_map = collate_active_info(src, act_vecs, inst_idx_to_position_map, active_inst_idx_list)
				
			def collect_hypothesis_and_scores(inst_dec_beams, n_best):
				all_hyp, all_scores = [], []
				for beam in inst_dec_beams:
					scores = beam.scores
					hyps = np.array([beam.get_hypothesis(i) for i in range(beam.size)], 'long')
					lengths = (hyps != Constants.PAD).sum(-1)
					normed_scores = [scores[i].item()/lengths[i] for i, hyp in enumerate(hyps)]
					idxs = np.argsort(normed_scores)[::-1]

					all_hyp.append([hyps[idx] for idx in idxs])
					all_scores.append([normed_scores[idx] for idx in idxs])
				return all_hyp, all_scores

			batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, n_bm)
			
		batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, n_bm)
		
		result = []
		for _ in batch_hyp:
			finished = False
			for r in _:
				if len(r) >= 8 and len(r) < max_sent_len:
					result.append(r)
					finished = True
					break
			if not finished:
				result.append(_[0])
		return result
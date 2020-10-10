## Joint model

import math, torch, torch.nn as nn, torch.nn.functional as F
from Beam import Beam
import numpy as np
import Constants

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)


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

		self.act_embedding = nn.Linear(44, self.ninp)		
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
	
	def forward(self, src, belief, da, tgt):
		max_sent_len = 50
		src_mask = torch.zeros(max_sent_len,max_sent_len, device=device)
		tgt_mask = _gen_mask_sent(tgt.shape[0])		
		batch_size = tgt.shape[1]
		
		max_dial_len = src.reshape(max_sent_len, -1, batch_size).shape[1]
		
		src_sent = src.reshape(max_sent_len, -1, batch_size).transpose(0,1).reshape(-1, batch_size)
		src_pad_mask_sent= (src_sent==0).transpose(0,1)
		
		# this mask depends on mdl, make dynamically
		src_mask_sent = self.mask_func(max_dial_len*max_sent_len, max_sent_len)

		src = src.reshape(max_sent_len, -1)

		src_pad_mask = (src==0).transpose(0,1)
		tgt_pad_mask = (tgt==0).transpose(0,1)

		src = self.encoder(src) * math.sqrt(self.ninp)
		src = self.pos_encoder(src)

		# encoder 1
		memory_inter = self.transformer_encoder(src, src_mask, src_pad_mask)

		memory_inter = memory_inter.view(max_sent_len, -1, batch_size, self.ninp).transpose(0,1).reshape(-1, batch_size, self.ninp)

		# encoder 2
		memory_inter = self.pos_encoder(memory_inter) # mdl*msl, bs, embed
		memory = self.transformer_encoder_sent(memory_inter, src_mask_sent, src_pad_mask_sent) 

		# Belief state decoder
		# Belief state:  #[belief start, domain slot_name, .., belief end] - msl, bs
		bs_mask = _gen_mask_sent(belief.shape[0])
		bs_pad_mask = (belief==0).transpose(0,1)
		belief = self.WHICH_encoder(belief)*math.sqrt(self.ninp) # 2*max_triplets, bs, embed
		# belief = self.pos_encoder(belief)
		pred_belief = self.bs_decoder(belief, memory, tgt_mask=bs_mask, tgt_key_padding_mask=bs_pad_mask) # 2*max_triplets, bs, embed
		pred_belief = pred_belief.transpose(0,1).reshape(batch_size, -1, 2*self.ninp)
		pred_belief_domains = self.linear1(pred_belief[:,:,0]) # bs, max_triplets, Vdomain
		pred_belief_slots = self.linear2(pred_belief[:,:,1]) # bs, max_triplets, Vslots
		out_belief = torch.stack([pred_belief_domains, pred_belief_slots], dim=-1)

		# Dialog Act decoder
		da_mask = _gen_mask_sent(da.shape[0])
		da_pad_mask = (da==0).transpose(0,1)
		da = self.WHICH_encoder(da)*math.sqrt(self.ninp) # 3*max_triplets, bs, embed
		da = self.pos_encoder(da)
		pred_da = self.da_decoder(da, torch.stack([memory,belief]) , tgt_mask=da_mask, tgt_key_padding_mask=da_pad_mask)
		pred_da = pred_belief.transpose(0,1).reshape(batch_size, -1, 3*self.ninp)
		pred_da_domains = self.linear1(pred_da[:,:,0]) # bs, max_triplets, Vdomain
		pred_da_actions = self.linear3(pred_da[:,:,1]) # bs, max_triplets, Vactions
		pred_da_slots = self.linear2(pred_da[:,:,2]) # bs, max_triplets, Vslots
		out_da = torch.stack([pred_da_domains, pred_da_actions, pred_da_slots], dim=-1)

		# Response decoder - 
		# tgt shape - (msl, batch_size, embed)- add act_vec of (None ,bs,embed)
		tgt = self.encoder(tgt) * math.sqrt(self.ninp)
		tgt = self.pos_encoder(tgt)
		output = self.response_decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask)
		output = self.decoder(output)
		return output, out_belief, out_da


	def greedy_search(self, src, batch_size):
		max_sent_len = 50
		max_dial_len = src.reshape(max_sent_len, -1, batch_size).shape[1]
		tgt = 2*torch.ones(1, batch_size , device=device).long()
		eos_tokens = 3*torch.ones(1, batch_size, device=device).long()

		_, belief, da = self.forward(src)
		for i in range(1, max_sent_len+1): # predict 48 words + sos+eos=50
			output, _, _ = self.forward(src, belief, da, tgt)[-1,:,:].unsqueeze(0)
			# print('output ', output.shape) # i, bs, vocab
			if i==1:
				logits = output
			else:
				logits = torch.cat([logits, output], dim=0)
			output_max = torch.max(output, dim=2)[1]
			tgt = torch.cat([tgt, output_max], dim=0)

		tgt = torch.cat([tgt[:49,:], eos_tokens], dim=0)
		return logits, tgt

		
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
import numpy as np

def get_files_action_pred(split): # dataset, dataset_counter, dataset_bs_kb, dataset_actvecs
	if split=='train':
		return train, train_counter, train_bs_kb, train_hierarchial_actvecs
	if split=='val':
		return val, val_counter, val_bs_kb, val_hierarchial_actvecs
	if split=='test':
		return test, test_counter, test_bs_kb, test_hierarchial_actvecs
	return ValueError
	
def train_action_pred(model, args, criterion, optimizer, scheduler):
	ntokens = len(idxtoword)
	# nbatches = len(train)//args.batch_size
	model.train()
	best_val_f1 = -float("inf")

	for e in range(1, 1+args.epochs):
		total_loss =0
		start_time = time.time()
		optimizer.zero_grad()

		for i, (data, bs, act_vecs) in enumerate(data_loader_action_pred(train, train_counter, train_bs_kb, train_hierarchial_actvecs, args.batch_size, wordtoidx)):

			batch_size_curr = data.shape[1]
			optimizer.zero_grad()	
			output = model(data, bs)
			loss = criterion(output.reshape(-1), act_vecs.reshape(-1))
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
			optimizer.step()
			total_loss += loss.item()*batch_size_curr
			
		elapsed = time.time()-start_time
		total_loss /= len(train)
		logger.debug('==>Train Epoch {}, Train \tLoss: {:0.4f}\tTime taken: {:0.1f}'.format(e,  total_loss, elapsed))

		val_precision, val_recall, val_f1 = evaluate_action_pred(model, args, criterion, optimizer, scheduler, 'val')
		logger.debug('Val Precision: {:0.2f}\tRecall: {:0.2f}\tF1 Score: {:0.2f}'.format(precision*100, recall*100, f1_score*100))
		if val_f1 > best_val_f1:
			best_val_f1 = val_f1
			save_model(model, args, 'checkpoint_ap.pt', total_loss, -1, -1)
		
	return 

def compute_metrics_binary(y_true, y_pred):
	# precision = tp/tp+fp, recall = tp/tp+fn
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)

	tp = sum(y_true*y_pred)
	fp = sum(np.logical_not(y_true)*y_pred)
	tn = sum(np.logical_not(y_true)*np.logical_not(y_pred))
	fn = sum(y_true*np.logical_not(y_pred))
	precision = tp/(tp+fp)
	recall = tp/(tp+fn)
	f1_score = 2*precision*recall/(precision+recall)
	return precision, recall, f1_score

def evaluate_action_pred(model, args, criterion, optimizer, scheduler, split):
	model.eval()

	total_loss =0
	start_time = time.time()
	optimizer.zero_grad()
	pred_actvecs =[]

	dataset, dataset_counter, dataset_bs_kb, dataset_actvecs = get_files(split)

	for i, (data, bs, act_vecs) in enumerate(data_loader_action_pred(dataset, dataset_counter, dataset_bs_kb, dataset_actvecs, args.batch_size, wordtoidx)):

		batch_size_curr = data.shape[1]
		output = model(data, bs)
		loss = criterion(output.reshape(-1), act_vecs.reshape(-1))	
		total_loss += loss.item()*batch_size_curr
		
		pred = (output>=0.5).transpose(0,1).long().numpy()
		pred_actvecs.extend(pred)

	# dataset_actvecs = dataset_actvecs[:args.batch_size] # uncomment for one batch
	
	assert(len(dataset_actvecs)==len(pred_actvecs))

	flat_dataset_actvecs = [item for sublist in dataset_actvecs for item in sublist]
	flat_pred_actvecs = [item for sublist in pred_actvecs for item in sublist]
	# print(len(flat_dataset_actvecs) ,len(flat_pred_actvecs))

	# calculate precision, recall, f1-score from sklearn/ compute_metrics

	precision, recall, f1_score = compute_metrics_binary(flat_dataset_actvecs, flat_pred_actvecs)
	logger.debug('Action Prediction: Precision: {:0.2f}\tRecall: {:0.2f}\tF1 Score: {:0.2f}'.format(precision*100, recall*100, f1_score*100))

	elapsed = time.time()-start_time
	total_loss /= len(dataset)
	logger.debug('==>Action Prediction: {} \t Loss: {:0.4f}\tTime taken: {:0.1f}'.format(split,  total_loss, elapsed))
	return


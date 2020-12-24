
### To do
**Bpe_nondelex** </br>
main.py runs the joint model.  </br>
make adjustments to run the nondelex setting from main_acts.py - DONE </br>
1. dataset.py - use bs values
2. tokenization - random chars in tokens - check
</br>Later: </br>
d_to_imap stuff? needed?

### DATASET: (From https://github.com/wenhuchen/HDSA-Dialog)

generated using create_delex_data.py in original multiwoz repository for multiwoz 2.1 version. <br>
add preprocessed data - train.json, val.json and test.json into hdsa_data/hdsa_data/ folder <br>
Add delex.json file into data folder, large file(~87MB)

```bash
wget --directory-prefix=data/  https://hdsa-dialog.s3-us-west-2.amazonaws.com/delex.json
```

### MODELS:
set, hier, mat, set++, hier++

Number of layers: e1,e2,d

#### SET/SET++:
initially, positional embeddings wrt individual utterances are added. After e1 layers, positional embedding wrt whole dialog are addded.  
For all e1+e2 layers, attention in between individual utterances.

#### HIER/HIER++:
For e1 layers, attention in between individual utterances.
For e2 layers, cross attention between current and last utterance.
For d layers in decoder, lower triangular mask to prevent from attending to future positions of response.

#### MAT:
For e2 layers, cross attention between current and last utterance.
For d layers in decoder, lower triangular mask to prevent from attending to future positions of response.


### HYPERPARAMETERS:
Number of layers: e1,e2,d
Number of heads in each encoder/decoder


### Reproducability:

For only response generation with da as input, run: </br>
python main_acts.py

For Joint model:(model code in joint_model.py, joint_model_v2.py), run: </br>
python main.py

HDSA Dialog action prediction(model code in model.py) </br>
python main.py -model_type action_pred




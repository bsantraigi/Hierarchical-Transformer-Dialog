
### DATASET:

generated using create_delex_data.py in original multiwoz repository for multiwoz 2.1 version. <br>
add preprocessed data - train.json, val.json and test.json into hdsa_data/hdsa_data/ folder
Add delex.json file into data folder, large file(~87MB)

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

For SET/HIER/MAT:
python main.py

For SET++/HIER++:
python main_acts.py






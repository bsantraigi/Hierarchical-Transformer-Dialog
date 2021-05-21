#!/home/bishal/miniconda3/envs/py2/bin/python
# coding: utf-8

# In[1]:

import json
import os

if __name__ == "__main__":
    # Test run

    # ## Evaluating model predictions
    model_folder = "../running/transformer_hier/"
    mode = "test"
    # for x in ['greedy', 'beam_2', 'beam_3', 'beam_5']:
    x = "greedy"
    budzianowski_eval(model_folder, mode, x)

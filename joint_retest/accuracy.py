import json
from sklearn.metrics import f1_score, accuracy_score
import sys
import numpy as np
import os
#os.chdir("simpletod")
from Constants import SLOT_VALS
from dst import ignore_none, default_cleaning, IGNORE_TURNS_TYPE2
import argparse
#os.chdir("..")


'''parser = argparse.ArgumentParser()
parser.add_argument('--eval_file', default=str,
                    help='evaluate file name (json)')
parser.add_argument('--default_cleaning', action='store_true',
                    help='use default cleaning from multiwoz')
parser.add_argument('--type2_cleaning', action='store_true',
                    help='use type 2 cleaning, refer to [https://arxiv.org/abs/2005.00796]')

data = json.load(open(args.eval_file, 'r'))

num_turns = 0
joint_acc = 0

clean_tokens = ['<|endoftext|>']

for dial in data:
    dialogue_pred = data[dial]['generated_turn_belief']
    dialogue_target = data[dial]['target_turn_belief']
    model_context = data[dial]['model_context']'''

def joint_accuracy(dialogue_target,dialogue_pred,d_c = True,type2_c = True):
    num_turns = 0
    joint_acc = 0
    clean_tokens = ['<|endoftext|>']    
    for turn_id, (turn_target, turn_pred) in enumerate(
            zip(dialogue_target, dialogue_pred)):
        
        # clean
        for bs in turn_pred:
            if len(bs.split(" ")) < 3:
                turn_pred.remove(bs)
        
        for bs in turn_target:
            if len(bs.split(" ")) < 3:
                turn_target.remove(bs)

        '''for bs in turn_pred:
            if bs in clean_tokens + ['', ' '] or bs.split()[-1] == 'none':
                turn_pred.remove(bs)

        new_turn_pred = []
        for bs in turn_pred:
            for tok in clean_tokens:
                bs = bs.replace(tok, '').strip()
                new_turn_pred.append(bs)
        turn_pred = new_turn_pred'''

        turn_pred, turn_target = ignore_none(turn_pred, turn_target)

        # MultiWOZ default cleaning
        if d_c:
            turn_pred, turn_target = default_cleaning(turn_pred, turn_target)

        join_flag = False
#         if set(turn_target) == set(turn_pred):
        if set(turn_target) == set(turn_pred[:len(turn_target)]):
            joint_acc += 1
            join_flag = True
        #slot_pred , slot_tgt = get_slots(turn_pred,turn_target)

        elif type2_c: # check for possible Type 2 noisy annotations
            flag = True
            for bs in turn_target:
                if bs not in turn_pred:
                    flag = False
                    break
            if flag:
                for bs in turn_pred:
                    if bs not in dialogue_target_final:
                        flag = False
                        break

            if flag: # model prediction might be correct if found in Type 2 list of noisy annotations
                dial_name = dial.split('.')[0]
                if dial_name in IGNORE_TURNS_TYPE2 and turn_id in IGNORE_TURNS_TYPE2[dial_name]: # ignore these turns
                    pass
                else:
                    joint_acc += 1
                    join_flag = True
#         if join_flag == False:
#             print(str(turn_target) + " ===> " + str(turn_pred))

        num_turns += 1

    joint_acc /= num_turns
    return joint_acc

#print('joint accuracy: {}'.format(joint_acc))

#calculates slot accuracy
def slot_accuracy(dialogue_target,dialogue_pred):
  num_turns = 0
  slot_acc = 0
  clean_tokens = ['<|endoftext|>']
  for turn_id, (turn_target, turn_pred) in enumerate(
            zip(dialogue_target, dialogue_pred)):
    for bs in turn_pred:
            if bs in clean_tokens + ['', ' '] or bs.split()[-1] == 'none':
                turn_pred.remove(bs)

    new_turn_pred = []
    for bs in turn_pred:
            for tok in clean_tokens:
                bs = bs.replace(tok, '').strip()
                new_turn_pred.append(bs)
    turn_pred = new_turn_pred
    
    slot_pred , slot_tgt = get_slots(turn_pred,turn_target)
#     print(set(slot_tgt), " ===> ", set(slot_pred))
    slot_flag = False
#     if set(slot_tgt) == set(slot_pred):
    if set(slot_tgt) == set(slot_pred[:len(slot_tgt)]):            #Please review this part
            slot_acc += 1
            slot_flag = True

    num_turns += 1
  slot_acc /= num_turns
  return slot_acc

def fix_mismatch_jason(slot, value):
    # miss match slot and value
    if slot == "type" and value in ["nigh", "moderate -ly priced", "bed and breakfast",
                                  "centre", "venetian", "intern", "a cheap -er hotel"] or \
            slot == "internet" and value == "4" or \
            slot == "pricerange" and value == "2" or \
            slot == "type" and value in ["gastropub", "la raza", "galleria", "gallery",
                                       "science", "m"] or \
            "area" in slot and value in ["moderate"] or \
            "day" in slot and value == "t":
        value = "none"
    elif slot == "type" and value in ["hotel with free parking and free wifi", "4",
                                    "3 star hotel"]:
        value = "hotel"
    elif slot == "star" and value == "3 star hotel":
        value = "3"
    elif "area" in slot:
        if value == "no":
            value = "north"
        elif value == "we":
            value = "west"
        elif value == "cent":
            value = "centre"
    elif "day" in slot:
        if value == "we":
            value = "wednesday"
        elif value == "no":
            value = "none"
    elif "price" in slot and value == "ch":
        value = "cheap"
    elif "internet" in slot and value == "free":
        value = "yes"

    # some out-of-define classification slot values
    if slot == "area" and value in ["stansted airport", "cambridge", "silver street"] or \
            slot == "area" and value in ["norwich", "ely", "museum", "same area as hotel"]:
        value = "none"
    return slot, value

#function for getting slots
def get_slots(turn_pred,turn_target):
#     print(turn_pred, turn_target)
    pred_belief_jason = []
    target_belief_jason = []
    
    for pred in turn_pred:
        if pred in ['',' ','  '] or len(pred.split()) < 3:
            continue
        if len(pred.split())==1 or len(pred.split())==0:
            continue
        domain = pred.split()[0]
        if 'book' in pred:
            slot = ' '.join(pred.split()[1:3])
            val = ' '.join(pred.split()[3:])
        else:
            slot = pred.split()[1]
            val = ' '.join(pred.split()[2:])

        #if slot in GENERAL_TYPO:
            #val = GENERAL_TYPO[slot]

        slot, val = fix_mismatch_jason(slot, val)
        pred_belief_jason.append('{}'.format(slot))
    for tgt in turn_target:
        if tgt in ['',' ', '  '] or len(tgt.split()) < 3:
            continue
        domain = tgt.split()[0]
        if 'book' in tgt:
            slot = ' '.join(tgt.split()[1:3])
            val = ' '.join(tgt.split()[3:])
        else:
            slot = tgt.split()[1]
            val = ' '.join(tgt.split()[2:])

        #if slot in GENERAL_TYPO:
            #val = GENERAL_TYPO[slot]

        slot, val = fix_mismatch_jason(slot, val)
        target_belief_jason.append('{}'.format(slot))
    slot_pred = pred_belief_jason
    slot_target = target_belief_jason
    return slot_pred,slot_target
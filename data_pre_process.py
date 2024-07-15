import os
import re
import csv
import time
import json
import multiprocessing
from multiprocessing import Pool
from functools import partial
import traceback

import torch
from sklearn.utils import shuffle
from transformers import BertTokenizer, BertModel

### pre-process the textual post and comments into pt file to speed up the pre-training 

def padding_func(x, max_len, pad=0):
    if type(x[0]) == list:
        newx = []
        for xi in x:
            pad_list = [pad] * (max_len - len(xi))                  
            newx.append(xi+pad_list)
        return newx
    else:
        pad_list = [pad] * (max_len - len(x))
        return x+pad_list

def xaa_data_to_pt(filename, infolder, outfolder):
    ## line format: did #EOS# post #EOS# comment1 #EOC# comment2 #EOC# comment3 #EOS# ...
    
    tokenizer = BertTokenizer(vocab_file="data/WV/vocab.txt")
    filepath = os.path.join(infolder, filename+".txt")
    outpath = os.path.join(outfolder, filename+".pt")
    
    id2post = {}
    id2cmt = {}
    all_id = []
    
    with open(filepath, 'r', encoding='utf-8') as f1:
        for i, line in enumerate(f1):
            if i%20000 == 0: print(filename, i)
            line_list = line.strip().split(' #EOS# ')
            
            did = line_list[0][2:-1]
            post = line_list[1]
            
            cmts = line_list[2].split(' #EOC# ')

            epost = tokenizer(post, padding=True, truncation=True, max_length = 40)["input_ids"]
            epost = padding_func(epost, max_len=40, pad=0)
            ecmts = tokenizer(cmts, padding=True, truncation=True, max_length = 25)["input_ids"]
            ecmts = padding_func(ecmts, max_len=25, pad=0)
            all_id.append(did)
            id2post[did] = epost
            id2cmt[did] = ecmts
     
    ret = {"all_did": all_id, "id2post": id2post, "id2cmt": id2cmt}      
    torch.save(ret, outpath)   
    print(f"filename: {filename} done ")
    
  
    
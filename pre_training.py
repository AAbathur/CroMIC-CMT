# -*- coding: utf-8 -*-
import os

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4"
import sys
import time
import math
import argparse
import random
import matplotlib
#matplotlib.use("Agg")
from matplotlib import pyplot as plt
import warnings
warnings.simplefilter("ignore", UserWarning)

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import sklearn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from functools import partial
from transformers import BertTokenizer, BertModel

from model import base_model
from dataset import pre_training_dataset
from utils import epoch_time, count_parameters

from metrics import BLEU, METEOR, process_for_metrics_flat_cmt, cal_entropy_and_distinct, process_for_metrics_stacked_tgt, Evaluator

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model_save_path', default=None, type=str, help='path to save model state dict')
parser.add_argument('--text_data_folder', default="data/pre_training_data", type=str)
parser.add_argument('--image_data_folder', default="data/pre_training_data", type=str)

parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument("--print_steps", type=int, default=300, help="step number of printing info. during training ")
parser.add_argument("--text_encoder", type=str, default="Transformer", choices=["BERT","Transformer"])
parser.add_argument("--image_encoder", type=str, default="ResNet50-base")
parser.add_argument("--cmt_decoder", type=str, default="Transformer", choices=["Transformer"])

parser.add_argument("--init_lr", type=float, default=0.0001)
parser.add_argument("--CLIP", type=float, default=1.0, help="gradient clipping")
parser.add_argument("--cls", type=int, default=101)
parser.add_argument("--sep", type=int, default=102)
parser.add_argument("--pad", type=int, default=0)
parser.add_argument("--embed_size", type=int, default=768)
parser.add_argument("--tgt_vocab_size", type=int, default=21128)
parser.add_argument("--embed_weight", type=str, default=None)
parser.add_argument("--GEN", type=bool, default=True, help="pre-training with Cmt Generative or not")
parser.add_argument("--total_step", type=int, help="total train step, calculate before training")

parser.add_argument("--local_rank", type=int)
parser.add_argument("--nodes", type=int, default=1)
parser.add_argument("--world_size", type=int, default=1)
parser.add_argument("--distributed", action="store_true", default=False)

args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda", args.local_rank)
else:
    device = torch.device("cpu")
    
def setup(rank):
    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()
    
def warm_linear_decay(step, total_step, warmup=0.01):
    x = step/total_step
    if x < warmup:
        return x/warmup
    return max((x-1.)/(warmup-1.), 0)
    
def compile_fn():
    
    model = base_model.BaseCMTGeneratorEarlyFusion(image_encoder_name=args.image_encoder, 
                                   text_encoder_name=args.text_encoder, 
                                   decoder_name=args.cmt_decoder, 
                                   encoder_layer_num=6, 
                                   decoder_layer_num=6, 
                                   vocab_size=args.tgt_vocab_size, 
                                   hid_dim=args.embed_size, 
                                   embed_weight=args.embed_weight,  
                                   GEN=args.GEN,
                                   CLSVQA=False,
                                   input_modality="t+v",
                                   )
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.init_lr)
    
    criterion = nn.CrossEntropyLoss(ignore_index=args.pad)
    partial_warm_linear_decay = partial(warm_linear_decay, total_step=args.total_step, warmup=0.01)
    scheduler = LambdaLR(optimizer, lr_lambda=partial_warm_linear_decay, last_epoch=-1)

    return model, optimizer, criterion, scheduler

def data_loader_generator(mode, batch_size, epoch, num_worker, distributed=False, world_size=-1):
    if mode == "train":     
        filenames = ["example"]
        shuffle=True
    elif mode == "val":
        filenames = ["example"]
        shuffle=False
    elif mode == "test":
        filenames =["example"]
        shuffle=False
    return pre_training_dataset.lazily_data_load_lmdb_image(filenames, args.text_data_folder, args.image_data_folder, batch_size, num_worker, shuffle, epoch, distributed=distributed, world_size=world_size)


def run_iters(device, model, ds_generator, optimizer, criterion, epoch, clip, scheduler, mode="train", **kwargs):

    if mode == "train":
        model.train()
    else:
        model.eval()
    gen_evaluator = Evaluator(["ROUGE-L","CIDEr"], args.cls, args.sep, args.pad)
    
    epoch_loss, avg_step_loss = 0, 0
    epoch_bleu1, epoch_bleu2, epoch_bleu3, epoch_bleu4 = 0, 0, 0, 0
    epoch_meteor = 0
    all_pred = []
    start_time = time.time()
    batch_count = 0

    for j, ds in enumerate(ds_generator):
        for i, batch_data in enumerate(ds):
            for k, v in batch_data.items():
                if k in ["batch_post", "batch_image"]:
                    batch_data[k] = v.to(device)
                    
            batch_count += 1
            
            if mode == "pred":
                with torch.no_grad():
                    single_pred, single_output = model.predict(batch_data, 25, 101)
                batch_cmts = batch_data["batch_cmts"]
             
            elif mode == "train" or mode == "val": 
                if mode == "train":
                    out = model(batch_data)
                elif mode == "val":
                    with torch.no_grad():
                        out = model(batch_data)
                gen_logits = out["gen_logits"] ## cmt_len-1, real_bs, ntoken
                gen_pred = gen_logits.argmax(2).transpose(0,1) ## bs, cmt_len-1
                gen_labels = out["gen_labels"] ## batch_cmt[:,1:]

                cmt_num_record = out["cmt_num_record"]
                all_cmts = out["batch_cmt"]
                batch_cmts = batch_data["batch_cmts"]
                
            if mode != "pred":
                loss = criterion(gen_logits.permute(1,2,0), gen_labels)
            
                epoch_loss += loss.item()
                avg_step_loss += loss.item()
                
            if mode == "train":
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                scheduler.step()

            if ((args.distributed and args.local_rank == 1) or (not args.distributed)) and mode == "train" and batch_count % args.print_steps == 0:
                
                end_time = time.time()
                avg_step_loss /= args.print_steps
                print('iter: {}, time_cost: {:.4f} seconds'.format( batch_count, end_time-start_time))
                print("\tcurrent lr: {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
                print("\tloss: {:.6f}, avg_step_loss: {:.6f}".format(loss.item(), avg_step_loss))
                
                avg_step_loss = 0
                start_time = time.time()
                
            elif mode == "val":
                gen_evaluator.stacked_batch_tracker(gen_pred.tolist(), all_cmts.tolist(), cmt_num_record)
                real_pred, real_targets = process_for_metrics_flat_cmt(gen_pred, all_cmts, args.cls, args.sep, args.pad, cmt_num_record)

                all_pred.extend(real_pred)
                bleu1, bleu2, bleu3, bleu4 = BLEU(real_pred, real_targets)
                meteor_score = METEOR(real_pred, real_targets)

                epoch_bleu1 += bleu1
                epoch_bleu2 += bleu2
                epoch_bleu3 += bleu3
                epoch_bleu4 += bleu4
                
                epoch_meteor += meteor_score
                
            elif mode == "pred":
                gen_evaluator.batch_tracker(single_pred.tolist(), batch_cmts)
                real_pred, real_targets = process_for_metrics_stacked_tgt(single_pred, batch_cmts, args.cls, args.sep, args.pad)
                all_pred.extend(real_pred)
                bleu1, bleu2, bleu3, bleu4 = BLEU(real_pred, real_targets) 
                meteor_score = METEOR(real_pred, real_targets)

                epoch_bleu1 += bleu1
                epoch_bleu2 += bleu2
                epoch_bleu3 += bleu3
                epoch_bleu4 += bleu4
                epoch_meteor += meteor_score
                
    runiter_output = {}
    
    if mode == "train":
        runiter_output["epoch_loss"] = epoch_loss / batch_count
    
    elif mode == "val" or mode == "pred":
        metrics_output = gen_evaluator.get_epoch_score()
        runiter_output.update(metrics_output)
        
        runiter_output["epoch_loss"] = epoch_loss / batch_count

        runiter_output["BLEU"] = {'avg_epoch_bleu1': epoch_bleu1/batch_count, 'avg_epoch_bleu2': epoch_bleu2/batch_count, 'avg_epoch_bleu3':epoch_bleu3/batch_count, 'avg_epoch_bleu4': epoch_bleu4/batch_count}
        runiter_output["METEOR"] = epoch_meteor / batch_count
        runiter_output["ENTROPY"], runiter_output["DISTINCT"] = cal_entropy_and_distinct(all_pred)

        
    return runiter_output

def train():
                      
    best_val_loss = float('inf')
    
    model, optimizer, criterion, scheduler = compile_fn()                  
    model.load_state_dict(torch.load(weight_path))
    
    model = model.to(device)
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank])

    print("The model has {} trainable parameters".format(count_parameters(model)))
    
    for epoch in range(args.epoch):
        
        train_data_loader_gen = data_loader_generator("train", args.batch_size, epoch, num_worker=4, distributed=args.distributed, world_size=args.world_size)
        
        start_time = time.time()
           
        train_output = run_iters(device, model, train_data_loader_gen, optimizer, criterion, epoch+1, args.CLIP, scheduler, mode="train")
        
        if (args.distributed and args.local_rank == 0) or (not args.distributed):
            t1 = time.time()  
            val_data_loader_gen = data_loader_generator("val", args.batch_size, epoch, num_worker=4, distributed=False, world_size=-1)
            if args.distributed:
                val_output = run_iters(device, model.module, val_data_loader_gen, optimizer, criterion, epoch+1, args.CLIP, scheduler, mode="val")
            else:
                val_output = run_iters(device, model, val_data_loader_gen, optimizer, criterion, epoch+1, args.CLIP, scheduler, mode="val")
            t2 = time.time()
        
            eval_min, eval_sec = epoch_time(t1, t2)
            print('evaluate time: {}m {}s'.format(eval_min, eval_sec))

            print("\tTrain Loss: {:.5f} | Train PPL: {:7.3f}".format(train_output["epoch_loss"], math.exp(train_output["epoch_loss"])))
            print("\t Val. Loss: {:.5f} |  Val. PPL: {:7.3f}".format(val_output["epoch_loss"], math.exp(val_output["epoch_loss"])))

            print("\t Val  Bleu1: {:.5f}, Val  Bleu2: {:.5f}, Val  Bleu3: {:.5f}, Val  Bleu4: {:.5f}".format(val_output["BLEU"]['avg_epoch_bleu1'], val_output["BLEU"]['avg_epoch_bleu2'],val_output["BLEU"]['avg_epoch_bleu3'],val_output["BLEU"]['avg_epoch_bleu4']))
            print("\t Val Rouge-L: {}".format(val_output["ROUGE-L"]))
            print("\t Val CIDEr: {}".format(val_output["CIDEr"]))
            print("\t Val Meteor: {:.5f}".format(val_output["METEOR"]))   
            print("\t Val Entropy: {}".format(val_output["ENTROPY"]))
            print("\t Val Distinct: {}".format(val_output["DISTINCT"]))
        
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print("Epoch: {:02} | Time: {}m {}s".format(epoch+1, epoch_mins, epoch_secs))

            if args.model_save_path and val_output["epoch_loss"] < best_val_loss:
                best_val_loss = val_output["epoch_loss"]
                real_save_path = args.model_save_path.format(epoch+1)
                torch.save(model.state_dict(), real_save_path)
                print("save model weight: {} at epoch: {}".format(real_save_path, epoch+1))
                
if __name__ == '__main__':
    args.total_step = 10
    
    if args.distributed:
        setup(args.local_rank)
        args.world_size = dist.get_world_size()
        
    print('--'*30)
    for k in list(vars(args).keys()):
        print("{}: {} ".format(k, vars(args)[k]))
    print('--'*30)
    
    train()
    
    
    
    
     


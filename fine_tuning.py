# -*- coding: utf-8 -*-
## load pre-trained model for multiple down stream tasks

import os
import time
import math
import json
import argparse
import warnings
warnings.simplefilter("ignore", UserWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, classification_report

from dataset import down_stream_dataset
from model import base_model
from utils import epoch_time, count_parameters, process_vqa_answer

from metrics import BLEU, METEOR, process_for_metrics, cal_entropy_and_distinct, process_for_metrics_stacked_tgt, vqa_batch_accuracy, VQAAnsTypeACC, Evaluator


parser = argparse.ArgumentParser()

parser.add_argument("--backbone_path", type=str, default=None, help="backbone model path")

parser.add_argument("-m", "--model_save_path", type=str, default=None, help="model path to save fine-tuned model")
parser.add_argument("-d", "--device_id", type=int, help="GPU device id")
parser.add_argument('--epoch', type=int, default=6)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument("--init_lr", type=float, default=0.0001)
parser.add_argument("--print_steps", type=int, default=100, help="print steps during training ")
parser.add_argument("--task", type=str, choices=["GEN_VQA", "IC_MUGE", "IC_Flickr30k", "IC_AIC_ICC", "CLS_VQA"], help="choose one down stream task: vqa, image caption, image text matching")
parser.add_argument("--text_encoder", type=str, default="Transformer")
parser.add_argument("--image_encoder", type=str, default="ResNet50-base")
parser.add_argument("--cmt_decoder", type=str, default="Transformer", choices=["Transformer"])
parser.add_argument("--embed_size", type=int, default=768)
parser.add_argument("--tgt_vocab_size", type=int, default=21128)
parser.add_argument("--embed_weight", type=str, default=None)
parser.add_argument("--GEN", type=bool, default=False, help="pre-training with Cmt Generative or not")
parser.add_argument("--CLSVQA", type=bool, default=False, help="fine tune on VQA 2.0 dataset")

parser.add_argument("--cls", type=int, default=101)
parser.add_argument("--sep", type=int, default=102)
parser.add_argument("--pad", type=int, default=0)
parser.add_argument("--vqa_ans_len", type=int, default=25)
parser.add_argument("--ic_post_len", type=int, default=72)

args = parser.parse_args()


if torch.cuda.is_available():
    device = torch.device("cuda", args.device_id)
else:
    device = torch.device("cpu")


def load_base_model():
    """
    return pre-trained model for down stream task
    """
    if args.task == "GEN_VQA":
        args.GEN = True
        input_modality = "t+v"
    elif args.task == "CLS_VQA":
        args.CLSVQA = True
        input_modality = "t+v"
    elif args.task in ["IC_MUGE", "IC_Flickr30k", "IC_AIC_ICC"]:
        args.GEN = True
        input_modality = "v"
    model = base_model.BaseCMTGeneratorEarlyFusion(image_encoder_name=args.image_encoder, 
                                   text_encoder_name=args.text_encoder, 
                                   decoder_name=args.cmt_decoder, 
                                   encoder_layer_num=6, 
                                   decoder_layer_num=6, 
                                   vocab_size=args.tgt_vocab_size, 
                                   hid_dim=args.embed_size, 
                                   embed_weight=args.embed_weight,  
                                   GEN=args.GEN,
                                   CLSVQA=args.CLSVQA,
                                   input_modality=input_modality,
                                   )
    
    if args.backbone_path:
        
        model.load_state_dict(torch.load(args.backbone_path, map_location="cpu"))

    
    return model

def compile_fn():
    """
    prepare model, criterion, optimizer and other things for down stream task
    """
    base_model = load_base_model()
    if args.task in ["GEN_VQA","IC_MUGE", "IC_Flickr30k", "IC_AIC_ICC"]:
        criterion = nn.CrossEntropyLoss(ignore_index=args.pad)
    elif args.task == "CLS_VQA":
        criterion = nn.BCEWithLogitsLoss(reduction="mean")
        
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, base_model.parameters()), lr=args.init_lr)
    
    return base_model, criterion, optimizer

def get_data_loader(task, batch_size, mode, shuffle, num_worker):

    if task == "GEN_VQA":
        print("calling FMIQA {} data loader ...".format(mode))
        ds = down_stream_dataset.FMIQADataset(mode)
    elif task == "IC_MUGE":
        print("calling IC MUGE {} data loader ...".format(mode))
        ds = down_stream_dataset.EcomICDataset(mode)
    elif task == "IC_Flickr30k":
        print("calling IC Flickr30k {} data loader ... ".format(mode))
        ds = down_stream_dataset.Flickr30kCN_IC_Dataset(mode)
    elif task == "IC_AIC_ICC":
        print("calling IC AIC ICC {} data loader ... ".format(mode))
        ds = down_stream_dataset.AIC_ICC_Dataset(mode)
    elif task == "CLS_VQA":
        print("calling VQA 2.0 {} data loader ...".format(mode))
        ds = down_stream_dataset.VQAv2Dataset(mode)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_worker, collate_fn=ds.collate_fn)

def fine_tuning(task, model, data_loader, criterion, optimizer, mode, device):
    if mode == "train":
        model.train()
    else:
        model.eval()

    if task == "CLS_VQA" and mode == "val":
        vqa_type_acc_tracker = VQAAnsTypeACC()
        

    epoch_loss, avg_step_loss = 0, 0
    epoch_vqa_acc, step_vqa_acc = 0, 0
    epoch_bleu1, epoch_bleu2, epoch_bleu3, epoch_bleu4 = 0., 0., 0., 0.

    epoch_meteor = 0.
    all_pred = []
    batch_count = 0
    start_time = time.time()
    gen_evaluator = Evaluator(["ROUGE-L","CIDEr"], args.cls, args.sep, args.pad)
    for i, batch_data in enumerate(data_loader):
        #if i>5: break
        batch_count += 1
        batch_data["batch_post"] = batch_data["batch_post"].to(device)
        batch_data["batch_image"] = batch_data["batch_image"].to(device)
        
        if task == "GEN_VQA":
            batch_data["batch_ans"] = batch_data["batch_ans"].to(device)
        if mode == "train":
            output = model.down_stream_forward(task, batch_data)
        elif mode == "val":
            with torch.no_grad():
                output = model.down_stream_forward(task, batch_data)
        elif mode == "pred":
            with torch.no_grad():
                pred, pred_prob = model.predict(batch_data, 40, 101, "greedy") 
        if task == "GEN_VQA":
            vqa_labels = output["vqa_labels"]
            vqa_logits = output["vqa_logits"]
            
            loss = criterion(vqa_logits.permute(1,2,0), vqa_labels)
            vqa_pred = vqa_logits.argmax(2).transpose(0,1)
            
        elif task in ["IC_MUGE", "IC_Flickr30k", "IC_AIC_ICC"]:
            if mode == "pred":
                ic_pred = pred
                ic_logits = pred_prob
                ic_labels = batch_data["batch_post"][:,1:]
            else:
                ic_logits = output["ic_logits"] # len, bs, ntoken
                ic_labels = output["ic_labels"] # bs, len
                ic_pred = ic_logits.argmax(2).transpose(0,1)
                
            loss = criterion(ic_logits.permute(1,2,0), ic_labels)
            
        elif task == "CLS_VQA":
            vqa_logits = output["vqa_cls_logits"]
            vqa_labels = batch_data["batch_ans_label"].to(device)
            vqa_labels = process_vqa_answer(vqa_labels)
            
            loss = criterion(vqa_logits, vqa_labels.float()) * 3129
            vqa_acc, scores = vqa_batch_accuracy(vqa_logits, vqa_labels)
            epoch_vqa_acc += vqa_acc
            step_vqa_acc += vqa_acc
            if mode == "val":
                vqa_type_acc_tracker.update_score(batch_data["batch_did"], scores.tolist())
            
        epoch_loss += loss.item()
        avg_step_loss += loss.item()
        
        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        if mode == "train" and batch_count%args.print_steps == 0:
            end_time = time.time()
            print("iter: {}, time_cost: {:.4f} seconds".format(batch_count, end_time-start_time))
            if task == "CLS_VQA":
                print("\tavg step loss: {:.4f}, avg step vqa acc: {:.4f}".format(avg_step_loss/args.print_steps, step_vqa_acc/args.print_steps))
                step_vqa_acc = 0.0

            avg_step_loss = 0.    
            start_time = time.time()
        
        if (mode == "val" or mode == "pred") and task not in ["CLS_VQA"]:
            if task == "GEN_VQA":
                real_pred, real_targets = process_for_metrics(vqa_pred, batch_data["batch_ans"], args.cls, args.sep, args.pad, args.vqa_ans_len)
                gen_evaluator.batch_tracker(vqa_pred.tolist(), batch_data["batch_ans"].tolist())
            if task in["IC_MUGE", "IC_Flickr30k", "IC_AIC_ICC"]:
                real_pred, real_targets = process_for_metrics_stacked_tgt(ic_pred, batch_data["batch_reference"], args.cls, args.sep, args.pad)
                gen_evaluator.batch_tracker(ic_pred.tolist(), batch_data["batch_reference"])
            all_pred.extend(real_pred)
            bleu1, bleu2, bleu3, bleu4 = BLEU(real_pred, real_targets)
            meteor_score = METEOR(real_pred, real_targets)

            epoch_bleu1 += bleu1
            epoch_bleu2 += bleu2
            epoch_bleu3 += bleu3
            epoch_bleu4 += bleu4
            epoch_meteor += meteor_score

    
    runiter_output = {}
    runiter_output["epoch_avg_loss"] = epoch_loss / batch_count
    
    if task == "CLS_VQA":
        runiter_output["epoch_avg_acc"] = epoch_vqa_acc / batch_count
        if mode == "val":
            vqa_type_acc_tracker.result()
            vqa_type_acc_tracker.reset()
        
    elif mode == "val" or mode == "pred":
        runiter_output["epoch_avg_loss"] = epoch_loss / batch_count
        runiter_output["BLEU"] = {'avg_epoch_bleu1': epoch_bleu1/batch_count, 'avg_epoch_bleu2': epoch_bleu2/batch_count, 'avg_epoch_bleu3':epoch_bleu3/batch_count, 'avg_epoch_bleu4':epoch_bleu4/batch_count}
        runiter_output["METEOR"] = epoch_meteor / batch_count
        runiter_output["ENTROPY"], runiter_output["DISTINCT"] = cal_entropy_and_distinct(all_pred)
        
        metrics_output = gen_evaluator.get_epoch_score()
        runiter_output.update(metrics_output)
        
    return runiter_output
            
                  
def main():
    best_val_loss = float('inf')
    model, criterion, optimizer = compile_fn()
    print("The model has {} trainable parameters".format(count_parameters(model)))
    model.to(device)
    for epoch in range(args.epoch):
        t1 = time.time()
        train_data_loader = get_data_loader(args.task, args.batch_size, "train", True, num_worker=4)
        train_output = fine_tuning(args.task, model, train_data_loader, criterion, optimizer, "train", device)
        t2 = time.time()
        
        val_data_loader = get_data_loader(args.task, args.batch_size, "val", False, num_worker=4)
        val_output = fine_tuning(args.task, model, val_data_loader, criterion, optimizer, "val", device)
        t3 = time.time()
        eval_min, eval_sec = epoch_time(t2, t3)
        print("evaluate time cost: {}m {}s".format(eval_min, eval_sec))
        
        epoch_min, epoch_sec = epoch_time(t1, t3)
        print("Epoch: {:02} | Time: {}m {}s".format(epoch+1, epoch_min, epoch_sec))
        
        print("\tTrain Loss: {:.5f} | Val. Loss: {:.5f}".format(train_output["epoch_avg_loss"],val_output["epoch_avg_loss"]))
        
        if args.task == "CLS_VQA":
            print("\tTrain VQA Acc: {:.5f} | Val VQA Acc: {:.5f}".format(train_output["epoch_avg_acc"], val_output["epoch_avg_acc"]))
        else:
            print("\t Val  Bleu1: {:.5f}, Val  Bleu2: {:.5f}, Val  Bleu3: {:.5f}, Val  Bleu4: {:.5f}".format(val_output["BLEU"]['avg_epoch_bleu1'], val_output["BLEU"]['avg_epoch_bleu2'],val_output["BLEU"]['avg_epoch_bleu3'],val_output["BLEU"]['avg_epoch_bleu4']))
            print("\t Val Rouge-L: {}".format(val_output["ROUGE-L"]))
            print("\t Val CIDEr: {}".format(val_output["CIDEr"]))
            print("\t Val Meteor: {:.5f}".format(val_output["METEOR"]))   
            print("\t Val Entropy: {}".format(val_output["ENTROPY"]))
            print("\t Val Distinct: {}".format(val_output["DISTINCT"]))
        
        
        if args.model_save_path and val_output["epoch_avg_loss"] < best_val_loss:
            best_val_loss = val_output["epoch_avg_loss"]
            print('Epoch {:02} saving model weight: {}'.format(epoch+1, args.model_save_path))
            torch.save(model.state_dict(), args.model_save_path)
        
def inference_evaluate():
    ## evaluate the generative_downstream tasks with inference mode
    data_mode = "test" if args.task != "IC_AIC_ICC" else "test_a"
    model, criterion, optimizer = compile_fn()
    fine_tuned_model_path = ""
    model.load_state_dict(torch.load(file_tuned_model_path, map_location="cpu"))
    print("The model has {} trainable parameters".format(count_parameters(model)))
    model.to(device)

    test_data_loader = get_data_loader(args.task, args.batch_size, data_mode, False, num_worker=8)
    test_output = fine_tuning(args.task, model, test_data_loader, criterion, optimizer, "pred", device)  

    print(test_output)
    print("\t Val  Bleu1: {:.5f}, Val  Bleu2: {:.5f}, Val  Bleu3: {:.5f}, Val  Bleu4: {:.5f}".format(test_output["BLEU"]['avg_epoch_bleu1'], test_output["BLEU"]['avg_epoch_bleu2'],test_output["BLEU"]['avg_epoch_bleu3'],test_output["BLEU"]['avg_epoch_bleu4']))
    print("\t Val Rouge-L: {}".format(test_output["ROUGE-L"]))
    print("\t Val CIDEr: {}".format(test_output["CIDEr"]))
    print("\t Val Meteor: {:.5f}".format(test_output["METEOR"]))   
    print("\t Val Entropy: {}".format(test_output["ENTROPY"]))
    print("\t Val Distinct: {}".format(test_output["DISTINCT"]))

        
if __name__ == "__main__":
    print('--'*30)
    for k in list(vars(args).keys()):
        print("{}: {} ".format(k, vars(args)[k]))
    print('--'*30)

    main()
    #inference_evaluate()

   
    
    





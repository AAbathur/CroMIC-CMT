import torch
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
from torchtext.data.metrics import bleu_score
from nltk import ngrams
from collections import Counter, defaultdict
import numpy as np

import pycocoevalcap
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge



def vqa_batch_accuracy(logits, labels):
    """
    follow Bilinear Attention Networks https://github.com/jnhwkim/ban-vqa.git
    and https://visualqa.org/evaluation.html
    """
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros_like(labels)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    acc = scores.sum() / labels.shape[0]
    return acc.item(), scores.sum(dim=1)

class VQAAnsTypeACC():
    def __init__(self):
        super().__init__()
        qid2ans_type_path = "data/downstream_data/VQA2.0/train_val_qid2ans_type.pt"
        print("**"*10)
        print("Noting: load qid2answer_type dict that is saved in .pt file: ", qid2ans_type_path)
        print("**"*10)
        self.id2type = torch.load(qid2ans_type_path)
        self.un_eval_id = []
        self.un_eval_score = []
        
    def update_score(self, batch_id, batch_score):
        assert len(batch_id) == len(batch_score)
        assert type(batch_id) == list
        assert type(batch_score) == list
        
        self.un_eval_id += batch_id
        self.un_eval_score += batch_score
        
    def result(self):
        type2score = {}
        for i in range(len(self.un_eval_id)):
            did = self.un_eval_id[i]
            score = self.un_eval_score[i]
            at = self.id2type[did]
            type2score[at] = type2score.get(at, [])
            type2score[at].append(score)
        total_score = 0
        total_len = 0
        for t, ss in type2score.items():
            print("AnswerType: ", t, "Number: ", len(ss), "Acc: ", sum(ss)/len(ss))
            total_score += sum(ss)
            total_len += len(ss)
        print("Total Number: ", total_len, "Total Avg Acc: ", total_score/total_len)
    
    def reset(self):
        self.un_eval_id = []
        self.un_eval_score = []
        
def get_real_cmt(cmt, SOS, EOS, PAD):
        real_cmt = []
        for i in range(len(cmt)): # remove SOS/EOS token
            if cmt[i] == EOS:
                break
            if cmt[i] != PAD and cmt[i] != SOS:
                real_cmt.append(str(cmt[i]))
        return real_cmt

def process_for_metrics_stacked_tgt(pred, targets, SOS, EOS, PAD):
    pred = pred.cpu()
    assert isinstance(targets, list)
    real_pred = []
    real_targets = []
    for i in range(pred.shape[0]):
        pred_cmt = pred[i].tolist()
        pred_cmt = get_real_cmt(pred_cmt, SOS, EOS, PAD)
        if pred_cmt:
            real_pred.append(pred_cmt)
        else:
            continue
        tgts = targets[i]
        same_post_cmt = []
        for tt in tgts:
            real_cmt = get_real_cmt(tt, SOS, EOS, PAD)
            if real_cmt:
                same_post_cmt.append(real_cmt)
        real_targets.append(same_post_cmt)
    assert len(real_pred) == len(real_targets)
    return real_pred, real_targets
    

def process_for_metrics(pred, targets, SOS, EOS, PAD, cmt_len):

    pred = pred.cpu()
    
    targets = targets.cpu()
    
    real_pred = []
    real_targets = []

    for i in range(targets.shape[0]):
        
        pred_cmt = pred[i].tolist()
        pred_cmt = get_real_cmt(pred_cmt, SOS, EOS, PAD)
        if pred_cmt:
            real_pred.append(pred_cmt)
        else:
            continue
        same_post_cmt = []
        target_i = targets[i].tolist()
        
        for j in range(targets.shape[1]//cmt_len):
            cmt = target_i[j*(cmt_len):(j+1)*(cmt_len)]
            real_cmt = get_real_cmt(cmt, SOS, EOS, PAD)  
            if real_cmt:
                same_post_cmt.append(real_cmt)
        real_targets.append(same_post_cmt)
    assert len(real_pred) == len(real_targets)  
    return real_pred, real_targets

def process_for_metrics_flat_cmt(pred, targets, SOS, EOS, PAD, cmt_num_record):
    pred = pred.tolist()
    targets = targets.tolist()
    
    real_pred = []
    real_targets = []
    
    def get_real_cmt(cmt):
        real_cmt = []
        for i in range(len(cmt)): # remove SOS/EOS token
            if cmt[i] == EOS :
                break
            if cmt[i] != PAD and cmt[i] != SOS:
                real_cmt.append(str(cmt[i]))
        return real_cmt
    
    pos = 0
    candidate = []
    refer =[]
    for num in cmt_num_record:
        pred_cmt = pred[pos: pos+num]
        real_cmt = targets[pos: pos+num]
        pos += num ## added 2022/08/10
        processed_target = []

        for i in range(num):
            tgt = real_cmt[i]
            real_tgt = get_real_cmt(tgt)
            if real_tgt == []:
                continue
            processed_target.append(real_tgt)
        if processed_target == []:
            continue
        for j in range(num):
            pcmt = pred_cmt[j]
            real_pcmt = get_real_cmt(pcmt)
            if real_pcmt == []:
                continue
            candidate.append(real_pcmt)
            refer.append(processed_target)
    assert len(candidate) == len(refer)
    return candidate, refer

def BLEU(real_pred, real_targets):
    if len(real_pred) == 0:
        return 0, 0, 0, 0
    else:
        bleu1 = bleu_score(real_pred, real_targets, max_n=1, weights=[1.0])
        bleu2 = bleu_score(real_pred, real_targets, max_n=2, weights=[0.5,0.5])
        bleu3 = bleu_score(real_pred, real_targets, max_n=3, weights=[0.333, 0.333, 0.334])
        bleu4 = bleu_score(real_pred, real_targets, max_n=4, weights=[0.25, 0.25, 0.25, 0.25])
        return bleu1, bleu2, bleu3, bleu4

def METEOR(real_pred, real_targets):
    new_pred = []
    new_targets =[]
    for i in range(len(real_pred)):
        pred = real_pred[i]
        sen = ' '.join(pred)
        new_pred.append(sen)
        
        cmts = real_targets[i]
        same_post_cmts = []
        for cmt in cmts:
            same_post_cmts.append(' '.join(cmt))
        new_targets.append(same_post_cmts)
     
    total = 0
    for k in range(len(new_pred)):
        score = meteor_score(new_targets[k], new_pred[k])
        total += score
    return total/len(new_pred) if len(new_pred) else 0
      
def ACC(pred, target):
    mask = (target > 0).float()
    cor = torch.sum((pred == target) * mask)
    number = torch.sum(mask)
    return cor / number

class Accuracy():
    def __init__(self):
        self.correct = 0
        self.total = 0
    def update_state(self, pred, target, ignore_index):
        if ignore_index == None:
            mask = torch.ones_like(target).float()
        else:
            mask = (target != ignore_index).float()
        self.correct += torch.sum((pred == target) * mask)
        self.total += torch.sum(mask)
    def result(self):
        return self.correct / self.total
    def reset(self):
        self.correct = 0
        self.total = 0
    
class RecallTopK():
    def __init__(self, topk):
        self.topk = topk
        self.total = 0
        self.count = 0
    def update_state(self, pred_index, target_index):
        for i in range(pred_index.shape[0]):
            self.total += 1
            gt = target_index[i]
            if gt in pred_index[i,:self.topk].tolist():
                self.count += 1
    def result(self):
        return self.count / self.total
    def reset(self):
        self.total = 0
        self.count = 0
        
def cal_entropy_and_distinct(generated):
    entropy_score = [0.0,0.0,0.0,0.0]
    distinct_score = [0.0,0.0,0.0,0.0]
    counter = [defaultdict(int),defaultdict(int),defaultdict(int),defaultdict(int)]
    for gg in generated:
        #g = gg.rstrip().split()
        g = gg
        for n in range(4):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter[n][ngram] += 1
    for n in range(4):
        total = sum(counter[n].values()) +1e-10
        for v in counter[n].values():
            entropy_score[n] += - (v+0.0) /total * (np.log(v+0.0) - np.log(total))
        distinct_score[n] = (len(counter[n].values())+0.0) / total
    return entropy_score, distinct_score
    
class Evaluator:
    def __init__(self, keys, SOS, EOS, PAD):
        self.SOS = SOS
        self.EOS = EOS
        self.PAD = PAD
        self.metrics_dict = {"ROUGE-L": Rouge(), "CIDEr": Cider()}
        self.keys = keys
        self.epoch_score_record = {}
        for key in keys:
            self.epoch_score_record[key] = []
            
    def get_real_sen(self, input_ids):
        output_ids = []
        for ids in input_ids:
            if ids == self.EOS:
                break
            elif ids != self.SOS and ids != self.PAD:
                output_ids.append(str(ids))
        return output_ids
    
    def process_input(self, batch_candidate, batch_references):
        assert len(batch_candidate) == len(batch_references)
        assert type(batch_candidate) == list
        assert type(batch_references) == list
        multi_ref = True if type(batch_references[0][0]) == list else False
        bs = len(batch_candidate)
        can = {}
        refs = {}
        for i in range(len(batch_candidate)):
            valid_tokens = self.get_real_sen(batch_candidate[i])
            can[i] = [" ".join(valid_tokens)]
            refs[i] = []
            if multi_ref == True:
                for ref in batch_references[i]:
                    valid_tokens = self.get_real_sen(ref)
                    refs[i].append(" ".join(valid_tokens))
            else:
                valid_tokens = self.get_real_sen(batch_references[i])
                refs[i].append(" ".join(valid_tokens))
        return can, refs

    def batch_compute_score(self, batch_candidate, batch_references):
        can, refs = self.process_input(batch_candidate, batch_references)
        key2score = {}
        for key in self.keys:
            if key == "BLEU":
                avg_score, score_list = self.metrics_dict[key].compute_score(refs, can, 0)
            else:
                avg_score, score_list = self.metrics_dict[key].compute_score(refs, can)
            key2score[key] = avg_score
            
        return key2score
    
    def batch_tracker(self, batch_candidate, batch_references):
        can, refs = self.process_input(batch_candidate, batch_references)
        
        for key in self.keys:
            avg_score, score_list = self.metrics_dict[key].compute_score(refs, can)
            
            self.epoch_score_record[key].append(avg_score)
    def stacked_batch_tracker(self, batch_candidate, batch_references, cmt_num_record):
        real_reference = []
        pos = 0
        for num in cmt_num_record:
      
            real_cmt = batch_references[pos: pos+num]
            pos += num
            
            for si in range(num):
                real_reference.append(real_cmt)
        
        self.batch_tracker(batch_candidate, real_reference)
            
    def get_epoch_score(self):
        output = {}
        print("epoch score record: ", self.epoch_score_record)
        for key, scores in self.epoch_score_record.items():
            if type(scores[0]) == list:
                num = len(scores[0])
                output[key] = []
                for j in range(num):
                    total = sum(scores[k][j] for k in range(len(scores)))
                    output[key].append(total/len(scores))
            else:
                output[key] = sum(scores)/len(scores)
        return output

    

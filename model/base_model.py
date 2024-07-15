import os
import operator
import random
import sys
sys.path.append("model")

import torch
from torch import nn
import numpy as np
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
from queue import PriorityQueue
import torch.nn.functional as F
from sklearn.utils import shuffle

from transformer_block import PositionalEncoding, TokenEmbedding, TFEncoderLayer, TFEncoder, TFDecoderLayer, TFDecoder


def l2norm(inputx, p=2.0, dim=1, eps=1e-12):
    # row-wise norm
    return inputx / inputx.norm(p, dim).clamp(min=eps).unsqueeze(dim).expand_as(inputx) 

def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float("-inf").Unmasked positions are filled with float(0.0).
        """
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class BertEncoder(nn.Module):
    def __init__(self, bert_path):
        super(BertEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
    
    def forward(self, input_ids):
        #self.bert.eval()
        att_mask = (input_ids != 0 ).long().to(input_ids.device)
        token_type_ids = torch.zeros_like(input_ids).to(input_ids.device)
        
        output = self.bert(input_ids, token_type_ids, att_mask)
        att_mask = 1.0 - att_mask
        encoded_text = output[0].transpose(0,1)
   
        return encoded_text, att_mask


class ResNet50Base(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*(list(model.children())[:-2]))
        self.conv2d = nn.Conv2d(2048, 768, (1,1), stride=1)
        self.pooling = nn.MaxPool2d((2,2), stride=1)
        self.batch_norm = nn.BatchNorm1d(768)
        
    def forward(self, img):
        bs = img.shape[0]
        img_vec = self.backbone(img) 
        img_vec = self.conv2d(img_vec) 
        img_vec = self.pooling(img_vec) 
        img_vec = img_vec.view(bs, 768, 36)
        img_vec = self.batch_norm(img_vec).transpose(1, 2)
        img_vec = l2norm(img_vec, dim=2)
        img_vec = img_vec.transpose(0,1)
        img_mask = torch.zeros((bs, 36))
        
        return img_vec, img_mask.to(img.device)
   
        
def transformer_decoder(d_model, nhead, dim_feedforward, decoder_layer_num, dropout=0.0):
    
    decoder_layer = TFDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation='relu')
    decoder_norm = nn.LayerNorm(d_model)
    tf_decoder = TFDecoder(decoder_layer, decoder_layer_num, decoder_norm)
    return tf_decoder    

def transformer_encoder(d_model, nhead, dim_feedforward, encoder_layer_num, dropout=0.0,):    
    encoder_layer = TFEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation='relu')
    encoder_norm = nn.LayerNorm(d_model)
    tf_encoder = TFEncoder(encoder_layer, encoder_layer_num, encoder_norm)
    return tf_encoder  

_IMAGE_ENCODER = {"ResNet50-base": ResNet50Base()}

_TEXT_ENCODER = {"Transformer": transformer_encoder} 
    
_CMT_DECODER = {"Transformer": transformer_decoder}


    

class BaseCMTGeneratorEarlyFusion(nn.Module):
    def __init__(self, image_encoder_name, text_encoder_name, decoder_name, encoder_layer_num, decoder_layer_num, vocab_size, hid_dim, embed_weight, GEN=False, CLSVQA=False, input_modality="t+v"):
        super().__init__()
        self.vocab_size = vocab_size
        self.hid_dim = hid_dim
        self.embed_weight = embed_weight
        self.input_modality = input_modality # t+v, v, t
        
        self.gen = GEN
        self.CLSVQA = CLSVQA
        
        self.pad = 0
        self.nhead = 8
        self.dim_feedforward = 1024
        self.encoder_layer_num = encoder_layer_num
        self.decoder_layer_num = decoder_layer_num

        self.image_encoder = _IMAGE_ENCODER[image_encoder_name]
        self.encoder = _TEXT_ENCODER[text_encoder_name](self.hid_dim, self.nhead, self.dim_feedforward, self.encoder_layer_num)
        
        self.token_embed = TokenEmbedding(self.vocab_size, self.hid_dim, self.embed_weight)
        self.positional_encoding = PositionalEncoding(self.hid_dim)
        self.token_type_embed = nn.Embedding(2, 768)
        self.token_type_embed.apply(init_weights)
        
        if self.gen:
            self.decoder = _CMT_DECODER[decoder_name](self.hid_dim, self.nhead, self.dim_feedforward, self.decoder_layer_num)
            self.generator = nn.Linear(self.hid_dim, self.vocab_size)
            
        if self.CLSVQA:
            
            self.vqa_classifier = nn.Sequential(
                nn.Linear(768, 768 * 2),
                nn.LayerNorm(768 * 2),
                nn.GELU(),
                nn.Linear(768 * 2, 3129),
            )
            self.vqa_classifier.apply(init_weights)

    def input_image_preprocess(self, image_input):
        img_vec, img_mask = self.image_encoder(image_input)
        self.img_length = img_vec.shape[0]
        
        return img_vec, img_mask

    def input_text_preprocess(self, text_input):
        text_vec = self.positional_encoding(self.token_embed(text_input.t()))
        text_mask = (text_input == self.pad).float().to(text_input.device)
        return text_vec, text_mask


    def encoder_input_process(self, batch_input):
        post = batch_input["batch_post"]
        image = batch_input["batch_image"]
    
        text_embed, text_mask = self.input_text_preprocess(post) ## len, bs, 768
        img_embed, img_mask = self.input_image_preprocess(image)

        text_embed, image_embed = (
                text_embed + self.token_type_embed(torch.zeros_like(text_mask.t()).long()),
                img_embed + self.token_type_embed(torch.ones_like(img_mask.t()).long()),
            )

        if self.input_modality == "t+v":
            co_embed = torch.cat([text_embed, image_embed], dim=0) ## post_len+49, bs, 768
            co_mask = torch.cat([text_mask, img_mask], dim=1).bool() ## bs, post_len+49
        elif self.input_modality == "t":
            co_embed = text_embed
            co_mask = text_mask.bool()
        elif self.input_modality == "v":
            co_embed = image_embed
            co_mask = img_mask.bool()

        return co_embed, co_mask

    def decoder_input_process(self, batch_input, memory, memory_padding_mask):
        batch_cmts = batch_input["batch_cmts"]
        all_cmts = []
        new_memory = []
        new_memory_padding_mask = []
        cmt_num_record = []

        for bi in range(len(batch_cmts)):
            cmts = batch_cmts[bi]
            cmt_num_record.append(len(cmts))
            
            for cmt in cmts:
                all_cmts.append(cmt)
                new_memory.append(memory[:,bi:bi+1])
                new_memory_padding_mask.append(memory_padding_mask[bi:bi+1])
        all_cmts = torch.tensor(all_cmts).to(memory.device) ## new_bs, cmt_len
        all_cmts_input = all_cmts[:,:-1]
        trg_len = all_cmts_input.shape[1]

        tgt_mask = generate_square_subsequent_mask(trg_len).to(memory.device)
        tgt_padding_mask = (all_cmts_input == self.pad)

        new_memory = torch.cat(new_memory, dim=1) ## post_len+49, new_bs, 768
        new_memory_padding_mask = torch.cat(new_memory_padding_mask, dim=0).bool() ## new_bs, post_len+49
        
        return all_cmts, all_cmts_input, new_memory, tgt_mask, tgt_padding_mask, new_memory_padding_mask, cmt_num_record
    
    def decode(self, tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask):
        tgt_emb = self.positional_encoding(self.token_embed(tgt))
        
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        decoded_output = self.generator(output)

        return output, decoded_output
    
    def forward(self, batch_input):
        co_embed, co_mask = self.encoder_input_process(batch_input)
        memory = self.encoder(src=co_embed, src_key_padding_mask=co_mask) # pos_len+49,bs,768
        output_dict = { 
                    "co_embed": co_embed,
                    "co_mask": co_mask,
                    "memory": memory,
                    }
 
        if self.gen:
            all_cmts, all_cmts_input, new_memory, tgt_mask, tgt_padding_mask, new_memory_padding_mask, cmt_num_record = self.decoder_input_process(batch_input, memory, co_mask)
            output, decoded_output = self.decode(all_cmts_input.t(), new_memory, tgt_mask, None, tgt_padding_mask, new_memory_padding_mask)
            output_dict["batch_cmt"] = all_cmts
            output_dict["cmt_num_record"] = cmt_num_record
            output_dict["decoder_output"] = output
            output_dict["gen_labels"] = all_cmts[:,1:]
            output_dict["gen_logits"] = decoded_output
            
        return output_dict
    
    def down_stream_forward(self, task, batch_input):
            
        co_embed, co_mask = self.encoder_input_process(batch_input)
        memory = self.encoder(src=co_embed, src_key_padding_mask=co_mask) # pos_len+49,bs,768
        output_dict = { 
                    "co_embed": co_embed,
                    "co_mask": co_mask,
                    "memory": memory,
                    }
        
        if task == "GEN_VQA":
            answer = batch_input["batch_ans"]
            decoder_input = answer[:,:-1]
            
            tgt_mask = generate_square_subsequent_mask(decoder_input.size(1)).type(torch.bool).to(memory.device)
            tgt_padding_mask = (decoder_input == self.pad)
            output, decoded_output = self.decode(decoder_input.t(), memory, tgt_mask, None, tgt_padding_mask, co_mask)
            output_dict["vqa_labels"] = answer[:,1:]
            output_dict["vqa_logits"] = decoded_output
            
        elif task in ["IC_Ecommerce", "IC_Flickr30k", "IC_AIC_ICC"]:
            assert self.input_modality == "v", "IC_GEN task only input img"
            caption = batch_input["batch_post"]
            decoder_input = caption[:,:-1]
            
            tgt_mask = generate_square_subsequent_mask(decoder_input.size(1)).type(torch.bool).to(memory.device)
            tgt_padding_mask = (decoder_input == self.pad)
            output, decoded_output = self.decode(decoder_input.t(), memory, tgt_mask, None, tgt_padding_mask, co_mask)
            output_dict["ic_logits"] = decoded_output
            output_dict["ic_labels"] = caption[:,1:]
            
        elif task == "CLS_VQA":
            cls_memory = memory[0] # bs, 768
            vqa_cls_logits = self.vqa_classifier(cls_memory) ## bs, 3129
            output_dict["vqa_cls_logits"] = vqa_cls_logits
            
        return output_dict
    
    def predict(self, batch_input, trg_len, SOS, decode_method="greedy"):
        co_embed, co_mask = self.encoder_input_process(batch_input)
        memory = self.encoder(src=co_embed, src_key_padding_mask=co_mask)
        
        N = memory.shape[1]
        
        d_input = torch.ones(1, N).fill_(SOS).type(torch.long).to(memory.device)
        all_output = []
        
        for i in range(trg_len - 1):
            tgt_mask = generate_square_subsequent_mask(d_input.size(0)).type(torch.bool).to(memory.device)
            output, output_prob = self.decode(d_input, memory, tgt_mask, None, None, co_mask) ## output_prob: len, bs, ntoken
            prob = output_prob[-1] ## prob: bs, ntoken
            all_output.append(prob)
            if decode_method == "greedy":
                _, next_word = torch.max(prob, dim=1)
                d_input = torch.cat([d_input, next_word.unsqueeze(0).type(torch.long)], dim=0)
                
            elif decode_method == "random_sampling":
                real_probs = F.softmax(prob, dim=-1).detach()
                select_ids = torch.multinomial(real_probs, num_samples=1, replacement=True) ## bs, 1
                select_ids = select_ids.t()
                d_input = torch.cat([d_input, select_ids.type(torch.long)], dim=0)
            elif decode_method == "topk_sampling":
                topk_prob, topk_word = torch.topk(prob, k=5, dim=-1)
                idx = torch.multinomial(topk_prob, num_samples=1)
                select_ids = torch.gather(topk_word, dim=1, index=idx)
                select_ids = select_ids.t()
                d_input = torch.cat([d_input, select_ids.type(torch.long)], dim=0)
                
        all_output = torch.stack(all_output, dim=0)
        d_input = d_input[1:].t()
        return d_input, all_output

    def predict_by_beam_search(self, batch_input, tgt_len, SOS, EOS, topk=1, beam_width=5):
        co_embed, co_mask = self.encoder_input_process(batch_input)
        memory = self.encoder(src=co_embed, src_key_padding_mask=co_mask)
        
        N = memory.shape[1]
        decoded_batch, batch_score = beam_decode_for_transformer_decoder(self.decode, N, memory, co_mask, memory.device, SOS, EOS, MAX_LEN=tgt_len, beam_width=beam_width, topk=topk, encoder_outputs=None, src_len=None)

        return decoded_batch, batch_score
        
    
class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        self.h = hiddenstate ## hiddenstate: memory, memory_padding_mask
        self.prevNode = previousNode
        self.wordid = wordId ## sequence of generated words
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        return -1 * self.logp
        
def beam_decode_for_transformer_decoder(decode_func, batch_size, memory, memory_padding_mask, device, SOS, EOS, MAX_LEN, beam_width=10, topk=1, encoder_outputs=None,src_len=None):
    print("top k: {}, beam width: {}".format(topk, beam_width))
    
    decoded_batch = []
    batch_score = []
    for idx in range(batch_size):
        hidden_memory = memory[:, idx:idx+1,:] # seq_len, 1, dim
        hidden_memory_padding_mask = memory_padding_mask[idx:idx+1] ## 1, seq_len
        decoder_hidden = [hidden_memory, hidden_memory_padding_mask]
        decoder_input = torch.ones(1, 1).fill_(SOS)
        decoder_input = decoder_input.to(device)
        
        endnodes = []
        num_required = min((topk+1), topk-len(endnodes))
        
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes = PriorityQueue()
        nodes.put((node.eval(), node))
        qsize = 1

        while True:
            if qsize >10000:
                print("break because of q size > threshold: 10000")
                break
                
            score, n = nodes.get()
            
            decoder_input = n.wordid
            decoder_hidden = n.h
            hidden_memory = decoder_hidden[0]
            hidden_memory_padding_mask = decoder_hidden[1]
            
            if n.wordid[-1].item() == EOS and n.prevNode != None:
                endnodes.append((score, n))
                if len(endnodes) >= num_required:
                    break
                else:
                    continue

            tgt_mask = generate_square_subsequent_mask(decoder_input.size(0)).type(torch.bool).to(device)
            output, output_prob = decode_func(decoder_input, hidden_memory, tgt_mask, None, None, hidden_memory_padding_mask)
            
            decoder_output = F.log_softmax(output_prob[-1], dim=1)
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                tmp = indexes[0][new_k].view(1,1)

                decoded_t = torch.cat((decoder_input, tmp), dim=0)
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp+log_p, n.leng+1)
                score = node.eval()
                nextnodes.append((score, node))

            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]

                nodes.put((score, nn))

            qsize += len(nextnodes) - 1

        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        scores = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
        
            utterance = n.wordid.squeeze(-1).long().tolist()
            utterances.append(utterance)
            scores.append(score)
            
        decoded_batch.append(utterances)
        batch_score.append(scores)
    
    return decoded_batch, batch_score
    

        
 
        
def beam_decode(decoder, batch_size, decoder_hiddens, SOS, EOS, MAX_LEN, topk=1, beam_width=10, encoder_outputs=None, src_len=None):
    device = decoder_hiddens.device
    print("topk: {}, beam_width: {}".format(topk, beam_width))
    decoded_batch = []
    batch_score = []
    for idx in range(batch_size):
        decoder_hidden = decoder_hiddens[:,idx,:].unsqueeze(0) ## 1,H

        decoder_input = torch.LongTensor([SOS])
        decoder_input = decoder_input.to(device)
        endnodes = []
        num_required = min((topk+1), topk-len(endnodes))

        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        nodes.put((node.eval(), node))
        qsize = 1

        while True:
            

            if qsize >10000:
                print("break because of q size > threshold: 10000")
                break
                
            score, n = nodes.get()
            decoder_input = n.wordid
            decoder_hidden = n.h
        
            if n.wordid.item() == EOS and n.prevNode != None:
                endnodes.append((score, n))
                if len(endnodes) >= num_required:
                    break
                else:
                    continue
            
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            decoder_output = F.log_softmax(decoder_output, dim=1)
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                tmp = indexes[0][new_k]
                decoded_t = indexes[0][new_k].view(-1) # 1*1
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp+log_p, n.leng+1)
                score = node.eval()
                nextnodes.append((score, node))

            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))

            qsize += len(nextnodes) - 1

        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        scores = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid.item())

            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid.item())
            utterance = utterance[::-1]
            utterances.append(utterance)
            scores.append(score)
        decoded_batch.append(utterances)
        batch_score.append(scores)
    
    return decoded_batch, batch_score





    
    
            
    

    

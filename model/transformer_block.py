import sys
import copy
import math
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
### pytorch version: 1.7.1


def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
    
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size, embed_weight, freeze_emb=False, padding_idx=0):
        super().__init__()
        if embed_weight:
            if embed_weight[-3:] == ".pt":
                embed_weight = torch.load(embed_weight)
            else:
                embed_weight = np.load(embed_weight)
                embed_weight = torch.Tensor(embed_weight)
                
            self.embedding = nn.Embedding.from_pretrained(embed_weight, padding_idx=padding_idx, freeze=freeze_emb)
            self.vocab_size = self.embedding.weight.shape[0]
            self.emb_size = self.embedding.weight.shape[1]
        else:
            self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
            self.vocab_size = vocab_size
            self.emb_size = emb_size
            
    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class TFEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0, activation="relu",
                    layer_norm_eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):

        src2, att_weight = self.self_attn(src, src, src, attn_mask=src_mask, 
                                key_padding_mask=src_key_padding_mask)
        
        src = src + self.dropout1(src2)
        src1 = self.norm1(src)
        src = self.linear1(src1)
        src = self.activation(src)
        src = self.dropout(src)
        src2 = self.linear2(src)
        src = src1 + self.dropout2(src2)
        src = self.norm2(src)
        return src, att_weight

class TFEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output, att_weight = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output

    
class TFDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = get_activation_fn(activation)
        
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            tgt: the sequence to the decoder layer (required)
            memory: the sequence from the last layer of the encoder (required)
            tgt_mask: the mask for the tgt sequence (optional)
            memory_mask: the mask for the memory sequence (optional)
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional)
            memeory_key_padding_mask: the mask for the memory keys per batch (optional)
        
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        tgt1 = self.linear1(tgt)
        tgt1 = self.activation(tgt1)
        tgt1 = self.dropout(tgt1)
        tgt1 = self.linear2(tgt1)
        tgt = tgt + self.dropout3(tgt1)
        tgt = self.norm3(tgt)
        return tgt
    
class TFDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        
        if self.norm is not None:
            output = self.norm(output)
        return output
    
class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, activation, custom_encoder=None, custom_decoder=None):
        super(Transformer, self).__init__()
        
        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TFEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            encoder_norm = nn.LayerNorm(d_model)
            self.encoder = TFEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TFDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TFDecoder(decoder_layer, num_decoder_layers, decoder_norm)
    
        self._reset_parameters()
        
        self.d_model = d_model
        self.nhead = nhead
    
    def forward(self, src, tgt, src_mask, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask):
        """
        Shape:
            - src: (S, N, E)
            - tgt: (T, N, E)
            - src_mask: (S, S)
            - tgt_mask: (T, T)
            - memory_mask: (T, S)
            - src_key_padding_mask: (N, S)
            - tgt_key_padding_mask: (N, T)
            - memory_key_padding_mask: (N, S)
        """
        
        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")
        
        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")
            
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output
    
    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float("-inf").Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        return mask
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
 
    

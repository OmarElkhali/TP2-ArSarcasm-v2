# Généré par python -m src.sync depuis le notebook ; modifier le notebook.
import math
from pathlib import Path
import torch
from torch import nn
from transformers import AutoModel
MAX_LEN=64
MODEL_ROOT=Path(__file__).resolve().parents[1]/'models/hf'

class AttentionPool(nn.Module):
    def __init__(self,dim):
        super().__init__(); self.w=nn.Linear(dim,64); self.v=nn.Linear(64,1,bias=False)
    def forward(self,h,mask):
        logits=self.v(torch.tanh(self.w(h))).squeeze(-1).masked_fill(~mask,-1e4)
        alpha=logits.softmax(-1); return (h*alpha.unsqueeze(-1)).sum(1),alpha

class StaticNet(nn.Module):
    def __init__(self,vocab_size,arch='attention',embedding=None,position=True,freeze=False):
        super().__init__(); self.arch=arch
        self.embedding=nn.Embedding(vocab_size,100,padding_idx=0)
        if embedding is not None: self.embedding.weight.data.copy_(torch.tensor(embedding))
        self.embedding.weight.requires_grad_(not freeze)
        self.dropout=nn.Dropout(.25)
        if arch=='transformer':
            self.proj=nn.Linear(100,128)
            layer=nn.TransformerEncoderLayer(128,4,256,dropout=.2,batch_first=True)
            self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
            pos=torch.arange(MAX_LEN).unsqueeze(1); div=torch.exp(torch.arange(0,128,2)*(-math.log(10000)/128))
            pe=torch.zeros(MAX_LEN,128); pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
            self.register_buffer('pe',pe if position else torch.zeros_like(pe))
        else:
            self.encoder=nn.LSTM(100,64,batch_first=True,bidirectional=True)
            self.attention=AttentionPool(128)
        self.head=nn.Linear(128,2)
    def forward(self,x,return_attention=False):
        mask=x.ne(0); h=self.dropout(self.embedding(x)); alpha=None
        if self.arch=='transformer':
            h=self.encoder(self.proj(h)+self.pe[:x.shape[1]],src_key_padding_mask=~mask)
            pooled=(h*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp(min=1)
        else:
            # pack empêche la direction arrière de traiter du padding avant le vrai texte.
            packed=nn.utils.rnn.pack_padded_sequence(h,mask.sum(1).cpu(),batch_first=True,enforce_sorted=False)
            packed,_=self.encoder(packed)
            h,_=nn.utils.rnn.pad_packed_sequence(packed,batch_first=True,total_length=x.shape[1])
            if self.arch=='attention': pooled,alpha=self.attention(h,mask)
            else: pooled=h.masked_fill(~mask.unsqueeze(-1),-1e4).max(1).values
        logits=self.head(self.dropout(pooled))
        return (logits,alpha) if return_attention else logits

class FrozenHead(nn.Module):
    def __init__(self,dim=768):
        super().__init__(); self.encoder=nn.LSTM(dim,64,batch_first=True,bidirectional=True)
        self.attention=AttentionPool(128); self.drop=nn.Dropout(.25); self.head=nn.Linear(128,2)
    def forward(self,h,mask,return_attention=False):
        h=h.float(); mask=mask.bool()
        packed=nn.utils.rnn.pack_padded_sequence(h,mask.sum(1).cpu(),batch_first=True,enforce_sorted=False)
        packed,_=self.encoder(packed); h,_=nn.utils.rnn.pad_packed_sequence(packed,batch_first=True,total_length=mask.shape[1])
        c,a=self.attention(h,mask); out=self.head(self.drop(c))
        return (out,a) if return_attention else out

class FineTuneNet(nn.Module):
    def __init__(self,name,multitask=False,encoder=None):
        super().__init__(); self.enc=AutoModel.from_pretrained(str(MODEL_ROOT/name.replace('/','--')),local_files_only=True) if encoder is None else encoder; h=self.enc.config.hidden_size
        # La tête utilise le CLS brut, pas le pooler ; ses poids ne reçoivent aucun gradient.
        if getattr(self.enc,'pooler',None) is not None: self.enc.pooler.requires_grad_(False)
        self.drop=nn.Dropout(.1); self.sarc=nn.Linear(h,2)
        self.sent=nn.Linear(h,3) if multitask else None
    def forward(self,input_ids,attention_mask):
        length=int(attention_mask.sum(1).max())
        input_ids=input_ids[:,:length]; attention_mask=attention_mask[:,:length]
        cls=self.drop(self.enc(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state[:,0])
        logits=self.sarc(cls)
        return (logits,self.sent(cls)) if self.sent is not None else logits

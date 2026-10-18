# Généré par python -m src.sync depuis le notebook ; modifier le notebook.
import numpy as np
import torch
from torch import nn
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def optimizer_step(model,loss,opt,scaler,scheduler):
    if scaler is None:
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if scheduler: scheduler.step()
    else:
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        previous_scale=scaler.get_scale(); scaler.step(opt); scaler.update()
        if scheduler and scaler.get_scale()>=previous_scale: scheduler.step()

def train_epoch(model,loader,opt,loss_fn,sent_loss,lam,scheduler,plm,name,seed,epoch,scaler=None):
    model.train(); total_loss=0.; steps=0
    for step,batch in enumerate(loader):
        *xx,yy,ss=[t.to(DEVICE,non_blocking=DEVICE.type=='cuda') for t in batch]
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type,dtype=torch.float16,enabled=scaler is not None):
            out=model(*xx); logits=out[0] if isinstance(out,tuple) else out
            loss=loss_fn(logits,yy)
            if lam: loss=loss+lam*sent_loss(out[1],ss)
        optimizer_step(model,loss,opt,scaler,scheduler)
        total_loss+=loss.item(); steps+=1
        if plm and steps%50==0:
            print(name,seed,'époque',epoch+1,'batch',steps,'/',len(loader),'perte',round(total_loss/steps,4),flush=True)
    return total_loss/max(steps,1)

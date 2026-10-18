"""Microbenchmarks CPU : forwards synthétiques et lots ; aucun apprentissage ni PLM téléchargé."""
from pathlib import Path
from copy import deepcopy
import json
import math
import os
import statistics
import time
import shutil

os.environ['USE_TF']='0'
os.environ['USE_FLAX']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset,RandomSampler,BatchSampler
from torch.utils.benchmark import Timer

ROOT=Path(__file__).resolve().parents[1]

def measure(function,threads):
    with torch.inference_mode():
        function(); function()
        return Timer('function()',globals={'function':function},num_threads=threads).blocked_autorange(min_run_time=.08).median*1000

def vector_loader(dataset,batch_size,seed):
    generator=torch.Generator().manual_seed(seed)
    sampler=RandomSampler(dataset,generator=generator)
    return DataLoader(dataset,batch_size=None,sampler=BatchSampler(sampler,batch_size,False),
                      generator=generator,num_workers=0,pin_memory=False)

def consume(loader):
    total=0
    for batch in loader: total+=len(batch[0])
    return total

def main():
    start=time.perf_counter()
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True,warn_only=True)
    from transformers import BertConfig,AutoModel
    from transformers.models.bert.modeling_bert import BertLayer
    config=BertConfig(hidden_size=768,intermediate_size=3072,num_attention_heads=12,
                      num_hidden_layers=1,vocab_size=32,layer_norm_eps=1e-12)
    config._attn_implementation='sdpa'
    layer=BertLayer(config).eval()
    shapes={length:(torch.randn(16,length,768),torch.zeros(16,1,1,length)) for length in [24,48]}
    timings=[]
    # Deux passages en ordre inverse limitent le biais d'échauffement/ordre des candidats.
    for round_id,threads_list in enumerate([[1,2,3,4,6,12],[12,6,4,3,2,1]]):
        for threads in threads_list:
            torch.set_num_threads(threads)
            for length,(x,mask) in shapes.items():
                ms=measure(lambda:layer(x,attention_mask=mask)[0],threads)
                timings.append({'round':round_id,'threads':threads,'length':length,'ms':ms})
            print('forward synthetic, round',round_id,'threads',threads,flush=True)
    means={t:statistics.mean(r['ms'] for r in timings if r['threads']==t) for t in [1,2,3,4,6,12]}
    best=min(means,key=means.get)
    stable=all(statistics.mean(r['ms'] for r in timings if r['round']==k and r['threads']==best)<
               .9*statistics.mean(r['ms'] for r in timings if r['round']==k and r['threads']==6) for k in [0,1])
    chosen=best if best!=6 and means[best]<.85*means[6] and stable else 6
    print('threads mean ms',means,'selected',chosen,flush=True)
    torch.set_num_threads(chosen)
    eager_config=deepcopy(config); eager_config._attn_implementation='eager'
    eager=BertLayer(eager_config).eval(); eager.load_state_dict(layer.state_dict())
    x,mask=shapes[48]
    with torch.inference_mode():
        a=eager(x,attention_mask=mask)[0]; b=layer(x,attention_mask=mask)[0]
        torch.testing.assert_close(a,b,rtol=1e-4,atol=2e-6)
    sdpa={'eager_layer_ms':measure(lambda:eager(x,attention_mask=mask)[0],chosen),
          'sdpa_layer_ms':measure(lambda:layer(x,attention_mask=mask)[0],chosen),
          'max_abs_difference_eval':float((a-b).abs().max())}
    q=torch.randn(16,12,48,64); k=torch.randn_like(q); v=torch.randn_like(q)
    for dropout in [0.,.1]:
        def manual():
            weights=(q@k.transpose(-1,-2)/math.sqrt(64)).softmax(-1)
            return F.dropout(weights,p=dropout,training=dropout>0)@v
        sdpa[f'kernel_eager_dropout_{dropout}_ms']=measure(manual,chosen)
        sdpa[f'kernel_sdpa_dropout_{dropout}_ms']=measure(lambda:F.scaled_dot_product_attention(q,k,v,dropout_p=dropout),chosen)
    tiny=AutoModel.from_config(BertConfig(vocab_size=32,hidden_size=32,intermediate_size=64,
                                         num_hidden_layers=1,num_attention_heads=4))
    sdpa['transformers_default_attention']=tiny.config._attn_implementation
    del tiny
    dataset=TensorDataset(torch.arange(4096*64).reshape(4096,64),torch.ones(4096,64,dtype=torch.long),
                          torch.zeros(4096,dtype=torch.long),torch.zeros(4096,dtype=torch.long))
    generator=torch.Generator().manual_seed(42)
    old=DataLoader(dataset,batch_size=16,shuffle=True,generator=generator,num_workers=0)
    new=vector_loader(dataset,16,42)
    # Deux époques de lecture, sans réseau ni gradients : ordre et tensors identiques.
    for _ in range(2):
        old_batches,new_batches=list(old),list(new)
        assert len(old_batches)==len(new_batches)
        for old_batch,new_batch in zip(old_batches,new_batches):
            assert all(torch.equal(a,b) for a,b in zip(old_batch,new_batch))
        assert torch.equal(generator.get_state(),new.generator.get_state())
    loader={'sample_collate_epoch_ms':measure(lambda:consume(old),chosen),
            'vectorized_epoch_ms':measure(lambda:consume(new),chosen),
            'exact_batch_order_and_rng':True}
    try:
        from torch._inductor.cpp_builder import get_cpp_compiler
        compiler=get_cpp_compiler()
        compilation={'available':True,'compiler':compiler,'enabled':False,'reason':'Pas de gain complet mesuré ; compilation non activée.'}
    except Exception as error:
        compilation={'available':False,'enabled':False,'reason':str(error).splitlines()[0]}
    report={'kind':'synthetic_forward_only_no_training','optimizer_steps':0,'backward_calls':0,
            'pretrained_models_loaded':0,'dataset_examples_used':0,'torch':torch.__version__,
            'cpu_capability':torch.backends.cpu.get_cpu_capability(),'mkldnn':torch.backends.mkldnn.enabled,
            'thread_measurements':timings,'thread_mean_ms':means,'selected_threads':chosen,
            'sdpa':sdpa,'dataloader':loader,'compile':compilation,'seconds':time.perf_counter()-start,
            'scope':'Une couche BERT aléatoire, tenseurs synthétiques. Aucun gain d’entraînement complet mesuré.'}
    (ROOT/'results/cpu_microbench.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='thread_measurements'},ensure_ascii=False,indent=2))

if __name__=='__main__': main()

"""Tests légers des optimisations : aucun entraînement, backward ou PLM préentraîné."""
from pathlib import Path
from types import SimpleNamespace
import ast
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hashlib
import html
import json
import os
import re
import shutil
import tempfile
import time
import unicodedata
os.environ['USE_TF']='0'; os.environ['USE_FLAX']='0'
import nbformat
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler,RandomSampler,BatchSampler
from transformers import AutoTokenizer,AutoModel,BertConfig
from transformers.trainer_pt_utils import LengthGroupedSampler
from arabert.preprocess import ArabertPreprocessor
import emoji
from src import preprocess
from src.models import FineTuneNet

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT

def reference_loader(x,y,sy,imbalance,plm):
    dataset=TensorDataset(*x,y,sy); generator=torch.Generator().manual_seed(42); sampler=None
    if imbalance=='oversample':
        weights=(1/np.bincount(y.numpy()))[y.numpy()]
        sampler=WeightedRandomSampler(weights,len(y),replacement=True,generator=generator)
    elif plm: sampler=LengthGroupedSampler(batch_size=16,lengths=x[1].sum(1).tolist(),generator=generator)
    return DataLoader(dataset,batch_size=16 if plm else 64,shuffle=sampler is None,sampler=sampler,
                      generator=generator,num_workers=0)

class SumClassifier(nn.Module):
    def __init__(self):
        super().__init__(); self.enc=nn.Identity(); self.calls=0
    def forward(self,ids,mask):
        self.calls+=1
        value=(ids*mask).sum(1).float()/1000
        return torch.stack([value,-value],1)

class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__(); self.config=SimpleNamespace(hidden_size=4)
    def forward(self,input_ids,attention_mask):
        states=input_ids.float().unsqueeze(-1)/1000*torch.arange(1,5)
        return SimpleNamespace(last_hidden_state=states)

def main():
    torch.set_num_threads(2); torch.manual_seed(42)
    nb=nbformat.read(WORK/'notebooks/TP2_Sarcasme_Arabe_T4.ipynb',4); nbformat.validate(nb)
    code=[c for c in nb.cells if c.cell_type=='code']
    assert len(code)==104 and all(c.execution_count is None and not c.outputs for c in code)
    source='\n\n'.join(c.source for c in code)
    nodes={n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    selected=['inference_signature','valid_cache','mark_cache','checkpoint_stat','compatible_metadata',
              'validation_scores','neural_predict','make_training_loader','tokenizer_for','plm_texts',
              'plm_encode','frozen_encoder','frozen_features']
    # Seules ces définitions sont extraites ; aucune cellule et aucune boucle d'apprentissage n'est appelée.
    definitions=ast.Module(body=[nodes[name] for name in selected],type_ignores=[])
    temporary_parent=ROOT/'tmp'; temporary_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='cpu_light_',dir=temporary_parent) as directory:
        temp=Path(directory).resolve(); assert temp.is_relative_to(temporary_parent.resolve())
        for folder in ['src','models','predictions','results']: (temp/folder).mkdir()
        shutil.copy2(ROOT/'src/preprocess.py',temp/'src/preprocess.py')
        revisions=json.loads((ROOT/'models/revisions.json').read_text())
        for record in revisions.values(): record['path']=str(ROOT/record['path'])
        ara='aubmindlab/bert-base-arabertv02-twitter'; mar='UBC-NLP/MARBERTv2'
        namespace=dict(globals(),ROOT=temp,BATCH=16,MAX_LEN=64,DEVICE=torch.device('cpu'),
            RUN_VERSION='tp2-v1',CACHE_CONTEXT={'synthetic_test':True},TOKENIZERS={},TOKENIZED={},
            FROZEN_MODELS={},REVISIONS=revisions,ARA_NAME=ara,
            ARA_PREP=ArabertPreprocessor(model_name=ara,remove_non_digit_repetition=False),
            prepare_text=preprocess.prepare_text,export_clean=preprocess.export_clean,
            URL_RE=preprocess.URL_RE,MENTION_RE=preprocess.MENTION_RE)
        exec(compile(definitions,'definitions-only','exec'),namespace)
        # Les quatre stratégies conservent tous les batches ET l'état final du générateur.
        ids=torch.randint(1,100,(135,64)); lengths=torch.arange(135)%64+1
        mask=(torch.arange(64)[None,:]<lengths[:,None]).long()
        y=(torch.arange(135)%5==0).long(); sy=torch.arange(135)%3
        checked=[]
        for imbalance,plm in [('weighted',False),('standard',False),('oversample',False),('weighted',True)]:
            old=reference_loader([ids,mask],y,sy,imbalance,plm)
            new=namespace['make_training_loader']([ids,mask],y,sy,42,imbalance,plm)
            for epoch in range(2):
                a,b=list(old),list(new); assert len(a)==len(b)
                assert all(all(torch.equal(x,z) for x,z in zip(batch_a,batch_b)) for batch_a,batch_b in zip(a,b))
                assert torch.equal(old.generator.get_state(),new.generator.get_state())
            checked.append([imbalance,plm])
        # Le regroupement d'inférence remet chaque score sur son tweet original.
        model=SumClassifier().eval()
        with torch.inference_mode(): expected=model(ids[:17],mask[:17]).softmax(-1)[:,1].numpy()
        actual=namespace['neural_predict'](model,[ids[:17],mask[:17]],batch=3)
        np.testing.assert_array_equal(actual,expected)
        # Petit BERT aléatoire, jamais entraîné : padding fixe vs tri/trim doivent rester proches.
        config=BertConfig(vocab_size=128,hidden_size=32,intermediate_size=64,
                          num_hidden_layers=1,num_attention_heads=4,max_position_embeddings=64)
        classifier=FineTuneNet('synthetic',encoder=AutoModel.from_config(config)).eval()
        with torch.inference_mode():
            full_states=classifier.enc(input_ids=ids[:17],attention_mask=mask[:17]).last_hidden_state[:,0]
            fixed=classifier.sarc(classifier.drop(full_states)).softmax(-1)[:,1].numpy()
        trimmed=namespace['neural_predict'](classifier,[ids[:17],mask[:17]],batch=3)
        np.testing.assert_allclose(trimmed,fixed,rtol=1e-4,atol=2e-6)
        # Le cache de validation évite le forward et se périme si les poids changent.
        checkpoint=temp/'models/toy_seed42.pt'; checkpoint.write_bytes(b'original-checkpoint')
        namespace['validation_scores'](model,[ids[:17],mask[:17]],checkpoint,'toy-signature',candidate=actual)
        calls=model.calls
        cached=namespace['validation_scores'](model,[ids[:17],mask[:17]],checkpoint,'toy-signature')
        assert model.calls==calls; np.testing.assert_array_equal(cached,actual)
        metadata=checkpoint.with_suffix('.json')
        metadata.write_text(json.dumps({'runtime_context':namespace['CACHE_CONTEXT'],
                                         'checkpoint_stat':namespace['checkpoint_stat'](checkpoint)}))
        assert namespace['compatible_metadata'](metadata)
        checkpoint.write_bytes(b'replaced-checkpoint-with-another-size')
        assert not namespace['compatible_metadata'](metadata)
        namespace['validation_scores'](model,[ids[:17],mask[:17]],checkpoint,'toy-signature')
        assert model.calls>calls
        # Quatre textes seulement, tokenizers locaux ; aucun poids de PLM n'est ouvert.
        texts=['يا سلاااام 😂 ههههه !!!','لَا أحب هذا أبداً','اختبار قصير','موضوع أطول قليلاً عن تجربة ممتازة وتأخير غير متوقع']
        for name in [ara,mar]:
            tokenizer=namespace['tokenizer_for'](name)
            expected=tokenizer(namespace['plm_texts'](texts,name),truncation=True,max_length=64,padding='max_length',return_tensors='pt')
            actual=namespace['plm_encode'](texts,name)
            assert torch.equal(actual[0],expected['input_ids']) and torch.equal(actual[1],expected['attention_mask'])
            namespace['TOKENIZED'].clear()  # Forcer le test du cache disque, sans retokeniser.
            disk=namespace['plm_encode'](texts,name)
            assert all(torch.equal(a,b) for a,b in zip(actual,disk))
        loads=[]
        def toy_load(name):
            loads.append(name); namespace['FROZEN_MODELS'].clear()
            return ToyEncoder().eval()
        namespace['load_hf']=toy_load
        h,m=namespace['frozen_features'](texts,'toy_train')
        assert np.isfinite(np.asarray(h)).all()
        encoded=namespace['plm_encode'](texts,ara)
        with torch.inference_mode(): full=ToyEncoder()(*encoded).last_hidden_state.numpy().astype('float16')
        assert np.array_equal(np.asarray(h)[m.astype(bool)],full[m.astype(bool)])
        assert np.array_equal(m,encoded[1].numpy())
        h._mmap.close()
        second,second_mask=namespace['frozen_features'](texts[::-1],'toy_val')
        second._mmap.close()
        assert len(loads)==1,'L’encodeur gelé devait être réutilisé.'
    audit={'passed':True,'notebook_executed':False,'optimizer_steps':0,'backward_calls':0,
           'pretrained_models_loaded':0,'corpus_inference_calls':0,'toy_tokenizer_texts':4,
           'loader_exact_for_two_epochs':checked,'prediction_order_exact':True,
           'toy_bert_padding_close':True,'toy_bert_max_abs_probability_difference':float(np.max(np.abs(trimmed-fixed))),
           'validation_cache_hit_and_invalidation':True,'tokenization_cache_exact':True,
           'frozen_trim_valid_states_and_row_order':True,'frozen_encoder_reused':True}
    (ROOT/'results/cpu_tests.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))

if __name__=='__main__': main()

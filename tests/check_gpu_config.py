"""Configuration CUDA/AMP vérifiée sans GPU, entraînement ou backward PyTorch."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import ast
import hashlib
import json
import os
os.environ['USE_TF']='0'; os.environ['USE_FLAX']='0'
import nbformat
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def main():
    book=nbformat.read(ROOT/'notebooks/TP2_Sarcasme_Arabe_T4.ipynb',4)
    code=[c for c in book.cells if c.cell_type=='code']
    nodes={n.name:n for n in ast.parse('\n\n'.join(c.source for c in code)).body
           if isinstance(n,ast.FunctionDef)}
    namespace={'torch':torch,'nn':nn}
    definitions=ast.Module(body=[nodes['select_device'],nodes['optimizer_step']],type_ignores=[])
    exec(compile(definitions,'definitions-only','exec'),namespace)
    with patch('torch.cuda.is_available',return_value=False):
        assert namespace['select_device'](False).type=='cpu'
        try: namespace['select_device'](True)
        except RuntimeError as error: assert 'GPU T4' in str(error)
        else: raise AssertionError('Colab CPU devait être refusé.')
    with patch('torch.cuda.is_available',return_value=True):
        assert namespace['select_device'](True).type=='cuda'

    # Ces doublures ne sont ni des tenseurs, ni un optimiseur PyTorch.
    def mock_step(amp,overflow):
        events=[]
        class FakeLoss:
            def backward(self): events.append('mock_backward')
        class FakeScaler:
            current=128.
            def scale(self,loss): events.append('scale'); return loss
            def unscale_(self,opt): events.append('unscale')
            def get_scale(self): return self.current
            def step(self,opt):
                events.append('scaler_step')
                if not overflow: opt.step()
            def update(self):
                events.append('update')
                if overflow: self.current/=2
        model=SimpleNamespace(parameters=lambda:[])
        optimizer=SimpleNamespace(step=lambda:events.append('mock_optimizer_step'))
        scheduler=SimpleNamespace(step=lambda:events.append('scheduler'))
        with patch.object(torch.Tensor,'backward',side_effect=AssertionError('Backward réel interdit.')):
            with patch.object(nn.utils,'clip_grad_norm_',side_effect=lambda *a:events.append('clip')):
                namespace['optimizer_step'](model,FakeLoss(),optimizer,FakeScaler() if amp else None,scheduler)
        return events
    assert mock_step(False,False)==['mock_backward','clip','mock_optimizer_step','scheduler']
    assert mock_step(True,False)==['scale','mock_backward','unscale','clip','scaler_step','mock_optimizer_step','update','scheduler']
    assert mock_step(True,True)==['scale','mock_backward','unscale','clip','scaler_step','update']
    # Configuration du contexte sans poids de modèle, CUDA simulé seulement.
    context=next(c.source for c in code if "STATIC_EPOCHS, PLM_EPOCHS" in c.source)
    import numpy as np
    import pandas as pd
    scope={'MAX_LEN':64,'BATCH':32,'torch':torch,'np':np,'pd':pd,'DEVICE':torch.device('cuda'),
           'PLM_PRECISION':'amp_float16','ROOT':ROOT,'json':json}
    with patch('torch.cuda.get_device_name',return_value='SIMULATED_T4'):
        exec(compile(context,'configuration-only','exec'),scope)
    assert scope['CACHE_CONTEXT']['plm_precision']=='amp_float16'
    assert scope['CACHE_CONTEXT']['batch_plm']==32 and scope['CACHE_CONTEXT']['device_type']=='cuda'
    scope.update(DEVICE=torch.device('cpu'),BATCH=16,PLM_PRECISION='float32')
    exec(compile(context,'configuration-only','exec'),scope)
    assert 'device_type' not in scope['CACHE_CONTEXT']
    assert scope['CACHE_CONTEXT']['batch_plm']==16
    # Architectures comparées à la copie avant adaptation, pas seulement au nom des classes.
    audit=json.loads((ROOT/'results/gpu_adaptation_audit.json').read_text())
    previous=nbformat.read(ROOT/audit['backup']/'notebooks/TP2_Sarcasme_Arabe.ipynb',4)
    classes=lambda b:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(
        '\n\n'.join(c.source for c in b.cells if c.cell_type=='code')).body if isinstance(n,ast.ClassDef)}
    assert classes(book)==classes(previous)
    assert hashlib.sha256((ROOT/'results/results.csv').read_bytes()).hexdigest()==audit['cpu_results_sha256']
    assert hashlib.sha256((ROOT/'src/preprocess.py').read_bytes()).hexdigest()==audit['preprocess_sha256']
    assert any("DEVICE=torch.device('cpu')" in c.source and 'comparison=[]' in c.source for c in code)
    result={'passed':True,'colab_cpu_rejected':True,'cuda_selection_simulated':True,
        'amp_unscale_before_clipping_verified_with_mocks':True,'scheduler_skip_on_overflow_verified_with_mocks':True,
        'cuda_batch_32_cpu_batch_16':True,'cpu_gpu_cache_context_distinct':True,
        'architectures_unchanged':True,'historical_results_unchanged':True,'g_latency_kept_on_cpu':True,
        'gpu_runtime_tested':False,'real_optimizer_steps':0,'pytorch_backward_calls':0,
        'pretrained_models_loaded':0,'notebook_executed':False}
    (ROOT/'results/gpu_config_tests.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()

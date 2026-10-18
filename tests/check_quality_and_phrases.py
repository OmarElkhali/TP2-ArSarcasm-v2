"""Contrôles légers : aucune boucle d'entraînement ni inférence de PLM préentraîné."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile
import ast
import base64
import hashlib
import io
import json
import os
import tempfile

os.environ['USE_TF']='0'
os.environ['USE_FLAX']='0'
os.environ['MPLBACKEND']='Agg'
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_recall_curve
from src.evaluate import scores, best_threshold
from src.predict import PhraseTester

ROOT=Path(__file__).resolve().parents[1]


def main():
    book=nbformat.read(ROOT/'notebooks/TP2_Sarcasme_Arabe_T4.ipynb',4)
    code=[c for c in book.cells if c.cell_type=='code']
    tree=ast.parse('\n\n'.join(c.source for c in code))
    nodes={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    selected=ast.Module(body=[nodes[n] for n in ['archive_previous_run','savefig']],type_ignores=[])
    scope={'torch':torch,'plt':plt}
    exec(compile(selected,'definitions-only','exec'),scope)
    assert 'PLM_EPOCHS, PATIENCE = 8, 5, 2' in code[3].source
    assert 'torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)' in code[49].source
    assert "optimizer_policy='adamw_uniform_wd_0p01_v1'" in code[49].source
    assert 'archive_previous_run(path,meta)' in code[49].source

    audit=json.loads((ROOT/'results/quality_and_plots_changes.json').read_text())
    prior=nbformat.read(ROOT/audit['backup']/'notebooks/TP2_Sarcasme_Arabe_T4.ipynb',4)
    prior_tree=ast.parse('\n\n'.join(c.source for c in prior.cells if c.cell_type=='code'))
    old_classes={n.name:ast.dump(n,include_attributes=False) for n in prior_tree.body if isinstance(n,ast.ClassDef)}
    new_classes={n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.ClassDef)}
    assert old_classes==new_classes
    assert sum(c.execution_count is not None for c in prior.cells if c.cell_type=='code')==104
    assert not any(c.execution_count is not None for c in code)

    with tempfile.TemporaryDirectory(dir=ROOT/'tmp',prefix='quality_') as directory:
        root=Path(directory)
        for name in ['models','predictions','results','figures']:(root/name).mkdir()
        for name in ['models/mock.pt','models/mock.json','predictions/mock_val.npy',
                     'predictions/mock_val.validation.json','results/results.csv','results/registry.json']:
            (root/name).write_text('fixture',encoding='utf-8')
        scope.update(ROOT=root,time=SimpleNamespace(time_ns=lambda:123),display=lambda fig:shown.append(fig))
        scope['archive_previous_run'](root/'models/mock.pt',root/'models/mock.json')
        assert not (root/'models/mock.pt').exists()
        assert (root/'results/history/mock_123/mock.pt').read_text()=='fixture'
        assert (root/'results/history/mock_123/registry.json').read_text()=='fixture'

        # Reproduire le cas d'une figure fermée par le backend inline, puis afficher l'objet précis.
        shown=[]
        fig,axes=plt.subplots(1,2)
        axes[0].plot([0,1],[.2,.7]);axes[1].plot([0,1],[.8,.3]);plt.close(fig)
        scope['savefig'](fig,'closed_figure_test')
        assert shown[-1] is fig and (root/'figures/closed_figure_test.png').stat().st_size>1000
        empty=plt.figure()
        try: scope['savefig'](empty,'empty')
        except ValueError: pass
        else: raise AssertionError('Une figure vide ne doit pas être présentée comme valide.')
        plt.close(empty)

        # La vraie cellule de courbes, avec des probabilités jouets, sans labels réels de test.
        scope.update(np=np,pd=pd,f1_score=f1_score,precision_recall_curve=precision_recall_curve,
                     best_threshold=best_threshold,scores=scores,test=SimpleNamespace(y=np.array([0,1,0,1])),
                     FINAL_GROUPS={'A':['a'],'C':['c']},TEST_P={'a':np.array([.1,.7,.3,.9]),'c':np.array([.2,.6,.7,.8])},
                     REGISTRY={'a':{'threshold':.5},'c':{'threshold':.5}})
        assert 'plt.subplots' not in code[84].source and 'plt.subplots' in code[85].source
        exec(compile(code[85].source,'plot-cell-only','exec'),scope)
        assert len(shown[-1].axes)==2 and all(ax.lines for ax in shown[-1].axes)
        assert not plt.get_fignums()
        # Même cellule pour la création et le tracé du prior ; pas d'axes vides en cellule 18.
        assert 'plt.subplots' not in code[17].source and 'plt.subplots' in code[18].source
        assert 'ConfusionMatrixDisplay.from_predictions' not in code[100].source
        assert 'ConfusionMatrixDisplay.from_predictions' in code[101].source

    # Inférence réelle légère de la baseline locale, sans fit ; aucun poids de PLM chargé.
    reader=PhraseTester(ROOT,device='cpu')
    examples=['يا سلام، تأخر القطار مرة أخرى 😂','لم يعجبني هذا الفيلم.']
    with patch('torch.Tensor.backward',side_effect=AssertionError('Backward interdit.')):
        first=reader.predict(examples,'A_tfidf_standard_seed42')
        model=reader.model
        second=reader.predict(examples,'A_tfidf_standard_seed42')
    assert reader.model is model and first.equals(second) and len(first)==2
    assert np.array_equal(first.prediction.eq('Sarcastique'),first.score.ge(first.seuil_val))
    try: reader.predict(['   '])
    except ValueError: pass
    else: raise AssertionError('Entrée vide acceptée.')
    try: reader.local_path('../outside')
    except ValueError: pass
    else: raise AssertionError('Chemin hors projet accepté.')
    try: reader.predict('phrase','absent')
    except KeyError: pass
    else: raise AssertionError('Modèle inconnu accepté.')
    reader.clear()

    # B2 : P(négatif)=0.4 peut être l'argmax à trois classes ; ce n'est pas un seuil 0.5.
    class FakeBatch(dict):
        def to(self,device):return self
    class FakeTokenizer:
        def __call__(self,*args,**kwargs):return FakeBatch()
    fake_model=SimpleNamespace(config=SimpleNamespace(id2label={0:'positive',1:'negative',2:'neutral'}),
        __call__=None)
    class FakeSentiment:
        config=fake_model.config
        def __call__(self,**kwargs):return SimpleNamespace(logits=torch.log(torch.tensor([[.35,.4,.25]])))
    reader.model=FakeSentiment();reader.tokenizer=FakeTokenizer()
    rec={'kind':'sentiment','config':{},'threshold':.5,'key':'B2_mock'}
    with patch.object(reader,'_load',return_value=rec):out=reader.predict(['texte'],'B2_mock')
    assert out.iloc[0].prediction=='Sarcastique' and np.isclose(out.iloc[0].score,.4)
    assert np.isnan(out.iloc[0].seuil_val)

    # Autres familles : tenseurs synthétiques et doublures seulement, sans poids préentraînés.
    class FakeTokens:
        def __call__(self,texts,*args,**kwargs):
            if kwargs.get('return_tensors')=='pt':
                return FakeBatch(input_ids=torch.ones((len(texts),4),dtype=torch.long),
                                 attention_mask=torch.ones((len(texts),4),dtype=torch.long))
            return {'input_ids':[[1,2] for _ in texts]}
        def batch_decode(self,ids,**kwargs):return ['مرحبا']*len(ids)
    class FakeHead:
        def __call__(self,*args):return torch.tensor([[0.,1.]]).repeat(len(args[0]),1)
    class FakePLM(FakeHead):
        def __call__(self,*args):return super().__call__(*args),torch.zeros((len(args[0]),3))
    class FakeEncoder:
        def __call__(self,**kwargs):
            return SimpleNamespace(last_hidden_state=torch.ones((len(kwargs['input_ids']),4,2)))
    class FakeLR:
        def predict_proba(self,x):return np.tile([.25,.75],(len(x),1))
    class FakeNLI:
        config=SimpleNamespace(id2label={0:'contradiction',1:'neutral',2:'entailment'})
        def __call__(self,**kwargs):
            n=len(kwargs['input_ids']);logits=torch.zeros((n,3));logits[::2,2]=1
            return SimpleNamespace(logits=logits)
    reader.tokenizer=FakeTokens();reader.preprocessor=SimpleNamespace(preprocess=lambda s:s)
    reader.encoder=FakeEncoder();reader.vocab={'<PAD>':0,'<UNK>':1};reader.max_len=64;reader.amp=False
    for kind,model,cfg in [
        ('static',FakeHead(),{}),('plm',FakePLM(),{'multitask':True}),
        ('frozen_lr',FakeLR(),{'pooling':'mean'}),('frozen_head',FakeHead(),{}),
        ('nli',FakeNLI(),{'language':'en'})]:
        reader.model=model
        rec={'kind':kind,'config':cfg,'threshold':.5,'key':kind+'_mock'}
        with patch.object(reader,'_load',return_value=rec):out=reader.predict(['مرحبا'],'mock')
        assert len(out)==1 and out.iloc[0].prediction=='Sarcastique' and .5<out.iloc[0].score<1,kind

    phrase_book=nbformat.read(ROOT/'notebooks/Tester_Phrases.ipynb',4)
    nbformat.validate(phrase_book)
    for cell in phrase_book.cells:
        if cell.cell_type=='code':
            compile(cell.source,'phrase-notebook-static','exec')
            assert cell.execution_count is None and not cell.outputs
    setup=next(c.source for c in phrase_book.cells if c.cell_type=='code' and 'SOURCE_ARCHIVE =' in c.source)
    archive_node=next(n for n in ast.parse(setup).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='SOURCE_ARCHIVE' for t in n.targets))
    payload=base64.b64decode(ast.literal_eval(archive_node.value))
    with ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():assert archive.read(name)==(ROOT/name).read_bytes()
    assert hashlib.sha256(payload).hexdigest()==phrase_book.metadata.tp2.source_archive_sha256
    result={'passed':True,'architectures_unchanged':True,'plm_max_epochs':5,
        'uniform_weight_decay_0p01_verified':True,'all_model_parameters_in_optimizer':True,
        'incompatible_run_archive_tested_with_fixtures':True,'closed_figure_explicitly_rendered':True,
        'empty_figure_rejected':True,'threshold_pr_cell_has_two_populated_axes':True,
        'phrase_baseline_inference_real':True,'phrase_model_reused':True,'b2_argmax_rule_verified_with_mock':True,
        'other_five_inference_branches_verified_with_mocks':True,
        'inference_notebook_bundle_verified':True,'pretrained_plm_models_loaded':0,
        'new_training_runs':0,'optimizer_steps':0,'backward_calls':0,'test_labels_read':False}
    (ROOT/'results/quality_and_phrases_tests.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

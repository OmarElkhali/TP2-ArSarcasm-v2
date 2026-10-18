"""Après votre Run All : contrôler les sorties réelles, sans relancer les modèles."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import nbformat
import pymupdf
from sklearn.metrics import f1_score,average_precision_score

ROOT=Path(__file__).resolve().parents[1]

def main():
    nb=nbformat.read(ROOT/'notebooks/TP2_Sarcasme_Arabe_T4.ipynb',4)
    code=[c for c in nb.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code),'Le notebook ne contient pas encore tous les compteurs.'
    assert [c.execution_count for c in code]==list(range(1,len(code)+1)), 'Exécuter dans un noyau neuf puis sauvegarder.'
    assert not [o for c in code for o in c.outputs if o.output_type=='error']
    official=pd.read_csv(ROOT/'data/raw/testing_data.csv').sarcasm.astype(int).to_numpy()
    manifest=json.loads((ROOT/'results/final_manifest.json').read_text(encoding='utf-8'))
    results=pd.read_csv(ROOT/'results/results.csv',float_precision='round_trip'); checked=[]
    for rec in manifest['models']:
        key=rec['key']; p=pd.read_csv(ROOT/f'predictions/{key}.csv',float_precision='round_trip')
        assert len(p)==3000 and np.array_equal(p.id,np.arange(3000))
        assert np.array_equal(p.y_true,official) and p.probability.between(0,1).all()
        row=results.loc[results.model.eq(rec['name'])&results.seed.eq(rec['seed'])].iloc[0]
        f=f1_score(p.y_true,p.y_pred,zero_division=0); ap=average_precision_score(p.y_true,p.probability)
        assert abs(f-row.test_f1_sarcastique)<1e-10 and abs(ap-row.test_pr_auc_ap)<1e-10,key
        if rec['kind']!='sentiment': assert np.array_equal(p.y_pred,p.probability.ge(rec['threshold']).astype(int))
        checked.append({'model':key,'f1':f,'ap':ap})
    for family,keys in manifest['groups'].items():
        seeds=[next(r['seed'] for r in manifest['models'] if r['key']==k) for k in keys]
        assert sorted(seeds)==[42,43,44],family
    assert len(checked)==17
    bootstrap=pd.read_csv(ROOT/'results/bootstrap.csv')
    assert len(bootstrap)==15 and bootstrap.resamples.eq(1000).all()
    errors=pd.read_csv(ROOT/'results/errors_30.csv')
    assert errors.type.value_counts().to_dict()=={'FN':15,'FP':15}
    pdf=pymupdf.open(ROOT/'Compte_Rendu_TP2.pdf')
    assert len(pdf)==8 and all(len(page.get_text().strip())>100 for page in pdf)
    audit={'code_cells':len(code),'executed':len(code),'errors':0,'prediction_files_verified':len(checked),
           'predictions':checked,'report_pages':len(pdf),'human_error_review_required':True}
    (ROOT/'results/final_audit.json').write_text(json.dumps(audit,indent=2))
    print('Audit final réussi :',len(code),'cellules, 17 fichiers de prédictions, 8 pages PDF. Relire les 30 erreurs et le PDF.')

if __name__=='__main__': main()

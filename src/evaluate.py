# Généré par python -m src.sync depuis le notebook ; modifier le notebook.
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, precision_recall_curve

def scores(y,p,threshold):
    y=np.asarray(y,dtype=int); p=np.asarray(p); pred=(p>=threshold).astype(int)
    return {'f1_sarcastique':float(f1_score(y,pred,zero_division=0)),
        'precision':float(precision_score(y,pred,zero_division=0)),
        'recall':float(recall_score(y,pred,zero_division=0)),
        'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
        'pr_auc_ap':float(average_precision_score(y,p)) if y.sum() else float('nan')}

def best_threshold(y,p):
    prec,rec,thresholds=precision_recall_curve(y,p)
    fs=2*prec[:-1]*rec[:-1]/np.maximum(prec[:-1]+rec[:-1],1e-12)
    i=np.flatnonzero(np.isclose(fs,fs.max(),atol=1e-12))[-1]
    return float(thresholds[i])

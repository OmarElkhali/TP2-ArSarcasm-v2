"""Inférence des checkpoints du TP sur des phrases, sans charger le notebook."""
from pathlib import Path
import gc
import json
import math
import os
import re

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from .models import FineTuneNet, FrozenHead, StaticNet
from .preprocess import export_clean, prepare_text, URL_RE, MENTION_RE

ARA = 'aubmindlab/bert-base-arabertv02-twitter'
PROMPTS = {'ar': ('هذه التغريدة ساخرة.', 'هذه التغريدة جادة.'),
           'fr': ('Ce tweet est sarcastique.', 'Ce tweet est sérieux.'),
           'en': ('This tweet is sarcastic.', 'This tweet is serious.')}


class PhraseTester:
    """Un seul modèle en mémoire ; les seuils proviennent du registre de validation."""

    def __init__(self, root, device=None):
        self.root = Path(root).resolve()
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = self.encoder = self.tokenizer = self.preprocessor = None
        self.loaded = None
        self.refresh()

    def local_path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Le chemin du modèle doit rester dans le projet.')
        return path

    def refresh(self):
        path = self.root/'results/registry.json'
        if not path.is_file():
            raise FileNotFoundError(f'Registre absent : {path}. Utiliser le dossier contenant les checkpoints entraînés.')
        self.records = json.loads(path.read_text(encoding='utf-8'))
        self.results = pd.read_csv(self.root/'results/results.csv')
        self.loaded = None

    def list_models(self):
        rows = []
        for key, rec in self.records.items():
            match = self.results.loc[self.results.model.eq(rec['name']) & self.results.seed.eq(rec['seed'])]
            rows.append({'key':key, 'modele':rec['name'], 'famille':rec['kind'], 'graine':rec['seed'],
                         'f1_validation':float(match.iloc[0].val_f1_sarcastique) if len(match) else np.nan,
                         'disponible':self.local_path(rec['path']).exists()})
        columns=['key','modele','famille','graine','f1_validation','disponible']
        return pd.DataFrame(rows,columns=columns).sort_values('f1_validation', ascending=False, kind='stable').reset_index(drop=True)

    def default_key(self):
        path = self.root/'results/decision.json'
        if path.is_file():
            key = json.loads(path.read_text(encoding='utf-8')).get('chosen_key')
            if key in self.records and self.local_path(self.records[key]['path']).exists():
                return key
        table = self.list_models()
        table = table.loc[table.disponible & ~table.modele.str.startswith('F_') & table.f1_validation.notna()]
        if table.empty:
            raise FileNotFoundError('Aucun modèle entraîné disponible dans ce dossier.')
        return table.iloc[0]['key']

    def clear(self):
        self.model = self.encoder = self.tokenizer = self.preprocessor = None
        self.loaded = None
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def _hf(self, name, classification=False):
        folder = self.local_path('models/hf/'+name.replace('/', '--'))
        cls = AutoModelForSequenceClassification if classification else AutoModel
        return cls.from_pretrained(str(folder), local_files_only=True).to(self.device).eval()

    def _load(self, key):
        if key not in self.records:
            raise KeyError(f'Modèle inconnu : {key}. Consulter list_models().')
        rec = self.records[key]
        path = self.local_path(rec['path'])
        if not path.exists():
            raise FileNotFoundError(f'Checkpoint absent : {path}. Récupérer les poids entraînés depuis Colab.')
        stamp = (key, path.stat().st_size, path.stat().st_mtime_ns)
        if stamp == self.loaded:
            return rec
        self.clear()
        cfg, kind = rec['config'], rec['kind']
        self.max_len = 64
        context = {}
        if path.is_file() and path.suffix == '.pt':
            metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            context = metadata.get('runtime_context', {})
            self.max_len = int(context.get('max_len', 64))
        self.amp = self.device.type == 'cuda' and context.get('plm_precision') == 'amp_float16'
        if kind in ['tfidf', 'frozen_lr']:
            self.model = joblib.load(path)
        if kind == 'static':
            self.vocab = json.loads((self.root/'models/vocab.json').read_text(encoding='utf-8'))
            self.model = StaticNet(len(self.vocab), cfg['arch'], position=cfg.get('position', True))
        elif kind in ['plm', 'frozen_lr', 'frozen_head', 'nli', 'sentiment']:
            name = cfg.get('model_name', ARA)
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.local_path('models/hf/'+name.replace('/', '--'))), local_files_only=True)
            if kind in ['nli', 'sentiment']:
                self.model = self._hf(name, True)
            elif kind == 'plm':
                self.model = FineTuneNet(name, cfg['multitask'], encoder=self._hf(name))
            else:
                self.encoder = self._hf(ARA)
                if kind == 'frozen_head':
                    self.model = FrozenHead(self.encoder.config.hidden_size)
            if name == ARA:
                from arabert.preprocess import ArabertPreprocessor
                self.preprocessor = ArabertPreprocessor(model_name=ARA, remove_non_digit_repetition=False)
        elif kind != 'tfidf':
            raise ValueError(f'Famille non prise en charge : {kind}')
        if kind in ['static', 'frozen_head', 'plm']:
            state = torch.load(path, map_location='cpu', weights_only=True)
            self.model.load_state_dict(state)
            del state
            self.model.to(self.device).eval()
        self.loaded = stamp
        return rec

    def _encode(self, texts, cfg):
        name = cfg.get('model_name', ARA)
        if name == ARA:
            texts = [export_clean(s) for s in texts]
            if cfg.get('drop_links'):
                texts = [MENTION_RE.sub(' ', URL_RE.sub(' ', s)) for s in texts]
            texts = [self.preprocessor.preprocess(s) for s in texts]
        else:
            texts = [prepare_text(s, 'plm', cfg.get('drop_links', False)) for s in texts]
        enc = self.tokenizer(texts, truncation=True, max_length=self.max_len,
                             padding='max_length', return_tensors='pt')
        return {k:enc[k].to(self.device) for k in ['input_ids', 'attention_mask']}

    def predict(self, phrases, key=None, batch_size=8):
        texts = [phrases] if isinstance(phrases, str) else list(phrases)
        if not texts or any(not isinstance(s, str) or not export_clean(s).strip() for s in texts):
            raise ValueError('Saisir au moins une phrase non vide ; chaque phrase doit être du texte.')
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size doit être un entier positif.')
        rec = self._load(key or self.default_key())
        cfg, kind = rec['config'], rec['kind']
        threshold = float(rec['threshold'])
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('Seuil de validation invalide dans le registre.')
        values, labels = [], []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start+batch_size]
                if cfg.get('drop_links') and kind in ['static', 'frozen_lr', 'frozen_head']:
                    batch = [MENTION_RE.sub(' ', URL_RE.sub(' ', s)) for s in batch]
                decision = None
                if kind == 'tfidf':
                    p = self.model.predict_proba([prepare_text(s,cfg['mode'],cfg.get('drop_links',False)) for s in batch])[:,1]
                elif kind == 'static':
                    x = np.zeros((len(batch), self.max_len), dtype=np.int64)
                    for i, s in enumerate(batch):
                        tokens = re.findall(r'\w+|[^\w\s]', prepare_text(s), re.UNICODE)
                        ids = [self.vocab.get(t, 1) for t in tokens[:self.max_len]] or [1]
                        x[i, :len(ids)] = ids
                    p = self.model(torch.tensor(x, device=self.device)).float().softmax(-1)[:,1].cpu().numpy()
                elif kind in ['plm', 'frozen_lr', 'frozen_head']:
                    enc = self._encode(batch, cfg)
                    if kind == 'plm':
                        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
                            out = self.model(enc['input_ids'], enc['attention_mask'])
                        logits = out[0] if isinstance(out, tuple) else out
                    else:
                        # Même quantification float16 des états que les caches utilisés en D.
                        h = self.encoder(**enc).last_hidden_state.half().float()
                        mask = enc['attention_mask']
                        if kind == 'frozen_lr':
                            h, m = h.cpu().numpy(), mask.cpu().numpy()
                            pooled = h[:,0] if cfg['pooling']=='cls' else ((h*m[...,None]).sum(1)/m.sum(1,keepdims=True)).astype(np.float32)
                            p = self.model.predict_proba(pooled)[:,1]
                        else:
                            logits = self.model(h, mask)
                    if kind != 'frozen_lr':
                        p = logits.float().softmax(-1)[:,1].cpu().numpy()
                elif kind == 'nli':
                    raw = [export_clean(s) for s in batch]
                    ids = self.tokenizer(raw, truncation=True, max_length=self.max_len, add_special_tokens=False)['input_ids']
                    premises = self.tokenizer.batch_decode(ids, skip_special_tokens=True)
                    enc = self.tokenizer([s for s in premises for _ in range(2)],
                        [h for _ in premises for h in PROMPTS[cfg['language']]],
                        padding=True, truncation='only_first', max_length=self.max_len+24, return_tensors='pt').to(self.device)
                    logits = self.model(**enc).logits
                    entail, = [int(i) for i,label in self.model.config.id2label.items() if 'entail' in label.lower()]
                    p = logits[:,entail].reshape(-1,2).float().softmax(1)[:,0].cpu().numpy()
                else:  # B2 conserve la règle argmax négatif, distincte du score continu affiché.
                    enc = self.tokenizer([export_clean(s) for s in batch], truncation=True,
                        max_length=self.max_len, padding=True, return_tensors='pt').to(self.device)
                    logits = self.model(**enc).logits
                    neg, = [int(i) for i,label in self.model.config.id2label.items() if 'neg' in label.lower()]
                    p = logits.float().softmax(-1)[:,neg].cpu().numpy()
                    decision = (logits.argmax(-1)==neg).cpu().numpy()
                values.extend(np.asarray(p).tolist())
                labels.extend((np.asarray(p)>=threshold if decision is None else decision).tolist())
        return pd.DataFrame({'phrase':texts, 'modele':rec['key'], 'score':values,
                             'nature_score':'P(sentiment négatif)' if kind=='sentiment' else 'score de sarcasme',
                             'seuil_val':np.nan if kind=='sentiment' else threshold,
                             'regle':'argmax sentiment = négatif' if kind=='sentiment' else 'score >= seuil_val',
                             'prediction':['Sarcastique' if b else 'Non sarcastique' for b in labels]})

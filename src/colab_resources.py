"""Préparer les ressources publiques épinglées du TP, sans charger de modèle."""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json
import urllib.request


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def download_verified(url,target,expected):
    target=Path(target); target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if digest(target)!=expected: raise ValueError(f'SHA256 incorrect : {target}')
        return target
    partial=target.with_name(target.name+'.part')
    print('Téléchargement officiel :',target.name,flush=True)
    urllib.request.urlretrieve(url,partial)
    if digest(partial)!=expected: raise ValueError(f'SHA256 incorrect : {partial}')
    partial.replace(target)
    return target


def prepare_data(root):
    root=Path(root).resolve()
    for folder in ['data/raw','data/processed','models','results','predictions','figures','src','logs']:
        (root/folder).mkdir(parents=True,exist_ok=True)
    records=json.loads((root/'data/raw/manifest.json').read_text(encoding='utf-8'))
    for record in records:
        target=(root/record['path']).resolve()
        if not target.is_relative_to(root/'data/raw'): raise ValueError(record['path'])
        download_verified(record['url'],target,record['sha256'])
    print('Données officielles disponibles dans',root/'data/raw',flush=True)


def prepare_models(root):
    from huggingface_hub import snapshot_download
    root=Path(root).resolve()
    revisions=json.loads((root/'models/revisions.json').read_text(encoding='utf-8'))
    for name,record in revisions.items():
        folder=root/'models/hf'/name.replace('/','--')
        if not folder.resolve().is_relative_to(root/'models/hf'): raise ValueError(name)
        filename='pytorch_model.bin' if name in ['UBC-NLP/MARBERTv2','CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment'] else 'model.safetensors'
        weight=folder/filename; stamp=folder/'tp2_download_verified.json'
        tokenizer_files=['spm.model','tokenizer.json','added_tokens.json'] if name.startswith('MoritzLaurer/') else ['vocab.txt']
        required=[folder/p for p in ['config.json','tokenizer_config.json','special_tokens_map.json',*tokenizer_files]]+[weight]
        try: saved=json.loads(stamp.read_text()) if stamp.exists() else {}
        except (OSError,ValueError): saved={}
        if all(p.is_file() for p in required) and saved.get('revision')==record['revision'] and saved.get('sha256')==record['sha256']:
            if saved.get('bytes')==weight.stat().st_size and saved.get('mtime_ns')==weight.stat().st_mtime_ns: continue
        if not all(p.is_file() for p in required):
            print('Préparation du PLM :',name,'révision',record['revision'],flush=True)
            snapshot_download(repo_id=name,revision=record['revision'],local_dir=folder,token=False,max_workers=2,
                allow_patterns=['config.json','tokenizer_config.json','special_tokens_map.json','tokenizer.json',
                                'vocab.txt','spm.model','added_tokens.json',filename])
        if digest(weight)!=record['sha256']: raise ValueError(f'Poids non conformes : {name}')
        draft=stamp.with_suffix('.json.tmp')
        draft.write_text(json.dumps({'revision':record['revision'],'sha256':record['sha256'],
            'bytes':weight.stat().st_size,'mtime_ns':weight.stat().st_mtime_ns}),encoding='utf-8')
        draft.replace(stamp)
    (root/'models/assets_complete.json').write_text(json.dumps({'models':list(revisions)}),encoding='utf-8')
    print('Les quatre PLM sont disponibles et vérifiés, sans chargement en mémoire.',flush=True)


def prepare_aravec(root):
    root=Path(root).resolve(); target=root/'models/aravec'; target.mkdir(exist_ok=True)
    for model in target.rglob('*.mdl'):
        if all(Path(str(model)+suffix).is_file() for suffix in ['.wv.vectors.npy','.trainables.syn1neg.npy']): return
    record=json.loads((root/'models/aravec_manifest.json').read_text(encoding='utf-8'))[0]
    archive_path=download_verified(record['url'],root/'models/full_uni_cbow_100_twitter.zip',record['sha256'])
    with ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if not (target/name).resolve().is_relative_to(target): raise ValueError(name)
        archive.extractall(target)
    print('AraVec extrait dans',target,flush=True)

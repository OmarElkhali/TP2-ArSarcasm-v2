"""Reprise des gros fichiers officiels par segments HTTP, vérification SHA256 obligatoire."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,hashlib,os,time,threading,shutil
import requests
from huggingface_hub import HfApi,snapshot_download,constants

ROOT=Path(__file__).resolve().parents[2]
NAMES=['aubmindlab/bert-base-arabertv02-twitter','UBC-NLP/MARBERTv2',
       'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7',
       'CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment']
CHUNK=4*1024*1024

def fetch_model(name):
    known=json.loads((ROOT/'models/revisions.json').read_text()) if (ROOT/'models/revisions.json').exists() else {}
    revision=known.get(name,{}).get('revision')
    info=HfApi().model_info(name,revision=revision,files_metadata=True)
    entries={f.rfilename:f for f in info.siblings}
    filename='model.safetensors' if 'model.safetensors' in entries else 'pytorch_model.bin'
    spec=entries[filename]; expected=spec.lfs.sha256; size=spec.size
    folder=ROOT/'models/hf'/name.replace('/','--');folder.mkdir(parents=True,exist_ok=True)
    # Les petits fichiers ne nécessitent pas de téléchargement segmenté.
    snapshot=snapshot_download(name,revision=info.sha,allow_patterns=['config.json','*token*','*vocab*','*.model','special_tokens_map.json'],max_workers=2)
    for p in Path(snapshot).iterdir():
        if p.is_file() and p.name!=filename:shutil.copyfile(p,folder/p.name)
    target=folder/filename; temp=folder/(filename+'.part'); state=folder/'segments.json'
    count=(size+CHUNK-1)//CHUNK
    if target.exists():
        with target.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected
        return name,{'path':str(folder),'revision':info.sha,'sha256':expected}
    if state.exists():
        record=json.loads(state.read_text());assert record['sha256']==expected
        done=set(record['done'])
    else:
        # Réutiliser le préfixe complet acquis par huggingface_hub avant les coupures réseau.
        blob=Path(constants.HF_HUB_CACHE)/('models--'+name.replace('/','--'))/'blobs'/(expected+'.incomplete')
        acquired=blob.stat().st_size if blob.exists() else 0
        if acquired:shutil.copyfile(blob,temp)
        with temp.open('a+b') as f:f.truncate(size)
        done=set(range(acquired//CHUNK))
    lock=threading.Lock()
    def persist():
        draft=state.with_suffix('.tmp');draft.write_text(json.dumps({'sha256':expected,'done':sorted(done)}));draft.replace(state)
    persist()
    print(name,'segments déjà présents',len(done),'/',count,flush=True)
    def segment(i):
        start=i*CHUNK;end=min(size,start+CHUNK)-1
        url=f'https://huggingface.co/{name}/resolve/{info.sha}/{filename}'
        for attempt in range(12):
            try:
                with requests.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=(30,90)) as r:
                    r.raise_for_status()
                    assert r.status_code==206 and r.headers.get('Content-Range')==f'bytes {start}-{end}/{size}',r.headers.get('Content-Range')
                    data=r.content
                    assert len(data)==end-start+1
                with temp.open('r+b') as f:f.seek(start);f.write(data);f.flush()
                with lock:
                    done.add(i);persist()
                    print(name,'segments',len(done),'/',count,flush=True)
                return
            except Exception as error:
                print(name,'segment',i,'tentative',attempt+1,type(error).__name__,str(error)[:160],flush=True)
                if attempt==11:raise
                time.sleep(3)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(segment,i) for i in range(count) if i not in done]):future.result()
    with temp.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
    assert actual==expected,(name,actual,expected)
    temp.replace(target)
    print('SHA256 vérifié',name,flush=True)
    return name,{'path':str(folder),'revision':info.sha,'sha256':expected}

if __name__=='__main__':
    revisions=json.loads((ROOT/'models/revisions.json').read_text()) if (ROOT/'models/revisions.json').exists() else {}
    for name in NAMES:
        key,value=fetch_model(name);revisions[key]=value
        (ROOT/'models/revisions.json').write_text(json.dumps(revisions,indent=2),encoding='utf-8')
    (ROOT/'models/assets_complete.json').write_text(json.dumps({'models':NAMES}),encoding='utf-8')
    print('TOUS LES PLM SONT PRÊTS ET VÉRIFIÉS.',flush=True)

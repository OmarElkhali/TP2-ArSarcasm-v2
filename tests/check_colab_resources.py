"""Préparation des ressources testée sur fichiers jouets, sans réseau ni modèle."""
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile
import ast
import base64
import hashlib
import io
import json
import os
import sys
import shutil
from types import SimpleNamespace
import tempfile
import nbformat
from src.colab_resources import prepare_data,prepare_models,prepare_aravec

ROOT=Path(__file__).resolve().parents[1]


def sha(data): return hashlib.sha256(data).hexdigest()


def main():
    with tempfile.TemporaryDirectory(dir=ROOT/'tmp',prefix='resources_') as directory:
        root=Path(directory); (root/'data/raw').mkdir(parents=True); (root/'models').mkdir()
        fixture=root/'fixture.csv'; fixture.write_bytes(b'tweet,sarcasm\nexample,False\n')
        records=[{'path':'data/raw/'+name,'url':fixture.as_uri(),'sha256':sha(fixture.read_bytes())}
                 for name in ['training_data.csv','testing_data.csv']]
        (root/'data/raw/manifest.json').write_text(json.dumps(records))
        prepare_data(root)
        assert all((root/r['path']).read_bytes()==fixture.read_bytes() for r in records)
        with patch('urllib.request.urlretrieve',side_effect=AssertionError('Fichier valide à réutiliser.')):
            prepare_data(root)
        (root/'data/raw/training_data.csv').write_bytes(b'wrong')
        try: prepare_data(root)
        except ValueError: pass
        else: raise AssertionError('Un CSV modifié devait être refusé.')

        name='test/model'; content=b'synthetic-weights-no-model'; expected_revision='pinned-test-revision'
        revisions={name:{'revision':expected_revision,'sha256':sha(content)}}
        (root/'models/revisions.json').write_text(json.dumps(revisions))
        def snapshot(**kwargs):
            assert kwargs['repo_id']==name and kwargs['revision']==expected_revision and kwargs['token'] is False
            folder=Path(kwargs['local_dir']); folder.mkdir(parents=True,exist_ok=True)
            for filename in kwargs['allow_patterns']:
                (folder/filename).write_bytes(content if filename=='model.safetensors' else b'{}')
        with patch('huggingface_hub.snapshot_download',side_effect=snapshot) as fetch:
            prepare_models(root); assert fetch.call_count==1
            prepare_models(root); assert fetch.call_count==1
        weight=root/'models/hf/test--model/model.safetensors'
        weight.write_bytes(b'corrupt')
        os.utime(weight,ns=(weight.stat().st_atime_ns,weight.stat().st_mtime_ns+1_000_000_000))
        try: prepare_models(root)
        except ValueError: pass
        else: raise AssertionError('Des poids corrompus devaient être refusés.')

        archive=root/'fixture.zip'
        with ZipFile(archive,'w') as z:
            for suffix in ['', '.wv.vectors.npy','.trainables.syn1neg.npy']:
                z.writestr('fixture.mdl'+suffix,b'fixture')
        (root/'models/aravec_manifest.json').write_text(json.dumps([{
            'url':archive.as_uri(),'sha256':sha(archive.read_bytes())}]))
        prepare_aravec(root)
        assert (root/'models/aravec/fixture.mdl.wv.vectors.npy').is_file()
        with patch('urllib.request.urlretrieve',side_effect=AssertionError('AraVec déjà présent.')):
            prepare_aravec(root)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp',prefix='unsafe_archive_') as unsafe:
            other=Path(unsafe); (other/'models').mkdir()
            bad=other/'bad.zip'
            with ZipFile(bad,'w') as z: z.writestr('../escape',b'bad')
            (other/'models/aravec_manifest.json').write_text(json.dumps([{'url':bad.as_uri(),'sha256':sha(bad.read_bytes())}]))
            try: prepare_aravec(other)
            except ValueError: pass
            else: raise AssertionError('Archive hors dossier cible acceptée.')

    # Vérifier l'archive réellement embarquée dans la cellule, sans l'exécuter.
    nb=nbformat.read(ROOT/'notebooks/TP2_Sarcasme_Arabe_T4.ipynb',4)
    layout=next(c.source for c in nb.cells if c.cell_type=='code' and '# TP2_LAYOUT' in c.source)
    assignment=next(n for n in ast.walk(ast.parse(layout)) if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='BOOTSTRAP_ARCHIVE' for t in n.targets))
    payload=base64.b64decode(ast.literal_eval(assignment.value))
    assert sha(payload)==nb.metadata.tp2.colab_bootstrap_sha256
    with ZipFile(io.BytesIO(payload)) as z:
        for name in z.namelist():
            assert z.read(name)==(ROOT/name).read_bytes(),name
            assert Path(name).suffix not in {'.csv','.pt','.safetensors','.bin'}
    # Exécuter le bloc des chemins dans un dossier vide. Les URL sont simulées
    # par la copie des deux CSV locaux déjà vérifiés, sans réseau ni labels lus.
    with tempfile.TemporaryDirectory(dir=ROOT/'tmp',prefix='empty_colab_') as directory:
        empty=Path(directory)
        def fake_download(url,destination):
            filename=url.rsplit('/',1)[-1]
            assert filename in ['training_data.csv','testing_data.csv']
            shutil.copy2(ROOT/'data/raw'/filename,destination)
        scope={'Path':Path,'os':SimpleNamespace(environ={'TP2_ROOT':str(empty)}),
               'IN_COLAB':True,'sys':sys,'hashlib':hashlib}
        with patch('urllib.request.urlretrieve',side_effect=fake_download) as retrieve:
            exec(compile(layout.split('# TP2_LAYOUT')[1],'empty-colab-paths-only','exec'),scope)
            assert retrieve.call_count==2
        assert scope['ROOT']==empty and scope['DATA_DIR']==empty/'data/raw'
        assert (empty/'src/preprocess.py').read_bytes()==(ROOT/'src/preprocess.py').read_bytes()
        assert not (empty/'models/assets_complete.json').exists(), 'Les poids ne sont pas téléchargés dans le bloc des chemins.'
    result={'passed':True,'csv_download_and_hash_verified_with_local_fixture':True,
        'hf_revision_pinned_and_token_disabled_verified_with_mock':True,'verified_download_reused':True,
        'corrupt_csv_and_weights_rejected':True,'aravec_extraction_and_reuse_verified':True,
        'archive_traversal_rejected':True,'embedded_bootstrap_matches_repository':True,
        'empty_colab_path_block_verified_with_simulated_downloads':True,
        'network_calls':0,'pretrained_models_loaded':0,'training_started':False,'notebook_executed':False}
    (ROOT/'results/colab_resources_tests.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()

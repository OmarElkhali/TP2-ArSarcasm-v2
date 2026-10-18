"""Téléchargements publics officiels, sans entraînement et sans lecture des labels test."""
from pathlib import Path
import hashlib
import json
import os
import urllib.request
import subprocess
import sys

os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
os.environ['HF_HUB_ETAG_TIMEOUT'] = '60'
ROOT = Path(__file__).resolve().parents[2]
for folder in ['data/raw', 'data/processed', 'models', 'results', 'predictions', 'figures', 'notebooks', 'src', 'tests', 'logs']:
    (ROOT / folder).mkdir(parents=True, exist_ok=True)

def download(url, dest):
    dest = ROOT / dest
    if not dest.exists():
        print('Téléchargement', url, flush=True)
        temp = dest.with_suffix(dest.suffix + '.part')
        urllib.request.urlretrieve(url, temp)
        temp.replace(dest)
    return {'url': url, 'path': str(dest.relative_to(ROOT)), 'bytes': dest.stat().st_size,
            'sha256': hashlib.file_digest(dest.open('rb'), 'sha256').hexdigest()}

if __name__ == '__main__':
    base = 'https://raw.githubusercontent.com/iabufarha/ArSarcasm-v2/main/ArSarcasm-v2/'
    manifest = [download(base + f, 'data/raw/' + f) for f in ['training_data.csv', 'testing_data.csv']]
    (ROOT / 'data/raw/manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    subprocess.run([sys.executable,'-m','src.assets.models'],cwd=ROOT,check=True)

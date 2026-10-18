"""AraVec Twitter CBOW unigramme 100 dimensions, source officielle."""
from pathlib import Path
from zipfile import ZipFile
import json
from .prepare import download,ROOT

if __name__=='__main__':
    record=download('https://bakrianoo.ewr1.vultrobjects.com/aravec/full_uni_cbow_100_twitter.zip','models/full_uni_cbow_100_twitter.zip')
    target=(ROOT/'models/aravec').resolve(); target.mkdir(exist_ok=True)
    with ZipFile(ROOT/record['path']) as archive:
        for name in archive.namelist():
            if not (target/name).resolve().is_relative_to(target): raise ValueError(name)
        if not list(target.glob('*.mdl')): archive.extractall(target)
    utility=download('https://raw.githubusercontent.com/bakrianoo/aravec/master/utilities.py','src/aravec_utilities.py')
    (ROOT/'models/aravec_manifest.json').write_text(json.dumps([record,utility],indent=2),encoding='utf-8')
    print('AraVec prêt.',record['sha256'])

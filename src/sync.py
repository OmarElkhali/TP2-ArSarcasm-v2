"""Extrait les fonctions du notebook pour les tests et le dépôt demandé par le TP."""
from pathlib import Path
import ast
import nbformat

ROOT = Path(__file__).resolve().parents[1]

def main():
    notebook = nbformat.read(ROOT/'notebooks/TP2_Sarcasme_Arabe_T4.ipynb', 4)
    source = '\n\n'.join(c.source for c in notebook.cells if c.cell_type == 'code')
    tree = ast.parse(source)
    nodes = {n.name: ast.get_source_segment(source, n) for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    header = '# Généré par python -m src.sync depuis le notebook ; modifier le notebook.\n'
    prelude = '''import re, html, unicodedata
import emoji
DIAC = re.compile(r'[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]')
URL_RE = re.compile(r'https?://\S+|www\.\S+', re.I)
MENTION_RE = re.compile(r'@[\w_]+')
ARABIC_MAP = str.maketrans({'أ':'ا','إ':'ا','آ':'ا','ٱ':'ا','ى':'ي'})
'''
    (ROOT/'src/preprocess.py').write_text(header+prelude+'\n\n'.join(
        nodes[n] for n in ['export_clean','duplicate_key','prepare_text','erase_cues'])+'\n', encoding='utf-8')
    evaluation = 'import numpy as np\nfrom sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, precision_recall_curve\n\n'
    (ROOT/'src/evaluate.py').write_text(header+evaluation+'\n\n'.join(
        nodes[n] for n in ['scores','best_threshold'])+'\n', encoding='utf-8')
    models = "import math\nfrom pathlib import Path\nimport torch\nfrom torch import nn\nfrom transformers import AutoModel\nMAX_LEN=64\nMODEL_ROOT=Path(__file__).resolve().parents[1]/'models/hf'\n\n"
    models += nodes['AttentionPool']+'\n\n'+nodes['StaticNet'].replace(
        "def __init__(self,arch='attention',embedding=None,position=True,freeze=False):",
        "def __init__(self,vocab_size,arch='attention',embedding=None,position=True,freeze=False):").replace(
        'nn.Embedding(len(VOCAB),100', 'nn.Embedding(vocab_size,100')
    models += '\n\n'+nodes['FrozenHead']+'\n\n'+nodes['FineTuneNet'].replace(
        'load_hf(name)', "AutoModel.from_pretrained(str(MODEL_ROOT/name.replace('/','--')),local_files_only=True)")+'\n'
    (ROOT/'src/models.py').write_text(header+models, encoding='utf-8')
    training = 'import numpy as np\nimport torch\nfrom torch import nn\nDEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")\n\n'
    (ROOT/'src/train.py').write_text(header+training+nodes['optimizer_step']+'\n\n'+nodes['train_epoch']+'\n', encoding='utf-8')
    print('Modules src synchronisés. Aucune cellule exécutée.')

if __name__ == '__main__':
    main()

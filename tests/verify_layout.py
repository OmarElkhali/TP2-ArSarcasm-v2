"""Contrôler la structure officielle du TP sans exécuter de cellule ou de modèle."""
from pathlib import Path
import ast
import hashlib
import io
import json
import os
import tempfile
import nbformat
from pyflakes.api import check
from pyflakes.reporter import Reporter

ROOT = Path(__file__).resolve().parents[1]


def main():
    for name in ['src', 'notebooks', 'tests', 'results', 'predictions', 'models',
                 'data/raw', 'data/processed', 'figures', '.git']:
        assert (ROOT / name).is_dir(), name
    for name in ['data', 'preprocess', 'models', 'train', 'evaluate']:
        assert (ROOT / 'src' / (name + '.py')).is_file(), name
    assert not (ROOT / 'checkpoints').exists()
    assert not (ROOT / 'TP2_Sarcasme_Arabe.ipynb').exists()
    notebook = nbformat.read(ROOT / 'notebooks/TP2_Sarcasme_Arabe_T4.ipynb', 4)
    nbformat.validate(notebook)
    code = [c for c in notebook.cells if c.cell_type == 'code']
    assert len(code) == 104
    for c in code: compile(c.source, 'notebook-static-check', 'exec')
    source = '\n\n'.join(c.source for c in code)
    stream = io.StringIO()
    check(source, 'notebook', Reporter(stream, stream))
    assert 'undefined name' not in stream.getvalue(), stream.getvalue()
    notes = [c for c in notebook.cells if c.cell_type == 'markdown'
             and c.source.startswith('### Cellule de code ')]
    assert len(notes) == 104
    assert notebook.metadata.tp2.layout == 'tp_repository'
    layout = next(c.source.split('# TP2_LAYOUT')[1] for c in code if '# TP2_LAYOUT' in c.source)

    def check_paths(root):
        for cwd in [root, root / 'notebooks']:
            proxy = type('LocalPath', (), {'cwd': staticmethod(lambda: cwd),
                                          '__new__': lambda cls,*args: Path(*args)})
            namespace = {'Path': proxy, 'os': type('Environment', (), {'environ': {}}),'IN_COLAB':False}
            exec(compile(layout, 'paths-only', 'exec'), namespace)
            assert namespace['ROOT'] == root
            assert namespace['DATA_DIR'] == root / 'data/raw'

    check_paths(ROOT)
    with tempfile.TemporaryDirectory(dir=ROOT / 'tmp', prefix='layout_') as directory:
        relocated = Path(directory)
        for name in ['data/raw', 'models', 'notebooks']: (relocated / name).mkdir(parents=True)
        for name in ['data/raw/training_data.csv', 'data/raw/testing_data.csv', 'models/assets_complete.json']:
            (relocated / name).touch()
        check_paths(relocated)
    for path in [ROOT / 'README.md', ROOT / 'notebooks/TP2_Sarcasme_Arabe_T4.ipynb',
                 *list((ROOT / 'src').rglob('*.py'))]:
        text = path.read_text(encoding='utf-8')
        assert 'TP2_DEPOT_FINAL_' not in text and 'TP2_ARCHIVE_' not in text, path
        if path.suffix == '.py': compile(text, str(path), 'exec')
    for path in (ROOT / 'models').rglob('*'):
        assert not path.is_symlink() and not getattr(path.stat(), 'st_file_attributes', 0) & 0x400

    def resource(relative):
        assert not Path(relative).is_absolute(), relative
        path = (ROOT / relative).resolve()
        assert path.is_relative_to(ROOT), relative
        return path

    for entry in json.loads((ROOT / 'data/raw/manifest.json').read_text()):
        assert hashlib.sha256(resource(entry['path']).read_bytes()).hexdigest() == entry['sha256']
    revisions = json.loads((ROOT / 'models/revisions.json').read_text())
    assert len(revisions) == 4
    for entry in revisions.values():
        folder = resource(entry['path'])
        assert (folder / 'config.json').is_file() and (folder / 'tokenizer_config.json').is_file()
        assert (folder / 'model.safetensors').is_file() or (folder / 'pytorch_model.bin').is_file()
    for entry in json.loads((ROOT / 'results/registry.json').read_text(encoding='utf-8')).values():
        assert resource(entry['path']).exists()
    cache_count = 0
    for path in (ROOT / 'predictions').glob('*_val.validation.json'):
        metadata = json.loads(path.read_text())
        checkpoint = ROOT / 'models' / (path.name.removesuffix('_val.validation.json') + '.pt')
        saved = json.loads(checkpoint.with_suffix('.json').read_text())
        stat = {'bytes': checkpoint.stat().st_size, 'mtime_ns': checkpoint.stat().st_mtime_ns}
        assert stat == saved['checkpoint_stat'] == metadata['checkpoint_stat']
        assert metadata['signature'] == saved['signature']
        assert metadata['context'] == saved['runtime_context']
        scores = path.with_name(path.name.removesuffix('.validation.json') + '.npy')
        assert hashlib.sha256(scores.read_bytes()).hexdigest() == metadata['probability_sha256']
        cache_count += 1
    tree = ast.parse(source)
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    assert {'StaticNet', 'FrozenHead', 'FineTuneNet', 'AttentionPool'} <= classes.keys()
    audit = {'layout': 'tp_repository', 'required_folders_and_modules': True,
             'notebook': 'notebooks/TP2_Sarcasme_Arabe_T4.ipynb', 'code_cells': len(code),
             'detailed_explanations': len(notes), 'paths_root_and_notebooks_checked': True,
             'relocated_paths_checked': True, 'all_resources_in_repository': True,
             'dataset_sha256_verified': True, 'historical_validation_caches_checked': cache_count,
             'notebook_executed_by_check': False, 'pretrained_models_loaded': 0,
             'optimizer_steps': 0, 'backward_calls': 0}
    (ROOT / 'results/layout_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()

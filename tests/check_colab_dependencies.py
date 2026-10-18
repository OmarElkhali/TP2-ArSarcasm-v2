"""Vérifier l'amorçage Colab avec pip simulé, sans réseau ni installation."""
from pathlib import Path
from unittest.mock import patch
from importlib.metadata import PackageNotFoundError
import ast
import contextlib
import io
import json
import sys
from types import SimpleNamespace
import nbformat

ROOT=Path(__file__).resolve().parents[1]


def main():
    nb=nbformat.read(ROOT/'notebooks/TP2_Sarcasme_Arabe_T4.ipynb',4)
    cell=next(c for c in nb.cells if c.cell_type=='code')
    assert 'TP2_COLAB_BOOTSTRAP_V2' in cell.source, 'La première cellule a été remplacée par une ancienne version.'
    node=next(n for n in ast.parse(cell.source).body if isinstance(n,ast.If)
              and isinstance(n.test,ast.Name) and n.test.id=='IN_COLAB')
    definition=ast.Module(body=[node],type_ignores=[])
    requirements=[line.strip() for line in (ROOT/'requirements-colab.txt').read_text(encoding='utf-8').splitlines()
                  if line.strip() and not line.lstrip().startswith('#')]
    expected=dict(r.split('==',1) for r in requirements)
    assert not {'torch','numpy','pandas','scikit-learn','scipy'} & expected.keys()
    def run(in_colab,absent=(),outdated=(),returncode=0):
        scope={'IN_COLAB':in_colab,'sys':sys}
        def installed(package):
            if package in absent: raise PackageNotFoundError(package)
            return '0.0' if package in outdated else expected[package]
        stopped=False
        output=io.StringIO()
        with patch('importlib.metadata.version',side_effect=installed), contextlib.redirect_stdout(output):
            with patch('subprocess.run',return_value=SimpleNamespace(
                    returncode=returncode,stdout='ERROR: simulated pip failure' if returncode else 'simulated success')) as pip:
                try: exec(compile(definition,'dependency-bootstrap-only','exec'),scope)
                except RuntimeError as e:
                    if returncode:
                        assert f'code {returncode}' in str(e) and 'ERROR' in str(e)
                        assert 'Installation terminée' not in str(e)
                    else:
                        assert 'Installation terminée' in str(e) and 'redémarre' in str(e)
                    stopped=True
                command=pip.call_args.args[0] if pip.called else None
                if pip.called:
                    assert pip.call_args.kwargs['text'] is True
                    assert pip.call_args.kwargs['stderr'] == -2
        if command and returncode: assert 'ERROR: simulated pip failure' in output.getvalue()
        return scope,command,stopped
    scope,command,stopped=run(False)
    assert command is None and not stopped and 'COLAB_REQUIREMENTS' not in scope
    scope,command,stopped=run(True)
    assert scope['COLAB_REQUIREMENTS']==requirements and command is None and not stopped
    _,command,stopped=run(True,absent=('emoji','arabert'))
    assert stopped and command[:4]==[sys.executable,'-m','pip','install']
    assert command[-2:]==['arabert==1.0.1','emoji==1.4.2']
    assert '--only-binary=safetensors,tokenizers,gensim,sentencepiece,pymupdf' in command
    _,command,stopped=run(True,outdated=('transformers',))
    assert stopped and command[-1]=='transformers==4.48.0'
    _,command,stopped=run(True,absent=('sentencepiece',),returncode=1)
    assert stopped and command[-1]=='sentencepiece==0.2.1'
    assert expected['safetensors']=='0.5.3' and expected['sentencepiece']=='0.2.1'
    result={'passed':True,'pip_calls_simulated_only':True,'network_calls':0,'packages_installed':0,
            'local_conda_unchanged':True,'already_ready_no_install':True,
            'missing_emoji_and_arabert_detected':True,'restart_required_after_install':True,
            'bootstrap_matches_requirements_file':True,'torch_not_requested':True,
            'pip_failure_output_visible':True,'failure_not_reported_as_success':True,
            'native_extensions_require_wheels':True,
            'notebook_executed':False,'training_started':False}
    (ROOT/'results/colab_dependency_tests.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()

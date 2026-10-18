"""Vérifier la provenance du compte rendu, sans modèle ni entraînement."""
from pathlib import Path
import hashlib
import json
import nbformat
import pandas as pd
import pymupdf

ROOT=Path(__file__).resolve().parents[1]


def main():
    evidence=ROOT/'results/report_data'
    manifest=json.loads((evidence/'provenance.json').read_text(encoding='utf-8'))
    bookpath=ROOT/manifest['execution_notebook']
    assert hashlib.sha256(bookpath.read_bytes()).hexdigest()==manifest['execution_sha256']
    code=[c for c in nbformat.read(bookpath,4).cells if c.cell_type=='code']
    assert len(code)==104 and all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.outputs)
    final=pd.read_csv(evidence/'cell_085_table_1.csv')
    assert len(final)==17
    for name,group in final.groupby('model'):
        assert len(group)==(1 if name.startswith('B') else 3)
    rows=final[final.model.eq('E2_AraBERT_lambda0p3')]
    assert abs(rows.test_f1_sarcastique.mean()-.6096)<.00005
    assert abs(rows.test_f1_sarcastique.std()-.0131)<.00005
    assert len(pd.read_csv(evidence/'cell_076_table_1.csv'))==16
    assert len(pd.read_csv(evidence/'cell_093_table_1.csv'))==15
    assert len(pd.read_csv(evidence/'cell_097_table_1.csv'))==30
    phrases=pd.read_csv(evidence/'phrases_capture.csv')
    assert len(phrases)==4 and (phrases.score_capture<phrases.seuil_capture).all()
    with pymupdf.open(ROOT/'Compte_Rendu_TP2.pdf') as d:
        assert len(d)==10
        assert d.metadata['author']=='Omar Elkhali'
        body='\n'.join(d[i].get_text() for i in range(6))
        assert all('Q'+str(q) in body for q in range(1,13))
        assert 'BDIOT' in body and 'Travail individuel' in body
        assert '0,015058' in body and '17 CSV' in body
        assert 'Annexe A' in d[6].get_text()
        assert any(link.get('uri')=='https://github.com/OmarElkhali/TP2-ArSarcasm-v2' for link in d[0].get_links())
        for page in d:
            for block in page.get_text('dict')['blocks']:
                x0,y0,x1,y1=block['bbox']
                assert x0>=0 and y0>=0 and x1<=page.rect.width+1 and y1<=page.rect.height+1
    print('Rapport : 104 cellules source, 17 runs test, 16 runs E, 15 IC, 30 erreurs, 4 phrases ; PDF 6 + 4 pages vérifié. Aucun entraînement.')


if __name__=='__main__': main()

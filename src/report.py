"""Compte rendu des sorties sauvegardées : aucune exécution scientifique."""
from pathlib import Path
import argparse
import base64
import hashlib
import html
import io
import json
import shutil
import nbformat
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak, Image

ROOT = Path(__file__).resolve().parents[1]
EXECUTION = ROOT/'notebooks/TP2_Sarcasme_Arabe_T4_Execution.ipynb'
EVIDENCE = ROOT/'results/report_data'


def collect_evidence():
    """Exporter intégralement les tables HTML et figures réellement enregistrées."""
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    book = nbformat.read(EXECUTION, 4)
    code = [c for c in book.cells if c.cell_type == 'code']
    assert len(code) == 104 and all(c.execution_count is not None for c in code)
    assert not any(o.output_type == 'error' for c in code for o in c.outputs)
    tables, streams = {}, []
    for number, cell in enumerate(code, 1):
        frames = []
        for index, output in enumerate(cell.outputs):
            data = output.get('data', {})
            markup = data.get('text/html', '')
            if '<table' in markup:
                for frame in pd.read_html(io.StringIO(markup), flavor='lxml'):
                    frames.append(frame)
                    frame.to_csv(EVIDENCE/f'cell_{number:03d}_table_{len(frames)}.csv', index=False)
            if 'image/png' in data:
                (EVIDENCE/f'cell_{number:03d}_image_{index}.png').write_bytes(base64.b64decode(data['image/png']))
            if output.output_type == 'stream': streams.append(f'CELL {number}\n{output.text}')
        tables[number] = frames
    (EVIDENCE/'execution_streams.txt').write_text('\n'.join(streams), encoding='utf-8')
    heatmaps = [o.data['text/html'] for o in code[66].outputs if 'text/html' in o.get('data', {})]
    (EVIDENCE/'attention_5_tweets.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Attention D2</title>'
        '<h1>Attention D2 - cinq tweets de validation, sortie T4</h1><p>Poids internes, sans interprétation causale.</p>'
        + '<hr>'.join(heatmaps), encoding='utf-8')
    phrases = pd.DataFrame([
        ['يا سلام، الإنترنت انقطع مرة أخرى، خدمة رائعة فعلاً 😂', 'Ironie attendue', .002247],
        ['الجو جميل اليوم وسأذهب للمشي.', 'Littéral attendu', .001082],
        ['شكراً على التأخير ساعتين، فعلاً تحترمون وقت الناس!', 'Ironie attendue', .015058],
        ['لم يعجبني هذا الفيلم.', 'Littéral attendu', .017326],
    ], columns=['phrase', 'interpretation_informelle', 'score_capture'])
    phrases['modele'] = 'E2_AraBERT_lambda0p5_seed42'
    phrases['seuil_capture'] = .579043
    phrases['prediction_capture'] = 'Non sarcastique'
    phrases['source'] = 'Transcription de la capture fournie par Omar ; sorties non enregistrées dans le fichier local du testeur'
    phrases.to_csv(EVIDENCE/'phrases_capture.csv', index=False, encoding='utf-8-sig')
    manifest = {
        'author':'Omar Elkhali', 'programme':'BDIOT', 'work':'individuel',
        'execution_notebook':EXECUTION.relative_to(ROOT).as_posix(),
        'execution_sha256':hashlib.sha256(EXECUTION.read_bytes()).hexdigest(),
        'executed_code_cells':len(code), 'errors':0,
        'plm_max_epochs_measured':3, 'plm_max_epochs_current_code':5,
        'tables_exported':sum(map(len,tables.values())),
        'phrase_evidence':'user screenshot; different lambda and threshold from final T4 selection',
        'test_prediction_csv_present_locally':len(list((ROOT/'predictions').glob('*.csv'))),
        'precision':'HTML tables saved to six decimals; comparison table rounded to four decimals',
    }
    (EVIDENCE/'provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return tables


def main(workdir=None):
    tables=collect_evidence()
    repo=json.loads((ROOT/'results/repository.json').read_text(encoding='utf-8'))['url']
    def frame(n,index=0):
        f=tables[n][index].copy()
        return f.drop(columns=[c for c in f if isinstance(c,str) and c.startswith('Unnamed:')])
    finals,comparison,grid=frame(85),frame(101),frame(76)
    cpu=pd.read_csv(ROOT/'results/results.csv')
    cpu.to_csv(EVIDENCE/'validation_cpu_historique.csv',index=False)
    fontdir=ROOT/'src/fonts'
    pdfmetrics.registerFont(TTFont('TP',str(fontdir/'arial.ttf')))
    pdfmetrics.registerFont(TTFont('TPBold',str(fontdir/'arialbd.ttf')))
    body=ParagraphStyle('body',fontName='TP',fontSize=9,leading=12.1,spaceAfter=6.5)
    heading=ParagraphStyle('heading',parent=body,fontName='TPBold',fontSize=10.7,leading=13.2,spaceBefore=6)
    title=ParagraphStyle('title',parent=heading,fontSize=17,leading=21,spaceAfter=11)
    small=ParagraphStyle('small',parent=body,fontSize=7.5,leading=9.6)
    cell_style=ParagraphStyle('cell',parent=small,fontSize=7.4,leading=9.3,spaceAfter=0)
    width=A4[0]-3.4*cm
    story,markdown=[],['# TP2 - Détection du sarcasme en arabe\n']
    def p(text,style=body):
        markup=html.escape(str(text))
        if str(text).startswith('Dépôt public : '):
            markup='Dépôt public : <link href="'+html.escape(repo,quote=True)+'" color="#24587c">'+html.escape(repo)+'</link>'
        story.append(Paragraph(markup,style)); markdown.append(str(text)+'\n')
    def h(text): p(text,heading)
    def page(text):
        if story: story.append(PageBreak())
        p(text,title)
    def table(data,columns=None,widths=None,compact=False):
        f=data.copy() if isinstance(data,pd.DataFrame) else pd.DataFrame(data,columns=columns)
        def fmt(v):
            if isinstance(v,float): return '-' if pd.isna(v) else f'{v:.4f}'
            return str(v)
        rows=[[Paragraph(html.escape(str(v)),cell_style) for v in f.columns]]
        rows += [[Paragraph(html.escape(fmt(v)),cell_style) for v in row] for row in f.itertuples(index=False,name=None)]
        t=Table(rows,colWidths=[width*x/sum(widths) for x in widths] if widths else [width/len(f.columns)]*len(f.columns),repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dfebf3')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7fa')]),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),
            ('TOPPADDING',(0,0),(-1,-1),2 if compact else 4),('BOTTOMPADDING',(0,0),(-1,-1),2 if compact else 4),
            ('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#52758e'))]))
        story.extend([t,Spacer(1,6)])
        markdown.append('```csv\n'+f.to_csv(index=False,lineterminator='\n')+'```\n')
    def figure(filename,maxwidth,maxheight):
        from PIL import Image as PILImage
        path=EVIDENCE/filename
        with PILImage.open(path) as im: w,height=im.size
        scale=min(maxwidth*cm/w,maxheight*cm/height)
        story.append(Image(str(path),width=w*scale,height=height*scale))
        markdown.append(f'![{filename}](report_data/{filename})\n')

    page('TP2 - Détection du sarcasme en arabe')
    p('Travail individuel - Omar Elkhali',heading)
    p('Filière : BDIOT | Module : Transformers & LLMs | Prof. Yassir Matrane')
    p('Compte rendu du 18 octobre 2026 - Corpus ArSarcasm-v2')
    p('Dépôt public : '+repo,small)
    h('Objectif et provenance des résultats')
    p("J'étudie la détection du sarcasme dans des tweets arabes, du TF-IDF au fine-tuning multi-tâche. Ce rapport exploite les sorties de l'exécution T4 archivée (104/104 cellules, aucune erreur enregistrée) et les essais du testeur transmis en capture. Les expériences A à G ont été exécutées dans cette version ; leurs limites et les livrables encore manquants sont explicités page 6.")
    p("L'exécution mesurée utilise trois époques PLM maximum, dans la plage de 3 à 5 du sujet. Le code courant propose cinq époques ; ses nouveaux résultats ne sont pas encore enregistrés et ne sont pas revendiqués ici. Les résultats CPU antérieurs restent séparés en annexe C.")
    h('Q1 - Données, déséquilibre et métrique retenue')
    table([['Train officiel',12548,'2 168 positifs ; 17,28 %'],['Train initial / nettoyé','11 293 / 10 842','1 864 positifs ; 17,19 %'],['Validation',1255,'217 positifs ; 17,29 %'],['Test officiel',3000,'821 positifs ; 27,37 %']],['Partition','Effectif','Sarcasme / remarque'],widths=[1.2,1,2])
    table([['egypt','2 320 ; 35,09 %','260 ; 36,15 %','306 ; 50,65 %'],['gulf','511 ; 22,90 %','68 ; 20,59 %','322 ; 17,70 %'],['levant','549 ; 21,68 %','55 ; 29,09 %','47 ; 29,79 %'],['magreb','35 ; 37,14 %','5 ; 0 %','2 ; 0 %'],['msa','7 427 ; 10,79 %','867 ; 10,73 %','2 323 ; 25,61 %']],['Dialecte','Train nettoyé : n ; taux','Val : n ; taux','Test : n ; taux'],widths=[.7,1.3,1.2,1.3],compact=True)
    p("Le modèle « jamais sarcastique » a une accuracy de 82,71 % sur val et 72,63 % sur test, mais un F1 sarcastique nul. Je retiens le F1 de la classe sarcastique : F1 = 2PR/(P+R). Macro-F1 et average precision (AP, convention PR-AUC utilisée ici) complètent l'analyse.")
    h('Q2 - Audit des fuites et des raccourcis')
    p("Split stratifié 90/10, graine 42, avant ajustement. L'audit trouve 2 collisions train/test normalisées, 72 train/val, 331 lignes dupliquées et 47 groupes contradictoires (96 lignes). Ces ensembles se recouvrent : leur somme n'est pas le nombre de lignes retirées. Seul train est réduit à 10 842 ; val et test sont conservés. Les seuils et modèles sont choisis sur val, avant ouverture des labels test en F.")
    p("Dans train nettoyé : sarcasme avec URL 6,29 %, sans URL 23,90 % ; avec guillemets d'export 11,93 %, sans 25,33 %. Une règle superficielle peut produire un score non nul sans comprendre l'ironie.")

    page('Prétraitement et modèles sans fine-tuning')
    h('Q2 - Valeur des règles simples sur validation')
    rules=frame(29,1)[['indice','f1_sarcastique','precision','recall']]
    rules.columns=['Indice (absence → sarcasme)','F1 val','Précision','Rappel']
    table(rules,widths=[1.8,1,1,1])
    p("Ces règles restent inférieures au TF-IDF, mais révèlent un risque de raccourci de provenance. Les guillemets extérieurs apparaissent dans 60,54 % du train brut contre 0,20 % du texte test ; leur suppression est un nettoyage d'export, pas une suppression générale de la ponctuation.")
    h('Q3 - Nettoyage selon la famille et ablation chiffrée')
    p("A/C : Unicode NFC, décodage HTML, retrait d'une paire de guillemets extérieurs, diacritiques et tatweel, normalisation alif/ya, URL et mentions remplacées par jetons. Je conserve négations, hashtags, emojis, rires, répétitions et ponctuation. AraVec clean_str sert à la recherche du vecteur. AraBERT emploie son préprocesseur officiel avec conservation des répétitions ; MARBERT reçoit un nettoyage minimal.")
    baseline=frame(29)[['model','val_f1_sarcastique','val_pr_auc_ap']]
    baseline.columns=['TF-IDF mots 1-2 + caractères 2-5','F1 val','AP val']
    table(baseline,widths=[2.7,1,1])
    p("Le nettoyage agressif fait passer le F1 de 0,5206 à 0,4959 (-2,47 points). Il détruit des signaux utiles et n'est pas retenu. Le vocabulaire et le TF-IDF sont ajustés sur train uniquement. La régression logistique utilise balanced pour la pondération des classes et un seuil optimisé sur val.")
    h('Q4 - Zero-shot NLI : sensibilité à la formulation')
    nli=frame(38)[['model','val_precision','val_recall','val_f1_sarcastique','val_pr_auc_ap']]
    nli.columns=['Modèle / hypothèse','Précision val','Rappel val','F1 val','AP val']
    table(nli,widths=[2,1,1,1,1])
    p("L'anglais obtient 0,4224, le français 0,4189 et l'arabe 0,4069 sur val : le prompt compte (+1,55 point entre arabe et anglais). B1 anglais atteint 0,4625 sur test, AP 0,4077. Le modèle NLI pré-entraîné compare une prémisse (tweet) à des hypothèses sarcastique/sérieuse sans apprendre sur nos exemples. L'auto-attention relie les tokens des deux segments ; le seuil est choisi sur val.")
    h('Q5 - « Négatif implique sarcastique » : un raccourci limité')
    p("B2 prédit sarcastique si l'argmax sentiment est negative, sans seuil optimisé sur P(negative). Sur val : précision 0,3327, rappel 0,7972, F1 0,4695 ; sur test : F1 0,5320 et AP 0,4928. Dans train, 90,45 % des sarcastiques sont NEG, mais seulement 41,89 % des NEG sont sarcastiques : une plainte sincère peut être négative, une louange ironique paraître positive.")
    p("Contamination B2 possible : CAMeLBERT sentiment a été entraîné sur ASTD/ArSAS/SemEval, et ArSarcasm-v1 réannote notamment ASTD/SemEval. Ce score n'est pas une preuve de généralisation sans recouvrement.")

    page('Réseaux statiques et représentations gelées')
    h('Q6 - Trois traitements du déséquilibre, meilleur C')
    imb=frame(56); imb.columns=['Configuration','F1 seuil 0,5','F1 seuil val','Seuil val']
    table(imb,widths=[2.6,1,1,1])
    p("Perte pondérée : 0,5427 ; sur-échantillonnage : 0,5410 ; perte standard avec seuil optimisé : 0,5347. Le seuil apporte +3,95 points au meilleur C par rapport à 0,5. Les écarts entre méthodes sont faibles à cette graine. Le sur-échantillonnage reste limité au train après split : le faire avant pourrait placer des copies du même tweet en train et val.")
    h('Q7 - Architectures, embeddings et position')
    static=frame(51)[['model','val_f1_sarcastique']].values.tolist()
    static += [['C1_AraVec',.5290],['C2_AraVec',.5362],['C3_AraVec',.542694]]
    table(static,['Architecture (graine 42)','F1 val'],widths=[3,1],compact=True)
    p("C1 : BiLSTM avec max-pooling masqué. C2 : BiLSTM et attention apprise, alpha = softmax(vT tanh(Wh)), puis somme pondérée. C3 : deux couches Transformer, dimension 128, quatre têtes, positions sinusoïdales, masque PAD. Sans positions, C3 aléatoire passe de 0,4627 à 0,4619 : effet faible sur ce run, pas une preuve d'inutilité. C4 char-CNN et focal loss sont optionnels et non exécutés.")
    p("Budget C commun : huit époques maximum, patience 2 ; AraVec Twitter 100D figé deux époques puis dégelé. Le meilleur checkpoint peut précéder le dégel (C3 : époque 1). Vocabulaire : fréquence train ≥ 2, PAD distinct de UNK. OOV tokens train/val/test : 15,19 % / 21,16 % / 27,71 %. AraVec couvre 94,13 % des types du vocabulaire et 71,94 % des tokens train ; les dénominateurs diffèrent.")
    h('Q8 - D1/D2 : extraction gelée et coût réel')
    table([['D1 - CLS + régression logistique',.6054,'4,7 s','769'],['D1 - moyenne masquée + régression',.6316,'7,3 s','769'],['D2 - tokens + BiLSTM / attention',.6625,'12,7 s','435 586']],['Encodeur AraBERT gelé','F1 val 42','Temps tête','Paramètres tête'],widths=[2.8,1,1,1.1])
    p("La moyenne masquée surpasse CLS de 2,62 points ; D2 gagne encore 3,09 points. Ces temps excluent extraction préalable, téléchargements et chargement des poids : ils ne représentent pas le coût total. L'encodeur reste nécessaire à l'inférence ; son cache évite des extractions répétées pendant la comparaison des têtes.")
    p("Les cinq attentions D2 (annexe D) ne portent pas seulement sur les marqueurs : le premier tweet donne environ 0,0964 à un point et 0,0904 à un sous-mot lexical. D'autres maxima concernent des mots de contenu. Une forte pondération n'est ni preuve de compréhension de l'ironie ni explication causale.")

    page('Fine-tuning et comparaison qualité / coût')
    h('Q8 et Q9 - Mono-tâche, sentiment auxiliaire et PLM')
    p("E1 ajuste tout l'encodeur par entropie croisée sarcasme pondérée. E2 partage l'encodeur et utilise deux têtes linéaires : L = Lsarc + lambda Lsent. Grille lambda {0,3 ; 0,5 ; 1} à graine 42, puis répétition du meilleur lambda et de E1 aux graines 43/44 pour chaque PLM. Le sentiment est une cible auxiliaire ; le dialecte n'est pas une entrée du classifieur.")
    means=[]
    for prefix in ['AraBERT','MARBERT']:
        mono=grid[grid.model.eq('E1_'+prefix+'_mono')].val_f1_sarcastique
        name='E2_'+prefix+'_lambda'+('0p3' if prefix=='AraBERT' else '0p5')
        multi=grid[grid.model.eq(name)].val_f1_sarcastique
        means.append([prefix,f'{mono.mean():.4f} ± {mono.std():.4f}',f'{multi.mean():.4f} ± {multi.std():.4f}',name.split('lambda')[1]])
    table(means,['PLM','E1 : F1 val (3 graines)','E2 : F1 val (3 graines)','Lambda retenu'],widths=[1,1.7,1.7,1])
    table([['AraBERT',.673114,.671785,.671845],['MARBERT',.660754,.666667,.662577]],['E2, graine 42','lambda 0,3','lambda 0,5','lambda 1'],widths=[1.6,1,1,1])
    p("AraBERT multi-tâche a le meilleur F1 moyen val (0,6824). Le gain E2/E1 est +0,81 point val mais seulement +0,03 point test moyen (0,6096 contre 0,6093) : aucune amélioration nette sur test n'est démontrée. MARBERT E1 a un F1 val moyen proche (0,6730), son E2 lambda 0,5 est inférieur (0,6703). Le sentiment peut régulariser l'encodeur, mais aussi privilégier la négativité ou créer un transfert négatif.")
    p("Réglages mesurés : AdamW, lr 2e-5, weight decay 0,01, warmup 10 %, batch 32, AMP FP16, trois époques maximum, patience 2, longueur 64. Aucun sous-échantillonnage pour accélérer. Troncature train > 64 : 76 tweets AraBERT (0,70 %) et 35 MARBERT (0,32 %). E1 AraBERT entraîne 134 604 290 paramètres, E2 134 606 597 ; MARBERT environ 162,25 millions.")
    h('G - Comparaison finale, seuils verrouillés sur validation')
    rows=[]
    for r in comparison.itertuples():
        rows.append([r.family,f'{r.test_f1_mean:.4f}'+(f' ± {r.test_f1_std:.4f}' if r.n_seeds==3 else ' (1 run)'),r.test_ap_mean,r.train_seconds_mean,r.cpu_batch1_latency_ms,r.complete_weights_mb])
    table(rows,['Famille','F1 test moyen ± ET','AP test','Train (s)','CPU ms/tweet','Poids (Mo)'],widths=[.7,1.6,.9,.9,1.1,1],compact=True)
    p("A = TF-IDF ; C = C3 AraVec ; D = D2 ; E1/E2 = AraBERT ; B1 = NLI anglais ; B2 = sentiment. A/C/D/E : trois graines ; B1/B2 : un passage sans entraînement sur le corpus. A est déterministe : ET nul attendu. Latences : CPU Colab, batch 1, médiane de trois passages après chauffe, vingt tweets. Elles ne mesurent pas le Ryzen local ni une latence de service en charge.",small)
    p("D2 → E1 : +1,60 point de F1 test moyen, environ 10,35 fois le temps d'ajustement de tête et 310 fois plus de paramètres entraînés. Les poids complets restent proches (environ 541 Mo), car D2 garde AraBERT. Ces temps excluent préparation et extractions ; ils ne permettent pas d'annoncer une durée totale.")

    page('Généralisation, robustesse et incertitude')
    h('Q10 - Décalage du prior et diagnostic des seuils')
    p("Le prior passe de 17,29 % val à 27,37 % test (+10,08 points). Ce changement et celui de composition affectent précision, rappel et F1. Le seuil déployable est celui de val ; les seuils test sont seulement des diagnostics a posteriori.")
    shift=frame(87); shift.columns=['Famille','Seuil val','Seuil test posthoc','F1 seuil val','F1 posthoc']
    table(shift,widths=[.8,1,1.4,1,1],compact=True)
    p("Pour E2, le seuil descend de 0,4908 à 0,2983 et le F1 diagnostic gagne 1,41 point ; ce n'est pas un réglage autorisé sur test. En production : suivi du prior, des dialectes et des scores ; annotation d'un nouveau lot temporel pour recalibrer ; contrôle sur un second lot indépendant.")
    h('Q11 - Dialecte non vu, URL et suppression des indices')
    p("Sans egypt, train passe à 8 522 exemples. Sur les 306 tweets egypt test, E2 passe de 0,7736 à 0,7674 : chute de 0,0063 (0,63 point). Le pré-entraînement peut déjà contenir de l'égyptien et l'architecture principale a été sélectionnée avec ce dialecte : cette faible chute ne démontre pas une indépendance au dialecte.")
    robustness,cues,groups=frame(92),frame(88,1),frame(88)
    rows=[]
    for fam in ['A','C','D','E1','E2']:
        group=groups[groups.famille.eq(fam)&groups.URL.notna()]
        rob=robustness[robustness.diagnostic.eq(fam+'_sans_url_mention')].iloc[0]
        cue=cues[cues.famille.eq(fam)].iloc[0]
        rows.append([fam,group.iloc[0].f1_sarcastique,group.iloc[1].f1_sarcastique,rob.f1_sarcastique,f'{cue.chute*100:.2f} points'])
    table(rows,['Famille','F1 sans URL','F1 avec URL','Réentraîné sans URL/mention','Chute détection sans indices'],widths=[.65,1,1,1.7,1.6],compact=True)
    p("Avec URL : 16 positifs sur 440 tweets ; sans URL : 805/2 560. Les F1 ne sont pas comparables à prior constant. Supprimer URL/mention à l'entraînement n'améliore aucune famille ici. F4 retire emojis, rires et ponctuation répétée aux vrais positifs initiaux, à seuil fixé : E2 garde 95,38 % des 584 détectés, contre 90,63 % des 480 de A. Ce test mesure une sensibilité, pas toute la compréhension.")
    p("Magreb : 35 tweets train et seulement 2 test, tous deux négatifs. Aucun F1 positif ni AP fiable ne peut soutenir un déploiement marocain. Il faut un test indépendant en darija, avec Arabizi, mélange français/arabe, nouveaux sujets et annotation relue.")
    h('Q12 - Incertitude et trente erreurs')
    p("Bootstrap, 1 000 rééchantillonnages, IC 95 % à graine 42 : A [0,5093 ; 0,5660], C [0,5172 ; 0,5729], D [0,5689 ; 0,6228], E1 [0,5997 ; 0,6487], E2 [0,5982 ; 0,6496]. McNemar exact E2/D : 151 erreurs gagnées, 156 perdues, p = 0,8195. La supériorité n'est pas significative selon ce test des erreurs appariées, qui ne teste pas directement le F1.")
    p("15 FN et 15 FP : catégories automatiques provisoires contexte/ironie implicite 19/30, dialecte-contexte 6/30, marqueur possiblement mal interprété 5/30. Aucun des trente n'a de clé contradictoire repérée ; cela ne démontre ni absence ni taux nul de bruit d'annotation. Une relecture contextuelle reste nécessaire : ces catégories ne sont pas une annotation humaine finalisée.")

    page('Essais sur phrases et conclusion personnelle')
    h('Du notebook scientifique à la saisie de phrases réelles')
    p("J'ai réalisé les expériences A à G prévues dans l'exécution T4 sauvegardée, avec les architectures demandées et les trois graines des modèles finaux. Cependant, lorsque j'ai saisi des phrases proches d'usages réels, je n'ai pas obtenu de bons résultats sur les exemples ironiques. Le respect d'un protocole et des scores positifs sur ArSarcasm-v2 ne garantissent donc pas une détection fiable en dehors de ce corpus.")
    p("Le testeur recharge checkpoint, prétraitement et seuil, puis utilise eval() et inference_mode(). La capture correspond à E2 AraBERT lambda 0,5, graine 42, seuil 0,579043. Ce n'est pas le checkpoint final lambda 0,3 au seuil 0,490785 de l'exécution archivée : ces évaluations ne doivent pas être fusionnées.")
    table([
        ['Internet encore coupé, « service vraiment excellent » + emoji rieur','Ironie','0,002247','Non sarcastique : manqué'],
        ["Il fait beau, je vais me promener.",'Littéral','0,001082','Non sarcastique : cohérent'],
        ['Merci des deux heures de retard, « vous respectez vraiment les gens »','Ironie','0,015058','Non sarcastique : manqué'],
        ["Je n'ai pas aimé ce film.",'Critique littérale','0,017326','Non sarcastique : cohérent'],
    ],['Sens des phrases arabes saisies','Lecture attendue','Score','Sortie observée'],widths=[2.6,1.1,.9,1.6])
    p("Le passage interactif de la phrase sur le retard donne 0,015073, toujours sous le seuil. Les deux exemples ironiques sont manqués ; les deux phrases littérales ont une sortie cohérente. Ces quatre phrases choisies à la main, sans annotation indépendante, ne permettent pas de publier un nouveau F1 ou une accuracy généralisable. Les originaux arabes et valeurs transcrites sont dans results/report_data/phrases_capture.csv.")
    h('Interprétation et décision de déploiement - Q12')
    p("Les exemples ironiques opposent une situation négative à une louange. Le modèle peut surtout exploiter des régularités lexicales, le sentiment et les indices du corpus sans résoudre ce contraste. Thèmes, formulations et dialectes changent ; le contexte manque. Un score n'est pas une garantie de probabilité calibrée. Abaisser le seuil pour réussir ces phrases reviendrait à ajuster le modèle sur la démonstration.")
    p("Je retiens E2 AraBERT lambda 0,3 comme candidat expérimental, choisi par le F1 moyen val : test 0,6096 ± 0,0131, environ 116,63 ms/tweet CPU Colab et 540,88 Mo. Le checkpoint représentatif graine 42 obtient précision 0,5557, rappel 0,7113, F1 0,6239, macro-F1 0,7267, AP 0,6355. Pour une contrainte CPU stricte, TF-IDF est un compromis : F1 0,5369, 2,15 ms, 5,84 Mo. Je ne validerais pas encore un déploiement automatique en darija ; je commencerais par un pilote avec contrôle humain et abstention, puis un test externe annoté.")
    h('Reproductibilité et limites des livrables')
    p("Le dépôt contient le notebook T4 exécuté, tous ses tableaux, le testeur, le code, les tests et ce rapport. Les 17 CSV individuels de prédictions test annoncés par Colab ne sont pas présents localement : il reste à les récupérer pour permettre un recalcul indépendant des métriques. Les agrégats ne les remplacent pas. Les poids volumineux ne sont pas publiés dans Git.")
    p("Les nouvelles cinq époques n'ont pas été exécutées par l'assistant. Les sorties du testeur viennent de la capture, leur provenance est explicite. Les anciennes mesures CPU ne sont pas des répétitions T4 supplémentaires. Graines fixées ne signifie pas déterminisme CUDA absolu. Ce bilan distingue réalisation des expériences, qualité réelle et pièces nécessaires à une remise entièrement vérifiable.")

    page('Annexe A - Courbes et matrice de confusion')
    p("Figures extraites de l'exécution T4, sans réentraînement. Courbes : F1 validation par époque sur trois graines. Matrice : checkpoint E2 lambda 0,3 graine 42, seuil val 0,490785.")
    figure('cell_101_image_1.png',17,10)
    figure('cell_101_image_3.png',13,10)
    p("Vrais négatifs 1 712 ; faux positifs 467 ; faux négatifs 237 ; vrais positifs 584. Total 3 000 ; 821 positifs réels. F1 = 1 168 / 1 872 = 0,623932.",small)

    page('Annexe B - Détail des runs T4')
    p("Scores affichés par le notebook. A/C/D/E1/E2 : graines 42/43/44 ; B1/B2 : un run. Le tableau G était arrondi à quatre décimales.",small)
    full=finals[['model','seed','threshold','test_f1_sarcastique','test_macro_f1','test_pr_auc_ap']].copy()
    full['model']=full.model.replace({'A_tfidf_conservateur':'A - TF-IDF','C3_AraVec':'C3 - AraVec','D2_BiLSTM_attention':'D2 - BiLSTM attention','E1_AraBERT_mono':'E1 - AraBERT mono','E2_AraBERT_lambda0p3':'E2 - AraBERT 0,3','B1_NLI_en':'B1 - NLI anglais','B2_negatif_sarcastique':'B2 - sentiment'})
    full.columns=['Modèle','Graine','Seuil val','F1 test','Macro-F1','AP test']
    table(full,widths=[2.3,.65,1,1,1,1],compact=True)
    h('Grille E complète sur validation : trois époques par run')
    ef=grid[['model','seed','val_f1_sarcastique','train_seconds']].copy()
    ef.columns=['Modèle / lambda','Graine','F1 val','Train (s)']
    table(ef,widths=[2.8,.7,1,1],compact=True)
    p("Tous les tableaux, dont 15 IC bootstrap individuels, 35 sous-groupes et les 30 erreurs, sont dans results/report_data/cell_*.csv. Les temps AraVec et D1 page 3 proviennent des journaux arrondis du notebook.",small)

    page('Annexe C - Historique CPU et traçabilité')
    p("Validation historique CPU, graine 42, même train nettoyé. Ces 24 lignes ne sont pas des résultats test ni une moyenne multi-graines. Les différences avec T4 sont conservées sans remplacer les sorties originales.",small)
    cf=cpu[['model','val_f1_sarcastique','val_pr_auc_ap','train_seconds']].copy()
    cf.columns=['Modèle historique CPU','F1 val','AP val','Temps (s)']
    table(cf,widths=[2.9,1,1,1],compact=True)
    h('Sources et reproduction')
    p('Corpus : https://github.com/iabufarha/ArSarcasm-v2 ; AraVec : https://github.com/bakrianoo/aravec.',small)
    p('Modèles Hugging Face : aubmindlab/bert-base-arabertv02-twitter ; UBC-NLP/MARBERTv2 ; MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 ; CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment. Révisions : models/revisions.json ; données : data/raw/manifest.json.',small)
    p('Régénérer le document : python -m src.report. Cette commande extrait les sorties enregistrées et compose le PDF ; elle ne charge aucun modèle et ne lance aucun entraînement. Tables HTML : pandas + lxml. Polices Arial, ReportLab. Corps : pages 1 à 6 ; annexes : pages 7 à 10.',small)
    p('Preuve : notebooks/TP2_Sarcasme_Arabe_T4_Execution.ipynb ; SHA256 et inventaire dans results/report_data/provenance.json. Le notebook de travail reste séparé pour préserver les résultats de cette exécution.',small)

    page('Annexe D - Attention et essais supplémentaires')
    p("D2 : cinq tweets sarcastiques de validation, poids par position extraits de l'exécution T4. Les tokens arabes et leur rendu droite à gauche figurent dans attention_5_tweets.html. Ces poids internes ne sont pas une attribution causale.")
    figure('cell_067_image_5.png',17.5,15)
    h('Contrôle local distinct du testeur Colab')
    p("Un diagnostic CPU antérieur sans entraînement a testé les quatre formulations avec un autre checkpoint E2 lambda 0,5, seuil 0,700477. Scores : 0,004811 ; 0,000933 ; 0,027889 ; 0,024360 ; toutes les sorties sont non sarcastiques. Ce n'est pas la répétition du checkpoint Colab au seuil 0,579043. Sur huit contrôles val choisis aux extrêmes (quatre positifs et quatre négatifs), le lecteur reproduit exactement les probabilités sauvegardées localement ; cela vérifie ce chargement, pas la qualité sur un nouvel échantillon représentatif.")
    p("Fichiers : results/phrase_inference_diagnostic.json et results/diagnostic_phrases.md. Le message de pooler BERT initialisé ne prouve pas à lui seul un chargement erroné : le classifieur utilise le CLS brut et le state_dict du checkpoint. La mauvaise généralisation est un constat ; sa cause technique exacte n'est pas démontrée.")

    def footer(canvas,doc):
        canvas.setFont('TP',7.7); canvas.setFillColor(colors.HexColor('#526577'))
        canvas.drawString(1.7*cm,.85*cm,'Omar Elkhali | BDIOT | Travail individuel | TP2 ArSarcasm-v2')
        canvas.drawRightString(A4[0]-1.7*cm,.85*cm,str(doc.page))
    output=ROOT/'Compte_Rendu_TP2.pdf'
    SimpleDocTemplate(str(output),pagesize=A4,leftMargin=1.7*cm,rightMargin=1.7*cm,topMargin=1.35*cm,bottomMargin=1.5*cm,
        title='TP2 - Sarcasme arabe - Omar Elkhali - BDIOT',author='Omar Elkhali').build(story,onFirstPage=footer,onLaterPages=footer)
    (ROOT/'results/compte_rendu.md').write_text('\n'.join(markdown),encoding='utf-8',newline='\n')
    import pymupdf
    with pymupdf.open(output) as document:
        assert len(document)==10,f'{len(document)} pages au lieu de 6 + 4 annexes : revoir la mise en page.'
        assert 'Annexe A - Courbes' in document[6].get_text(), 'Le corps doit tenir sur six pages'
    if workdir:
        target=Path(workdir).resolve()/'Compte_Rendu_TP2.pdf'
        if target!=output: shutil.copy2(output,target)
    print(f'{output} | 6 pages + 4 annexes | aucun entraînement')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir',type=Path)
    main(parser.parse_args().workdir)

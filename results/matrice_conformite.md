# Matrice de conformité - énoncé TP2 Version B

> Cette matrice décrit l'état de préparation historique. L'exécution T4 a depuis été sauvegardée, puis le code a été modifié. Voir l'[audit actualisé du 18 octobre 2026](audit_conformite_20261018.md) pour les preuves et les réserves qui empêchent encore une validation à 100 %.

Source : document officiel de Prof. Yassir Matrane. Le code est préparé ; la version refactorisée n'a pas été exécutée intégralement. « Historique » désigne des sorties antérieures contrôlées, pas de nouveaux calculs.

| Exigence | Implémentation / preuve | État actuel |
|---|---|---|
| Git, environnement figé, src, notebooks, tests | `.git/`, `requirements.txt`, modules imposés ; `src.sync` extrait le code du notebook | Préparé, Git local seulement |
| CSV officiels 12 548 / 3 000 | `data/raw/`, manifeste SHA256 | Présents, hashes vérifiés |
| Split 90/10, graine 42, stratification | `src.data`, notebook section 1 | Vérifié sans entraînement |
| Pas d'apprentissage sur test | Texte test seulement pour audit/exclusions ; labels ouverts en F après manifeste | Ordre statique contrôlé ; exécution future à vérifier |
| Audit 30 train / 30 test | Deux affichages bruts et tableau des indices | Code présent ; sorties historiques archivées |
| Artefact, indices arabes, doublons, contradictions | `text_audit.csv`, `split_audit.json`, `contradictions_train.csv` | Historique conservé ; train final 10 842 |
| Audit corrélations test sans régler les modèles | Complément de l'audit en F | À exécuter |
| Prétraitements par famille, ablation destructrice | Normalisation conservatrice, minimale PLM, AraBERT officiel, clean_str AraVec | Invariants synthétiques réussis ; scores A disponibles |
| A : trivial, mots 1-2 + caractères 2-5 + LR balanced | Partie A, `results.csv`, règles URL/mention/artefact | Quatre variantes historiques, métriques val recalculées |
| B1 : checkpoint exact, trois langues, seuil val | Partie B, trois prompts NLI | Trois scores historiques ; poids SHA vérifiés |
| B2 : checkpoint exact, id2label, argmax négatif | Partie B, règle distincte de P(négatif) pour AP | Historique authentifié ; contamination explicitée |
| C : vocabulaire train freq ≥ 2, OOV val/test | Vocabulaire et token_stats ; test OOV seulement en F | Val historique ; test à exécuter |
| C : aléatoire et AraVec 100D, figé puis dégelé | clean_str par token, deux époques figées | Checkpoints et couverture historiques |
| C1 BiLSTM, C2 attention implémentée, C3 2 couches/128/4 têtes | StaticNet ; positions sinusoïdales, masques, sans PE | Forward/masques/invariance sans PE vérifiés sur jouets |
| C4 char-CNN / focal loss | Explicitement optionnels | Non réalisés, aucune obligation retirée |
| C déséquilibre : poids / suréchantillonnage / standard + seuil | `imbalance.csv`, même architecture/budget | Historique disponible |
| D1 CLS vs mean + LR | Pooling masqué ; moyenne par blocs RAM bornée | Historiques ; équivalence du pooling synthétique vérifiée |
| D2 token states + BiLSTM attention, cinq tweets | FrozenHead ; figures et CSV attention | Historique disponible, encodeur gelé |
| E1 deux PLM, ajustement complet | AraBERT et MARBERT ; toutes couches utiles entraînables | AraBERT graine 42 terminé ; MARBERT à exécuter |
| E2 encodeur partagé + sentiment, λ 0,3 / 0,5 / 1 | Grille pour chaque encodeur puis sélection val | Trois lambda AraBERT historiques ; MARBERT à exécuter |
| E paramètres de départ et trois graines finales | lr 2e-5, wd .01 uniforme, warmup .1, 5 époques max, patience 2 ; 42/43/44 | Ancienne exécution T4 à 3 époques sauvegardée ; réglages actuels à mesurer, batch 32/AMP FP16 |
| AraBERT pooler hors prédiction et apprentissage | CLS last_hidden_state ; pooler.requires_grad=False | Sans influence ni gradient vérifié avec encodeur jouet |
| F1 test officiel, prior, F1=f(seuil) posthoc | Seuil val verrouillé avant test ; courbes diagnostiques séparées | Code présent, aucun résultat test inventé |
| F2 retrait egypt, réentraînement meilleur E | Train et sélection checkpoint/seuil sans egypt | À exécuter ; limite du lambda hérité explicitée |
| F3 avec/sans URL + réentraînement sans liens A/C/D/E1/E2 | Sous-groupes et six expériences de robustesse au total avec F2 | À exécuter, aucune famille supprimée |
| F4 retrait emojis/rires/ponctuation sur mêmes TP | Ensemble conditionnel fixe, seuil inchangé | À exécuter |
| F5 trois graines, bootstrap 1 000, McNemar | Cinq familles entraînées, deux meilleures sur val | À exécuter |
| F6 30 erreurs, 15 FN/15 FP, catégories et contradictions | Export avec hypothèses provisoires + champ human_review | À calculer puis annoter humainement |
| G qualité/robustesse/temps/latence CPU/poids/CM/décision | CPU batch 1, 20 tweets × 3 après warmup, poids encodeur D inclus | À exécuter ; aucun modèle final retenu à ce stade |
| CSV résultats de tous les runs et prédictions finales | `results/results.csv`, `predictions/<modèle>_seed<k>.csv` | 24 lignes val historiques ; 17 exports finaux attendus |
| Compte rendu officiel Q1-Q12 ≤ 6 pages hors annexes | `Compte_Rendu_TP2.pdf`, six pages + deux annexes | PDF partiel produit ; actualisable via src.report |
| Tests sans Run All | `tests.verify_layout`, `tests.check_cpu_optimisations`, `tests.check_gpu_config` ; `tests.verify_final` réservé après exécution | Structure et contrôles légers vérifiés ; CUDA simulé seulement, essai T4 et exécution intégrale laissés à l'étudiant |

L'audit initial du texte test et le scellement des labels sont distingués pour résoudre la tension de l'énoncé. Les exclusions de train fondées sur le texte test sont expressément demandées par le sujet. Elles ne consultent pas les labels test.

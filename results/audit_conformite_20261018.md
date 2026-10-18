# Audit de conformité au TP2 — 18 octobre 2026

**Verdict : les architectures et l'essentiel des expériences obligatoires sont présents ; le projet ne peut pas encore être déclaré conforme à 100 % ni prêt à remettre.**

Référence : texte du TP joint par l'utilisateur dans `d6206b63-99da-48d4-9c60-0f3b8036db4c/Pasted text.txt`. Il est utilisé comme cahier des charges, sans exécuter ses exemples de code.

## Preuves distinguées

- **Version actuelle** : `notebooks/TP2_Sarcasme_Arabe_T4.ipynb`, 104 cellules de code, plafond PLM de 5 époques. Aucun compteur d'exécution sauvegardé après les modifications. Leurs gains ne sont pas mesurés.
- **Exécution T4 précédente** : `results/history/before_quality_and_plots_20261018_091651/notebooks/TP2_Sarcasme_Arabe_T4.ipynb`, 104/104 cellules exécutées et aucune sortie d'erreur enregistrée. Cela atteste une exécution enregistrée, pas une vérification indépendante de chaque fichier distant.
- **CSV local** : `results/results.csv` contient les 24 anciens runs CPU de validation. Les tableaux `execution_t4_precedente_*.csv`, extraits des sorties T4, en sont séparés.
- **Livrables distants** : aucun CSV test n'est présent dans `predictions/` local. `final_manifest.json`, `bootstrap.csv`, `mcnemar.json`, `errors_30.csv`, `final_comparison.csv`, `decision.json` et `validation_checks.json` sont également absents de `results/` à la racine locale. Les sorties T4 montrent leur génération sur Colab ; ils restent à rapatrier et vérifier.

## Comparaison avec les obligations

« Présent » désigne le code inspecté ; cela ne signifie pas que la version modifiée a été exécutée.

| Obligation | Vérification | État |
|---|---|---|
| Données officielles, split 90/10 stratifié graine 42 | Cellules 012–014, `src/data.py`, manifestes | Conforme |
| Exclusions des collisions et contradictions du train | Clés normalisées, train final 10 842, val/test intacts | Conforme à la politique explicitée |
| Transformations apprises sur train, texte seul en entrée | TF-IDF, vocabulaire fréquence ≥ 2, poids de classes, suréchantillonnage ; annotations auxiliaires jamais concaténées aux entrées | Conforme sur les chemins inspectés |
| Audit de 30 textes bruts train/test et indices arabes | Cellules 014–018 ; texte brut conservé ; corrélations de labels test reportées en F | Présent, réordonnancement expliqué plus bas |
| A : trivial, mots 1–2 + caractères 2–5, LR balanced, seuil val, règles de raccourcis | Cellules 026–029 | Conforme |
| B1 : checkpoint NLI imposé, trois langues, seuil val | Trois paires d'hypothèses arabe/français/anglais | Conforme |
| B2 : checkpoint CAMeLBERT imposé, id2label, règle négatif ⇒ sarcasme, contamination | Décision argmax négatif ; AP sur P(négatif) ; limites expliquées | Conforme |
| C : aléatoire/AraVec 100D, clean_str, gel puis dégel, OOV | Cellules C ; deux époques figées ; OOV test en F | Conforme |
| C1 BiLSTM max-pooling | Branche max-pooling masqué de `StaticNet` | Prédiction conforme ; comptage des paramètres à corriger |
| C2 attention implémentée | `softmax(vᵀ tanh(Wh))`, pooling masqué | Conforme |
| C3 Transformer | 2 couches, dimension 128, 4 têtes, sinusoïdes, masque, ablation sans positions | Conforme |
| C : même budget et comparaison du déséquilibre | 8 époques max / patience 2 ; pondération, suréchantillonnage, standard + seuil | Conforme |
| D1 : AraBERT gelé CLS/moyenne masquée + LR | Pooling et entraînement LR inspectés | Conforme |
| D2 : états gelés → BiLSTM/attention, cinq tweets sarcastiques | `FrozenHead`, cellules 066–067 | Conforme |
| E1 : AraBERT Twitter et MARBERTv2, perte pondérée | Grille et répétitions pour les deux PLM | Présent ; les deux modèles ont des sorties dans l'archive T4 |
| E2 : encodeur partagé, deux têtes, λ 0,3/0,5/1 | `FineTuneNet`, grille puis choix sur validation | Conforme |
| Réglages E et trois graines | T4 batch 32, AMP FP16, lr 2e-5, warmup 10 %, plafond 5, patience 2 ; graines 42/43/44 | Dans la plage du sujet ; résultats du nouveau réglage à mesurer |
| F1 : prior, seuil val, courbe test posthoc | Cellules 081, 085–087 | Conforme en code ; test exclu de la sélection |
| F2 : retrait egypt et réentraînement du meilleur E | Train sans egypt ; checkpoint/seuil choisis sur val sans egypt ; test egypt | Conforme en code |
| F3 : avec/sans URL et réentraînement sans liens | A/C/D/E1/E2, cellules 088–092 | Conforme en code |
| F4 : retrait des indices sur les mêmes vrais positifs | Ensemble initial fixé et seuil inchangé | Conforme en code |
| F5 : trois graines, bootstrap 1 000, IC 95 %, McNemar | Cellules 093–095 ; 15 IC et McNemar visibles dans l'archive | Présent ; fichiers distants à contrôler |
| F6 : 30 erreurs, 15 FN/15 FP, catégories et contradictions | Cellules 096–097 | Partiel : catégories automatiques provisoires et seul modèle retenu |
| G : qualité, robustesse, temps, latence CPU, taille, matrice et décision | Cellules 098–103 | Présent ; robustesse à consolider dans le tableau final et comptage C1 à corriger |
| Journal et prédictions finales | `register`, export des 17 modèles finaux et diagnostics | Génération visible dans l'archive, exports locaux absents |
| Compte rendu Q1–Q12, six pages hors annexes | PDF inspecté : 6 pages de corps et 2 annexes | Volume conforme, contenu encore partiel |
| Dépôt Git et environnement | Structure, README, requirements, src, notebooks, tests présents | Git sans commit ; dépendances et versions Colab à finaliser |
| C4, focal loss, troisième tête dialectale | Explicitement optionnels | Leur absence n'est pas un manquement |

## Réserves précises à lever

**1. Comptage des paramètres C1.** `StaticNet` instancie `AttentionPool(128)` aussi pour C1, mais le forward de C1 n'utilise que le max-pooling. Les paramètres de cette attention restent `requires_grad=True` et sont inclus dans le nombre annoncé comme entraîné. Cela ajoute **8 320 paramètres inutilisés** : 128 × 64 + 64 pour W et son biais, puis 64 pour v. Le nombre a été confirmé sur une instanciation jouet, sans forward ni entraînement. La correction du comptage doit préserver la compatibilité des anciens checkpoints.

**2. Analyse humaine F6.** Les colonnes produites sont `category_provisional` et `human_review`. La présence d'un emoji ou d'un dialecte ne permet pas d'établir la vraie cause d'une erreur. Les cinq catégories du sujet doivent être examinées sur les textes, renseignées et discutées dans Q12. Le code ne sélectionne les 30 erreurs que pour le modèle global retenu. Le préambule F demande F1–F6 pour A/C/D/E1/E2, alors que F2 précise ensuite le meilleur E : pour une lecture stricte, compléter F6 par famille ou faire préciser la portée attendue par l'enseignant.

**3. Synthèse G et décision de production.** Les diagnostics de robustesse sont dans des tableaux séparés (`subgroups.csv`, `cues_ablation.csv`, `retraining_robustness.csv`). `final_comparison.csv` ne les regroupe pas en colonnes. La synthèse demandée doit relier qualité, robustesse, temps, latence et taille. Le choix automatique sur moyenne du F1 de validation ne remplace pas l'argumentation qualité/coût/robustesse demandée en Q12.

**4. Rapatriement et vérification finale.** Les 17 CSV finaux attendus correspondent à A/C/D/E1/E2 sur trois graines, plus B1/B2. Récupérer depuis Colab ces fichiers, les exports diagnostiques, résultats complets, manifestes, versions, checkpoints et figures. Vérifier les IDs, labels, graines et seuils, puis recalculer F1/AP depuis les CSV. `tests.verify_final` n'a pas été déclaré réussi ici : la version actuelle n'a pas de compteurs sauvegardés et les principaux artefacts sont absents localement. Une sortie de notebook annonçant que les fichiers existent ne suffit pas à les certifier.

**5. Compte rendu.** Le PDF local contient encore « ÉTAT PARTIEL », des figures à produire et des champs identité/groupe/lien du dépôt vides. Il respecte actuellement six pages de corps et deux annexes, mais ne répond pas encore complètement aux Q1–Q12 avec les résultats T4. Après actualisation, son rendu et sa pagination doivent être vérifiés de nouveau. Le générateur fourni ne remplace pas la rédaction des conclusions et l'analyse humaine des erreurs.

**6. Reproductibilité.** `.git/` existe mais `main` ne contient aucun commit. Il faut versionner le rendu ; la publication distante dépend du mode de remise demandé. Le sujet cite `datasets` et `evaluate`, absents des requirements actuels : les fonctions correspondantes sont assurées par pandas/scikit-learn. Une conformité littérale de l'environnement demande de compléter ces dépendances ou de documenter leur remplacement. L'overlay Colab conserve volontairement PyTorch CUDA et les bibliothèques numériques préinstallées ; les versions effectives doivent accompagner le rendu.

## Adaptations à ne pas confondre avec des obligations supprimées

- Le passage de **3 à 5 époques** reste dans la plage du sujet. Aucun gain nouveau n'est affirmé.
- Après la clarification de l'utilisateur sur le respect strict des réglages, le weight decay est revenu à **0,01 uniformément**, sans exception pour les biais ou normalisations. Les couches et pertes demandées restent inchangées.
- Les runs CPU batch 16/float32 sont distincts des runs T4 batch 32/FP16.
- Le sujet demande des statistiques de labels test avant F tout en déclarant ce test scellé : le code utilise d'abord son texte pour l'audit/exclusion, puis ses labels après verrouillage. Ce réordonnancement doit être expliqué.
- « PR-AUC » utilise la convention **average precision**, annoncée dans le projet. Si l'enseignant attend l'intégrale trapézoïdale, ajouter cette mesure explicitement plutôt que les confondre.
- Le préentraînement des PLM n'est pas purgé des dialectes : F2 teste l'absence d'exemples egypt dans le fine-tuning, pas une absence garantie d'exposition durant le préentraînement.

Cet audit a modifié uniquement la documentation de conformité. Aucun modèle, checkpoint, seuil ni résultat scientifique n'a été modifié ; aucun entraînement n'a été lancé.

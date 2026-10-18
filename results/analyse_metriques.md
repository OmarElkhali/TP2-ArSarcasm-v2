> Mise ? jour : cette analyse portait sur les 24 anciens runs CPU disponibles lors de sa r?daction. Une ex?cution T4 compl?te a depuis ?t? sauvegard?e par l?utilisateur ; ses tableaux sont maintenant conserv?s s?par?ment dans `execution_t4_precedente_*.csv`, et son notebook int?gral dans le dossier indiqu? par `quality_and_plots_changes.json`. Les nouveaux r?glages ? cinq ?poques n?ont pas encore de scores mesur?s.

# Analyse des métriques disponibles — TP2 ArSarcasm-v2

Les résultats sont encourageants pour ce TP : le meilleur F1 de la classe sarcastique atteint **67,96 %**, contre **52,16 %** pour la meilleure baseline TF-IDF. Le gain observé est de **15,81 points**. Cela justifie l'intérêt des représentations contextuelles, mais ne constitue pas encore une mesure finale de généralisation.

## Périmètre réellement vérifié

Cette analyse porte sur les **24 expériences historiques enregistrées dans `results/results.csv`**, toutes avec la graine 42. Les métadonnées des checkpoints indiquent une exécution CPU float32, avec un batch PLM de 16. Ces valeurs ne sont donc pas présentées comme des résultats de la nouvelle exécution T4, qui utilise un batch de 32 et AMP float16.

Le notebook T4 sauvegardé localement ne contient aucune sortie exécutée au moment de cette analyse. De nouveaux résultats peuvent exister sur le serveur Colab ; ils ne sont pas accessibles dans les fichiers inspectés ici.

Les cinq métriques de chacune des 24 expériences ont été recalculées à partir des prédictions et des labels de validation : elles correspondent au CSV à la tolérance numérique près. Les 1 255 tweets de validation contiennent 217 exemples sarcastiques, soit **17,29 %**. Les IDs et les clés de textes normalisés sont disjoints entre les fichiers train et validation. Cela vérifie ces deux formes de recouvrement ; ce n'est pas une preuve d'absence de toute proximité sémantique ou contamination du préentraînement.

Pour B2, la décision « argmax sentiment négatif » sert au F1, tandis que la probabilité continue P(négatif) sert à l'average precision. Cette distinction du notebook explique les deux représentations sauvegardées et a été respectée lors de la vérification.

Aucun entraînement, aucune inférence de PLM et aucune ouverture des labels test n'ont été effectués. La tokenisation locale du train et le recalcul des métriques sur les prédictions existantes sont les seuls calculs supplémentaires.

## Comparaison des principaux modèles

Toutes les valeurs ci-dessous sont des pourcentages de **validation**, aux seuils sélectionnés sur cette même validation. L'AP est l'average precision, pas une aire PR trapézoïdale.

| Modèle | F1 sarcastique | Précision | Rappel | Macro-F1 | AP |
|---|---:|---:|---:|---:|---:|
| AraBERT multitâche, λ = 1 | **67,96** | **71,79** | 64,52 | **80,83** | **70,69** |
| AraBERT multitâche, λ = 0,5 | 67,87 | 66,67 | 69,12 | 80,50 | 70,08 |
| D2 : AraBERT gelé + BiLSTM/attention | 67,74 | 59,93 | **77,88** | 79,86 | 69,13 |
| AraBERT multitâche, λ = 0,3 | 67,61 | 68,90 | 66,36 | 80,49 | 69,30 |
| AraBERT monotâche | 67,23 | 62,45 | 72,81 | 79,84 | 67,51 |
| D1 : moyenne des représentations + LR | 63,16 | 60,25 | 66,36 | 77,49 | 63,01 |
| C1 : AraVec + BiLSTM | 53,70 | 46,46 | 63,59 | 70,89 | 51,93 |
| A : TF-IDF standard | 52,16 | 48,99 | 55,76 | 70,65 | 51,70 |
| B2 : sentiment négatif ⇒ sarcasme | 46,95 | 33,27 | 79,72 | 62,45 | 47,46 |
| B1 : NLI anglais, meilleur des trois prompts | 42,24 | 32,90 | 58,99 | 61,93 | 34,88 |

Le classement intégral des 24 expériences, les matrices de confusion et les empreintes des fichiers de prédictions figurent dans `analyse_metriques.csv`.

## Interprétation

### Le F1 sarcastique est la mesure principale

Un classifieur qui prédit toujours « non sarcastique » obtient déjà **82,71 % d'accuracy** sur cette validation, mais **0 % de F1 sarcastique**. Une accuracy supérieure à 80 % ne suffit donc pas à qualifier un modèle de bon. Le TP demande bien de privilégier le F1 de la classe sarcastique ; le macro-F1 décrit les deux classes et ne doit pas être confondu avec lui.

L'AP du meilleur modèle vaut 70,69 %, contre un niveau de référence sans pouvoir de classement égal à la prévalence positive, 17,29 %. Cela indique un classement informatif des tweets sarcastiques sur cette validation, sans démontrer une calibration parfaite des probabilités.

À titre de contexte historique, le rapport officiel WANLP 2021 donne un meilleur **F1 sarcastique de 62,25 % sur le test officiel**. Ce chiffre ne constitue pas un plafond ni un état de l'art actuel. Notre score de 67,96 % est obtenu sur une autre partition, utilisée pour choisir les seuils et les modèles : il ne démontre pas une supériorité sur ces systèmes. Source : [Abu Farha et al., WANLP 2021](https://aclanthology.org/2021.wanlp-1.36/).

### Le meilleur score correspond à un compromis précision/rappel

Sur les 217 tweets sarcastiques de validation, AraBERT multitâche λ = 1 détecte **140 vrais positifs** et en manque **77** ; il produit **55 faux positifs** et **983 vrais négatifs**. Il est relativement précis lorsqu'il annonce du sarcasme, mais manque encore environ 35,5 % des exemples positifs.

D2 détecte **169** de ces 217 tweets et en manque **48**, au prix de **113 faux positifs**. Il offre donc un meilleur rappel, avec davantage de fausses alertes. Pour l'objectif F1 imposé, λ = 1 reste le meilleur score ponctuel disponible ; pour une application où manquer du sarcasme coûterait davantage, le compromis D2 mériterait discussion.

Le seuil de **0,8014028072** appartient au checkpoint historique `E2_AraBERT_lambda1p0_seed42`. Il ne faut pas le recopier comme une constante pour un modèle réentraîné sur T4 : le notebook doit sélectionner son propre seuil sur validation, puis le verrouiller avant l'évaluation test.

### Les meilleurs modèles sont très proches

L'écart entre λ = 1 et λ = 0,5 n'est que de **0,09 point de F1**. Celui entre λ = 1 et D2 est de **0,23 point**. L'amélioration sur AraBERT monotâche est de **0,73 point**. Une seule graine ne permet pas de conclure à un avantage stable du multitâche ou d'une valeur de λ.

Un bootstrap apparié exploratoire a été calculé sur les mêmes 1 255 tweets, avec **1 000 rééchantillonnages, graine 42 et seuils fixes** :

| Quantité | Estimation | Intervalle percentile à 95 % |
|---|---:|---:|
| F1 de E2 λ = 1 | 67,96 % | [62,75 % ; 73,16 %] |
| F1 de D2 | 67,74 % | [63,09 % ; 72,20 %] |
| Écart F1 E2 − D2 | +0,23 point | [−4,83 ; +4,77] points |
| Écart F1 E2 − E1 | +0,73 point | [−2,54 ; +3,82] points |

Les intervalles des écarts contiennent zéro : ces données n'apportent pas de preuve nette d'un avantage. Ces intervalles sont descriptifs et conditionnels aux modèles et seuils déjà retenus ; ils ne corrigent ni la sélection parmi plusieurs expériences ni la variabilité d'entraînement entre graines. **Ils ne remplacent pas les trois graines, le bootstrap sur test et le test de McNemar exigés dans F5.**

### Les prétraitements donnent déjà des indications utiles

- TF-IDF conservateur : **52,06 %**, contre **49,69 %** avec le nettoyage agressif. Supprimer davantage de ponctuation, rires, emojis ou répétitions n'est pas justifié par ces résultats.
- TF-IDF standard : **52,16 %**, contre **47,43 %** avec suréchantillonnage. Le suréchantillonnage n'améliore pas automatiquement cette tâche.
- C1 aléatoire : **45,35 %**, contre **53,70 %** avec AraVec. Les embeddings préentraînés apportent un gain observé de **8,35 points** à cette architecture.
- D1 CLS : **60,54 %**, contre **63,16 %** avec moyenne masquée des représentations. Le choix du pooling compte.
- Les modèles statiques et le zero-shot restent utiles au TP, même lorsqu'ils sont moins bons : leur comparaison explique l'apport des modèles contextualisés et du fine-tuning.

## Ce qui manque avant de choisir définitivement

Les fichiers inspectés ne contiennent pas de résultats MARBERTv2, de répétitions avec les graines 43 et 44, de métriques test, ni les diagnostics complets F/G. Le meilleur modèle actuel est donc un **candidat provisoire**, pas un vainqueur final validé. Le classement qualité/coût nécessite aussi la latence CPU et les tailles complètes demandées en G ; les anciens temps CPU ne prédisent pas les durées T4.

## Priorités pour rechercher mieux en respectant le TP

1. **Terminer d'abord la grille MARBERTv2 déjà prévue**, monotâche puis multitâche λ ∈ {0,3 ; 0,5 ; 1}. Le score ne peut pas être annoncé avant exécution. La [fiche officielle du modèle](https://huggingface.co/UBC-NLP/MARBERTv2) documente un modèle arabe préentraîné ; elle ne fournit pas une garantie de gain sur ce split de sarcasme.
2. **Exécuter les trois graines prévues** pour les configurations retenues et rapporter moyenne/dispersion. Conserver la règle de sélection du TP plutôt que de publier seulement la meilleure graine.
3. **Pour une expérience supplémentaire sur la durée, comparer un plafond de 5 époques avec le plafond actuel de 3**, avec patience 2 et conservation du meilleur checkpoint de validation. Le sujet autorise 3 à 5 époques. Les courbes historiques montrent déjà des F1 proches de leur maximum dès la première époque : cinq époques ne garantissent pas une hausse et peuvent surapprendre. Un tel essai change le budget et le calendrier du scheduler ; il faut le déclarer comme un nouveau run, sans le mélanger avec les résultats historiques ni écraser leur provenance. Le plafond du notebook n'a pas été modifié ici.
4. **Conserver le prétraitement propre à chaque famille** : AraBERT officiel avec répétitions préservées, MARBERT minimal, négations, contenu des hashtags, emojis et dialectes conservés dans le protocole principal. Garder le nettoyage destructif et le retrait des liens comme ablations demandées.
5. **Conserver 64 tokens pour l'instant.** L'audit du train, avec les mêmes prétraitements et tokenizers locaux, trouve 76/10 842 tweets au-delà de 64 tokens AraBERT (**0,70 %**) et 35/10 842 pour MARBERTv2 (**0,32 %**). À 128 tokens, il en reste respectivement 1 et 0. Une longueur doublée n'est donc pas une priorité démontrée pour améliorer les scores et augmente le coût. Le test n'a pas servi à ce choix.
6. **Verrouiller modèles et seuils avant le test**, puis compléter les diagnostics dialectaux, les ablations, les erreurs et les mesures de coût. Utiliser le test pour le bilan final, pas pour chercher un meilleur réglage.

La meilleure configuration démontrée à ce stade est **AraBERT Twitter multitâche sarcasme + sentiment, λ = 1, graine 42, lr 2e-5, 3 époques, meilleur checkpoint et seuil sélectionnés sur validation**. Ses métriques sont celles de l'ancien run CPU décrit plus haut. Le réglage T4 déjà prévu dans le notebook conserve les architectures et utilise batch 32/AMP float16 ; il doit être mesuré à son tour.

Une nouvelle valeur « meilleure possible » n'est pas connaissable sans les expériences manquantes. Aucun score n'a été remplacé ou inventé et aucun nouvel entraînement n'a été lancé pour cette analyse.

## Fichiers de preuve

- `results/results.csv` : résultats historiques inchangés, SHA256 `ed642881c3de252cfea444d7759644aeb70a850180fdd3a3eab5535c7f7c10b3`.
- `results/analyse_metriques.csv` : classement complet, métriques et TP/FP/FN/TN recalculés.
- `results/analyse_metriques.json` : tailles, empreinte source, bootstrap et audit des longueurs.
- `predictions/*_seed42_val.npy`, `predictions/B2_val_raw.npz` et `data/processed/val.csv` : sources des recalculs.
- `tmp/analyse_metriques.py` : script ponctuel de cette analyse, sans entraînement ni labels test.

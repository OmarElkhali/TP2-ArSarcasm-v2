# Notes de présentation

## 01. LMA3NA

Présenter le projet comme une comparaison scientifique. Les tableaux T4 sont les résultats exécutés sauvegardés. Les expériences d’amélioration restent identifiées séparément.

Sources : results/report_data/provenance.json

## 02. Le contexte peut inverser le sens des mots

L’exemple vient des essais qualitatifs transmis par l’utilisateur. La lecture ironique est informelle et ne remplace pas une annotation indépendante.

Sources : results/report_data/phrases_capture.csv

## 03. Une démarche complète, du corpus à la démonstration

Présenter les étapes dans leur ordre logique. La séparation des données précède les apprentissages. Les familles A à E sont des modèles, F et G des évaluations. L’application restitue les mesures existantes. R-Drop reste une extension ajoutée.

Sources :

## 04. Les partitions restent distinctes pendant le protocole

Le CSV officiel comprend 12 548 tweets. Le split stratifié réserve 1 255 tweets à la validation. Les exclusions s’appliquent ensuite au train, qui passe à 10 842. Le test reste officiel, avec 3 000 tweets. Le texte test peut intervenir dans la détection des collisions sans utiliser ses labels.

Sources : results/split_audit.json

## 05. L’accuracy masque facilement la classe rare

Expliquer le piège du déséquilibre : prédire toujours non sarcastique donne une accuracy élevée, mais aucun rappel sur la classe utile. Le F1 combine précision et rappel. L’Average Precision exploite le classement des scores et complète un F1 dépendant du seuil.

Sources : results/compte_rendu.md, Q1

## 06. Les collisions et contradictions sont traitées avant l’apprentissage

Les clés normalisées recherchent les mêmes contenus malgré des variantes d’écriture. Les ensembles de doublons, collisions et contradictions se recouvrent. Leur somme serait trompeuse. Préserver validation et test empêche de modifier l’évaluation pour améliorer artificiellement les scores.

Sources : results/compte_rendu.md, Q2

## 07. Le nettoyage préserve les marqueurs utiles à l’ironie

Un nettoyage trop fort détruit le signal. Les rires, emojis, répétitions et négations peuvent expliquer le contraste ironique. Le traitement doit rester compatible avec le tokenizer du modèle préentraîné. En inférence, on reprend exactement le même prétraitement.

Sources : src/preprocess.py ; notebook de présentation, Dataset et prétraitement

## 08. Les indices d’export créent des raccourcis de prédiction

Les fréquences de marqueurs diffèrent entre les exports. Un modèle peut apprendre leur provenance plutôt que l’ironie. La règle à une seule variable est un diagnostic de raccourci. Les résultats ne prouvent pas qu’un indice cause le sarcasme.

Sources : results/report_data/cell_017_table_1.csv

## 09. Le nettoyage conservateur gagne la comparaison TF-IDF

TF-IDF combine n-grammes de mots et de caractères pour couvrir aussi des variantes morphologiques. La régression logistique fournit un score, puis le seuil de validation fournit la décision. Le nettoyage conservateur gagne cette ablation, mais le résultat ne vaut pas universellement pour tous les corpus.

Sources : results/report_data/cell_029_table_1.csv

## 10. Le zero-shot dépend du prompt et du signal de sentiment

NLI : mDeBERTa multilingue XNLI. Sentiment : CAMeLBERT. Aucun ajustement de ces encodeurs sur les exemples du TP pour B. Une contamination avec des sources de sentiment réannotées reste possible pour B2.

Sources : results/report_data/cell_038_table_1.csv

## 11. Trois architectures apprennent l’ordre et les relations

C1 résume les états bidirectionnels par max-pooling. C2 apprend une pondération des états. C3 représente les relations entre tokens par auto-attention et ajoute une information de position. Le masque évite d’apprendre à partir du padding. Ces architectures gardent le budget du TP.

Sources : src/models.py

## 12. AraVec améliore les trois réseaux statiques

AraVec améliore toutes les architectures C sur la validation de cette exécution. Les embeddings apportent des régularités lexicales apprises sur Twitter. Il faut distinguer le calendrier de dégel de l’époque du meilleur checkpoint, qui peut être antérieure.

Sources : results/report_data/cell_051_table_1.csv ; notebook de présentation, AraVec

## 13. Le seuil compte autant que le traitement du déséquilibre

Les trois stratégies de déséquilibre ne produisent pas les mêmes distributions de scores. Optimiser le seuil sur validation améliore chaque configuration. Il serait incorrect de choisir le seuil sur test ou de sur-échantillonner avant la séparation.

Sources : results/report_data/cell_056_table_1.csv

## 14. AraBERT gelé fournit des représentations contextuelles

D conserve les poids de l’encodeur et n’ajuste que la tête supervisée. Les représentations contextuelles sont plus riches que les embeddings statiques. Les temps d’ajustement de tête excluent l’extraction et ne représentent donc pas le temps total du système.

Sources : results/compte_rendu.md, Q8

## 15. Le multi-tâche partage l’encodeur entre sarcasme et sentiment

E1 ajuste l’encodeur pour le sarcasme. E2 partage l’encodeur avec une tâche de sentiment. Le coefficient lambda contrôle le signal auxiliaire. Une corrélation sentiment-sarcasme peut aider, mais la négativité ne définit pas l’ironie. La tête sentiment peut aussi détourner l’apprentissage.

Sources : src/models.py ; résultats T4 E1/E2

## 16. La validation sélectionne AraBERT E2 avec λ = 0,3

La comparaison utilise la validation pour sélectionner l’encodeur et lambda. Les trois graines mesurent la variabilité de l’apprentissage. Le choix final retient AraBERT E2 lambda 0,3. Le score validation ne doit pas être présenté comme un score test.

Sources : results/report_data/cell_076_table_1.csv

## 17. Les checkpoints reposent sur les courbes de validation

Le suivi par époque choisit le checkpoint sur validation avec early stopping. La campagne historique limite les PLM à trois époques. AMP FP16 concerne les entraînements T4. Les nouvelles valeurs de code non exécutées ne créent pas de nouveaux résultats.

Sources : results/report_data/cell_101_image_1.png

## 18. E1 et E2 obtiennent presque le même F1 test moyen

Le classement donne les moyennes de test des modèles retenus auparavant sur validation. Les scores E1 et E2 sont presque identiques. L’écart de 0,03 point est très inférieur à la dispersion entre graines. A/C/D/E utilisent trois graines et B1/B2 un passage.

Sources : results/report_data/cell_101_table_1.csv

## 19. Les trois graines rendent la variabilité visible

Le tableau permet de vérifier les agrégats. TF-IDF est déterministe dans cette configuration, donc ses trois résultats identiques ne sont pas un bug. E1 varie davantage. Le meilleur run test ne devient pas automatiquement le modèle à choisir après coup.

Sources : results/report_data/cell_085_table_1.csv

## 20. La matrice révèle encore 237 sarcasmes manqués

La matrice correspond à une seule graine E2, pas à la moyenne de trois modèles. Les 237 faux négatifs sont des sarcasmes manqués et les 467 faux positifs des textes pris à tort pour sarcastiques. Le total est bien de 3 000 et les positifs réels de 821.

Sources : results/report_data/cell_101_image_3.png

## 21. TF-IDF reste un compromis intéressant sur CPU

La latence doit être interprétée avec son protocole : CPU Colab, batch un, après chauffe. TF-IDF offre une solution légère. AraBERT améliore la qualité mais nécessite les poids complets, même avec un encodeur gelé. Les temps de tête ne comparent pas tous les coûts.

Sources : results/report_data/cell_101_table_1.csv

## 22. Le taux de sarcasme change entre validation et test

Le prior de sarcasme change de plus de dix points entre validation et test. Cela affecte précision et F1. Les écarts ne prouvent pas à eux seuls une mauvaise implémentation. Les diagnostics de seuils post-hoc expliquent une sensibilité, sans autoriser un réglage sur test.

Sources : results/report_data/cell_081_table_1.csv

## 23. Les sous-groupes URL ont des distributions différentes

Le sous-groupe avec URL contient seulement 16 sarcasmes. Son prior diffère radicalement du groupe sans URL. Un F1 très faible doit donc être discuté avec les effectifs. Les ablations réentraînées complètent cette observation sans isoler parfaitement une cause.

Sources : results/report_data/cell_088_table_1.csv

## 24. Retirer les indices ne garantit pas de meilleurs résultats

La suppression des URL et mentions ne suffit pas à améliorer les résultats. La généralisation sans égyptien reste limitée comme preuve, car le préentraînement peut déjà contenir le dialecte. Une ablation contrôlée teste une hypothèse précise, sans garantir une invariance générale.

Sources : results/report_data/cell_092_table_1.csv

## 25. Les dialectes rares limitent les conclusions de déploiement

Le nombre de tweets maghrébins est insuffisant pour conclure sur la darija. Deux exemples test négatifs ne renseignent pas la détection des sarcasmes en darija. Les performances en égyptien ne se transfèrent pas automatiquement au marocain ou à l’Arabizi.

Sources : results/compte_rendu.md, Q1 et Q11

## 26. Les intervalles et les désaccords nuancent le classement

Le bootstrap rééchantillonne les tweets du test pour chaque modèle et graine. L’écart-type entre graines répond à une autre question. McNemar compare les désaccords de classification appariés et ne teste pas directement une différence de F1. Ne pas confondre absence de différence détectée et équivalence.

Sources : results/report_data/cell_093_table_1.csv ; cell_095_table_1.csv

## 27. L’attention indique des poids, sans prouver la compréhension

L’attention indique des poids, sans prouver la compréhension
Cinq tweets de validation
La tête D2 pondère les représentations des tokens. Des mots de contenu et de la ponctuation reçoivent de l’attention.

Un poids élevé n’est pas une explication causale de la décision.
Lecture qualitative à rapprocher du texte arabe, de la prédiction et des erreurs. Aucun score supplémentaire ne résulte de cette figure.

Sources : results/report_data/cell_067_image_5.png ; notebook de présentation, attention D2

## 28. L’analyse des erreurs demande encore une relecture humaine

L’analyse des erreurs demande encore une relecture humaine
15 + 15
Faux négatifs et faux positifs
Échantillon d’erreurs à analyser dans leur contexte
19 / 30
Contexte ou ironie implicite
Catégorie proposée automatiquement, encore provisoire
6 + 5
Dialecte / marqueur
6 cas dialecte–contexte et 5 marqueurs potentiellement mal interprétés
Ces catégories ne sont pas une annotation humaine finalisée. Aucun taux global de bruit ne peut en être déduit.

Sources : results/report_data/cell_097_table_1.csv ; cell_097_table_2.csv

## 29. Les phrases réelles exposent une limite visible du modèle

Les deux exemples ironiques sont manqués. La capture constitue une transcription qualitative, pas une expérience indépendante donnant une accuracy généralisable. Ne pas abaisser le seuil pour réussir ces seuls exemples.

Sources : results/report_data/phrases_capture.csv

## 30. Les difficultés techniques ont des solutions identifiées

Les difficultés techniques ont des solutions identifiées
Problème
Solution dans le projet
Dépendances absentes sur Colab
Installation dans le Python du noyau, puis redémarrage
Disque Windows invisible depuis Colab
Chemins du projet distant et ressources vérifiées
Chargements et tokenisations répétés
Caches, padding dynamique et réemploi des ressources
Avertissement bert.pooler
CLS brut utilisé, pooler hors de la tête de décision
Graphiques historiques vides
Affichage conditionnel des données, exclusion des axes vides
CPU local, RAM 16 Go
Inférences séquentielles et chargement différé

Sources : notebook de présentation ; src/predict.py ; src/reprise_t4.py

## 31. LMA3NA rassemble les résultats et leurs sources

LMA3NA rassemble les résultats et leurs sources
Un espace de restitution
50 tableaux enregistrés
19 figures exploitables
17 évaluations T4
Modèles et techniques
Provenance et limites
Application locale. Les métriques T4 historiques et la validation CPU locale restent séparées.

Sources : webapp/README.md ; output/webapp/LMA3NA.jpg

## 32. La saisie compare les checkpoints disponibles localement

La vérification locale enregistrée a répondu en environ 49,9 secondes, chargements inclus, sans erreur de modèle. Ce délai ne remplace pas les latences expérimentales batch 1 du tableau G. Les scores affichés ne sont pas des probabilités calibrées de justesse.

Sources : output/webapp/inference_verification.json ; webapp/README.md

## 33. R-Drop ajoute une contrainte de cohérence entre deux passages

R-Drop ajoute une contrainte de cohérence entre deux passages
Même tweet
Aucune donnée supplémentaire
Deux passages
Dropout différent à chaque fois
Deux distributions
p₁ et p₂ sur le sarcasme
Perte = CE moyenne + λ × CE sentiment moyenne
+ α × KL symétrique des prédictions de sarcasme
Pendant l’entraînement seulement. À l’inférence : un passage en mode eval(). Gain encore à mesurer sur ce projet.

Sources : src/ameliorations.py, loss_parts ; Liang et al. 2021; https://proceedings.neurips.cc/paper_files/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html

## 34. L’extension R-Drop dispose d’un protocole sans résultat annoncé

L’extension R-Drop dispose d’un protocole sans résultat annoncé
Comparaison appariée
E2 AraBERT, λ 0,3
Référence : α = 0
R-Drop : α = 1
Graines 42, 43, 44
Réglages identiques
Même train et validation
LR 2e−5, longueur 64
Batch effectif 32
3 époques max., patience 2
État vérifié
3 cellules ajoutées
Exécution désactivée
Aucun run local terminé
Aucun gain revendiqué
Six runs au total. Sélection sur validation. Les résultats seront séparés des expériences historiques A–G.

Sources : notebooks/TP2_Sarcasme_Arabe_Presentation.ipynb, H.1–H.3

## 35. Les prochaines étapes privilégient des comparaisons contrôlées

Les prochaines étapes privilégient des comparaisons contrôlées
Priorité
Action
Critère de décision
1
Revalider checkpoint, prétraitement et seuil
Même chaîne en évaluation et en saisie
2
Exécuter référence / R-Drop
Écarts appariés sur trois graines
3
Comparer LR, longueur et prétraitement
Gain validation et coût mesuré
4
Étudier ensemble ou TAPT sur train
Extension isolée, sans labels test
5
Finaliser les preuves et l’analyse
Exports test et relecture des erreurs
Pas de modification des réglages pour réussir quelques phrases de démonstration. Aucune amélioration n’est garantie avant mesure.

Sources :

## 36. La reproductibilité dépend aussi des artefacts conservés

La reproductibilité dépend aussi des artefacts conservés
Disponible
Code et notebooks
Tableaux T4 exportés
Figures historiques
Registre et checkpoints CPU
À compléter
17 CSV individuels test
Poids T4 nécessaires au service
Relecture des 30 erreurs
Résultats R-Drop exécutés
Séparations explicites
CPU / T4
Validation / test
Agrégats / prédictions
Exécuté / seulement ajouté
Le dépôt public permet de consulter le projet : github.com/OmarElkhali/TP2-ArSarcasm-v2

Sources : results/report_data/provenance.json ; webapp/README.md

## 37. Un résultat de corpus ne suffit pas à valider l’usage réel

Conclusion personnelle : les PLM améliorent le score par rapport à TF-IDF, mais E2 et E1 restent presque équivalents sur test. Les exemples manuels montrent une limite hors du corpus. L’application doit rester un laboratoire pédagogique, avec contrôle humain pour un usage réel.

Sources :

## 38. Sources scientifiques et preuves du projet

Sources scientifiques et preuves du projet
ArSarcasm-v2
iabufarha/ArSarcasm-v2, jeu officiel et partitions
AraBERT
Antoun, Baly et Hajj, 2020
ARBERT & MARBERT
Abdul-Mageed, Elmadany et Nagoudi, 2021
R-Drop
Liang et al., NeurIPS 2021

Sources : https://github.com/iabufarha/ArSarcasm-v2; https://aclanthology.org/2020.osact-1.2/; https://aclanthology.org/2021.acl-long.551/; https://proceedings.neurips.cc/paper_files/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html

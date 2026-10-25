# Présenter les résultats du TP2


Ces phrases accompagnent les sorties historiques du notebook de présentation. Elles ne décrivent pas une nouvelle exécution. Les scores de validation servent à choisir ; le test sert à évaluer.


**Introduction à dire :** « Mon objectif est de détecter le sarcasme dans des tweets arabes. Je compare des approches classiques, des réseaux de neurones et des modèles préentraînés. Je mesure aussi leurs limites, leur robustesse et leur coût. »


### Présenter les résultats — étape 01

**Phrase de présentation**

> Je commence par vérifier que les bibliothèques et les ressources sont accessibles. Les messages affichés décrivent la préparation de la session ; ils ne sont pas des scores de détection.


### Présenter les résultats — étape 02

**Phrase de présentation**

> Je fixe les paramètres avant les expériences afin de conserver un protocole comparable. Cette cellule de configuration ne produit pas de résultat scientifique.


### Présenter les résultats — étape 03

**Phrase de présentation**

> Cette exécution historique a utilisé une Tesla T4 avec précision mixte et batch 32. Je précise ces conditions car elles influencent les temps et la reproductibilité.


### Présenter les résultats — étape 04

**Phrase de présentation**

> Je définis les règles de nettoyage sans encore mesurer de performance. Leur intérêt sera évalué sur les exemples et les variantes de modèles.


### Présenter les résultats — étape 05

**Support 1 — Audit des partitions**

> « Ce tableau vérifie la séparation des données. Après les exclusions, je conserve 10 842 tweets pour apprendre, 1 255 pour valider et 3 000 pour tester. Les collisions sont mesurées avant l’entraînement ; les différentes catégories de doublons peuvent se recouper, donc je ne les additionne pas directement. »

**Support 2 — 30 tweets bruts du train**

> « Je montre ces exemples avant normalisation pour identifier les caractéristiques réelles du corpus : arabe dialectal, liens, emojis et guillemets d’export. L’objectif est de comprendre les données avant de les transformer. »

**Support 3 — 30 tweets bruts du test**

> « Ces exemples servent à auditer le format du texte test. À ce stade, je n’utilise pas ses labels pour régler les modèles. La comparaison avec le train révèle des différences de présentation des tweets. »


### Présenter les résultats — étape 06

**Support 1 — Fréquence des indices textuels**

> « Les URL apparaissent dans 38,09 % du train brut contre 14,67 % du test. Les guillemets d’export passent de 60,54 % à 0,20 %. Ces différences montrent qu’un modèle peut apprendre des indices liés au corpus plutôt qu’au sarcasme. »

**Support 2 — Corrélation entre indices et sarcasme**

> « Dans le train nettoyé des collisions, le taux de sarcasme est de 23,90 % sans URL contre 6,29 % avec URL. Cette corrélation peut devenir un raccourci ; elle ne signifie pas qu’un lien empêche un tweet d’être sarcastique. »

**Support 3 — Exemples avant et après nettoyage**

> « Le nettoyage conservateur garde les emojis, les rires et la ponctuation, alors que le nettoyage agressif en supprime une partie. Je conserve ces indices quand ils peuvent porter l’ironie, puis je vérifie leur intérêt par les scores. »

**Support 4 — Train : répartition par dialecte**

> « Le train contient surtout de l’arabe standard, avec 7 427 tweets, contre seulement 35 tweets maghrébins. Cette différence limite ce que je peux conclure sur les dialectes peu représentés. »

**Support 5 — Train : répartition par sentiment**

> « Dans le train, environ 41,89 % des tweets négatifs sont sarcastiques, contre 2,69 % des neutres et 2,41 % des positifs. Le sentiment est donc lié au sarcasme, mais ces deux notions ne sont pas équivalentes. »

**Support 6 — Validation : répartition par dialecte**

> « La validation ne contient que cinq tweets maghrébins. Un score sur un groupe aussi petit serait trop instable pour juger un usage sur des tweets marocains. »

**Support 7 — Validation : répartition par sentiment**

> « Le lien entre sentiment négatif et sarcasme se retrouve dans la validation : environ 41,94 % des tweets négatifs sont sarcastiques. Cette observation motive la tâche auxiliaire de sentiment. »

**Support 8 — Graphique du taux de sarcasme par dialecte**

> « L’axe horizontal représente les dialectes et la hauteur des barres leur proportion de sarcasme dans le train. Je lis ces proportions avec les effectifs du tableau : une barre élevée sur 35 tweets est beaucoup moins représentative qu’une barre calculée sur plusieurs milliers. »


### Présenter les résultats — étape 07

**Phrase de présentation**

> Je choisis le F1 de la classe sarcastique comme métrique principale, puis je complète avec la précision, le rappel, le macro-F1 et l’average precision, utilisée ici comme mesure PR-AUC. Ces mesures ne doivent pas être confondues avec l’accuracy.


### Présenter les résultats — étape 08

**Phrase de présentation**

> Le modèle qui prédit toujours « non sarcastique » atteint 82,71 % d’accuracy, mais son F1 sarcastique est nul. Ce résultat montre pourquoi une accuracy élevée peut être trompeuse sur des classes déséquilibrées.


### Présenter les résultats — étape 09

**Phrase de présentation**

> Je prépare une baseline TF-IDF de mots et de caractères avec régression logistique. Aucun score n’est produit tant que cette chaîne n’est pas entraînée et évaluée.


### Présenter les résultats — étape 10

**Phrase de présentation**

> Cette fonction applique le même protocole à toutes les variantes TF-IDF. Le résultat à commenter se trouve dans le tableau de comparaison suivant.


### Présenter les résultats — étape 11

**Support 1 — Comparaison des variantes TF-IDF**

> « Le nettoyage conservateur donne le meilleur F1 validation de cette comparaison : 0,5206. Le nettoyage agressif descend à 0,4959 et le sur-échantillonnage à 0,4743. La solution la plus complexe ne gagne donc pas automatiquement. »

**Support 2 — Classifieurs à une seule règle**

> « Une règle fondée sur les URL atteint déjà un F1 de 0,3618 ; celles fondées sur les mentions et les guillemets atteignent 0,2664 et 0,3006. Ces scores révèlent des raccourcis dans les données, sans démontrer une compréhension de l’ironie. »


### Présenter les résultats — étape 12

**Phrase de présentation**

> Les modèles sont téléchargés et leurs fichiers vérifiés. Les barres de progression sont des journaux de préparation, pas des mesures de performance.


### Présenter les résultats — étape 13

**Phrase de présentation**

> Le prompt arabe atteint un F1 validation de 0,4069, le français 0,4189 et l’anglais 0,4224. Le choix de formulation influence donc le score, même sans entraînement sur le sarcasme. Je retiens l’anglais sur validation.


### Présenter les résultats — étape 14

**Support 1 — Comparaison zero-shot et sentiment**

> « Le meilleur prompt NLI, en anglais, atteint 0,4224 de F1 validation. La règle « négatif implique sarcastique » atteint 0,4695, avec une précision de 0,3327 et un rappel de 0,7972 : elle détecte beaucoup de sarcasmes, mais produit aussi de nombreuses fausses alertes. Le modèle B2 peut avoir déjà vu des tweets issus de corpus réutilisés dans ArSarcasm-v2 ; son score n’est donc pas garanti exempt de contamination. Zéro seconde d’entraînement ne signifie pas zéro coût d’inférence. »


### Présenter les résultats — étape 15

**Support 1 — Tokens inconnus et troncature**

> « En validation, environ 21,16 % des tokens sont hors vocabulaire, contre 15,19 % dans le train. Le vocabulaire exclut les mots trop rares même dans le train. Cette limite explique l’intérêt de représentations préentraînées et de modèles à sous-mots. »


### Présenter les résultats — étape 16

**Phrase de présentation**

> Je définis trois architectures : C1 BiLSTM, C2 BiLSTM avec attention et C3 Transformer. Cette étape prépare les modèles ; seule leur évaluation permettra de les départager.


### Présenter les résultats — étape 17

**Support 1 — Architectures avec embeddings aléatoires**

> « C1 atteint 0,4774, C2 0,4683 et C3 0,4627 de F1 validation. Sans positions, C3 atteint 0,4619 : l’écart est très faible pour cet essai. Je ne peux donc pas attribuer ici un gain solide à l’encodage positionnel ou à l’attention seule. »

**Pour lire les journaux :** la perte décrit l’apprentissage, mais le checkpoint est choisi avec le F1 validation. À l’oral, je présente le score retenu et la comparaison ; je ne lis pas chaque batch. Les avertissements techniques ne sont pas des métriques.


### Présenter les résultats — étape 18

**Phrase de présentation**

> Avec AraVec, C1 atteint 0,5290, C2 0,5362 et C3 0,5427 de F1 validation. Les trois architectures progressent par rapport à leurs embeddings aléatoires. C3 AraVec est le meilleur de cette famille pour la graine 42.

**Support 1 — Couverture AraVec**

> « AraVec retrouve 14 078 des 14 956 types du vocabulaire, soit environ 94,13 %. La couverture des occurrences du train est de 71,94 %, ce qui est une mesure différente. Un mot du vocabulaire couvert et une occurrence couverte ne doivent pas être confondus. »


### Présenter les résultats — étape 19

**Support 1 — Déséquilibre et optimisation du seuil**

> « Avec un seuil choisi sur validation, la perte pondérée atteint 0,5427 de F1, le sur-échantillonnage 0,5410 et la perte standard 0,5347. Le réglage du seuil améliore les trois résultats par rapport à 0,5. Dans cette expérience, le sur-échantillonnage ne dépasse pas la pondération. »


### Présenter les résultats — étape 20

**Phrase de présentation**

> AraBERT utilise son prétraitement officiel en conservant explicitement les répétitions. L’objectif est de respecter les attentes de l’encodeur sans effacer inutilement les indices expressifs.


### Présenter les résultats — étape 21

**Phrase de présentation**

> Avec AraBERT gelé, la moyenne masquée atteint 0,6316 de F1 validation contre 0,6054 pour CLS. La moyenne est meilleure dans cette comparaison ; les temps d’ajustement de la régression ne comprennent pas toute l’extraction des représentations.


### Présenter les résultats — étape 22

**Phrase de présentation**

> La tête BiLSTM avec attention sur AraBERT gelé atteint 0,6625 de F1 validation pour la graine 42. Elle dépasse ici les deux têtes linéaires, au prix d’une tête plus complexe.


### Présenter les résultats — étape 23

**Support 1 — Attention : tweet 1**

> « Ce 1e exemple montre comment la tête d’attention répartit son poids entre les tokens. Je repère les mots les plus mis en évidence, puis je les relis dans la phrase arabe. Un poids élevé indique une importance dans ce calcul ; il ne prouve pas que le modèle comprend le sarcasme. »

**Support 2 — Attention : tweet 2**

> « Ce 2e exemple montre comment la tête d’attention répartit son poids entre les tokens. Je repère les mots les plus mis en évidence, puis je les relis dans la phrase arabe. Un poids élevé indique une importance dans ce calcul ; il ne prouve pas que le modèle comprend le sarcasme. »

**Support 3 — Attention : tweet 3**

> « Ce 3e exemple montre comment la tête d’attention répartit son poids entre les tokens. Je repère les mots les plus mis en évidence, puis je les relis dans la phrase arabe. Un poids élevé indique une importance dans ce calcul ; il ne prouve pas que le modèle comprend le sarcasme. »

**Support 4 — Attention : tweet 4**

> « Ce 4e exemple montre comment la tête d’attention répartit son poids entre les tokens. Je repère les mots les plus mis en évidence, puis je les relis dans la phrase arabe. Un poids élevé indique une importance dans ce calcul ; il ne prouve pas que le modèle comprend le sarcasme. »

**Support 5 — Attention : tweet 5**

> « Ce 5e exemple montre comment la tête d’attention répartit son poids entre les tokens. Je repère les mots les plus mis en évidence, puis je les relis dans la phrase arabe. Un poids élevé indique une importance dans ce calcul ; il ne prouve pas que le modèle comprend le sarcasme. »

**Support 6 — Comparaison des cinq profils d’attention**

> « Les cinq panneaux présentent les profils d’attention des tweets précédents. Je compare la concentration des poids : certains tokens peuvent dominer, mais leur rôle doit être interprété avec le sens de chaque tweet. Cette visualisation complète les métriques sans constituer une explication causale. »

**À l’oral :** je lis chaque tweet arabe avant de commenter ses tokens. Je n’attribue pas une signification linguistique aux couleurs sans vérifier le contexte.


### Présenter les résultats — étape 24

**Phrase de présentation**

> E1 apprend seulement le sarcasme ; E2 partage un encodeur entre sarcasme et sentiment. La définition de l’architecture ne suffit pas à prouver que la tâche auxiliaire aide.


### Présenter les résultats — étape 25

**Phrase de présentation**

> Je teste lambda 0,3, 0,5 et 1 sur validation. Lambda règle le poids du sentiment ; je ne choisis jamais sa valeur à partir du test.


### Présenter les résultats — étape 26

**Phrase de présentation**

> Pour AraBERT graine 42, E1 atteint 0,6653 de F1 validation. E2 atteint 0,6731 avec lambda 0,3, contre 0,6718 avec 0,5 et 1. Le gain reste modeste ; les autres graines sont nécessaires pour le mettre en perspective.

**Pour lire les journaux :** la perte décrit l’apprentissage, mais le checkpoint est choisi avec le F1 validation. À l’oral, je présente le score retenu et la comparaison ; je ne lis pas chaque batch. Les avertissements techniques ne sont pas des métriques.


### Présenter les résultats — étape 27

**Support 1 — AraBERT, MARBERT, mono-tâche et multi-tâche**

> « Ce tableau rassemble les essais de fine-tuning. Pour la graine 42, AraBERT multi-tâche obtient son meilleur F1 validation avec lambda 0,3 : 0,6731. Pour MARBERT multi-tâche, lambda 0,5 atteint 0,6667. Les répétitions montrent une variabilité ; le choix final repose sur les moyennes de validation, jamais sur le test. »

**Pour lire les journaux :** la perte décrit l’apprentissage, mais le checkpoint est choisi avec le F1 validation. À l’oral, je présente le score retenu et la comparaison ; je ne lis pas chaque batch. Les avertissements techniques ne sont pas des métriques.


### Présenter les résultats — étape 28

**Phrase de présentation**

> Je sélectionne les modèles à partir de leurs résultats de validation. Cette cellule prépare les candidats finaux ; elle n’utilise pas les performances test pour décider.


### Présenter les résultats — étape 29

**Phrase de présentation**

> Avant de consulter les labels test, je verrouille les modèles et leurs seuils. Le manifeste documente cette séparation entre sélection et évaluation.

**Support 1 — Répétitions finales sur trois graines**

> « Chaque ligne correspond à une famille et une graine. Les seuils peuvent changer entre graines, car ils sont choisis sur validation. Ces répétitions permettent de présenter une moyenne et un écart-type, plutôt que de retenir seulement le run le plus favorable. »

**Pour lire les journaux :** la perte décrit l’apprentissage, mais le checkpoint est choisi avec le F1 validation. À l’oral, je présente le score retenu et la comparaison ; je ne lis pas chaque batch. Les avertissements techniques ne sont pas des métriques.


### Présenter les résultats — étape 30

**Support 1 — Décalage du taux de sarcasme**

> « Le sarcasme représente 17,29 % de la validation contre 27,37 % du test, soit environ 10,08 points de plus. Le test est donc différent de la validation ; un seuil réglé sur validation peut y être moins adapté. »

**Support 2 — Dialectes dans train, validation et test**

> « Le test contient 306 tweets égyptiens, mais seulement deux tweets maghrébins. Les performances par dialecte doivent être lues avec leurs effectifs ; ce corpus ne permet pas de valider sérieusement un déploiement sur des tweets marocains. »

**Support 3 — Sentiment et sarcasme dans les trois partitions**

> « Le sarcasme reste fréquent parmi les tweets négatifs du test, environ 42,28 %. Cependant, le taux des tweets neutres augmente à 14,17 %. La relation sentiment–sarcasme n’est donc pas parfaitement stable entre partitions. »

**Support 4 — Raccourcis dans le train**

> « Je rappelle ici les corrélations observées pendant l’apprentissage. Par exemple, les tweets avec URL sont moins souvent sarcastiques. Ce tableau sert de référence pour comprendre les écarts avec le test. »

**Support 5 — Raccourcis dans le test**

> « Dans le test, 3,64 % des tweets avec URL sont sarcastiques contre 31,45 % sans URL. Les mentions et les guillemets concernent très peu de tweets ; leurs taux doivent être interprétés avec prudence. »

**Support 6 — Effectifs maghrébins**

> « Je dispose de 35 tweets maghrébins dans le train et de deux dans le test. Je ne peux pas transformer ces effectifs en preuve de qualité sur l’arabe marocain ; un jeu de test adapté est nécessaire. »

**Support 7 — OOV sur le test**

> « Le taux de tokens inconnus passe de 21,16 % en validation à 27,71 % sur test. Le modèle statique rencontre donc davantage de mots qu’il ne sait pas représenter précisément, ce qui constitue un défi de généralisation. »


### Présenter les résultats — étape 31

**Support 1 — Scores test par modèle et par graine**

> « Ce tableau donne les performances sur le test officiel aux seuils déjà fixés. Il distingue les runs individuels : par exemple E1 graine 42 atteint 0,6248 et E2 graine 42 atteint 0,6239. Le classement global doit ensuite être examiné sur les trois graines. »

**Support 2 — Courbes du seuil et précision–rappel**

> « À gauche, l’axe horizontal est le seuil et l’axe vertical le F1 test ; les points repèrent les seuils choisis sur validation. À droite, les courbes montrent le compromis entre précision et rappel. Les courbes test servent au diagnostic après évaluation, pas à sélectionner un nouveau modèle. »

**Support 3 — Seuil validation et optimum test a posteriori**

> « Pour E2, le seuil validation est 0,4908 et le seuil optimal a posteriori sur test 0,2983. Le F1 passerait de 0,6239 à 0,6380, mais ce dernier chiffre exploite les labels test. Je conserve donc 0,6239 comme résultat officiel de ce run. »


### Présenter les résultats — étape 32

**Support 1 — Scores par dialecte et présence d’URL**

> « La baseline atteint un F1 de 0,5437 sans URL contre 0,0769 avec URL. Le second groupe ne contient que 16 tweets sarcastiques : il est difficile et son score est instable. Les lignes par dialecte et par URL sont des découpages différents ; elles ne doivent pas être additionnées. »

**Support 2 — Retrait des indices du sarcasme**

> « Je ne garde ici que les tweets sarcastiques initialement bien détectés. Après suppression des emojis, rires et ponctuations répétées, la détection baisse de 9,38 points pour A et de 4,62 points pour E2. Les modèles utilisent ces indices ; il ne s’agit pas d’une baisse du F1 global. »


### Présenter les résultats — étape 33

**Support 1 — Ablations par réentraînement**

> « Sans URL ni mentions, E1 passe de 0,6248 à 0,6027 sur test et E2 de 0,6239 à 0,6199 pour les runs comparés. Sans tweets égyptiens dans le train, le meilleur E atteint 0,7674 sur le test égyptien contre 0,7736 avec tout le train : la baisse est faible dans cet essai. Cela ne signifie pas que le dialecte était absent du préentraînement. »

**Pour lire les journaux :** la perte décrit l’apprentissage, mais le checkpoint est choisi avec le F1 validation. À l’oral, je présente le score retenu et la comparaison ; je ne lis pas chaque batch. Les avertissements techniques ne sont pas des métriques.


### Présenter les résultats — étape 34

**Support 1 — Intervalles bootstrap à 95 %**

> « Chaque intervalle est obtenu avec 1 000 rééchantillonnages du test. Pour E2 graine 42, le F1 est 0,6239 et l’intervalle va environ de 0,5982 à 0,6496. Cet intervalle décrit la sensibilité à l’échantillon test ; l’écart-type entre graines mesure une autre source de variation. Le simple chevauchement des intervalles ne remplace pas un test de différence. »


### Présenter les résultats — étape 35

**Support 1 — Test de McNemar**

> « E2 et D2 ont respectivement 151 et 156 exemples qu’ils réussissent seuls. La p-valeur vaut 0,8195 : au seuil de 5 %, ce test ne met pas en évidence de différence de leurs erreurs. Cela ne prouve pas que les deux modèles sont équivalents et ce n’est pas un test direct du F1. »


### Présenter les résultats — étape 36

**Support 1 — Lecture de 30 erreurs**

> « Je présente 15 faux négatifs, des sarcasmes manqués, et 15 faux positifs, des tweets non sarcastiques signalés à tort. Pour chaque ligne, je compare le texte, le label et la prédiction avant de proposer une explication. »

**Support 2 — Catégories provisoires d’erreurs**

> « Les catégories automatiques regroupent 19 erreurs sous « ironie sans marqueur ou contexte requis », six autour du dialecte/contexte et cinq autour de marqueurs possiblement mal interprétés. Ce sont des hypothèses à vérifier humainement, pas des causes démontrées. Aucune contradiction certaine n’est repérée dans cette sélection ; cela ne prouve pas l’absence générale de bruit d’annotation. »


### Présenter les résultats — étape 37

**Phrase de présentation**

> Je mesure ensuite la latence sur CPU avec un protocole commun. Ces journaux ne donnent pas encore la synthèse ; les valeurs comparables apparaissent dans le tableau final.


### Présenter les résultats — étape 38

**Support 1 — Qualité, coût et taille des modèles**

> « E2 a le meilleur F1 test moyen affiché, 0,6096 ± 0,0131, contre 0,6093 ± 0,0221 pour E1. Cet écart de 0,0003 est trop faible pour affirmer un avantage général du multi-tâche. TF-IDF atteint 0,5369 avec environ 2,15 ms par tweet sur le CPU mesuré, contre 116,63 ms et environ 541 Mo pour E2. Le temps d’ajustement du tableau ne représente pas tout le temps du projet : téléchargement et extraction gelée doivent être considérés séparément. »

**Support 2 — Courbes d’apprentissage C, D, E1 et E2**

> « Chaque panneau montre le F1 validation en fonction de l’époque et chaque courbe correspond à une graine. Je retiens le meilleur checkpoint de validation, pas automatiquement la dernière époque. Les variations entre courbes montrent pourquoi il faut répéter les expériences. »

**Support 3 — Matrice de confusion du modèle retenu**

> « Les lignes représentent les classes réelles et les colonnes les classes prédites. La diagonale contient les bonnes décisions, les deux autres cases les fausses alertes et les sarcasmes manqués. Le run retenu a une précision de 55,57 % et un rappel de 71,13 % pour le sarcasme : il détecte une majorité des sarcasmes, avec un nombre important de fausses alertes. »

**Support 4 — Comparaison graphique du F1 test**

> « La hauteur des barres indique le F1 test moyen. Les barres d’erreur des familles entraînées montrent l’écart-type entre trois graines, pas un intervalle bootstrap. B1 et B2 n’ont qu’une exécution : l’absence de barre d’erreur ne signifie pas absence d’incertitude. E1 et E2 sont très proches, devant D, C et A dans ce tableau. »


### Présenter les résultats — étape 39

**Phrase de présentation**

> Le modèle retenu sur validation est E2 AraBERT, lambda 0,3, graine 42, au seuil 0,4908. Son F1 test individuel est 0,6239, sa précision 0,5557 et son rappel 0,7113. Je distingue ce run de la moyenne sur trois graines, 0,6096. L’export annoncé concerne la session historique.


### Présenter les résultats — étape 40

**Support 1 — Contrôles de reproductibilité**

> « Les contrôles enregistrés indiquent des partitions disjointes, le bon effectif test et trois graines par famille finale. Ils décrivent la session T4 d’origine. Pour remettre le projet, je dois aussi récupérer les CSV de prédictions : un message enregistré ne certifie pas leur présence dans le dossier local. »


## Conclusion à présenter

« Les modèles préentraînés améliorent la détection du sarcasme dans cette expérience. AraVec aide les réseaux statiques et le fine-tuning d’AraBERT donne les meilleurs scores test moyens du tableau. Cependant, E1 et E2 restent presque à égalité : je ne peux pas affirmer un gain général du multi-tâche. La baseline TF-IDF garde un intérêt pour sa faible latence. Enfin, les différences entre dialectes, les indices superficiels et le manque de tweets maghrébins limitent la généralisation. Avant un usage sur des tweets marocains, je prévoirais une évaluation dédiée et une analyse humaine des erreurs. »


## Visualisations complémentaires

Ces graphiques sont calculés à partir des tableaux T4 enregistrés. Le code du notebook utilise les valeurs courantes lors d’une nouvelle exécution.

### Étape 6 — Indices textuels — train brut et test

Les différences de fréquence révèlent un changement de distribution. Les guillemets d’export, notamment, sont beaucoup plus fréquents dans le train.

### Étape 6 — Validation — taux de sarcasme par dialecte

Je compare les proportions avec les effectifs : cinq tweets maghrébins ne permettent pas de conclure à une absence de sarcasme dans ce dialecte.

### Étape 6 — Validation — sarcasme selon le sentiment

Les tweets négatifs sont plus souvent sarcastiques, mais la majorité ne l’est pas. Le sentiment est un indice, pas une définition du sarcasme.

### Étape 11 — TF-IDF — comparaison des variantes sur validation

La variante conservatrice obtient le meilleur F1 de cette comparaison. Le nettoyage agressif retire des informations utiles dans cet essai.

### Étape 18 — C1–C3 — embeddings aléatoires contre AraVec

AraVec améliore les trois architectures dans ces essais à la graine 42. C3 AraVec atteint le meilleur F1 validation de cette famille, 0,5427.

### Étape 19 — Déséquilibre — seuil 0,5 contre seuil validation

Choisir le seuil sur validation améliore les trois stratégies. Le sur-échantillonnage ne dépasse pas la perte pondérée dans cet essai.

### Étape 27 — Fine-tuning — F1 validation moyen par modèle

Chaque barre résume les graines disponibles du modèle. Les lambda non retenus n’ont qu’une graine : cette comparaison ne donne pas la même précision pour chaque barre.

### Étape 30 — Train, validation et test — décalage du taux de sarcasme

Le taux passe de 17,29 % en validation à 27,37 % sur test. Ce changement aide à comprendre le décalage des seuils optimaux.

### Étape 32 — Indices supprimés — baisse de détection sur les vrais positifs

Les barres mesurent une baisse de détection sur un ensemble fixé de sarcasmes bien détectés, pas une baisse de F1 global.

### Étape 33 — Ablations — référence contre modèle réentraîné

Chaque paire compare le même sous-ensemble test. La ligne sans égyptien porte seulement sur 306 tweets égyptiens ; elle ne doit pas être comparée directement aux autres lignes calculées sur tout le test.

### Étape 38 — Comparaison finale — qualité et latence CPU

Les modèles fine-tunés ont les meilleurs F1 moyens, mais E1 et E2 sont presque à égalité. La latence est analysée dans la visualisation suivante.

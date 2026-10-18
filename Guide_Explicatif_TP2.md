# Guide explicatif complet du TP2 : détection du sarcasme en arabe

**Omar Elkhali — BDIOT — travail individuel**  
Guide établi le **18 octobre 2026**, à partir du code et des sorties enregistrés localement.  
Dépôt : <https://github.com/OmarElkhali/TP2-ArSarcasm-v2>.

Ce document explique **chaque partie du projet et chaque cellule de code**. Il accompagne le compte rendu scientifique ; il ne remplace pas le sujet du TP. La numérotation désigne **l'ordre des cellules de type code**, pas le compteur d'exécution affiché entre crochets par Jupyter, qui peut changer lors d'une relance.

Le parcours principal suit les **104 cellules du notebook T4**, puis les **6 cellules du testeur**. Les deux autres fichiers présents sont également couverts par une correspondance cellule par cellule et l'explication de leurs différences. Les notebooks d'origine ne sont pas modifiés pour fabriquer ce guide et aucun de leurs calculs n'est exécuté par sa génération.

## Sommaire

- [1. Comprendre le problème et le protocole](#comprendre)
- [2. Architecture du projet et rôle des fichiers](#architecture)
- [3. Comprendre les modèles A à G](#modeles)
- [4. Comprendre les tenseurs, pertes et métriques](#notions)
- [5. Suivre les étapes sans confondre apprentissage et inférence](#execution)
- [6. Guide des 104 cellules du notebook principal](#cellules-principales)
- [7. Guide des 6 cellules du testeur de phrases](#cellules-testeur)
- [8. Anciennes versions : correspondance de toutes les cellules](#variantes)
- [9. Lire les résultats et diagnostiquer les problèmes](#diagnostic)
- [10. Vérification de la couverture du guide](#couverture)

<a id="comprendre"></a>
## 1. Comprendre le problème et le protocole

### 1.1 Que cherche-t-on à prédire ?

L'entrée est **uniquement le texte d'un tweet arabe**. La cible principale est `y` : `0` signifie non sarcastique et `1` sarcastique. Le sentiment `NEG/NEU/POS` est une **cible auxiliaire** pour E2 ; il n'est pas fourni à l'encodeur comme une information déjà connue lors d'une prédiction. Le dialecte sert à décrire et tester les sous-groupes, sans être ajouté aux entrées du classifieur principal.

Exemple : « يا سلام، الإنترنت انقطع مرة أخرى، خدمة رائعة فعلاً 😂 » exprime une louange alors que la situation est négative. Comprendre cette contradiction exige davantage que détecter un mot positif ou un emoji. À l'inverse, « لم يعجبني هذا الفيلم. » exprime une opinion négative sans nécessairement être sarcastique. L'étiquette attendue sur un exemple saisi reste une interprétation informelle tant qu'elle n'a pas été annotée indépendamment.

### 1.2 Pourquoi trois partitions ?

| Partition | Rôle | Peut-elle servir à entraîner les poids ? | Peut-elle choisir le seuil ? |
|---|---|---|---|
| Train | Ajuster vocabulaire, paramètres et poids de classes | Oui | Ce n'est pas sa fonction dans ce projet |
| Validation | Choisir configuration, lambda, checkpoint et seuil | Aucun gradient utilisé pour entraîner les modèles du TP | Oui |
| Test officiel | Évaluer après verrouillage et analyser la généralisation | Non | Non, sauf diagnostic **a posteriori** explicitement non déployable |

Le train officiel contient **12 548 tweets**, le test **3 000**. Le split stratifié à 10 % et graine 42 donne **11 293 exemples de train initial** et **1 255 de validation**. Après les exclusions de collisions, contradictions et doublons, le train utilisé contient **10 842 tweets**. Les exclusions peuvent se recouvrir : on ne doit pas additionner aveuglément les compteurs d'audit. Validation et test ne sont pas raccourcis pour accélérer les calculs.

« Test scellé » signifie ici que **ses labels** ne servent pas aux choix de développement avant F. Le texte test est consulté en amont pour détecter les collisions et décrire des indices non supervisés. Il ne s'agit donc pas d'un protocole où absolument aucun octet du test ne serait accessible avant F ; cette distinction doit rester explicite.

### 1.3 Pourquoi conserver des marqueurs ?

Le nettoyage n'est pas une course à la suppression. Rires, répétitions, négations, hashtags, emojis et ponctuation peuvent aider à interpréter une intention. Supprimer `!!!` ou 😂 peut enlever un indice. Une normalisation raisonnable corrige surtout des variantes de forme et des artefacts d'export. L'ablation agressive sert à **mesurer** ce qui se perd, pas à supposer qu'un texte plus vide produit de meilleurs scores.

### 1.4 Ce que signifie « reproductible »

On fixe les graines, les partitions, les versions et les révisions des modèles. On conserve la configuration, le checkpoint, le seuil et les probabilités associés à chaque expérience. Une graine seule ne suffit pas : matériel, précision numérique et certains kernels CUDA peuvent introduire des différences. Une sortie enregistrée atteste ce qui a été affiché ; elle ne garantit pas qu'un checkpoint distant existe encore après la fermeture de Colab.

<a id="architecture"></a>
## 2. Architecture du projet et rôle des fichiers

### 2.1 Organisation générale

```text
Tweets bruts
   └─ audit + split + nettoyage adapté
        ├─ A : TF-IDF → régression logistique
        ├─ B1 : prémisse + hypothèses → modèle NLI
        ├─ B2 : modèle de sentiment → règle « négatif »
        ├─ C : tokens → embeddings → C1 / C2 / C3
        ├─ D : AraBERT gelé → CLS / moyenne / BiLSTM-attention
        └─ E : AraBERT ou MARBERT ajusté → une ou deux têtes
             ↓
      Validation : checkpoint + seuil + configuration
             ↓
      Modèles verrouillés → F : test et robustesse
             ↓
      G : qualité, incertitude, coût, décision
             ↓
      Checkpoint + registre → testeur de nouvelles phrases
```

| Élément | Contenu et utilité | À ne pas confondre |
|---|---|---|
| `notebooks/TP2_Sarcasme_Arabe_T4.ipynb` | Parcours pédagogique principal A–G, 104 cellules de code | Son code courant prévoit au maximum 5 époques PLM |
| `notebooks/Tester_Phrases.ipynb` | Saisie de phrases et inférence des checkpoints existants | N'entraîne aucun modèle et ne crée pas de checkpoint manquant |
| `notebooks/TP2_Sarcasme_Arabe_T4_Execution.ipynb` | Copie préservée d'une exécution antérieure complète | Budget PLM de 3 époques ; ses résultats ne sont pas ceux de la nouvelle exécution |
| `notebooks/TP2_Sarcasme_Arabe.ipynb` | Ancienne variante présente localement | Pas le fichier de référence T4 ; sa correspondance est donnée plus bas |
| `src/data.py` | Préparation/vérification des données locales | Le notebook expose aussi le code expérimental pour la lecture |
| `src/preprocess.py` | Nettoyage, normalisation et expressions régulières | Doit rester cohérent avec le prétraitement appris par le checkpoint |
| `src/models.py` | `AttentionPool`, `StaticNet`, `FrozenHead`, `FineTuneNet` | Définit des architectures ; importer une classe ne l'entraîne pas |
| `src/train.py` | Pas d'optimisation et fonction d'époque exportés du notebook | N'est pas à lui seul un script qui lance tout le TP |
| `src/evaluate.py` | Métriques et recherche du seuil | Un score ne doit jamais être associé au seuil d'un autre run |
| `src/sync.py` | Exporte certaines définitions du notebook vers `src/` | Ne lance pas le notebook ; peut remplacer des modifications manuelles dans les modules générés |
| `src/predict.py` | Classe `PhraseTester`, chargement et inférence de chaque famille | Réutilise les poids ajustés ; les poids pré-entraînés seuls ne suffisent pas pour E |
| `src/colab_resources.py`, `src/assets/` | Téléchargement et vérification des ressources officielles | Télécharger des poids n'est pas les ajuster sur ArSarcasm |
| `src/report.py` | Reconstruit le rapport à partir de sources enregistrées précises | Le générateur actuel cible l'exécution archivée ; une nouvelle sauvegarde ne met pas automatiquement le rapport à jour |
| `data/raw/` | CSV officiels et manifeste | Données non suivies dans Git, à préparer lors d'un nouveau clonage |
| `data/processed/` | Train/validation après traitement | Ne pas les remplacer par un échantillon pour accélérer silencieusement |
| `models/` | Poids pré-entraînés, poids ajustés, tokenizers, vocabulaire et caches | Ressources volumineuses exclues de Git ; à sauvegarder séparément |
| `results/registry.json` | Clé → architecture, configuration, chemin et seuil | Doit provenir de la même session que les checkpoints utilisés |
| `results/results.csv` | Mesures enregistrées localement | Peut être un historique CPU différent des sorties T4 du notebook sauvegardé |
| `results/history/` | Archives locales des exécutions remplacées | Dossier exclu du dépôt public |
| `results/report_data/` | Tables/figures extraites pour le rapport précédent | Elles restent liées à leur manifeste de provenance |
| `predictions/` | Probabilités val, métadonnées, CSV test s'ils ont été récupérés | Un tableau affiché dans VS Code ne rapatrie pas les CSV du serveur Colab |
| `figures/` | PNG sauvegardés par les cellules de tracé | Une image inline vide peut coexister avec une figure sauvegardée ailleurs |
| `tests/` | Contrôles de structure, configurations, lecteur de phrases et rapport | Vérifier un contrôle léger ne signifie pas avoir réentraîné un PLM |

### 2.2 Où se trouvent réellement les fichiers ?

Le noyau exécute le Python du **serveur sélectionné**, pas nécessairement celui du PC. Avec Colab T4 dans VS Code, l'interface est locale mais le calcul et les écritures de fichiers sont distants, généralement sous `/content/TP2_ArSarcasme`. Sauvegarder le notebook avec Ctrl+S conserve ses sorties dans le fichier local ouvert ; cela ne télécharge pas `models/`, `predictions/` ou `results/` depuis Colab.

En Conda local, `ROOT` doit pointer vers ce projet, sur le disque Windows. `TP2_ROOT` permet un choix explicite. Les sous-dossiers utiles restent dans cette racine ; aucune dépendance à un autre projet n'est requise. Un clone Git ne contient pas les gigaoctets de poids ignorés par Git.

<a id="modeles"></a>
## 3. Comprendre les modèles A à G

### A — Construire une baseline solide

Le TF-IDF représente un tweet par des fréquences pondérées de mots et caractères. Les mots 1–2 capturent un mot et des paires de mots ; les caractères 2–5 capturent notamment des fragments et variantes orthographiques. Les fréquences fréquentes dans le corpus sont moins discriminantes grâce à l'IDF. `sublinear_tf=True` réduit l'impact d'un terme répété par une transformation logarithmique.

`FeatureUnion` concatène les deux familles de caractéristiques ; la régression logistique apprend un poids par caractéristique et produit un score de classe. `class_weight='balanced'` rend les erreurs sur la minorité plus coûteuses. Cette méthode sert de référence de qualité et de coût ; un réseau beaucoup plus lourd doit apporter un gain réel pour justifier son utilisation.

### B1 — Classifier à partir d'un modèle NLI

NLI signifie inférence en langage naturel. On compare le tweet à deux hypothèses, sarcastique et sérieuse. Le modèle a déjà appris, sur d'autres corpus, à comparer prémisse et hypothèse. Le code repère l'indice `entailment` dans `id2label`, puis normalise les deux scores de cette classe. Il ne s'agit pas d'entraîner le modèle sur nos tweets ; le choix du prompt et du seuil sur validation reste néanmoins une forme d'adaptation au problème.

L'auto-attention permet aux tokens des deux segments de s'influencer au sein de l'encodage conjoint. L'hypothèse et sa langue peuvent modifier les scores ; les formulations arabe, française et anglaise sont donc comparées.

### B2 — Utiliser le sentiment comme heuristique

Le modèle de sentiment renvoie trois scores. On choisit la classe au logit maximal : si elle est négative, la règle prédit sarcastique. **Cette règle n'est pas `P(négatif) ≥ 0,5`.** Exemple : `[positif=0,30 ; négatif=0,40 ; neutre=0,30]` suffit pour prédire négatif par argmax, alors que 0,40 est inférieur à 0,5.

L'AP doit utiliser le score continu `P(négatif)` ; précision/rappel/F1 doivent utiliser la règle argmax. Le modèle de sentiment peut avoir vu des corpus qui recouvrent l'origine d'ArSarcasm : cette limite est discutée dans le rapport et ne disparaît pas avec un bon score.

### C — Réseaux avec embeddings statiques

Une table `Embedding` transforme chaque identifiant de token en vecteur de 100 valeurs. La même occurrence lexicale reçoit initialement le même vecteur, contrairement aux représentations contextuelles BERT.

| Architecture | Chemin du calcul | Comment le tweet devient un vecteur |
|---|---|---|
| C1 | Embedding 100D → BiLSTM, 64 unités par direction | Max-pooling masqué sur les 128 dimensions |
| C2 | Embedding 100D → même BiLSTM | Attention apprise sur les positions valides |
| C3 | Embedding 100D → projection 128D → deux couches Transformer à quatre têtes | Moyenne masquée des états ; positions sinusoïdales ou ablation sans positions |

Un BiLSTM lit la séquence dans les deux sens. Les deux états de 64 dimensions sont concaténés en un état de 128 dimensions par position. `pack_padded_sequence` empêche les positions de remplissage d'être prises pour de vrais tokens par la récurrence arrière.

Les embeddings sont soit aléatoires, soit initialisés depuis AraVec Twitter. Lors de l'entraînement AraVec, ils sont d'abord figés, puis dégelés à partir de la troisième époque si l'exécution l'atteint. **Le meilleur checkpoint peut être celui de la première ou deuxième époque**, donc avant le dégel : on ne doit pas lui attribuer des améliorations qu'il n'a pas apprises.

### D — Encodeur pré-entraîné gelé

AraBERT transforme chaque token en représentation contextualisée, mais ses poids restent figés et le modèle est en `eval()`. D1 utilise soit l'état du token spécial CLS, soit la moyenne des tokens valides, puis apprend une régression logistique. D2 garde tous les états et entraîne une tête BiLSTM + attention. Il faut encore l'encodeur à l'inférence : « gelé » signifie qu'il n'est pas ajusté, pas qu'il est supprimé du pipeline.

Le cache des états est écrit en float16 sur disque pour limiter la mémoire ; les blocs nécessaires sont reconvertis en float32 pour les têtes. Cette quantification existe dans le protocole numérique et doit être reproduite par le lecteur de phrases pour D. Le cache n'est pas une nouvelle source de données et ne réentraîne pas AraBERT.

### E — Fine-tuning mono-tâche et multi-tâche

Les couches utiles de l'encodeur reçoivent cette fois des gradients. Le résumé est le **CLS brut**, `last_hidden_state[:, 0]`, après dropout. E1 ajoute une tête à deux logits sarcasme. E2 ajoute une seconde tête à trois logits sentiment ; l'encodeur reçoit la somme des signaux des deux tâches.

`L = L_sarcasme + lambda × L_sentiment`, pour lambda dans `{0,3 ; 0,5 ; 1}`. La perte sarcasme est pondérée ; la perte sentiment ne l'est pas. Lambda règle l'influence de l'auxiliaire, pas le seuil de classification. Une tâche auxiliaire peut aider par régularisation ou nuire en renforçant un raccourci sentiment/sarcasme.

Pour chaque encodeur, la grille à graine 42 contient E1 et trois E2. Le meilleur lambda sur validation et E1 sont ensuite répétés à 43/44. Cela donne **8 runs E par encodeur, donc 16 pour AraBERT + MARBERT**, sans compter les réentraînements F.

### F — Tester la généralisation

| Test | Question | Lecture correcte |
|---|---|---|
| F1 | Le taux de sarcasme et le seuil optimal changent-ils ? | Le meilleur seuil test est un diagnostic a posteriori, pas un nouveau réglage exploitable |
| F2 | Que se passe-t-il sans tweets `egypt` dans l'apprentissage ? | Réentraîner le meilleur E sur le train non égyptien, sélectionner aussi sans `egypt`, évaluer le sous-groupe test égyptien |
| F3 | URL/mentions servent-elles de raccourcis ? | Distinguer évaluation avec/sans URL et véritable réentraînement sans ces indices |
| F4 | Les marqueurs explicites aident-ils ? | Retirer des indices des **vrais positifs initiaux**, conserver le seuil, mesurer la part encore détectée |
| F5 | Les écarts sont-ils stables ? | Trois graines, bootstrap de 1 000 échantillons, McNemar exact pour deux modèles |
| F6 | Pourquoi les modèles se trompent-ils ? | Relire 15 FN et 15 FP ; les catégories automatiques restent provisoires |

Les sous-groupes peuvent avoir des proportions de positifs et des tailles très différentes. Le F1 d'un groupe minuscule, en particulier `magreb`, ne permet pas une conclusion robuste sur la darija marocaine. Une absence de doublon contradictoire ne démontre pas l'absence de bruit d'annotation.

### G — Choisir un candidat, pas seulement le meilleur nombre

Le F1 test, son écart-type, l'AP, la robustesse, le temps, la taille des poids et la latence CPU sont mis en regard. Le choix du candidat scientifique vient de la validation, pour éviter une sélection sur test. Les latences sont mesurées sur le **CPU du serveur qui exécute G** : après entraînement T4, cela reste le CPU Colab et non le Ryzen du PC.

Les temps de tête D excluent l'extraction gelée ; les temps d'ajustement excluent les téléchargements. Une taille de tête minuscule n'est pas la taille du système complet si AraBERT doit aussi être chargé.

<a id="notions"></a>
## 4. Comprendre les tenseurs, pertes et métriques

### 4.1 Dimensions à savoir lire

`N` : nombre de tweets ; `B` : taille du lot ; `L` : longueur ; `H` : dimension cachée du PLM, 768 ici pour les encodeurs BERT concernés.

| Objet | Forme | Type / sens |
|---|---|---|
| `XW_TRAIN` | `(N, 64)` | Entiers des tokens statiques ; PAD=0, UNK=1 |
| `input_ids` | `(N, 64)` | Entiers issus du tokenizer propre au PLM |
| `attention_mask` | `(N, 64)` | 1 aux tokens valides, 0 aux PAD ; inclut les tokens spéciaux valides |
| Sortie embedding C | `(B, L, 100)` | Vecteurs float |
| États BiLSTM C | `(B, L, 128)` | Deux directions de 64 dimensions |
| États BERT | `(B, L, H)` | Vecteur contextualisé par token |
| Poids d'attention | `(B, L)` | Somme proche de 1 sur les positions valides |
| Logits sarcasme | `(B, 2)` | Scores bruts, pas encore des probabilités |
| Logits sentiment E2 | `(B, 3)` | Scores bruts des cibles auxiliaires |
| `YTRAIN` | `(N,)` | Entiers 0/1, `torch.long` pour CrossEntropyLoss |
| `STRAIN` | `(N,)` | Entiers NEG=0, NEU=1, POS=2 dans **ce** modèle multi-tâche |
| Probabilités sarcasme | `(N,)` | `softmax(logits)[:,1]`, ordre des tweets restauré après regroupement |

**Attention :** le codage NEG=0 de `SENT_MAP` n'est pas nécessairement celui du modèle CAMeLBERT de B2. Pour B2 et NLI, les indices sont retrouvés dans la configuration du modèle, pas supposés à partir de `SENT_MAP`.

### 4.2 Exemple de padding et masque

```text
Token IDs : [CLS, mot1, mot2, SEP, PAD, PAD]
Masque    : [  1,    1,    1,   1,   0,   0]
```

Un pooling moyen doit exclure les PAD. Dans l'attention, on donne aux positions PAD un très grand score négatif avant softmax. Dans E, si tous les tweets du lot tiennent sur 24 positions, on coupe les 40 colonnes PAD inutiles avant le forward, tout en conservant les positions des vrais tokens. Le stockage peut rester `(N,64)` et le calcul utiliser une longueur de lot plus courte : c'est le **padding dynamique**.

### 4.3 Du logit à la décision

Pour les deux logits `z0,z1`, softmax transforme les scores en valeurs positives dont la somme vaut 1. On retient `p = exp(z1)/(exp(z0)+exp(z1))`. La décision habituelle du TP est ensuite `p >= seuil_validation`. Ce seuil n'est pas obligatoirement 0,5. Un score élevé signifie que le modèle classe fortement la phrase comme sarcastique selon ses représentations ; il n'est pas une garantie de justesse ou de calibration sur une nouvelle population.

### 4.4 Perte et apprentissage

`CrossEntropyLoss` reçoit les logits bruts et la classe attendue, sans softmax externe préalable. Elle pénalise une faible masse sur la classe correcte. Pour la classe `c`, le poids équilibré est `N/(2 × n_c)` : la minorité reçoit un poids plus grand. Le sur-échantillonnage agit autrement, en répétant davantage de lignes minoritaires. Le déplacement de seuil agit après calcul du score. Ces trois mécanismes ne sont donc pas interchangeables.

Une itération suit : remise à zéro des gradients → forward → perte → backward → clipping → pas AdamW → scheduler. Une époque parcourt tous les lots définis pour ce run. Le clipping borne la **norme du gradient**, pas chaque poids à 1. Le warmup augmente d'abord le taux d'apprentissage, puis la planification linéaire le réduit.

Sur T4, `autocast` utilise la précision mixte pour certaines opérations. `GradScaler` multiplie la perte avant backward pour limiter les sous-flux, puis rétablit l'échelle avant clipping. Les poids ne sont pas tous définitivement convertis en float16. Si un overflow fait sauter le pas d'optimiseur, le scheduler ne doit pas progresser comme si ce pas avait réussi.

### 4.5 Métriques : petit calcul manuel

Soit `y=[1,1,0,0]`, `p=[0.8,0.4,0.6,0.1]`, seuil=0,5. Les décisions sont `[1,0,1,0]`. On obtient TP=1, FN=1, FP=1, TN=1. Précision=1/2, rappel=1/2, F1=0,5, accuracy=2/4. Avec un seuil de 0,75, une seule alerte reste : précision=1, rappel=1/2, F1=2/3. Cet exemple explique l'intérêt du seuil ; sur le vrai projet, on le choisit sur **validation**, pas sur les phrases de démonstration.

- **Précision** `TP/(TP+FP)` : parmi les alertes, combien sont correctes ?
- **Rappel** `TP/(TP+FN)` : parmi les sarcastiques réels, combien sont retrouvés ?
- **F1 sarcastique** `2TP/(2TP+FP+FN)` : compromis pour la classe minoritaire.
- **Macro-F1** : moyenne du F1 des deux classes, sans pondération par effectif.
- **Accuracy** : proportion totale correcte ; elle peut être élevée avec une classe majoritaire toujours prédite.
- **AP** : average precision sur le classement des scores. Le projet l'appelle PR-AUC par convention ; ce n'est pas l'aire trapézoïdale calculée sur une grille.

Le bootstrap rééchantillonne les tweets **avec remise**, à modèle et seuil fixés. Il ne réentraîne pas 1 000 modèles. Les trois graines mesurent une autre variabilité : celle de l'apprentissage. McNemar compare les désaccords de correction sur des observations appariées, pas directement la différence de F1.

<a id="execution"></a>
## 5. Suivre les étapes sans confondre apprentissage et inférence

| Cellules principales | Étape | Conséquence lors d'une exécution réelle |
|---|---|---|
| 001–006 | Noyau, chemins, matériel et paramètres | Installation/téléchargement préparatoire possible dans Colab |
| 007–019 | Nettoyage, split et audit | Lecture/écriture de données et graphiques |
| 020–026 | Métriques et suivi | Définitions et référence triviale |
| 027–029 | A | Ajustement TF-IDF/logistique si aucun cache compatible |
| 030–038 | B | Inférence des PLM directs, sans ajustement sur ArSarcasm |
| 039–050 | Outils C et boucle commune | Définitions ; `def train_neural` ne lance pas l'apprentissage à lui seul |
| 051–056 | C | Entraînement de réseaux statiques et ablation de déséquilibre |
| 057–067 | D | Extraction PLM gelée et entraînement des têtes |
| 068–076 | E | Nettoyage mémoire puis fine-tunings complets |
| 077–080 | Sélection finale | Répétitions A/C/D possibles et verrouillage |
| 081–088 | F1/F3/F4 | Ouverture des labels test et nouvelles inférences |
| 089–092 | F2/F3 | **Réentraînements réels**, pas seulement des filtres de résultats |
| 093–097 | F5/F6 | Statistiques et analyse d'erreurs |
| 098–104 | G | Inférence CPU, graphiques, exports et contrôles |
| Testeur 1–6 | Utilisation des poids existants | Préparation puis inférence ; aucun entraînement |

Relancer un bloc d'initialisation comme `RESULTS=[]`, `e_keys=[]` ou `comparison=[]` peut effacer des variables **en mémoire**, même si les fichiers restent présents. Une cellule qui définit une fonction peut être rapide et silencieuse ; une cellule qui appelle une boucle de fine-tuning peut durer longtemps. Le cache n'évite un calcul que s'il est compatible avec données, configuration, checkpoint et contexte.

Ce guide contient les sources à titre de lecture. **Il ne faut pas exécuter tous ses blocs de code les uns après les autres hors du notebook** : ils dépendent du même noyau et de l'ordre des cellules. Les longues archives base64 sont abrégées volontairement dans les extraits ; leur contenu original reste dans les notebooks. Les résultats affichés dans les notes pédagogiques sont des repères ; les chiffres d'un run doivent être lus dans ses propres sorties, et non repris d'une autre version.

<a id="cellules-principales"></a>
## 6. Guide des 104 cellules du notebook principal

Chaque fiche contient l'explication pédagogique, une lecture complémentaire précise et le code correspondant. « Résultat attendu » décrit l'effet du code ; ce n'est pas une preuve d'exécution faite pendant la création du guide.


### Index rapide des cellules principales

| Numéro | Accéder directement à la fiche |
|---|---|
| 001 | [Cellule de code 001 — Importer les outils de base](#tp2-001) |
| 002 | [Cellule de code 002 — Charger les outils NLP et fixer les chemins](#tp2-002) |
| 003 | [Cellule de code 003 — Choisir le matériel et les paramètres communs](#tp2-003) |
| 004 | [Cellule de code 004 — Définir les budgets d’époques et la compatibilité des caches](#tp2-004) |
| 005 | [Cellule de code 005 — Définir la fonction de fixation des graines](#tp2-005) |
| 006 | [Cellule de code 006 — Initialiser la session et vérifier le noyau Conda](#tp2-006) |
| 007 | [Cellule de code 007 — Préparer les règles de normalisation de l’arabe](#tp2-007) |
| 008 | [Cellule de code 008 — Retirer les artefacts d’export avec export_clean](#tp2-008) |
| 009 | [Cellule de code 009 — Construire une clé pour détecter les doublons](#tp2-009) |
| 010 | [Cellule de code 010 — Préparer le texte selon la famille de modèles](#tp2-010) |
| 011 | [Cellule de code 011 — Définir la suppression contrôlée des indices](#tp2-011) |
| 012 | [Cellule de code 012 — Lire le train et créer la validation imposée](#tp2-012) |
| 013 | [Cellule de code 013 — Écarter les collisions et les doublons du train](#tp2-013) |
| 014 | [Cellule de code 014 — Sauvegarder l’audit et afficher les tweets bruts](#tp2-014) |
| 015 | [Cellule de code 015 — Décrire les marqueurs à compter](#tp2-015) |
| 016 | [Cellule de code 016 — Extraire les indicateurs de surface](#tp2-016) |
| 017 | [Cellule de code 017 — Mesurer les indices et comparer deux nettoyages](#tp2-017) |
| 018 | [Cellule de code 018 — Tester les invariants et décrire les sous-groupes](#tp2-018) |
| 019 | [Cellule de code 019 — Visualiser la prévalence par dialecte](#tp2-019) |
| 020 | [Cellule de code 020 — Définir les métriques de classification](#tp2-020) |
| 021 | [Cellule de code 021 — Choisir un seuil sur la validation](#tp2-021) |
| 022 | [Cellule de code 022 — Initialiser le journal et identifier les données](#tp2-022) |
| 023 | [Cellule de code 023 — Enregistrer une expérience de validation](#tp2-023) |
| 024 | [Cellule de code 024 — Contrôler la reprise d’un checkpoint](#tp2-024) |
| 025 | [Cellule de code 025 — Uniformiser la sauvegarde des figures](#tp2-025) |
| 026 | [Cellule de code 026 — Évaluer le classifieur jamais sarcastique](#tp2-026) |
| 027 | [Cellule de code 027 — Construire la baseline TF-IDF et régression logistique](#tp2-027) |
| 028 | [Cellule de code 028 — Exécuter ou reprendre une variante TF-IDF](#tp2-028) |
| 029 | [Cellule de code 029 — Comparer les baselines et les règles superficielles](#tp2-029) |
| 030 | [Cellule de code 030 — Déclarer les modèles imposés et les hypothèses NLI](#tp2-030) |
| 031 | [Cellule de code 031 — Définir un chargement local des PLM](#tp2-031) |
| 032 | [Cellule de code 032 — Identifier une inférence mise en cache](#tp2-032) |
| 033 | [Cellule de code 033 — Vérifier la validité d’un cache de scores](#tp2-033) |
| 034 | [Cellule de code 034 — Enregistrer les métadonnées d’un cache d’inférence](#tp2-034) |
| 035 | [Cellule de code 035 — Calculer le score zero-shot par NLI](#tp2-035) |
| 036 | [Cellule de code 036 — Comparer les trois langues d’hypothèse](#tp2-036) |
| 037 | [Cellule de code 037 — Prédire le sentiment et construire la règle B2](#tp2-037) |
| 038 | [Cellule de code 038 — Évaluer la règle de sentiment sur validation](#tp2-038) |
| 039 | [Cellule de code 039 — Définir une tokenisation Unicode pour les réseaux statiques](#tp2-039) |
| 040 | [Cellule de code 040 — Associer nettoyage conservateur et tokenisation](#tp2-040) |
| 041 | [Cellule de code 041 — Apprendre le vocabulaire sur le train uniquement](#tp2-041) |
| 042 | [Cellule de code 042 — Encoder les listes de tokens en tenseurs](#tp2-042) |
| 043 | [Cellule de code 043 — Mesurer les inconnus et la troncature](#tp2-043) |
| 044 | [Cellule de code 044 — Préparer entrées, cibles et poids de classes](#tp2-044) |
| 045 | [Cellule de code 045 — Implémenter le pooling attentionnel](#tp2-045) |
| 046 | [Cellule de code 046 — Construire C1, C2 et C3 dans une même classe](#tp2-046) |
| 047 | [Cellule de code 047 — Définir une inférence neuronale par lots](#tp2-047) |
| 048 | [Cellule de code 048 — Décrire les opérations d’une époque d’apprentissage](#tp2-048) |
| 049 | [Cellule de code 049 — Assembler les lots et gérer leur échantillonnage](#tp2-049) |
| 050 | [Cellule de code 050 — Orchestrer entraînement, validation et checkpoint](#tp2-050) |
| 051 | [Cellule de code 051 — Comparer les réseaux à embeddings aléatoires](#tp2-051) |
| 052 | [Cellule de code 052 — Intégrer la normalisation officielle AraVec](#tp2-052) |
| 053 | [Cellule de code 053 — Identifier le cache réduit des embeddings AraVec](#tp2-053) |
| 054 | [Cellule de code 054 — Construire la matrice AraVec et mesurer sa couverture](#tp2-054) |
| 055 | [Cellule de code 055 — Comparer C1/C2/C3 avec AraVec](#tp2-055) |
| 056 | [Cellule de code 056 — Isoler l’effet du traitement du déséquilibre](#tp2-056) |
| 057 | [Cellule de code 057 — Configurer le préprocesseur officiel AraBERT Twitter](#tp2-057) |
| 058 | [Cellule de code 058 — Adapter les textes à chaque encodeur](#tp2-058) |
| 059 | [Cellule de code 059 — Réutiliser les tokenizers en mémoire](#tp2-059) |
| 060 | [Cellule de code 060 — Transformer les tweets en entrées BERT](#tp2-060) |
| 061 | [Cellule de code 061 — Extraire les états gelés et les stocker sur disque](#tp2-061) |
| 062 | [Cellule de code 062 — Obtenir les représentations de train et val pour D](#tp2-062) |
| 063 | [Cellule de code 063 — Comparer résumé CLS et moyenne masquée](#tp2-063) |
| 064 | [Cellule de code 064 — Apprendre les deux classifieurs linéaires D1](#tp2-064) |
| 065 | [Cellule de code 065 — Définir une tête BiLSTM attention sur états BERT](#tp2-065) |
| 066 | [Cellule de code 066 — Apprendre D2 et sélectionner cinq tweets à visualiser](#tp2-066) |
| 067 | [Cellule de code 067 — Afficher et exporter l’attention sur cinq tweets](#tp2-067) |
| 068 | [Cellule de code 068 — Demander le nettoyage des objets Python inutilisés](#tp2-068) |
| 069 | [Cellule de code 069 — Définir le fine-tuning mono et multi-tâche](#tp2-069) |
| 070 | [Cellule de code 070 — Préparer le suivi des expériences E](#tp2-070) |
| 071 | [Cellule de code 071 — Définir la grille mono/multi pour un encodeur](#tp2-071) |
| 072 | [Cellule de code 072 — Définir les répétitions finales d’un encodeur](#tp2-072) |
| 073 | [Cellule de code 073 — Exécuter la grille AraBERT Twitter](#tp2-073) |
| 074 | [Cellule de code 074 — Répéter les deux configurations finales AraBERT](#tp2-074) |
| 075 | [Cellule de code 075 — Exécuter la grille MARBERTv2](#tp2-075) |
| 076 | [Cellule de code 076 — Répéter MARBERT et récapituler E](#tp2-076) |
| 077 | [Cellule de code 077 — Retrouver une ligne de résultat à partir de sa clé](#tp2-077) |
| 078 | [Cellule de code 078 — Choisir les configurations finales sans consulter le test](#tp2-078) |
| 079 | [Cellule de code 079 — Compléter les trois graines de A, C et D](#tp2-079) |
| 080 | [Cellule de code 080 — Verrouiller modèles et seuils avant le test](#tp2-080) |
| 081 | [Cellule de code 081 — Ouvrir le test officiel en partie F](#tp2-081) |
| 082 | [Cellule de code 082 — Compléter l’audit post-test et extraire ses états gelés](#tp2-082) |
| 083 | [Cellule de code 083 — Recharger un modèle enregistré](#tp2-083) |
| 084 | [Cellule de code 084 — Uniformiser la prédiction entre les familles](#tp2-084) |
| 085 | [Cellule de code 085 — Évaluer les modèles finaux et exporter leurs prédictions](#tp2-085) |
| 086 | [Cellule de code 086 — Tracer l’effet du seuil sur le test à titre diagnostique](#tp2-086) |
| 087 | [Cellule de code 087 — Présenter et sauvegarder le diagnostic de seuil](#tp2-087) |
| 088 | [Cellule de code 088 — Évaluer les sous-groupes et retirer les indices des vrais positifs](#tp2-088) |
| 089 | [Cellule de code 089 — Fixer le meilleur modèle E pour l’ablation dialectale](#tp2-089) |
| 090 | [Cellule de code 090 — Réentraîner A, C et D sans URL ni mention](#tp2-090) |
| 091 | [Cellule de code 091 — Réentraîner E sans egypt et sans liens](#tp2-091) |
| 092 | [Cellule de code 092 — Rassembler les résultats des réentraînements de robustesse](#tp2-092) |
| 093 | [Cellule de code 093 — Calculer les intervalles bootstrap à 95 %](#tp2-093) |
| 094 | [Cellule de code 094 — Comparer deux modèles avec McNemar exact](#tp2-094) |
| 095 | [Cellule de code 095 — Exporter le test statistique](#tp2-095) |
| 096 | [Cellule de code 096 — Prélever les erreurs du modèle retenu](#tp2-096) |
| 097 | [Cellule de code 097 — Afficher les erreurs et leurs catégories provisoires](#tp2-097) |
| 098 | [Cellule de code 098 — Imposer le CPU pour la comparaison de latence](#tp2-098) |
| 099 | [Cellule de code 099 — Mesurer qualité, coût et latence des familles entraînées](#tp2-099) |
| 100 | [Cellule de code 100 — Mesurer le coût des deux méthodes directes](#tp2-100) |
| 101 | [Cellule de code 101 — Exporter la comparaison et visualiser l’apprentissage](#tp2-101) |
| 102 | [Cellule de code 102 — Présenter les erreurs globales et préparer la décision](#tp2-102) |
| 103 | [Cellule de code 103 — Sauvegarder la décision et le journal final](#tp2-103) |
| 104 | [Cellule de code 104 — Consigner les versions et vérifier les livrables calculés](#tp2-104) |



### Préparation du noyau et des ressources

<a id="tp2-001"></a>

#### Cellule de code 001 — Importer les outils de base

**Objectif.** Préparer les bibliothèques utilisées dans tout le projet, sans encore lire le corpus ni entraîner de modèle.

**Comprendre le code.** `Path` représente les chemins de fichiers. `numpy` manipule des tableaux numériques ; `pandas` organise les tweets en tables ; `matplotlib` dessine les figures. PyTorch fournit les tenseurs, les couches et les lots de données. `Counter` servira à compter les tokens du train. Les variables `USE_TF` et `USE_FLAX` désactivent les autres moteurs avant le chargement de Transformers.

**Résultat attendu à l’exécution.** Ces objets deviennent disponibles dans le noyau Python. Une cellule d’import réussie peut ne produire aucune sortie visible.

**À retenir.** Importer PyTorch ne signifie pas entraîner BERT. Une erreur `ModuleNotFoundError` indique un problème de noyau ou de dépendance, pas un défaut du jeu de données.

**Optimisation CPU ajoutée.** Six threads intra-op sont configurés avant NumPy/PyTorch via OMP et MKL ; la précision reste float32. Ce réglage de threads concerne le CPU ; le choix CUDA et AMP est effectué dans les cellules suivantes. Aucun BF16 n’est imposé.

**Préparation automatique dans Colab.** Avant les imports scientifiques, la cellule vérifie les versions des compléments du TP (`emoji`, `arabert`, Gensim, Transformers, etc.). Elle installe seulement ceux qui manquent ou diffèrent des versions prévues, avec le Python du noyau courant. Elle ne demande aucune installation de PyTorch, CUDA, NumPy, pandas ou scikit-learn. Après une installation, elle s'arrête volontairement avec le message « Installation terminée » : redémarrer le noyau pour ne pas mélanger des modules déjà importés et de nouveaux fichiers. Lors de la relance, les versions conformes ne sont pas réinstallées. Ce bloc ne s'exécute pas dans le Conda local. Aucun modèle n'est entraîné et aucun poids de PLM n'est chargé par cette préparation.

**Compatibilité Python 3.13.** Le runtime sélectionné peut utiliser Python 3.13 : les anciens épinglages `safetensors==0.4.3` et `sentencepiece==0.2.0` ne fournissent pas de wheel Linux correspondante. Le complément Colab utilise donc `safetensors==0.5.3` et `sentencepiece==0.2.1`. Les poids, architectures et hyperparamètres du TP restent identiques. Pour les extensions compilées, `--only-binary` exige un paquet déjà compilé et évite une construction locale C++/Rust ; le paquet Python pur `emoji==1.4.2`, requis par AraBERT, reste autorisé depuis ses sources. La sortie pip est récupérée avec stderr puis affichée entièrement. Un code de retour non nul signifie une installation échouée : lire les lignes `ERROR:`. Le message « Installation terminée » n'apparaît qu'après un retour zéro et demande le redémarrage du noyau. La vérification locale des wheels ne remplace pas une installation réelle dans votre session distante.

Sources : [SentencePiece 0.2.1](https://pypi.org/project/sentencepiece/0.2.1/), [safetensors 0.5.3](https://pypi.org/project/safetensors/0.5.3/).

**Lecture complémentaire, dans l’ordre des opérations.**

Le bloc `IN_COLAB` compare les versions installées à la liste figée ; il ne touche pas au Conda local. Un arrêt « Installation terminée » demande de redémarrer le noyau, alors qu'un arrêt « Échec de pip » demande de lire les erreurs d'installation. Les variables OMP/MKL sont fixées avant les imports numériques : six threads localement, au plus deux dans Colab. Le code actuel supporte bien le GPU dans les cellules suivantes ; l'ancienne mention « aucun réglage GPU » dans la fiche d'origine ne décrit que le réglage des threads CPU.

**Code de référence.**

```python
from pathlib import Path

import os, sys, re, json, time, random, hashlib, gc, math, copy, platform, warnings

IN_COLAB = bool(os.environ.get('COLAB_RELEASE_TAG')) or Path('/content').is_dir()

if IN_COLAB:
    print('TP2_COLAB_BOOTSTRAP_V2 — préparation compatible Python 3.13',flush=True)
    from importlib.metadata import version, PackageNotFoundError
    import subprocess
    # Copie autonome des dépendances, synchronisée avec requirements-colab.txt.
    COLAB_REQUIREMENTS = ['transformers==4.48.0', 'huggingface-hub==0.29.1', 'safetensors==0.5.3', 'tokenizers==0.21.0', 'gensim==4.4.0', 'arabert==1.0.1', 'pyarabic==0.6.15', 'emoji==1.4.2', 'sentencepiece==0.2.1', 'nbformat==5.9.2', 'reportlab==5.0.1', 'pymupdf==1.28.2', 'requests==2.32.3', 'pyflakes==3.0.1']
    missing=[]
    for specification in COLAB_REQUIREMENTS:
        package,wanted=specification.split('==',1)
        try: installed=version(package)
        except PackageNotFoundError: installed=None
        if installed!=wanted: missing.append(specification)
    if missing:
        print('Préparation Colab — Python',sys.version.split()[0],':',', '.join(missing),flush=True)
        # Ces extensions doivent avoir une wheel : éviter une compilation C++/Rust.
        command=[sys.executable,'-m','pip','install','--disable-pip-version-check',
                 '--progress-bar','off',
                 '--only-binary=safetensors,tokenizers,gensim,sentencepiece,pymupdf',*missing]
        installation=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                    text=True,errors='replace')
        print(installation.stdout,flush=True)
        if installation.returncode:
            raise RuntimeError(f"Échec de pip (code {installation.returncode}). "
                               "Copier les lignes ERROR affichées ci-dessus pour le diagnostic. "
                               "Ne pas poursuivre les cellules scientifiques.")
        raise RuntimeError('Installation terminée : redémarre le noyau Colab puis relance depuis la première cellule.')

CPU_THREADS=min(2,os.cpu_count() or 1) if IN_COLAB else 6  # 12 testés : gain forward limité, backward non mesuré.
os.environ['OMP_NUM_THREADS']=str(CPU_THREADS)
os.environ['MKL_NUM_THREADS']=str(CPU_THREADS)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ['USE_TF']='0'
os.environ['USE_FLAX']='0'
from collections import Counter

from datetime import datetime, timezone

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from IPython.display import display, Markdown, HTML

import torch

from torch import nn

from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler, RandomSampler, BatchSampler

from sklearn.model_selection import train_test_split
```

<a id="tp2-002"></a>

#### Cellule de code 002 — Charger les outils NLP et fixer les chemins

**Objectif.** Respecter l'organisation du dépôt demandée dans le TP et utiliser uniquement ses ressources locales.

**Comprendre le code.** Les métriques évaluent les prédictions ; `Pipeline` et `FeatureUnion` construisent la baseline ; les classes `Auto*` ouvrent les poids locaux. `ROOT` désigne la racine du dépôt. Si le noyau démarre dans `notebooks/`, le bloc remonte d'un niveau ; s'il démarre à la racine, il y reste. `DATA_DIR = ROOT/'data/raw'` indique l'emplacement des deux CSV officiels. Les autres fonctions construisent leurs chemins sous `ROOT`, par exemple `models/hf/` ou `results/results.csv`. Les assertions détectent des données ou poids manquants avant les calculs.

**Résultat attendu à l'exécution.** Les chemins du dépôt sont prêts sans chargement de modèle, téléchargement ni apprentissage. Les chemins de checkpoints enregistrés sont relatifs à `ROOT`, ce qui permet de déplacer l'ensemble du dépôt.

**À retenir.** Ouvrir VS Code à la racine contenant `src/`, `notebooks/`, `data/` et `requirements.txt`. Le notebook reste la source pédagogique visible ; les modules exportés servent aux tests et à la reproductibilité. Aucune ressource n'est recherchée dans un dossier voisin.

**Sur Colab.** Les chemins Windows ne sont pas accessibles au serveur. Transférer le projet complet via « Upload to Colab », ou transférer le code puis préparer les ressources avec les scripts officiels. Si le dossier courant est `/content`, un unique projet enfant est détecté ; sinon définir `TP2_ROOT` avant les imports. Les assertions s'arrêtent avant tout apprentissage si données ou poids manquent.

**Session Colab vide.** Le noyau distant ne voit pas le disque Windows. Un petit paquet intégré fournit les scripts de préparation et les manifestes exacts, sans poids ni données. L'archive est contrôlée par SHA256 et extraite seulement sous la racine choisie, sans écraser les fichiers existants. `prepare_data` télécharge les deux CSV officiels et contrôle leurs SHA256, sans consulter les labels test. Avant B, `prepare_models` récupère les quatre révisions Hugging Face épinglées et vérifie leurs poids. AraVec est préparé à son étape C. Aucun de ces téléchargements n'entraîne de modèle. Les sources de ces opérations sont lisibles dans `src/colab_resources.py`.

**Lecture complémentaire, dans l’ordre des opérations.**

La priorité des chemins est `TP2_ROOT`, puis le répertoire courant, avec remontée si celui-ci s'appelle `notebooks`. Dans Colab, l'archive intégrée fournit des sources, pas des checkpoints entraînés ; son SHA256 et le confinement des chemins sont contrôlés avant extraction. `prepare_data(ROOT)` peut télécharger les CSV officiels. Les assertions vérifient la présence des fichiers ; elles ne calculent aucun F1.

**Code de référence.**

```python
from sklearn.metrics import (f1_score, precision_score, recall_score, accuracy_score,
    average_precision_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve)

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import FeatureUnion, Pipeline

from sklearn.linear_model import LogisticRegression

import joblib

import emoji

warnings.filterwarnings('ignore',message='This pattern is interpreted as a regular expression, and has match groups')

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from transformers import get_linear_schedule_with_warmup, set_seed

# TP2_LAYOUT
ROOT = Path(os.environ.get('TP2_ROOT',Path.cwd())).resolve()
if ROOT.name == 'notebooks': ROOT = ROOT.parent
if IN_COLAB and not (ROOT/'data/raw/training_data.csv').is_file() and 'TP2_ROOT' not in os.environ:
    candidates=[p for p in ROOT.iterdir() if p.is_dir() and (p/'src/preprocess.py').is_file()]
    ROOT=candidates[0] if len(candidates)==1 else Path('/content/TP2_ArSarcasme')
if IN_COLAB:
    import base64, io, zipfile
    # BOOTSTRAP_ARCHIVE : archive intégrée abrégée pour la lecture (15,082 caractères sur cette ligne).
    BOOTSTRAP_ARCHIVE = "<archive base64 : voir le notebook original>"
    payload=base64.b64decode(BOOTSTRAP_ARCHIVE)
    assert hashlib.sha256(payload).hexdigest()=='a73d1a98482328a362f033677bf3d6f0411acf24ce7239d2cfc5f4f85391a0f6'
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for info in archive.infolist():
            target=(ROOT/info.filename).resolve()
            assert target.is_relative_to(ROOT), 'Chemin de préparation invalide.'
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(archive.read(info))
    if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
    from src.colab_resources import prepare_data, prepare_models, prepare_aravec
    prepare_data(ROOT)
DATA_DIR = ROOT/'data/raw'
assert (DATA_DIR/'training_data.csv').is_file(), 'CSV manquant dans data/raw/. Sur Colab utiliser le notebook T4 depuis la première cellule.'
assert (DATA_DIR/'testing_data.csv').is_file(), 'CSV officiel manquant dans data/raw/.'
print('Dossier du projet :',ROOT)
```

<a id="tp2-003"></a>

#### Cellule de code 003 — Choisir le matériel et les paramètres communs

**Objectif.** Définir un budget compatible avec les 16 Go de RAM installés et organiser les emplacements de sortie.

**Comprendre le code.** `mkdir(..., exist_ok=True)` prépare les dossiers du dépôt local sans effacer leur contenu. `DEVICE` utilise CUDA si disponible ; une session Colab sans GPU provoque une erreur explicite. Six threads correspondent au réglage retenu sur le Ryzen 5 5600G. `SEEDS` fixe les trois répétitions finales, `MAX_LEN=64` limite les séquences et `BATCH` vaut 32 sur GPU et 16 sur le CPU local ; les lots statiques restent de taille 64. `TOKENIZERS` est un cache en mémoire.

**Résultat attendu à l’exécution.** Les variables de configuration sont définies ; cette cellule ne lance aucun entraînement.

**À retenir.** La présence d’un GPU Radeon intégré ne suffit pas à fournir CUDA. La longueur 64 est un choix de budget aligné sur le préentraînement Twitter d’AraBERT ; son effet sera mesuré par les taux de troncature, pas supposé négligeable.

**Optimisation CPU ajoutée.** Le périphérique est maintenant explicitement CPU. Intra-op=6, inter-op=1 ; le réglage inter-op demande un noyau neuf. Les autres budgets, graines et tailles de lots restent identiques.

**T4.** Les fine-tunings PLM utilisent AMP float16 avec poids maîtres float32 ; les autres familles restent en float32. Cette précision et le batch sont journalisés et entrent dans la compatibilité des checkpoints.

**Lecture complémentaire, dans l’ordre des opérations.**

`select_device(True)` refuse une session Colab sans CUDA, pour éviter un entraînement CPU involontaire. Sur T4 : batch PLM 32 et précision AMP float16 ; sur CPU local : batch 16 et float32. `MAX_LEN=64` compte des positions, pas 64 caractères. Les dictionnaires de cache initialisés ici sont partagés par les fonctions suivantes.

**Définitions importantes :** `select_device`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
assert (DATA_DIR / 'training_data.csv').exists(), 'Conserver les deux CSV dans data/raw/.'

os.environ['USE_TF'] = '0'

os.environ['USE_FLAX'] = '0'

for folder in ['results','predictions','figures','models','data/processed','logs']:
    (ROOT / folder).mkdir(exist_ok=True, parents=True)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

TOKENIZERS = {}
TOKENIZED = {}  # Cache borné : quatre jeux de tenseurs CPU.
FROZEN_MODELS = {}  # Un seul encodeur gelé ; vidé avant tout autre chargement.

os.environ['HF_HUB_DISABLE_XET'] = '1'

torch.set_num_threads(CPU_THREADS)
if torch.get_num_interop_threads()!=1:
    try: torch.set_num_interop_threads(1)
    except RuntimeError: print('Pour appliquer inter-op=1, redémarrer le noyau avant Run All.')

def select_device(in_colab):
    if in_colab and not torch.cuda.is_available():
        raise RuntimeError('Colab sans CUDA : choisir un serveur GPU T4, puis redémarrer le noyau.')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEVICE = select_device(IN_COLAB)
PLM_PRECISION = 'amp_float16' if DEVICE.type=='cuda' else 'float32' 

SEEDS = [42, 43, 44]

MAX_LEN = 64

BATCH = 32 if DEVICE.type=='cuda' else 16
```

<a id="tp2-004"></a>

#### Cellule de code 004 — Définir les budgets d’époques et la compatibilité des caches

**Objectif.** Fixer les règles communes de durée et empêcher la reprise de checkpoints provenant d’un autre contexte.

**Comprendre le code.** Les petits réseaux ont au maximum huit époques ; les PLM en ont cinq au maximum dans le code courant ; `PATIENCE=2` autorise deux validations sans amélioration avant arrêt. `RUN_VERSION` identifie la logique expérimentale. `CACHE_CONTEXT` rassemble longueur, tailles de lots, versions et précision. Le manifeste `REVISIONS` précise les versions des quatre PLM ; leurs chemins sont recalculés à partir du dossier actuel.

**Résultat attendu à l’exécution.** Un contexte reproductible et un dictionnaire de ressources locales sont disponibles.

**À retenir.** Le cache économise des calculs, mais ne crée pas une nouvelle expérience. Une modification scientifique doit être accompagnée d’un changement de version et d’une réévaluation des caches concernés.

**Mise à jour qualité / affichage.** Le plafond PLM passe à **5 époques**, autorisé par le TP, avec patience 2. Le meilleur checkpoint reste choisi sur le F1 de validation. Cela permet une recherche plus longue sans garantie de gain ; le scheduler linéaire utilise désormais le budget de cinq époques. Aucun score de l'ancienne exécution n'est attribué à ce nouveau réglage.

**Lecture complémentaire, dans l’ordre des opérations.**

Le code courant fixe `STATIC_EPOCHS=8`, `PLM_EPOCHS=5`, `PATIENCE=2`. Ce sont des plafonds ; l'arrêt anticipé peut réduire les époques réalisées. `CACHE_CONTEXT` décrit notamment matériel, versions et batch ; il sert à refuser la réutilisation silencieuse d'un run incompatible. Les chemins de révision sont replacés sous la racine de cette session.

**Code de référence.**

```python
STATIC_EPOCHS, PLM_EPOCHS, PATIENCE = 8, 5, 2

RUN_VERSION = 'tp2-v1'

CACHE_CONTEXT={'max_len':MAX_LEN,'batch_plm':BATCH,'batch_static':64,
    'torch':torch.__version__,'transformers':__import__('transformers').__version__,
    'precision':'float32','protocol':'conservative-tp2-v1'}

if DEVICE.type=='cuda':
    CACHE_CONTEXT.update(device_type='cuda',plm_precision=PLM_PRECISION,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),numpy=np.__version__,pandas=pd.__version__,
        sklearn=__import__('sklearn').__version__)

REVISIONS = json.loads((ROOT/'models/revisions.json').read_text()) if (ROOT/'models/revisions.json').exists() else {}

for model_name,entry in REVISIONS.items():
    entry['path']=str(ROOT/'models/hf'/model_name.replace('/','--'))
```

<a id="tp2-005"></a>

#### Cellule de code 005 — Définir la fonction de fixation des graines

**Objectif.** Centraliser l’initialisation des générateurs aléatoires utilisés par les bibliothèques.

**Comprendre le code.** `random.seed`, `np.random.seed`, `set_seed` et `torch.manual_seed` contrôlent notamment les initialisations et les tirages. `torch.use_deterministic_algorithms(..., warn_only=True)` demande des opérations déterministes lorsque c’est possible ; si une opération pose problème, PyTorch avertit au lieu d’interrompre systématiquement le calcul.

**Résultat attendu à l’exécution.** La fonction `seed_all` est définie. Son corps ne s’exécute que lorsqu’elle est appelée.

**À retenir.** Une graine facilite la reproduction sur une même configuration ; elle ne garantit pas des résultats identiques entre tous les matériels et toutes les versions. Trois graines servent à observer une partie de la variabilité d’apprentissage.

**Lecture complémentaire, dans l’ordre des opérations.**

Définir `seed_all` ne l'applique pas encore : l'appel arrive cellule 006 puis avant les runs. `warn_only=True` avertit si une opération n'a pas d'algorithme déterministe, plutôt que bloquer. Des résultats CPU et T4 peuvent donc différer malgré les mêmes graines.

**Définitions importantes :** `seed_all`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); set_seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
```

<a id="tp2-006"></a>

#### Cellule de code 006 — Initialiser la session et vérifier le noyau Conda

**Objectif.** Appliquer la graine de référence et rendre l’environnement réellement utilisé visible dans les sorties.

**Comprendre le code.** `seed_all(42)` initialise cette session. Les options pandas permettent de lire des tweets assez longs et les options matplotlib harmonisent les graphiques. `sys.executable` affiche le véritable interpréteur du noyau, ce qui est plus fiable que le nom du terminal ouvert. L’assertion finale contrôle la présence du chemin Anaconda local ou Colab attendu.

**Résultat attendu à l’exécution.** Un récapitulatif indique Python, sa version, PyTorch, le périphérique et le nombre de threads.

**À retenir.** Vérifier ce récapitulatif avant les calculs longs. Un terminal Anaconda local ou Colab ouvert ailleurs ne force pas automatiquement le notebook à utiliser cet interpréteur.

**Colab.** Le Python distant est accepté ; l’assertion Anaconda reste limitée à l’exécution locale. Le nom réel du GPU, CUDA, la précision et le batch sont affichés et enregistrés dans `results/runtime_hardware.json`.

**Lecture complémentaire, dans l’ordre des opérations.**

`sys.executable` révèle le Python du noyau réellement actif. Le code accepte Conda local ou Colab et affiche les informations GPU si CUDA est choisi. Les options pandas changent seulement l'affichage, pas les exemples utilisés ; voir une table tronquée n'est pas une réduction du dataset.

**Code de référence.**

```python
seed_all(42)

pd.set_option('display.max_colwidth', 220)

pd.set_option('display.max_rows', 65)

plt.rcParams.update({'figure.dpi':120, 'axes.spines.top':False, 'axes.spines.right':False})

print('Python :',sys.executable, '\nVersion :',sys.version, '\nPyTorch :',torch.__version__,
      '\nPériphérique :',DEVICE, '\nCœurs utilisés :',torch.get_num_threads())

assert IN_COLAB or 'anaconda3' in sys.executable.lower(), 'Utiliser Conda local ou le noyau Colab.'
if DEVICE.type=='cuda': print('GPU :',torch.cuda.get_device_name(0),'— PLM :',PLM_PRECISION,'— batch :',BATCH)
(ROOT/'results/runtime_hardware.json').write_text(json.dumps({
    'training_device':str(DEVICE),'gpu':torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else None,
    'plm_precision':PLM_PRECISION,'batch_plm':BATCH,'cpu_threads':CPU_THREADS,
    'torch':torch.__version__,'cuda':torch.version.cuda,'platform':platform.platform(),
    'cpu_model':platform.processor() or 'unknown','logical_cpus':os.cpu_count()},indent=2))
```

**Petit extrait de sortie déjà enregistrée dans ce fichier (non recalculée pour le guide).**

```text
Python : /usr/bin/python3 
Version : 3.13.15 (main, Aug  6 2026, 11:06:22) [GCC 13.3.0] 
PyTorch : 2.11.0+cu130 
Périphérique : cuda 
Cœurs utilisés : 2
GPU : Tesla T4 — PLM : amp_float16 — batch : 32
```

### Audit et préparation des données

<a id="tp2-007"></a>

#### Cellule de code 007 — Préparer les règles de normalisation de l’arabe

**Objectif.** Décrire les caractères et motifs à reconnaître avant d’appliquer un nettoyage.

**Comprendre le code.** `DIAC` repère les diacritiques arabes ; `URL_RE` et `MENTION_RE` détectent les liens et les mentions. `ARABIC_MAP` rapproche certaines variantes d’alif et remplace alif maqsura par ya. `html` et `unicodedata` serviront à décoder les caractères exportés et à stabiliser leur représentation Unicode.

**Résultat attendu à l’exécution.** Des motifs compilés et une table de traduction sont disponibles pour les fonctions suivantes.

**À retenir.** Définir une règle n’altère encore aucun tweet. Une normalisation réduit la variation graphique, mais peut aussi supprimer de l’information ; elle ne sera donc pas appliquée de façon identique à toutes les familles.

**Lecture complémentaire, dans l’ordre des opérations.**

Les motifs compilés sont réutilisés sur tous les tweets. `DIAC` vise des plages de signes diacritiques arabes ; `ARABIC_MAP` rapproche certaines variantes d'alif et de ya. Ces règles sont déterministes : elles n'apprennent ni vocabulaire ni statistique sur validation/test.

**Code de référence.**

```python
import html, unicodedata

DIAC = re.compile(r'[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]')

URL_RE = re.compile(r'https?://\S+|www\.\S+', re.I)

MENTION_RE = re.compile(r'@[\w_]+')

ARABIC_MAP = str.maketrans({'أ':'ا','إ':'ا','آ':'ا','ٱ':'ا','ى':'ي'})
```

<a id="tp2-008"></a>

#### Cellule de code 008 — Retirer les artefacts d’export avec export_clean

**Objectif.** Corriger le format du texte avant de toucher aux indices linguistiques du sarcasme.

**Comprendre le code.** Le texte est converti en chaîne, décodé avec `html.unescape`, normalisé en NFC puis débarrassé des espaces extérieurs. Une seule paire de guillemets ASCII extérieurs est retirée lorsqu’elle encadre tout le tweet. Certains contrôles invisibles de direction et le BOM sont ensuite supprimés. Les citations internes et les marqueurs visibles ne sont pas effacés par cette fonction.

**Exemple ou repère.** Une cellule CSV contenant `"texte du tweet"` devient `texte du tweet` ; une citation située au milieu du tweet est conservée.

**Résultat attendu à l’exécution.** Lors d’un appel, on obtient une chaîne débarrassée des principaux artefacts techniques d’export.

**À retenir.** Il faut distinguer un guillemet ajouté par le CSV d’un guillemet écrit par l’auteur. Ce nettoyage ne constitue ni une tokenisation ni une compréhension du sarcasme.

**Lecture complémentaire, dans l’ordre des opérations.**

`html.unescape` transforme les entités d'export, NFC unifie des représentations Unicode et `strip` retire les espaces extérieurs. Une seule paire de guillemets ASCII entourant tout le tweet est retirée ; les citations internes ne sont pas supprimées. Les caractères de direction invisibles sont nettoyés pour stabiliser les comparaisons.

**Définitions importantes :** `export_clean`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def export_clean(text):
    text = unicodedata.normalize('NFC', html.unescape(str(text))).strip()
    # Une seule paire extérieure ASCII ; ne pas supprimer les citations internes de l'auteur.
    if len(text)>1 and text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    return re.sub(r'[\u200e\u200f\u202a-\u202e\ufeff]', '', text)
```

<a id="tp2-009"></a>

#### Cellule de code 009 — Construire une clé pour détecter les doublons

**Objectif.** Repérer des tweets équivalents après quelques normalisations déterministes.

**Comprendre le code.** `duplicate_key` applique le nettoyage d’export, retire les diacritiques et le tatweel, normalise les lettres choisies et compacte les espaces. Elle conserve le contenu des URL et des mentions : deux tweets visant des personnes différentes ne sont pas fusionnés simplement parce qu’ils contiennent tous deux une mention.

**Résultat attendu à l’exécution.** Chaque tweet reçoit une chaîne `key`, utilisée pour rechercher répétitions, collisions entre partitions et groupes contradictoires.

**À retenir.** Cette clé sert à l’audit, pas de cible à prédire. L’égalité de clés est une convention de dédoublonnage ; elle ne prouve pas que deux tweets avaient exactement le même contexte d’origine.

**Lecture complémentaire, dans l’ordre des opérations.**

La clé est destinée à repérer les collisions, pas à devenir automatiquement l'entrée de tous les modèles. Les URL et mentions gardent ici leur contenu, pour ne pas fusionner toutes les phrases qui ne diffèrent que par leur destinataire ou lien. Deux textes de même clé mais de labels différents forment un groupe contradictoire.

**Définitions importantes :** `duplicate_key`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def duplicate_key(text):
    return re.sub(r'\s+', ' ', DIAC.sub('',export_clean(text)).replace('ـ','').translate(ARABIC_MAP)).strip()
```

<a id="tp2-010"></a>

#### Cellule de code 010 — Préparer le texte selon la famille de modèles

**Objectif.** Disposer d’un nettoyage conservateur et d’une ablation agressive explicitement distingués.

**Comprendre le code.** Le mode conservateur retire diacritiques/tatweel et normalise les variantes retenues. Liens et mentions deviennent `URLTOKEN` et `USERTOKEN`, sauf si `drop_links=True`, qui les retire. Le mode `plm` conserve davantage la graphie. Le mode agressif enlève en plus emojis, rires, répétitions et ponctuation : il sert à mesurer une perte d’information, pas à annoncer un meilleur nettoyage. Les espaces sont finalement compactés.

**Résultat attendu à l’exécution.** Les appels produisent des chaînes adaptées à l’expérience choisie. Ici, seule la fonction est définie.

**À retenir.** Les négations et le contenu des hashtags restent disponibles. Le prétraitement doit être comparé sur validation ; un texte plus court ou plus uniforme n’est pas nécessairement plus utile pour détecter l’ironie.

**Lecture complémentaire, dans l’ordre des opérations.**

`drop_links=False` remplace URL/mentions par des marqueurs ; `True` les supprime pour les ablations. Le mode `plm` évite la normalisation statique de l'arabe ; `aggressive` retire volontairement des indices supplémentaires. Une même configuration doit être conservée de l'apprentissage à la phrase testée.

**Définitions importantes :** `prepare_text`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def prepare_text(text, mode='conservative', drop_links=False):
    text = export_clean(text)
    if mode != 'plm':
        text = DIAC.sub('',text).replace('ـ','').translate(ARABIC_MAP)
    text = URL_RE.sub(' ' if drop_links else ' URLTOKEN ',text)
    text = MENTION_RE.sub(' ' if drop_links else ' USERTOKEN ',text)
    if mode == 'aggressive':
        text = emoji.get_emoji_regexp().sub(' ',text)
        text = re.sub(r'ه{2,}|خ{2,}', ' ', text)
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        text = re.sub(r'[^\w\s]', ' ', text)
    # Hashtags : contenu conservé ; ni stopwords ni stemming (négation/dialecte préservés).
    return re.sub(r'\s+', ' ',text).strip()
```

<a id="tp2-011"></a>

#### Cellule de code 011 — Définir la suppression contrôlée des indices

**Objectif.** Préparer l’ablation F4, qui retirera des marqueurs sur des tweets déjà correctement détectés.

**Comprendre le code.** `erase_cues` supprime les emojis reconnus, les suites de rires en ه ou خ et certaines répétitions de ponctuation. Elle n’apprend rien et n’utilise aucun label. Elle est distincte du mode agressif : son rôle est de modifier les indices ciblés lors de l’évaluation, sans réentraîner le modèle ni changer son seuil.

**Résultat attendu à l’exécution.** Une fonction permet de créer une version modifiée d’un même tweet pour une comparaison avant/après.

**À retenir.** Une chute de détection après suppression mesure une sensibilité à cette transformation. Elle ne suffit pas à établir une explication causale complète du sarcasme.

**Lecture complémentaire, dans l’ordre des opérations.**

`erase_cues` est une intervention expérimentale F4, pas le nettoyage habituel de toutes les données. On supprime emojis, certaines répétitions de rire et ponctuation répétée ; on ne supprime pas arbitrairement tout le vocabulaire d'opinion. L'effet sera mesuré sur les vrais positifs de départ.

**Définitions importantes :** `erase_cues`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def erase_cues(text):
    text = emoji.get_emoji_regexp().sub(' ',str(text))
    text = re.sub(r'ه{2,}|خ{2,}', ' ', text)
    return re.sub(r'([!?؟.،])\1+', ' ',text)
```

<a id="tp2-012"></a>

#### Cellule de code 012 — Lire le train et créer la validation imposée

**Objectif.** Respecter la séparation officielle tout en gardant les labels test hors des décisions de développement.

**Comprendre le code.** Les 12 548 lignes du train reçoivent un identifiant officiel et une cible binaire `y`. `usecols=["tweet"]` lit uniquement le texte des 3 000 lignes test pour l’audit demandé. `train_test_split` réserve 10 % du train à la validation, avec graine 42 et stratification sarcasme. Des copies évitent les modifications ambiguës de vues pandas. Les clés contradictoires du train sont repérées par plusieurs valeurs de `y` pour la même clé.

**Résultat attendu à l’exécution.** Avant les exclusions, on attend 11 293 exemples train et 1 255 exemples val. Les conflits sont stockés pour inspection.

**À retenir.** Stratifier conserve approximativement les proportions de classes. Ni sentiment ni dialecte ne deviennent des variables d’entrée : ces annotations servent aux cibles auxiliaires ou aux analyses.

**Lecture complémentaire, dans l’ordre des opérations.**

`id` conserve la position officielle avant le split ; `reset_index` ultérieur ne doit pas être confondu avec cet identifiant. `usecols=['tweet']` charge le texte test sans ses labels. La stratification conserve approximativement le taux de sarcasme entre train initial et val, sans faire de stratification par dialecte.

**Code de référence.**

```python
full = pd.read_csv(DATA_DIR/'training_data.csv')

full['id'] = np.arange(len(full))

full['y'] = full.sarcasm.astype(int)

assert len(full)==12548 and full[['tweet','sarcasm','sentiment','dialect']].notna().all().all()

sealed_text = pd.read_csv(DATA_DIR/'testing_data.csv', usecols=['tweet'])

assert len(sealed_text)==3000

train_original, val = train_test_split(full, test_size=.10, random_state=42, stratify=full.sarcasm)

train_original, val = train_original.copy(), val.copy()

for frame in [train_original, val, sealed_text]: frame['key'] = frame.tweet.map(duplicate_key)

train = train_original.copy()

conflict_keys = set(train.groupby('key').y.nunique().loc[lambda x:x>1].index)

conflicts = train.loc[train.key.isin(conflict_keys)].copy()
```

<a id="tp2-013"></a>

#### Cellule de code 013 — Écarter les collisions et les doublons du train

**Objectif.** Réduire la fuite d’information qui rendrait les scores artificiellement favorables.

**Comprendre le code.** Deux masques repèrent les clés du train également présentes en test ou en validation. Le dictionnaire `audit_split` enregistre les effectifs avant retrait, les doublons et les conflits. On retire uniquement du train les collisions, groupes contradictoires et clés vides ; on garde ensuite un exemplaire par clé. Les indices pandas sont remis à zéro pour aligner les futurs tableaux et tenseurs.

**Résultat attendu à l’exécution.** Le train admissible est constitué et son absence de collision avec val/test est vérifiée. Pour les fichiers actuels, l’audit historique donne 10 842 exemples train.

**À retenir.** Val et test restent intacts. Le retrait repose sur les textes et les contradictions du train, jamais sur le fait qu’un modèle se trompe sur une ligne test.

**Lecture complémentaire, dans l’ordre des opérations.**

Les masques `overlap_test` et `overlap_val` sont calculés sur train avant exclusion. Le filtre retire du train collisions et clés contradictoires, puis `drop_duplicates('key')` garde une occurrence par clé. Les assertions finales contrôlent l'absence de clés communes ; elles ne garantissent pas l'absence de paraphrases sémantiques.

**Code de référence.**

```python
overlap_test = train.key.isin(set(sealed_text.key))

overlap_val = train.key.isin(set(val.key))

audit_split = {'train_initial':len(train), 'val':len(val), 'test':len(sealed_text),
    'train_test_exact':int(train.tweet.isin(set(sealed_text.tweet)).sum()),
    'train_test_normalized':int(overlap_test.sum()), 'train_val_normalized':int(overlap_val.sum()),
    'train_duplicate_rows':int(train.duplicated('key').sum()),
    'contradictory_groups_train':len(conflict_keys), 'contradictory_rows_train':len(conflicts)}

train = train.loc[~overlap_test & ~overlap_val & ~train.key.isin(conflict_keys)]

train = train.loc[train.key.ne('')].drop_duplicates('key').reset_index(drop=True)

val = val.reset_index(drop=True)

audit_split['train_final'] = len(train)

assert set(train.key).isdisjoint(val.key) and set(train.key).isdisjoint(sealed_text.key)
```

<a id="tp2-014"></a>

#### Cellule de code 014 — Sauvegarder l’audit et afficher les tweets bruts

**Objectif.** Rendre les décisions de préparation vérifiables avant les expériences.

**Comprendre le code.** Les partitions traitées et les conflits sont exportés en CSV ; le récapitulatif des exclusions est enregistré en JSON. `utf-8-sig` facilite l’ouverture des textes arabes avec certains outils Windows. Les 30 premières lignes du train original sont affichées avec annotations, puis 30 textes test sans labels. L’affichage brut permet de constater les artefacts au lieu de supposer leur existence.

**Résultat attendu à l’exécution.** Des tableaux lisibles apparaissent dans le notebook et des fichiers d’audit sont écrits dans le dépôt local.

**À retenir.** Lire quelques tweets n’est pas une preuve de performance. Cette inspection documente le format et les indices ; les décisions de modèles restent fondées sur train/validation.

**Lecture complémentaire, dans l’ordre des opérations.**

Les tables et exemples bruts donnent une trace de ce qui a été retiré et conservé. Les trente tweets affichés sont des illustrations, pas une réduction du corpus à trente lignes. Les fichiers `train.csv` et `val.csv` matérialisent les partitions après audit.

**Code de référence.**

```python
assert not train.key.duplicated().any()

for name, frame in [('train',train),('val',val)]:
    frame.to_csv(ROOT/f'data/processed/{name}.csv',index=False,encoding='utf-8-sig')

conflicts.to_csv(ROOT/'results/contradictions_train.csv',index=False,encoding='utf-8-sig')

(ROOT/'results/split_audit.json').write_text(json.dumps(audit_split,indent=2),encoding='utf-8')

display(pd.Series(audit_split,name='Effectif').to_frame())

display(Markdown('**30 tweets bruts de train avant toute normalisation**'))

display(train_original[['id','tweet','sarcasm','sentiment','dialect']].head(30))

display(Markdown('**30 textes bruts du test pour audit uniquement**'))

display(sealed_text[['tweet']].head(30))
```

<a id="tp2-015"></a>

#### Cellule de code 015 — Décrire les marqueurs à compter

**Objectif.** Transformer l’audit linguistique en mesures explicites et reproductibles.

**Comprendre le code.** `PATTERNS` associe un nom à un motif : URL, mention, hashtag, rires, ponctuation répétée, diacritiques, tatweel, répétition de lettres, alphabet latin et chiffres arabes orientaux. Les motifs de répétition utilisent une référence au caractère capturé, afin de reconnaître une suite du même signe plutôt qu’une simple longueur de texte.

**Résultat attendu à l’exécution.** Un dictionnaire de règles de détection est défini ; les taux seront calculés ensuite.

**À retenir.** Ces motifs sont des approximations de surface. Des rires peuvent avoir d’autres graphies et le mélange de langues ne se réduit pas à la présence d’une lettre latine. L’audit doit être lu avec ces limites.

**Lecture complémentaire, dans l’ordre des opérations.**

Un motif d'indice reconnaît une présence dans le texte ; il ne classe pas encore le sarcasme. Un hashtag peut être un thème ordinaire et un emoji peut être sincère. La corrélation de ces indicateurs avec `y` sera calculée séparément.

**Code de référence.**

```python
PATTERNS = {'url':r'https?://|www\.', 'mention':r'@\w+', 'hashtag':r'#\w+',
    'rires':r'ه{2,}|خ{2,}', 'ponctuation_repetee':r'([!?؟.،])\1+',
    'diacritiques':DIAC.pattern, 'tatweel':'ـ', 'lettres_repetees':r'([^\W\d_])\1{2,}',
    'latin':r'[A-Za-z]', 'chiffres_indiens':r'[٠-٩]'}
```

<a id="tp2-016"></a>

#### Cellule de code 016 — Extraire les indicateurs de surface

**Objectif.** Obtenir, pour chaque tweet, la présence ou l’absence des indices de l’audit.

**Comprendre le code.** `str.contains` produit une colonne booléenne par motif. Deux colonnes supplémentaires détectent les emojis et les guillemets ASCII encadrant tout le texte. Le résultat est une table de forme nombre de tweets × nombre d’indices. Une valeur vraie signifie au moins une occurrence, pas le nombre total d’occurrences.

**Exemple ou repère.** Si 25 indicateurs sur 100 valent vrai, la moyenne vaut 0,25 ; multipliée par 100, elle donne 25 %.

**Résultat attendu à l’exécution.** La fonction `text_features` pourra alimenter les taux, les règles simples et les sous-groupes URL.

**À retenir.** La moyenne d’une colonne booléenne représente une proportion : Python compte vrai comme 1 et faux comme 0.

**Lecture complémentaire, dans l’ordre des opérations.**

`text_features` renvoie une ligne par tweet et une colonne booléenne par indice. La moyenne d'une colonne booléenne vaut la proportion de tweets contenant l'indice, pas le nombre moyen d'occurrences. Cela explique le `.mean().mul(100)` de la cellule suivante.

**Définitions importantes :** `text_features`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def text_features(series):
    result = pd.DataFrame({k:series.str.contains(p,regex=True) for k,p in PATTERNS.items()})
    result['emoji'] = series.map(lambda s: bool(emoji.get_emoji_regexp().search(s)))
    result['artefact_guillemets'] = series.str.strip().str.startswith('"') & series.str.strip().str.endswith('"')
    return result
```

<a id="tp2-017"></a>

#### Cellule de code 017 — Mesurer les indices et comparer deux nettoyages

**Objectif.** Quantifier le format du corpus et illustrer les informations retirées par un nettoyage agressif.

**Comprendre le code.** Les taux de présence sont calculés sur le train brut et sur les textes test, sans lire leurs labels. Sur le train nettoyé, on compare ensuite le taux sarcastique selon présence/absence d’URL, mention et guillemets. Trois exemples arabes sont affichés sous leurs versions brute, conservatrice et agressive. Une assertion vérifie explicitement que la version conservatrice garde emoji, rire, ponctuation, allongement et hashtag de l’exemple.

**Résultat attendu à l’exécution.** Le notebook montre des pourcentages, des sous-groupes train et une comparaison de textes ; `text_audit.csv` conserve les fréquences.

**À retenir.** Un indice corrélé au sarcasme n’est pas nécessairement une cause. Une corrélation peut refléter une source de collecte, un export ou un thème dominant.

**Lecture complémentaire, dans l’ordre des opérations.**

`raw_audit` compare les fréquences d'indices sans lire de labels test. `shortcut_rates` compare ensuite les taux de sarcasme dans les sous-groupes du train nettoyé. Les exemples conservateur/agressif servent à vérifier concrètement que négation, rire et emoji survivent au prétraitement choisi.

**Code de référence.**

```python
raw_audit = pd.concat({name:text_features(frame.tweet).mean().mul(100)
    for name,frame in [('train_brut',train_original),('test_texte',sealed_text)]},axis=1)

display(raw_audit.round(2).rename_axis('Indice présent (%)'))

raw_audit.to_csv(ROOT/'results/text_audit.csv')

shortcut_rates = []

features_train = text_features(train.tweet)

for feature in ['url','mention','artefact_guillemets']:
    for flag in [False,True]:
        selected = train.loc[features_train[feature].eq(flag)]
        shortcut_rates.append({'indice':feature,'present':flag,'n':len(selected),'taux_sarcasme':selected.y.mean()})

display(pd.DataFrame(shortcut_rates))

examples = ['"يا سلاااام 😂 ههههه !!! #جميل https://t.co/abc @ami"', 'لَا أحب هـذا أبداً؟!','ما شاء الله، خدمة ممتازة... بعد ساعتين انتظار!']

display(pd.DataFrame({'brut':examples, 'conservateur':[prepare_text(s) for s in examples],
                      'agressif':[prepare_text(s,'aggressive') for s in examples]}))

assert all(c in prepare_text(examples[0]) for c in ['😂','ههههه','!!!','سلاااام','#جميل'])
```

<a id="tp2-018"></a>

#### Cellule de code 018 — Tester les invariants et décrire les sous-groupes

**Objectif.** Vérifier des propriétés essentielles du nettoyage avant d’interpréter les distributions.

**Comprendre le code.** Les assertions contrôlent la conservation de la négation لا, l’idempotence du nettoyage et le remplacement des liens/mentions. L’idempotence signifie que nettoyer une seconde fois ne doit pas continuer à déformer le texte. Pour train et val, `groupby` rapporte taille, nombre de sarcastiques et proportion par dialecte puis sentiment. Une figure vide est préparée pour la cellule suivante.

**Résultat attendu à l’exécution.** Les contrôles passent ou arrêtent l’exécution avec une assertion ; les tables décrivent les partitions de développement.

**À retenir.** Un taux calculé sur un très petit sous-groupe est instable. La colonne `size` compte autant que la colonne `mean` pour interpréter les différences entre dialectes.

**Mise à jour qualité / affichage.** La création de la figure a été déplacée dans la cellule 019, avec son tracé, afin que Jupyter n'affiche pas des axes encore vides à la fin de cette cellule.

**Lecture complémentaire, dans l’ordre des opérations.**

Les assertions testent des propriétés du nettoyage sur des exemples contrôlés. Les agrégations `size/sum/mean` signifient effectif, nombre de sarcastiques et proportion. Les données test étiquetées ne sont pas encore introduites dans ces descriptions supervisées.

**Code de référence.**

```python
assert 'لا' in prepare_text('لَا أحب هذا')

assert prepare_text('مرحبا   بالعالم') == prepare_text(prepare_text('مرحبا   بالعالم'))

assert 'http' not in prepare_text(examples[0]) and '@ami' not in prepare_text(examples[0])

print('Contrôles du prétraitement et de séparation : OK.')

for name,frame in [('train',train),('val',val)]:
    print(name, 'n =',len(frame),'sarcasme =',round(frame.y.mean(),4))
    display(frame.groupby('dialect').y.agg(['size','sum','mean']))
    display(frame.groupby('sentiment').y.agg(['size','sum','mean']))
```

<a id="tp2-019"></a>

#### Cellule de code 019 — Visualiser la prévalence par dialecte

**Objectif.** Lire rapidement les différences de proportion sarcastique dans le train nettoyé.

**Comprendre le code.** La moyenne de `y` est calculée pour chaque dialecte puis dessinée sur la figure préparée auparavant. Les étiquettes précisent qu’il s’agit d’une proportion, et non d’un nombre de tweets. `tight_layout` ajuste les marges ; `savefig` conserve une version PNG avant l’affichage.

**Résultat attendu à l’exécution.** Un diagramme en barres apparaît et `figures/prior_train.png` est créé dans le dépôt local.

**À retenir.** Ce graphique montre la distribution des annotations du corpus, pas un taux d’erreur du modèle ni une propriété universelle d’un dialecte. Revenir aux effectifs du tableau précédent pour juger sa stabilité.

**Mise à jour qualité / affichage.** La figure est créée, remplie, sauvegardée puis explicitement affichée dans cette même cellule. `plt.close(fig)` évite un deuxième affichage automatique vide ou en double.

**Lecture complémentaire, dans l’ordre des opérations.**

Le graphique compare les taux de sarcasme par dialecte de train et val. Les effectifs doivent être lus avec les taux : 0 % sur cinq exemples ne représente pas la même information que 0 % sur des milliers. Création, remplissage et affichage sont réunis pour éviter des axes vides en sortie inline.

**Code de référence.**

```python
fig,ax=plt.subplots(figsize=(7,3))

train.groupby('dialect').y.mean().plot.bar(ax=ax,color='#275D88')

ax.set(ylabel='Proportion sarcastique',title='Train nettoyé par dialecte',xlabel='Dialecte')

plt.tight_layout()

fig.savefig(ROOT/'figures/prior_train.png')

display(fig)
plt.close(fig)
```

### Métriques et traçabilité

<a id="tp2-020"></a>

#### Cellule de code 020 — Définir les métriques de classification

**Objectif.** Évaluer toutes les méthodes avec les mêmes conventions.

**Comprendre le code.** La décision est `p >= threshold`. La précision mesure la part des alertes correctes ; le rappel, la part des sarcastiques retrouvés. Le F1 sarcastique combine ces deux quantités. Le macro-F1 moyenne les F1 des deux classes sans les pondérer par leur taille. L’AP évalue le classement des scores sans choisir un seuil unique. `zero_division=0` traite les cas sans prédiction positive.

**Exemple ou repère.** $P=TP/(TP+FP)$, $R=TP/(TP+FN)$ et $F1=2TP/(2TP+FP+FN)$. TP : vrai positif ; FP : fausse alerte ; FN : sarcasme manqué.

**Résultat attendu à l’exécution.** Un appel à `scores` renverra un dictionnaire de métriques ; cette définition n’évalue encore aucun modèle.

**À retenir.** L’AP est la convention PR-AUC retenue ici ; ce n’est pas l’intégrale trapézoïdale. Si un sous-groupe ne contient aucun positif, son AP est notée non définie plutôt qu’interprétée comme une qualité normale.

**Lecture complémentaire, dans l’ordre des opérations.**

Le vecteur `p` doit être aligné sur `y` et le seuil doit correspondre au run. Si un sous-groupe ne contient aucun positif réel, l'AP est renvoyée comme NaN : ce n'est pas une performance nulle démontrée. `zero_division=0` rend les métriques bien définies lorsque le dénominateur s'annule.

**Définitions importantes :** `scores`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def scores(y,p,threshold):
    y=np.asarray(y,dtype=int); p=np.asarray(p); pred=(p>=threshold).astype(int)
    return {'f1_sarcastique':float(f1_score(y,pred,zero_division=0)),
        'precision':float(precision_score(y,pred,zero_division=0)),
        'recall':float(recall_score(y,pred,zero_division=0)),
        'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
        'pr_auc_ap':float(average_precision_score(y,p)) if y.sum() else float('nan')}
```

<a id="tp2-021"></a>

#### Cellule de code 021 — Choisir un seuil sur la validation

**Objectif.** Adapter la règle de décision au déséquilibre sans utiliser les réponses du test.

**Comprendre le code.** `precision_recall_curve` fournit les précisions et rappels associés aux seuils des scores. Le code calcule leurs F1, évite une division par zéro et recherche le maximum. Parmi les valeurs jugées numériquement équivalentes avec `np.isclose`, il retient la dernière, donc le seuil le plus élevé dans cet ordre. Le dernier point de la courbe PR n’a pas de seuil et est écarté.

**Résultat attendu à l’exécution.** La fonction renverra un scalaire à enregistrer avec le modèle puis à conserver pendant le test.

**À retenir.** Le seuil 0,5 n’est pas automatiquement optimal, surtout après pondération de la perte. Sélectionner un seuil améliore la décision sur val, mais ne garantit pas une probabilité bien calibrée.

**Lecture complémentaire, dans l’ordre des opérations.**

`precision_recall_curve` fournit un seuil de moins que de valeurs précision/rappel ; d'où `[:-1]`. La formule calcule le F1 de chaque seuil et retient le dernier des maxima proches, donc le seuil le plus élevé parmi les ex aequo retournés. Ce choix est adapté à val ; l'utiliser sur test n'est permis ici que pour un diagnostic posthoc.

**Définitions importantes :** `best_threshold`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def best_threshold(y,p):
    prec,rec,thresholds=precision_recall_curve(y,p)
    fs=2*prec[:-1]*rec[:-1]/np.maximum(prec[:-1]+rec[:-1],1e-12)
    i=np.flatnonzero(np.isclose(fs,fs.max(),atol=1e-12))[-1]
    return float(thresholds[i])
```

<a id="tp2-022"></a>

#### Cellule de code 022 — Initialiser le journal et identifier les données

**Objectif.** Créer les structures de suivi et relier les futurs caches à la version réelle des partitions.

**Comprendre le code.** `RESULTS` contiendra les lignes de métriques, `REGISTRY` les chemins/configurations et `HISTORY` les courbes par époque. `DATA_SIG` résume les identifiants, textes et cibles du train. Une empreinte plus large inclut train/val, sentiment, dialecte et partition ; les révisions PLM sont ajoutées au contexte. Ces empreintes permettent de distinguer deux expériences aux entrées différentes.

**Résultat attendu à l’exécution.** Les journaux en mémoire sont initialement vides et les signatures des données sont disponibles.

**À retenir.** Cette cellule réinitialise le suivi de la session. Ne pas la relancer isolément au milieu d’un calcul ; suivre l’ordre du notebook dans un noyau neuf permet ensuite de reconstruire le journal via les checkpoints compatibles.

**Lecture complémentaire, dans l’ordre des opérations.**

`RESULTS` garde les mesures, `REGISTRY` les ressources nécessaires à l'inférence et `HISTORY` les trajectoires par époque. La signature inclut notamment partition, identifiant, texte et labels autorisés. Relancer cette cellule vide ces collections en mémoire ; les fichiers sur disque ne sont pas effacés par ces affectations.

**Code de référence.**

```python
RESULTS=[]

REGISTRY={}

HISTORY=[]

DATA_SIG=hashlib.sha256(pd.util.hash_pandas_object(train[['id','tweet','y']],index=False).values.tobytes()).hexdigest()
protocol_frames=pd.concat([train.assign(partition='train'),val.assign(partition='val')],ignore_index=True)
CACHE_CONTEXT.update(train_val_sha256=hashlib.sha256(pd.util.hash_pandas_object(
    protocol_frames[['partition','id','tweet','y','sentiment','dialect']],index=False).values.tobytes()).hexdigest(),
    model_revisions={name:r['revision'] for name,r in REVISIONS.items()})
```

<a id="tp2-023"></a>

#### Cellule de code 023 — Enregistrer une expérience de validation

**Objectif.** Associer une mesure réelle à une configuration, une graine et un fichier de modèle.

**Comprendre le code.** `register` choisit le seuil sur val sauf règle particulière fournie, calcule les métriques et construit une clé comme `nom_seed42`. Une éventuelle ligne portant la même clé est remplacée pour éviter les doublons du journal. Les probabilités val sont sauvegardées en NumPy ; `results.csv` et `registry.json` rendent les chiffres et paramètres consultables hors du noyau. Les champs supplémentaires décrivent notamment le nombre d’époques.

**Résultat attendu à l’exécution.** Lors d’un appel, une ligne de suivi est affichée et les artefacts de validation sont exportés.

**À retenir.** Le temps transmis est celui du run mesuré ou du run historique repris ; il ne faut pas présenter une simple lecture de cache comme un nouvel entraînement. Aucun score test n’est calculé ici.

**Lecture complémentaire, dans l’ordre des opérations.**

La clé combine nom et graine : `E2_AraBERT_lambda0p5_seed42` distingue un run exact. Une ligne antérieure de même nom/graine est remplacée, pas ajoutée comme une répétition indépendante. Les chemins stockés sont relatifs à `ROOT`, ce qui permet de déplacer le projet avec ses poids.

**Définitions importantes :** `register`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def register(name,seed,pval,seconds,params,preprocessing,kind,path,config,threshold=None,extra=None):
    t=best_threshold(val.y,pval) if threshold is None else threshold
    key=f'{name}_seed{seed}'
    row={'model':name,'seed':seed,'preprocessing':preprocessing,'threshold':t,
         'train_seconds':seconds,'trainable_parameters':int(params),'n_train':len(train),
         'data_sha256':DATA_SIG,'kind':kind,**{'val_'+k:v for k,v in scores(val.y,pval,t).items()}}
    if extra: row.update(extra)
    RESULTS[:]=[r for r in RESULTS if (r['model'],r['seed'])!=(name,seed)]
    RESULTS.append(row)
    REGISTRY[key]={'key':key,'name':name,'seed':seed,'kind':kind,'path':Path(path).relative_to(ROOT).as_posix(),'config':config,'threshold':t}
    np.save(ROOT/f'predictions/{key}_val.npy',np.asarray(pval))
    pd.DataFrame(RESULTS).to_csv(ROOT/'results/results.csv',index=False)
    (ROOT/'results/registry.json').write_text(json.dumps(REGISTRY,ensure_ascii=False,indent=2),encoding='utf-8')
    print(key,'F1 val =',round(row['val_f1_sarcastique'],4),'seuil =',round(t,4),'temps =',round(seconds,1),'s',flush=True)
    return key
```

<a id="tp2-024"></a>

#### Cellule de code 024 — Contrôler la reprise d’un checkpoint

**Objectif.** Réutiliser un modèle terminé seulement lorsqu’il correspond à l’expérience demandée.

**Comprendre le code.** `cache_signature` combine version logique, empreinte des données et configuration dans un SHA256. `compatible_metadata` exige aussi l’égalité du contexte d’exécution. Les fonctions d’entraînement vérifieront la présence du modèle, des métadonnées et la correspondance de ces informations avant de charger des poids anciens.

**Résultat attendu à l’exécution.** Deux fonctions de contrôle sont définies ; elles ne lancent aucun calcul de réseau.

**À retenir.** Une empreinte identifie des entrées ; elle ne certifie pas à elle seule la correction scientifique du programme. Elle complète les métadonnées et le contrôle des fichiers, sans remplacer la nécessité de changer `RUN_VERSION` après une modification de logique.

**Optimisation CPU ajoutée.** Les métadonnées neuronales lient aussi taille et date du checkpoint terminé. Un fichier de poids remplacé ne peut pas être associé à d’anciennes probabilités par simple correspondance de nom.

**Lecture complémentaire, dans l’ordre des opérations.**

Le hash de configuration identifie les choix ; le contexte compare versions et matériel ; la taille et la date nanoseconde décrivent le checkpoint. Un fichier de même nom n'est pas forcément compatible. Cette validation évite qu'un cache ancien produise un score présenté comme celui de nouveaux paramètres.

**Définitions importantes :** `cache_signature`, `compatible_metadata`, `checkpoint_stat`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def cache_signature(config):
    return hashlib.sha256(json.dumps([RUN_VERSION,DATA_SIG,config],sort_keys=True).encode()).hexdigest()

def compatible_metadata(meta):
    if not meta.exists(): return False
    saved=json.loads(meta.read_text())
    if saved.get('runtime_context')!=CACHE_CONTEXT: return False
    expected=saved.get('checkpoint_stat')
    checkpoint=meta.with_suffix('.pt')
    return expected is None or (checkpoint.exists() and expected==checkpoint_stat(checkpoint))

def checkpoint_stat(path):
    stat=path.stat()
    return {'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
```

<a id="tp2-025"></a>

#### Cellule de code 025 — Uniformiser la sauvegarde des figures

**Objectif.** Conserver les graphiques consultables dans le notebook et dans les livrables.

**Comprendre le code.** `savefig` ajuste la disposition, construit un chemin sous `ROOT/figures/`, enregistre le PNG et affiche la figure. `bbox_inches="tight"` réduit les marges superflues et limite les étiquettes coupées. Le nom est fourni par la cellule qui produit le graphique.

**Résultat attendu à l’exécution.** Une fonction utilitaire est définie. Aucun fichier graphique n’est créé tant qu’elle n’est pas appelée avec une figure.

**À retenir.** Une image sauvegardée n’apporte pas une mesure nouvelle : elle représente les tableaux ou calculs réellement produits. Une légende précise doit indiquer partition, métrique et seuil concernés.

**Mise à jour qualité / affichage.** `savefig` vérifie que la figure contient des données ou du texte, sauvegarde l'objet reçu, puis appelle `display(fig)` et le ferme. L'affichage ne dépend plus de la figure courante de pyplot.

**Lecture complémentaire, dans l’ordre des opérations.**

`ax.has_data()` ou des textes vérifient qu'au moins un axe possède du contenu. `display(fig)` affiche explicitement cette figure puis `plt.close(fig)` libère son état Matplotlib. Cela évite de laisser le backend choisir une autre figure courante ; le contrôle ne prouve toutefois pas que tous les sous-axes d'une figure sont sémantiquement complets.

**Définitions importantes :** `savefig`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def savefig(fig,name):
    if not any(ax.has_data() or ax.texts for ax in fig.axes):
        raise ValueError(f'Figure vide : {name}. Exécuter la cellule de tracé complète.')
    fig.tight_layout()
    fig.savefig(ROOT/f'figures/{name}.png',bbox_inches='tight')
    display(fig)  # Afficher cet objet précis, même si le backend inline l'a fermé.
    plt.close(fig)
```

<a id="tp2-026"></a>

#### Cellule de code 026 — Évaluer le classifieur jamais sarcastique

**Objectif.** Montrer pourquoi l’accuracy seule serait trompeuse dans ce problème déséquilibré.

**Comprendre le code.** `np.zeros(len(val))` prédit la classe non sarcastique pour chaque tweet, sans analyser son contenu. L’accuracy compte ses décisions correctes sur la majorité ; le F1 sarcastique mesure son incapacité à détecter la classe d’intérêt. Le nombre réel de sarcastiques val est affiché pour contextualiser les deux scores.

**Résultat attendu à l’exécution.** L’accuracy vaut la proportion de non-sarcastiques ; le F1 sarcastique vaut 0. Les chiffres exacts sont calculés sur la validation.

**À retenir.** Une méthode doit être comparée à cette référence avant de qualifier une accuracy élevée de bonne performance. La métrique principale du TP reste le F1 sarcastique, complété par AP et macro-F1.

**Lecture complémentaire, dans l’ordre des opérations.**

Le vecteur de zéros prédit toujours la classe 0. L'accuracy peut sembler élevée parce que les non-sarcastiques sont majoritaires, mais le rappel et le F1 sarcastiques sont nuls. Cette référence interdit de présenter l'accuracy seule comme une preuve de qualité.

**Code de référence.**

```python
trivial=np.zeros(len(val))

print('Jamais sarcastique : accuracy =',accuracy_score(val.y,trivial),'F1 =',f1_score(val.y,trivial))

print('Sarcastiques val :',int(val.y.sum()),'sur',len(val))
```

**Petit extrait de sortie déjà enregistrée dans ce fichier (non recalculée pour le guide).**

```text
Jamais sarcastique : accuracy = 0.8270916334661355 F1 = 0.0
Sarcastiques val : 217 sur 1255
```

### Partie A : TF-IDF

<a id="tp2-027"></a>

#### Cellule de code 027 — Construire la baseline TF-IDF et régression logistique

**Objectif.** Combiner des indices lexicaux et des indices de forme dans un modèle linéaire interprétable.

**Comprendre le code.** `FeatureUnion` concatène TF-IDF de mots 1–2 et de caractères 2–5. Les mots décrivent expressions et thèmes ; les caractères captent graphies, rires et allongements. `min_df=2` ignore les unités trop rares et les limites de vocabulaire bornent la taille. La fréquence est transformée de façon sous-linéaire. La régression logistique prédit un score binaire ; `class_weight="balanced"` est activé selon la variante.

**Exemple ou repère.** Pour une unité, l’IDF lissé suit $log((1+n)/(1+df))+1$. Une unité présente dans presque tous les tweets est moins discriminante qu’une unité plus rare.

**Résultat attendu à l’exécution.** La fonction renvoie un pipeline non ajusté. Le vocabulaire, l’IDF et les coefficients seront appris dans `fit`, uniquement sur train.

**À retenir.** Dans cette implémentation, le nettoyage est réalisé par `run_tfidf` avant le pipeline. Le vecteur TF-IDF est sparse : la plupart de ses composantes sont nulles pour un tweet donné.

**Lecture complémentaire, dans l’ordre des opérations.**

Le pipeline concatène deux matrices TF-IDF creuses ; il ne concatène pas le sentiment ou le dialecte. Les plafonds de 60 000 mots/ngrammes et 100 000 caractères/ngrammes limitent les caractéristiques apprises. `solver='liblinear'` est utilisé pour cette régression binaire ; aucune époque BERT n'est impliquée.

**Définitions importantes :** `make_tfidf`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def make_tfidf(mode='conservative',balanced=True,seed=42):
    # top-level prepare_text est enregistré avec le modèle ; aucun label auxiliaire en entrée.
    return Pipeline([('features',FeatureUnion([
        ('word',TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=60000,token_pattern=r'(?u)\b\w+\b',sublinear_tf=True)),
        ('char',TfidfVectorizer(analyzer='char',ngram_range=(2,5),min_df=2,max_features=100000,sublinear_tf=True))])),
        ('clf',LogisticRegression(C=1,max_iter=1200,class_weight='balanced' if balanced else None,random_state=seed,solver='liblinear'))])
```

<a id="tp2-028"></a>

#### Cellule de code 028 — Exécuter ou reprendre une variante TF-IDF

**Objectif.** Appliquer les mêmes règles de préparation et de validation à plusieurs stratégies de déséquilibre.

**Comprendre le code.** Le texte est nettoyé selon `mode` et `drop_links`. En suréchantillonnage, des positifs du train sont tirés avec remise jusqu’à équilibrer les deux classes. La signature inclut graine, choix et identifiants. Un cache compatible est chargé ; sinon le pipeline est ajusté puis sauvegardé. Les probabilités val sont ensuite calculées et transmises à `register`.

**Résultat attendu à l’exécution.** Un appel produit une clé d’expérience, un checkpoint `joblib`, ses métadonnées et les probabilités val.

**À retenir.** Le suréchantillonnage ne modifie ni val ni test. La durée historique reste associée au modèle repris ; le seuil est déterminé sur validation, sans toucher aux labels test.

**Lecture complémentaire, dans l’ordre des opérations.**

Le sur-échantillonnage est effectué sur les indices du train de cette variante, avec remise pour la minorité. Un cache compatible recharge le pipeline joblib ; sinon `.fit(x,y)` entraîne réellement la baseline. Les probabilités val sont toujours calculées avec le même mode de préparation que celui du run.

**Définitions importantes :** `run_tfidf`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def run_tfidf(name,mode='conservative',imbalance='weighted',seed=42,drop_links=False,train_frame=None):
    frame=train if train_frame is None else train_frame
    x=[prepare_text(s,mode,drop_links) for s in frame.tweet]; y=frame.y.to_numpy()
    if imbalance=='oversample':
        rng=np.random.default_rng(seed); idx=np.concatenate([np.flatnonzero(y==0),rng.choice(np.flatnonzero(y==1),sum(y==0),replace=True)])
        x=[x[i] for i in idx]; y=y[idx]
    config={'mode':mode,'imbalance':imbalance,'drop_links':drop_links,'seed':seed,'ids':frame.id.tolist()}
    path=ROOT/f'models/{name}_seed{seed}.joblib'; meta=path.with_suffix('.json'); sig=cache_signature(config)
    if path.exists() and compatible_metadata(meta) and json.loads(meta.read_text())['signature']==sig:
        model=joblib.load(path); seconds=json.loads(meta.read_text())['seconds']
    else:
        model=make_tfidf(mode,imbalance=='weighted',seed); start=time.perf_counter(); model.fit(x,y); seconds=time.perf_counter()-start
        joblib.dump(model,path); meta.write_text(json.dumps({'signature':sig,'seconds':seconds,'runtime_context':CACHE_CONTEXT}))
    p=model.predict_proba([prepare_text(s,mode,drop_links) for s in val.tweet])[:,1]
    params=model.named_steps['clf'].coef_.size+1
    return register(name,seed,p,seconds,params,mode,'tfidf',path,config)
```

<a id="tp2-029"></a>

#### Cellule de code 029 — Comparer les baselines et les règles superficielles

**Objectif.** Distinguer une vraie baseline textuelle de raccourcis pouvant exploiter la provenance du corpus.

**Comprendre le code.** Quatre variantes sont évaluées : nettoyage conservateur pondéré, agressif pondéré, perte standard et suréchantillonnage. Puis chaque indicateur URL/mention/guillemets est utilisé seul. L’orientation de la règle est choisie sur train : prédire sarcastique quand l’indice est présent ou absent, selon les taux observés. La qualité de cette règle est mesurée sur val.

**Résultat attendu à l’exécution.** Deux tableaux montrent métriques/seuils des variantes et résultats des trois règles ; ces derniers sont exportés dans `shortcut_rules_val.csv`.

**À retenir.** Un bon score d’une règle à un indice signale un raccourci possible. Il ne prouve pas une compréhension de l’ironie ; les sous-groupes et réentraînements sans liens de F permettront de compléter ce diagnostic.

**Lecture complémentaire, dans l’ordre des opérations.**

Les quatre variantes isolent nettoyage, pondération et fréquence des classes. Pour la règle URL/mention/guillemet, le sens de la corrélation est choisi sur train, puis la règle est évaluée sur val. Ces règles de surface ne démontrent pas une compréhension du contraste ironique.

**Code de référence.**

```python
baseline_keys=[run_tfidf('A_tfidf_conservateur'),run_tfidf('A_tfidf_agressif',mode='aggressive'),
               run_tfidf('A_tfidf_standard',imbalance='standard'),run_tfidf('A_tfidf_sur_echantillonne',imbalance='oversample')]

display(pd.DataFrame(RESULTS)[['model','val_f1_sarcastique','val_pr_auc_ap','threshold','train_seconds']])

rule_rows=[]

features_val=text_features(val.tweet)

for feature in ['url','mention','artefact_guillemets']:
    flag_train=features_train[feature].to_numpy(); flag_val=features_val[feature].to_numpy()
    # Deux orientations ; orientation choisie sur train pour ne pas ajouter de sélection test.
    direction=1 if train.y[flag_train].mean()>=train.y[~flag_train].mean() else 0
    p=(flag_val==bool(direction)).astype(float)
    rule_rows.append({'indice':feature,'orientation':direction,**scores(val.y,p,.5)})

rules=pd.DataFrame(rule_rows)

rules.to_csv(ROOT/'results/shortcut_rules_val.csv',index=False)

display(rules)
```

### Partie B : NLI et sentiment

<a id="tp2-030"></a>

#### Cellule de code 030 — Déclarer les modèles imposés et les hypothèses NLI

**Objectif.** Fixer les checkpoints et les formulations qui seront comparés sans entraînement sur le sarcasme.

**Comprendre le code.** Les noms identifient exactement NLI multilingue, CAMeLBERT sentiment, AraBERT Twitter et MARBERTv2. Le manifeste local doit être présent ; cette assertion évite une attente ou un téléchargement implicite. `PROMPTS` contient, en arabe/français/anglais, une hypothèse sarcastique puis une hypothèse sérieuse. Les chemins locaux des modèles sont recalculés après lecture des révisions.

**Résultat attendu à l’exécution.** Les trois paires d’hypothèses sont prêtes et les modèles sont associés à leurs ressources locales.

**À retenir.** L’ordre des deux hypothèses compte : la première représente ici le sarcasme. Le choix ultérieur du meilleur prompt se fera sur val ; aucune formulation n’est sélectionnée à partir du test.

**Colab.** Les quatre PLM manquants sont téléchargés ici aux révisions exactes du projet, puis leurs poids sont contrôlés par SHA256. Cette préparation peut prendre plusieurs minutes et ne produit aucune métrique. Une trace de taille/date/hash permet de réutiliser les téléchargements vérifiés.

**Lecture complémentaire, dans l’ordre des opérations.**

Cette cellule peut télécharger les poids nécessaires dans Colab, avant d'annoncer leurs noms. AraBERT et MARBERT sont les encodeurs de D/E ; mDeBERTa sert au NLI ; CAMeLBERT sert au sentiment. Le couple d'hypothèses conserve l'ordre sarcastique puis sérieux dans chaque langue.

**Code de référence.**

```python
if IN_COLAB: prepare_models(ROOT)

NLI_NAME='MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'

assert (ROOT/'models/assets_complete.json').exists(), 'Poids locaux manquants dans models/hf.'

REVISIONS.update(json.loads((ROOT/'models/revisions.json').read_text()))

SA_NAME='CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment'

ARA_NAME='aubmindlab/bert-base-arabertv02-twitter'

MAR_NAME='UBC-NLP/MARBERTv2'

PROMPTS={'ar':('هذه التغريدة ساخرة.','هذه التغريدة جادة.'),
         'fr':('Ce tweet est sarcastique.','Ce tweet est sérieux.'),
         'en':('This tweet is sarcastic.','This tweet is serious.')}
for model_name,entry in REVISIONS.items():
    entry['path']=str(ROOT/'models/hf'/model_name.replace('/','--'))
```

<a id="tp2-031"></a>

#### Cellule de code 031 — Définir un chargement local des PLM

**Objectif.** Utiliser le type de modèle adapté à l’expérience sans remplacer le checkpoint demandé.

**Comprendre le code.** `AutoModelForSequenceClassification` charge aussi la tête déjà entraînée pour NLI ou sentiment. `AutoModel` charge l’encodeur dont on exploitera les états cachés dans D/E. `local_files_only=True` impose l’utilisation des ressources présentes. Le modèle est envoyé sur `DEVICE`, qui est le CPU dans l’environnement actuel.

**Résultat attendu à l’exécution.** La fonction est définie ; ses appels chargeront effectivement les poids du modèle demandé.

**À retenir.** Charger un modèle est plus coûteux qu’importer sa classe. Une absence de poids doit être résolue explicitement, sans substituer silencieusement un petit modèle ni annoncer que le calcul a été réalisé.

**Optimisation CPU ajoutée.** SDPA, déjà choisi par défaut dans la version locale pour BERT, est rendu explicite pour les checkpoints BERT compatibles. DeBERTa conserve son implémentation. Un seul encodeur gelé peut être réutilisé ; il est libéré avant un autre chargement PLM.

**Lecture complémentaire, dans l’ordre des opérations.**

`local_files_only=True` force l'utilisation des ressources déjà préparées, après éventuels téléchargements de la cellule 030. SDPA est demandé uniquement aux modèles de type BERT pris en charge par ce code. `frozen_encoder` met l'encodeur en eval et désactive ses gradients ; le cache ne conserve qu'un extracteur gelé à la fois.

**Définitions importantes :** `load_hf`, `frozen_encoder`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def load_hf(name,classification=False):
    FROZEN_MODELS.clear()  # Libérer l'extracteur avant un nouveau PLM, sous 16 Go.
    if name not in REVISIONS: REVISIONS.update(json.loads((ROOT/'models/revisions.json').read_text()))
    folder=ROOT/'models/hf'/name.replace('/','--')
    config=json.loads((folder/'config.json').read_text())
    options={'attn_implementation':'sdpa'} if config['model_type']=='bert' else {}
    cls=AutoModelForSequenceClassification if classification else AutoModel
    return cls.from_pretrained(str(folder),local_files_only=True,**options).to(DEVICE)

def frozen_encoder():
    key=(ARA_NAME,str(DEVICE))
    if key not in FROZEN_MODELS:
        encoder=load_hf(ARA_NAME).eval().requires_grad_(False)
        FROZEN_MODELS[key]=encoder
    return FROZEN_MODELS[key]
```

<a id="tp2-032"></a>

#### Cellule de code 032 — Identifier une inférence mise en cache

**Objectif.** Empêcher la réutilisation de scores calculés sur d’autres textes, prompts ou versions.

**Comprendre le code.** L’empreinte couvre la liste ordonnée des textes, un tag décrivant l’usage, la longueur maximale, la version logique, PyTorch, les révisions PLM et le hash du prétraitement exporté. Les textes sont ordonnés : une permutation doit aussi être détectée puisque les scores sauvegardés sont alignés ligne par ligne.

**Résultat attendu à l’exécution.** Un appel retourne une signature SHA256 pour une inférence précise.

**À retenir.** Le cache d’inférence est distinct du checkpoint d’apprentissage. Des poids identiques ne suffisent pas si le texte d’entrée, l’ordre ou la préparation a changé.

**Lecture complémentaire, dans l’ordre des opérations.**

La signature d'inférence dépend des textes, révisions, longueur et fichier de prétraitement ; elle ne dépend pas seulement du nom `val`. Sur GPU, le contexte CUDA entre aussi dans la signature. Cette identité aide à éviter le mélange de scores issus de textes ou conventions différents.

**Définitions importantes :** `inference_signature`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def inference_signature(texts,tag):
    payload={'texts':list(texts),'tag':tag,'max_len':MAX_LEN,'run_version':RUN_VERSION,'torch':torch.__version__,
             'revisions':{k:v['revision'] for k,v in REVISIONS.items()},
             'preprocess_sha256':hashlib.sha256((ROOT/'src/preprocess.py').read_bytes()).hexdigest()}
    if CACHE_CONTEXT.get('device_type')=='cuda': payload['gpu_context']=CACHE_CONTEXT
    return hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
```

<a id="tp2-033"></a>

#### Cellule de code 033 — Vérifier la validité d’un cache de scores

**Objectif.** Savoir si une extraction ou une inférence peut être relue sans recalcul.

**Comprendre le code.** La fonction construit le nom du fichier de métadonnées voisin, calcule la signature attendue puis compare la signature et la taille du fichier. Si l’un des fichiers manque ou si les entrées ne correspondent plus, elle renvoie faux. Les fonctions NLI, sentiment et extraction gelée utilisent ce contrôle.

**Résultat attendu à l’exécution.** Un booléen indique si le cache correspond à l’appel courant.

**À retenir.** Cette vérification n’est pas une nouvelle inférence. Le contrôle de taille n’est pas un hash complet du fichier de scores ; les audits des probabilités et des ressources complètent donc la vérification de provenance.

**Lecture complémentaire, dans l’ordre des opérations.**

Le cache doit posséder son fichier compagnon `.cache.json`, une signature égale et la bonne taille. Ces vérifications ne sont pas un recalcul de métrique ; elles décident si une sortie antérieure peut être réutilisée. Le cache plus spécifique de validation, cellule 034, ajoute un hash des probabilités.

**Définitions importantes :** `valid_cache`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def valid_cache(path,texts,tag):
    sidecar=path.with_name(path.name+'.cache.json')
    signature=inference_signature(texts,tag)
    if path.exists() and sidecar.exists():
        meta=json.loads(sidecar.read_text())
        return meta['signature']==signature and meta['bytes']==path.stat().st_size
    return False
```

<a id="tp2-034"></a>

#### Cellule de code 034 — Enregistrer les métadonnées d’un cache d’inférence

**Objectif.** Conserver la signature nécessaire à une reprise ultérieure contrôlée.

**Comprendre le code.** `mark_cache` écrit un petit JSON à côté du fichier de probabilités ou de représentations. Il contient la signature calculée à partir des entrées et la taille réelle de l’artefact. Il doit être appelé après la sauvegarde effective de cet artefact.

**Résultat attendu à l’exécution.** Lors d’un appel, un fichier suffixé `.cache.json` est créé ou actualisé.

**À retenir.** La présence d’un fichier ne suffit pas à prouver son origine. Ces métadonnées rendent explicite à quels textes et à quelle configuration il correspond, sans modifier les scores ni les labels.

**Optimisation CPU ajoutée.** La fonction additionnelle validation_scores conserve les scores du meilleur checkpoint. Les signatures de données/configuration, le fichier de poids et le hash des probabilités sont vérifiés avant de sauter une validation redondante.

**Lecture complémentaire, dans l’ordre des opérations.**

`candidate` peut contenir les probabilités déjà calculées pour le meilleur checkpoint : on évite alors un second forward identique. Sans candidat, on vérifie hash, forme `(N,)`, valeurs finies et plage [0,1] du cache. En cas d'incompatibilité, le code refait une vraie inférence plutôt que fabriquer les scores.

**Définitions importantes :** `mark_cache`, `validation_scores`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def mark_cache(path,texts,tag):
    path.with_name(path.name+'.cache.json').write_text(json.dumps({
        'signature':inference_signature(texts,tag),'bytes':path.stat().st_size},indent=2))

def validation_scores(model,inputs,path,signature,candidate=None,plm=False):
    target=ROOT/f'predictions/{path.stem}_val.npy'
    meta=target.with_suffix('.validation.json')
    expected={'signature':signature,'checkpoint_stat':checkpoint_stat(path),'context':CACHE_CONTEXT}
    if candidate is None and target.exists() and meta.exists():
        try:
            saved=json.loads(meta.read_text())
            digest=hashlib.sha256(target.read_bytes()).hexdigest()
            if all(saved.get(k)==v for k,v in expected.items()) and saved['probability_sha256']==digest:
                p=np.load(target,allow_pickle=False)
                if p.shape==(len(inputs[0]),) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all(): return p
        except (OSError,ValueError,KeyError): pass  # Cache invalide : vraie inférence ci-dessous.
    p=candidate if candidate is not None else neural_predict(model,inputs,batch=BATCH if plm else 64)
    np.save(target,p)
    expected['probability_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
    expected['producer']='checkpoint_validation'
    meta.write_text(json.dumps(expected,indent=2))
    return p
```

<a id="tp2-035"></a>

#### Cellule de code 035 — Calculer le score zero-shot par NLI

**Objectif.** Reformuler le sarcasme comme une comparaison entre un tweet et deux hypothèses.

**Comprendre le code.** Pour chaque tweet, deux paires prémisse/hypothèse sont créées. Le code trouve le label entailment dans `id2label`, extrait un logit par hypothèse, regroupe les deux logits par tweet puis applique un softmax entre eux. Le premier score devient celui du sarcasme. La prémisse est bornée à 64 tokens avant la construction des paires ; une marge conserve les hypothèses. `eval()` et `inference_mode()` évitent l’apprentissage.

**Résultat attendu à l’exécution.** Un vecteur de scores aligné sur les textes est renvoyé avec un temps d’inférence ; un cache compatible peut le fournir directement.

**À retenir.** Ce score est une compatibilité relative entre deux hypothèses, pas une probabilité de sarcasme garantie calibrée. Le modèle n’apprend pas sur ArSarcasm dans B1, mais prompt et seuil seront adaptés sur val.

**Lecture complémentaire, dans l’ordre des opérations.**

Pour chaque tweet, deux paires prémisse/hypothèse sont créées. Le logit entailment est remis en forme `(B,2)` et softmax compare les deux hypothèses, pas les trois classes NLI originales. `inference_mode()` interdit la construction du graphe de gradients ; aucun ajustement ArSarcasm n'a lieu.

**Définitions importantes :** `cached_nli`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def cached_nli(texts,language,split):
    path=ROOT/f'predictions/B1_{language}_{split}_raw.npy'
    if valid_cache(path,texts,['nli',language,PROMPTS[language]]): return np.load(path),0.0
    tokenizer=AutoTokenizer.from_pretrained(REVISIONS[NLI_NAME]['path'],local_files_only=True)
    model=load_hf(NLI_NAME,True).eval()
    print('NLI labels :',model.config.id2label,flush=True)
    entail=[i for i,label in model.config.id2label.items() if 'entail' in label.lower()]
    assert len(entail)==1
    probs=[]; start=time.perf_counter()
    with torch.inference_mode():
        for j in range(0,len(texts),BATCH):
            batch=[export_clean(s) for s in texts[j:j+BATCH]]
            # Tronquer la prémisse seule garde les deux hypothèses complètes.
            batch=tokenizer.batch_decode(tokenizer(batch,truncation=True,max_length=MAX_LEN,add_special_tokens=False)['input_ids'],skip_special_tokens=True)
            premises=[s for s in batch for _ in range(2)]; hypotheses=[h for _ in batch for h in PROMPTS[language]]
            enc=tokenizer(premises,hypotheses,padding=True,truncation='only_first',max_length=MAX_LEN+24,return_tensors='pt').to(DEVICE)
            logits=model(**enc).logits[:,entail[0]].reshape(-1,2)
            probs.extend(logits.softmax(1)[:,0].cpu().numpy())
            if (j//BATCH+1)%20==0: print('B1',language,split,j+len(batch),'/',len(texts),flush=True)
    elapsed=time.perf_counter()-start
    np.save(path,np.asarray(probs)); mark_cache(path,texts,['nli',language,PROMPTS[language]]); del model; gc.collect()
    return np.asarray(probs),elapsed
```

<a id="tp2-036"></a>

#### Cellule de code 036 — Comparer les trois langues d’hypothèse

**Objectif.** Mesurer la sensibilité du zero-shot à la formulation sur la même validation.

**Comprendre le code.** La boucle appelle NLI pour arabe, français et anglais, puis enregistre chaque variante avec son propre seuil val. Les champs `train_seconds=0` et paramètres entraînés égaux à 0 signifient qu’aucun poids n’est ajusté sur le train du TP. Le temps d’inférence est suivi séparément ; à la lecture d’un cache, le temps renvoyé par cette lecture ne mesure pas l’inférence initiale.

**Résultat attendu à l’exécution.** `nli_keys` contient trois références comparables sur 1 255 tweets val ; les probabilités et métriques sont sauvegardées.

**À retenir.** Comparer les trois résultats permet de discuter la formulation et l’alignement linguistique. On choisira le meilleur sur val avant le test, au lieu de retenir après coup la langue la plus favorable sur test.

**Lecture complémentaire, dans l’ordre des opérations.**

La boucle réalise trois évaluations val et enregistre un seuil pour chaque langue. Le temps d'entraînement est nul parce que le modèle n'est pas ajusté ; le temps d'inférence est conservé dans une colonne différente. Un cache peut rendre ce dernier temps retourné nul sans rendre l'inférence conceptuellement gratuite.

**Code de référence.**

```python
nli_keys=[]

for language in PROMPTS:
    p,elapsed=cached_nli(val.tweet.tolist(),language,'val')
    key=register(f'B1_NLI_{language}',42,p,0,0,'export_only','nli',REVISIONS[NLI_NAME]['path'],
        {'language':language,'model_name':NLI_NAME},extra={'val_inference_seconds':elapsed})
    nli_keys.append(key)
```

<a id="tp2-037"></a>

#### Cellule de code 037 — Prédire le sentiment et construire la règle B2

**Objectif.** Tester l’heuristique négatif implique sarcastique avec le modèle de sentiment imposé.

**Comprendre le code.** Le code inspecte `id2label` pour trouver la classe négative sans supposer son indice. Le softmax donne les probabilités des sentiments. Deux sorties sont gardées : `p`, la probabilité négative, et `rule`, vraie seulement si la classe la plus probable est négative. Le modèle est en évaluation sans gradients et les sorties peuvent être mises en cache.

**Résultat attendu à l’exécution.** Un appel fournit une probabilité, une décision de règle et un temps pour chaque texte.

**À retenir.** La règle `argmax = négatif` diffère de `P(négatif) >= 0,5` : une classe peut gagner avec 0,4 si les deux autres ont chacune 0,3. La contamination possible par ASTD/SemEval devra limiter l’interprétation.

**Lecture complémentaire, dans l’ordre des opérations.**

La fonction renvoie trois objets : `p` (probabilité du sentiment négatif), `rule` (argmax négatif) et le temps. Ces deux vecteurs ne doivent pas être intervertis. L'indice négatif est retrouvé dans la configuration du modèle, qui peut différer du codage `SENT_MAP` de E2.

**Définitions importantes :** `sentiment_predict`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def sentiment_predict(texts,split):
    path=ROOT/f'predictions/B2_{split}_raw.npz'
    if valid_cache(path,texts,'sentiment'):
        a=np.load(path); return a['p'],a['rule'],0.0
    tokenizer=AutoTokenizer.from_pretrained(REVISIONS[SA_NAME]['path'],local_files_only=True)
    model=load_hf(SA_NAME,True).eval(); print('CAMeLBERT labels :',model.config.id2label,flush=True)
    neg=[i for i,label in model.config.id2label.items() if 'neg' in label.lower()]
    assert len(neg)==1
    probs=[]; rule=[]; start=time.perf_counter()
    with torch.inference_mode():
        for j in range(0,len(texts),BATCH):
            enc=tokenizer([export_clean(s) for s in texts[j:j+BATCH]],truncation=True,max_length=MAX_LEN,padding=True,return_tensors='pt').to(DEVICE)
            pred=model(**enc).logits.softmax(-1)
            probs.extend(pred[:,neg[0]].cpu().numpy()); rule.extend((pred.argmax(-1)==neg[0]).cpu().numpy())
    elapsed=time.perf_counter()-start; np.savez(path,p=probs,rule=rule)
    mark_cache(path,texts,'sentiment')
    del model; gc.collect(); return np.asarray(probs),np.asarray(rule),elapsed
```

<a id="tp2-038"></a>

#### Cellule de code 038 — Évaluer la règle de sentiment sur validation

**Objectif.** Enregistrer la vraie heuristique B2 sans la remplacer par un seuil optimisé sur P(négatif).

**Comprendre le code.** `rule_sent` est convertie en 0/1 et passée à `register` avec seuil 0,5 ; cela conserve exactement sa décision binaire. Pour l’AP, le champ est remplacé par la valeur calculée sur `p_sent`, qui contient les scores continus. Le tableau filtre ensuite les résultats B1/B2 pour une lecture commune.

**Résultat attendu à l’exécution.** Le notebook affiche précision, rappel, F1 et AP des méthodes directes disponibles.

**À retenir.** Une plainte peut être négative sans être ironique ; une louange ironique peut paraître positive. Le score de B2 mesure donc une corrélation imparfaite entre tâches, avec en plus un recouvrement potentiel des corpus de préentraînement supervisé.

**Lecture complémentaire, dans l’ordre des opérations.**

`register` reçoit la décision binaire pour les métriques de B2 et un seuil formel de 0,5 sur ce vecteur 0/1. L'AP calculée sur la vraie probabilité négative remplace la valeur qui aurait été obtenue sur les décisions binaires. Le seuil affiché ici ne signifie donc pas `P(négatif) ≥ 0,5`.

**Code de référence.**

```python
p_sent,rule_sent,elapsed=sentiment_predict(val.tweet.tolist(),'val')

b2_key=register('B2_negatif_sarcastique',42,rule_sent.astype(float),0,0,'export_only','sentiment',
    REVISIONS[SA_NAME]['path'],{'model_name':SA_NAME},threshold=.5,extra={'val_inference_seconds':elapsed,'val_pr_auc_ap':float(average_precision_score(val.y,p_sent))})

display(pd.DataFrame(RESULTS).loc[lambda d:d.model.str.startswith('B')])
```

### Partie C : outils et réseaux statiques

<a id="tp2-039"></a>

#### Cellule de code 039 — Définir une tokenisation Unicode pour les réseaux statiques

**Objectif.** Choisir les unités auxquelles seront associés les embeddings de la partie C.

**Comprendre le code.** Le motif `\w+|[^\w\s]` reconnaît soit une suite de caractères de mot, soit un signe isolé qui n’est ni un caractère de mot ni un espace. Avec Unicode, les mots arabes ne sont pas traités comme du bruit. La ponctuation et les caractères d’emoji restent accessibles, contrairement à une simple suppression de tous les signes.

**Résultat attendu à l’exécution.** Une expression régulière est définie. Les textes seront découpés par `word_tokens` dans la cellule suivante.

**À retenir.** Cette tokenisation ne correspond pas aux sous-mots de BERT. Certains emojis composés peuvent être répartis en plusieurs caractères ; on ne prétend pas que chaque unité corresponde à un mot linguistique complet.

**Lecture complémentaire, dans l’ordre des opérations.**

La regex reconnaît soit une suite de caractères de mot, soit un signe non espace. Le mode Unicode évite de perdre l'arabe. Cette définition reste plus simple qu'une segmentation morphologique complète et peut découper des emojis composés en plusieurs unités.

**Code de référence.**

```python
TOKEN_RE=re.compile(r'\w+|[^\w\s]',re.UNICODE)
```

<a id="tp2-040"></a>

#### Cellule de code 040 — Associer nettoyage conservateur et tokenisation

**Objectif.** Définir une entrée identique pour construire le vocabulaire et encoder les textes des réseaux C.

**Comprendre le code.** `word_tokens` commence par `prepare_text` dans son mode conservateur, puis applique le motif Unicode. La fonction renvoie une liste de chaînes. Ainsi, les variantes graphiques normalisées de la même manière sont comptées de façon cohérente, et les marqueurs gardés par le nettoyage peuvent devenir des tokens.

**Résultat attendu à l’exécution.** Chaque appel produit une liste de tokens ; définir cette fonction ne construit pas encore le vocabulaire.

**À retenir.** Il faut utiliser la même préparation sur train, val et test. En revanche, seules les fréquences du train peuvent décider quelles unités obtiennent une entrée dans le vocabulaire.

**Lecture complémentaire, dans l’ordre des opérations.**

`word_tokens` appelle le nettoyage conservateur puis la regex. Tous les calculs de vocabulaire, OOV et encodage statique doivent réutiliser cette même convention. Les tokenizers BERT, eux, possèdent leur vocabulaire pré-entraîné indépendant.

**Définitions importantes :** `word_tokens`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def word_tokens(text): return TOKEN_RE.findall(prepare_text(text))
```

<a id="tp2-041"></a>

#### Cellule de code 041 — Apprendre le vocabulaire sur le train uniquement

**Objectif.** Transformer les tokens en identifiants sans consulter les distributions de val ou test.

**Comprendre le code.** `Counter` compte toutes les occurrences dans le train admissible. Les unités de fréquence au moins 2 sont triées pour obtenir un ordre stable, puis numérotées à partir de 2. L’identifiant 0 est réservé à PAD et 1 à UNK. Le dictionnaire est exporté en JSON avec les caractères arabes lisibles.

**Résultat attendu à l’exécution.** `VOCAB` associe chaque token connu à un entier. Le fichier `models/vocab.json` permet de retrouver exactement cette correspondance.

**À retenir.** Un mot présent une seule fois sur train pourra déjà devenir UNK ; le taux OOV du train n’est donc pas forcément nul. Ajouter les mots de val/test au vocabulaire introduirait une transformation apprise hors du train.

**Lecture complémentaire, dans l’ordre des opérations.**

Seuls les tokens observés au moins deux fois dans train reçoivent un identifiant propre. Le tri rend l'attribution des indices stable ; PAD=0 et UNK=1 sont réservés. Un token fréquent dans val mais absent de train reste UNK : il ne doit pas enrichir le vocabulaire après le split.

**Code de référence.**

```python
counts=Counter(t for s in train.tweet for t in word_tokens(s))

VOCAB={'<PAD>':0,'<UNK>':1,**{t:i+2 for i,t in enumerate(sorted(t for t,n in counts.items() if n>=2))}}

(ROOT/'models/vocab.json').write_text(json.dumps(VOCAB,ensure_ascii=False),encoding='utf-8')
```

<a id="tp2-042"></a>

#### Cellule de code 042 — Encoder les listes de tokens en tenseurs

**Objectif.** Créer les matrices d’identifiants attendues par la couche d’embedding.

**Comprendre le code.** Un tableau entier de forme `(nombre_de_tweets, MAX_LEN)` est initialisé avec PAD=0. Chaque token connu est remplacé par son indice ; sinon il devient UNK=1. Les premières positions sont remplies, les dernières restent du padding. Les textes trop longs sont tronqués. Le repli `[1]` évite une séquence entièrement vide, nécessaire pour le masquage et le BiLSTM compacté.

**Exemple ou repère.** Si trois tokens deviennent `[7, 1, 12]`, la ligne commence par ces identifiants puis contient 61 zéros. UNK et PAD ont des rôles différents.

**Résultat attendu à l’exécution.** La fonction renvoie un tenseur PyTorch entier. Avec la configuration actuelle, chaque ligne comporte 64 positions.

**À retenir.** Un identifiant n’est pas encore un vecteur sémantique. La couche `Embedding` effectuera ensuite la recherche du vecteur correspondant.

**Lecture complémentaire, dans l’ordre des opérations.**

Le résultat est un tableau d'entiers `(nombre de textes,64)` initialisé à zéro. Les premiers identifiants sont copiés, les derniers restent PAD ; un texte vidé par le nettoyage reçoit au moins UNK pour éviter une séquence de longueur nulle dans le LSTM. La troncature conserve les 64 premières unités statiques.

**Définitions importantes :** `word_encode`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def word_encode(texts):
    result=np.zeros((len(texts),MAX_LEN),dtype=np.int64)
    for i,s in enumerate(texts):
        ids=[VOCAB.get(t,1) for t in word_tokens(s)[:MAX_LEN]] or [1]
        result[i,:len(ids)]=ids
    return torch.tensor(result)
```

<a id="tp2-043"></a>

#### Cellule de code 043 — Mesurer les inconnus et la troncature

**Objectif.** Vérifier si la représentation choisie couvre suffisamment les textes.

**Comprendre le code.** La fonction compte tous les tokens avant la troncature. Le taux OOV est le nombre d’occurrences absentes de `VOCAB` divisé par le nombre total de tokens ; ce n’est pas une proportion de types distincts. Le taux `truncated` est la proportion de tweets dont la longueur dépasse `MAX_LEN`.

**Résultat attendu à l’exécution.** Un dictionnaire donne le volume de tokens, le taux OOV et le taux de tweets tronqués pour une partition.

**À retenir.** Un OOV élevé peut limiter les réseaux statiques, même avec de bons embeddings. Un faible taux de tweets tronqués ne prouve pas que les tokens retirés étaient sans importance pour le sarcasme.

**Lecture complémentaire, dans l’ordre des opérations.**

Le taux OOV concerne les tokens absents du vocabulaire propre au train, pas les tokens absents d'AraVec. La troncature concerne les tweets plus longs que MAX_LEN, avec une autre unité de comptage. Un OOV élevé sur test signale un décalage lexical sans prouver seul sa cause.

**Définitions importantes :** `token_stats`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def token_stats(texts):
    seqs=[word_tokens(s) for s in texts]; n=sum(map(len,seqs))
    return {'tokens':n,'oov':sum(t not in VOCAB for seq in seqs for t in seq)/max(n,1),
        'truncated':sum(len(seq)>MAX_LEN for seq in seqs)/len(seqs)}
```

<a id="tp2-044"></a>

#### Cellule de code 044 — Préparer entrées, cibles et poids de classes

**Objectif.** Aligner les textes et annotations dans des tenseurs de types compatibles avec PyTorch.

**Comprendre le code.** `XW_TRAIN` et `XW_VAL` contiennent les identifiants de tokens ; `YTRAIN` et `YVAL` les classes sarcasme. Les sentiments sont codés NEG=0, NEU=1, POS=2, dans des tenseurs séparés. `torch.long` est le type attendu pour les indices et les cibles de classification. Les poids `n/(2 n_c)` donnent davantage de poids à la classe rare. La boucle commune recalculera ces poids pour son sous-ensemble effectif.

**Résultat attendu à l’exécution.** Les statistiques train/val sont affichées et les tenseurs prêts à être groupés en lots.

**À retenir.** La présence de `STRAIN` dans le jeu de données ne signifie pas qu’il est une entrée textuelle : il sera utilisé seulement comme cible de perte auxiliaire dans les expériences multi-tâches.

**Lecture complémentaire, dans l’ordre des opérations.**

`YTRAIN/YVAL` sont les cibles sarcasme, `STRAIN/SVAL` les cibles sentiment. Les tableaux d'entrée ne contiennent pas ces labels sous forme de tokens supplémentaires. Les poids de classes sont calculés à partir du train : aucune fréquence de label test ne décide la perte d'apprentissage.

**Code de référence.**

```python
display(pd.DataFrame({name:token_stats(frame.tweet) for name,frame in [('train',train),('val',val)]}).T)

XW_TRAIN=word_encode(train.tweet.tolist())

XW_VAL=word_encode(val.tweet.tolist())

YTRAIN=torch.tensor(train.y.to_numpy(),dtype=torch.long)

YVAL=torch.tensor(val.y.to_numpy(),dtype=torch.long)

SENT_MAP={'NEG':0,'NEU':1,'POS':2}

STRAIN=torch.tensor(train.sentiment.map(SENT_MAP).to_numpy(),dtype=torch.long)

SVAL=torch.tensor(val.sentiment.map(SENT_MAP).to_numpy(),dtype=torch.long)

CLASS_WEIGHTS=torch.tensor(len(train)/(2*np.bincount(train.y)),dtype=torch.float32)
```

<a id="tp2-045"></a>

#### Cellule de code 045 — Implémenter le pooling attentionnel

**Objectif.** Apprendre quelles positions résument le mieux un tweet pour la classification.

**Comprendre le code.** Pour des états `h` de forme `(B, L, d)`, une couche linéaire projette vers 64 dimensions, `tanh` apporte une non-linéarité puis une seconde couche produit un score par position. Les positions PAD reçoivent un score très négatif avant softmax. Les poids obtenus servent à sommer les états, ce qui produit un vecteur de forme `(B, d)` et des poids de forme `(B, L)`.

**Exemple ou repère.** $e_i=v^T\tanh(Wh_i)$, $\alpha_i=\mathrm{softmax}(e_i)$ et $c=\sum_i\alpha_i h_i$. Le masque empêche les positions ajoutées pour le padding d’influencer ce résumé.

**Résultat attendu à l’exécution.** La classe `AttentionPool` est définie ; ses poids seront appris avec le réseau qui l’utilise.

**À retenir.** Les poids totalisent environ 1 sur une séquence valide. Une forte attention représente une pondération interne, pas une preuve que le token explique à lui seul la décision.

**Lecture complémentaire, dans l’ordre des opérations.**

La projection 128→64 puis 64→1 produit un score par position ; softmax convertit ces scores en poids. Le masque élimine la contribution des PAD avant la normalisation. La sortie résumée garde la dimension cachée d'origine, 128 dans C2/D2.

**Définitions importantes :** `AttentionPool`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
class AttentionPool(nn.Module):
    def __init__(self,dim):
        super().__init__(); self.w=nn.Linear(dim,64); self.v=nn.Linear(64,1,bias=False)
    def forward(self,h,mask):
        logits=self.v(torch.tanh(self.w(h))).squeeze(-1).masked_fill(~mask,-1e4)
        alpha=logits.softmax(-1); return (h*alpha.unsqueeze(-1)).sum(1),alpha
```

<a id="tp2-046"></a>

#### Cellule de code 046 — Construire C1, C2 et C3 dans une même classe

**Objectif.** Comparer les architectures tout en gardant une interface et une dimension de sortie communes.

**Comprendre le code.** La couche d’embedding produit 100 valeurs par token, aléatoires ou copiées depuis AraVec. C1/C2 utilisent un LSTM bidirectionnel de 64 unités par sens, donc des états de dimension 128 : C1 fait un max-pooling masqué, C2 une somme attentionnelle. C3 projette les embeddings vers 128 dimensions et applique deux couches Transformer à quatre têtes, avec positions sinusoïdales optionnelles et masque PAD. Le dropout régularise le réseau. La tête finale produit deux logits par tweet.

**Résultat attendu à l’exécution.** Une instance de `StaticNet` pourra recevoir un tenseur `(B, L)` et renvoyer `(B, 2)` ; avec attention demandée, elle peut aussi renvoyer les poids.

**À retenir.** Le compactage du BiLSTM évite que sa direction arrière traite du padding avant le texte. Sans positions, le Transformer et son pooling deviennent insensibles à une permutation des tokens en mode évaluation ; cette propriété motive l’ablation.

**Lecture complémentaire, dans l’ordre des opérations.**

La branche Transformer projette 100→128 et ajoute les positions avant deux couches ; sa moyenne finale est masquée. La branche récurrente utilise des séquences compactées et max-pooling ou attention selon `arch`. Un module `AttentionPool` est aussi instancié dans C1 mais n'est pas appelé en max-pooling : compter tous les paramètres `requires_grad` peut donc inclure des paramètres sans gradient effectif, sans changer le forward C1.

**Définitions importantes :** `StaticNet`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
class StaticNet(nn.Module):
    def __init__(self,arch='attention',embedding=None,position=True,freeze=False):
        super().__init__(); self.arch=arch
        self.embedding=nn.Embedding(len(VOCAB),100,padding_idx=0)
        if embedding is not None: self.embedding.weight.data.copy_(torch.tensor(embedding))
        self.embedding.weight.requires_grad_(not freeze)
        self.dropout=nn.Dropout(.25)
        if arch=='transformer':
            self.proj=nn.Linear(100,128)
            layer=nn.TransformerEncoderLayer(128,4,256,dropout=.2,batch_first=True)
            self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
            pos=torch.arange(MAX_LEN).unsqueeze(1); div=torch.exp(torch.arange(0,128,2)*(-math.log(10000)/128))
            pe=torch.zeros(MAX_LEN,128); pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
            self.register_buffer('pe',pe if position else torch.zeros_like(pe))
        else:
            self.encoder=nn.LSTM(100,64,batch_first=True,bidirectional=True)
            self.attention=AttentionPool(128)
        self.head=nn.Linear(128,2)
    def forward(self,x,return_attention=False):
        mask=x.ne(0); h=self.dropout(self.embedding(x)); alpha=None
        if self.arch=='transformer':
            h=self.encoder(self.proj(h)+self.pe[:x.shape[1]],src_key_padding_mask=~mask)
            pooled=(h*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp(min=1)
        else:
            # pack empêche la direction arrière de traiter du padding avant le vrai texte.
            packed=nn.utils.rnn.pack_padded_sequence(h,mask.sum(1).cpu(),batch_first=True,enforce_sorted=False)
            packed,_=self.encoder(packed)
            h,_=nn.utils.rnn.pad_packed_sequence(packed,batch_first=True,total_length=x.shape[1])
            if self.arch=='attention': pooled,alpha=self.attention(h,mask)
            else: pooled=h.masked_fill(~mask.unsqueeze(-1),-1e4).max(1).values
        logits=self.head(self.dropout(pooled))
        return (logits,alpha) if return_attention else logits
```

<a id="tp2-047"></a>

#### Cellule de code 047 — Définir une inférence neuronale par lots

**Objectif.** Convertir les logits d’un modèle en scores sans modifier ses poids.

**Comprendre le code.** `model.eval()` désactive les comportements d’apprentissage tels que le dropout ; `torch.inference_mode()` évite le suivi des gradients. Les entrées sont découpées en lots et envoyées sur le périphérique courant. Pour un modèle multi-tâche, seule la première sortie, sarcasme, est utilisée ici. Le softmax donne un score de classe 1 ; les poids attentionnels sont également collectés si demandés.

**Résultat attendu à l’exécution.** La fonction renvoie un vecteur de longueur nombre de tweets, ou une paire scores/attention.

**À retenir.** `eval()` et l’absence de gradients ont des rôles différents : le premier règle le comportement des couches, la seconde évite le coût de la rétropropagation. Aucun seuil n’est choisi à l’intérieur de cette fonction.

**Optimisation CPU ajoutée.** L’inférence PLM regroupe les longueurs similaires, puis réécrit chaque score à sa position d’origine. La taille des lots et tous les textes sont conservés ; C/D2 gardent leur parcours habituel. Des arrondis float32 peuvent varier selon la taille effective des calculs.

**Lecture complémentaire, dans l’ordre des opérations.**

`eval()` désactive le dropout ; `inference_mode()` évite les gradients. Le regroupement par longueur réduit le padding PLM, mais `probs[index]=...` restaure l'ordre officiel avant métriques/export. Dans le multi-tâche, seul le premier tenseur de logits (sarcasme) fournit le score principal.

**Définitions importantes :** `neural_predict`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def neural_predict(model,inputs,batch=64,attention=False):
    model.eval(); n=len(inputs[0]); probs=np.empty(n,dtype=np.float32); att=[]
    grouped=hasattr(model,'enc') and len(inputs)==2 and not attention
    order=np.argsort(-inputs[1].sum(1).cpu().numpy(),kind='stable') if grouped else np.arange(n)
    with torch.inference_mode():
        for start in range(0,n,batch):
            index=order[start:start+batch]
            selection=index if grouped else slice(start,start+batch)
            args=[x[selection].to(DEVICE,non_blocking=DEVICE.type=='cuda') for x in inputs]
            with torch.autocast(device_type=DEVICE.type,dtype=torch.float16,enabled=grouped and DEVICE.type=='cuda'):
                out=model(*args,return_attention=True) if attention else model(*args)
            if attention: out,a=out; att.extend(a.cpu().numpy())
            if isinstance(out,tuple): out=out[0]
            probs[index]=out.float().softmax(-1)[:,1].cpu().numpy()
    return (probs,np.asarray(att)) if attention else probs
```

<a id="tp2-048"></a>

#### Cellule de code 048 — Décrire les opérations d’une époque d’apprentissage

**Objectif.** Réunir la mise à jour des poids effectuée sur tous les lots du train.

**Comprendre le code.** Chaque lot contient les entrées `xx`, la cible sarcasme `yy` et la cible sentiment `ss`. Les gradients précédents sont remis à zéro ; le modèle calcule ses logits ; la perte sarcasme est calculée. Si `lam` est non nul, la perte sentiment est ajoutée. `backward` calcule les gradients, le clipping borne leur norme à 1 et AdamW met à jour les paramètres. Le scheduler ajuste ensuite le taux d’apprentissage. Les messages de progression PLM sont espacés de 50 lots.

**Exemple ou repère.** En multi-tâche : $L=L_{sarc}+\lambda L_{sent}$. Les sentiments sont fournis à la perte, pas concaténés aux tokens du tweet.

**Résultat attendu à l’exécution.** Lors d’un appel, une époque réelle est parcourue et la moyenne des pertes de lots est renvoyée.

**À retenir.** Une perte train qui baisse ne garantit pas une meilleure généralisation. Les choix de checkpoint et d’arrêt se feront avec le F1 val dans la fonction suivante.

**Optimisation CPU ajoutée.** La somme des pertes est maintenue au fil des lots au lieu de refaire une moyenne de toute la liste à chaque message. Le backward, le clipping, AdamW, le scheduler et l’ordre des lots restent identiques. Aucun gain global d’apprentissage n’a été mesuré.

**Précision mixte T4.** `autocast` entoure seulement le forward et les pertes. `GradScaler` adapte l'échelle avant backward, puis `unscale_` rétablit les vrais gradients avant le clipping à 1. Si un débordement fait sauter le pas d'AdamW, le scheduler ne progresse pas. Le scaler persiste entre les époques du même run. Sans scaler, l'ordre CPU précédent est conservé ; aucune conversion globale des poids en float16 n'est faite.

**Lecture complémentaire, dans l’ordre des opérations.**

Cette cellule définit `optimizer_step` et `train_epoch` ; elle n'entraîne rien tant qu'elles ne sont pas appelées. En AMP, `unscale_` précède le clipping et le scheduler ne progresse pas après un overflow avec pas sauté. La perte affichée est une moyenne des pertes de lots, pas une métrique F1.

**Définitions importantes :** `optimizer_step`, `train_epoch`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def optimizer_step(model,loss,opt,scaler,scheduler):
    if scaler is None:
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if scheduler: scheduler.step()
    else:
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        previous_scale=scaler.get_scale(); scaler.step(opt); scaler.update()
        if scheduler and scaler.get_scale()>=previous_scale: scheduler.step()

def train_epoch(model,loader,opt,loss_fn,sent_loss,lam,scheduler,plm,name,seed,epoch,scaler=None):
    model.train(); total_loss=0.; steps=0
    for step,batch in enumerate(loader):
        *xx,yy,ss=[t.to(DEVICE,non_blocking=DEVICE.type=='cuda') for t in batch]
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type,dtype=torch.float16,enabled=scaler is not None):
            out=model(*xx); logits=out[0] if isinstance(out,tuple) else out
            loss=loss_fn(logits,yy)
            if lam: loss=loss+lam*sent_loss(out[1],ss)
        optimizer_step(model,loss,opt,scaler,scheduler)
        total_loss+=loss.item(); steps+=1
        if plm and steps%50==0:
            print(name,seed,'époque',epoch+1,'batch',steps,'/',len(loader),'perte',round(total_loss/steps,4),flush=True)
    return total_loss/max(steps,1)
```

<a id="tp2-049"></a>

#### Cellule de code 049 — Assembler les lots et gérer leur échantillonnage

**Objectif.** Fournir les mêmes exemples d’apprentissage au modèle avec une stratégie de tirage explicite.

**Comprendre le code.** `TensorDataset` aligne entrées et cibles. En suréchantillonnage, `WeightedRandomSampler` tire avec remise selon l’inverse des fréquences de classes : certains tweets peuvent apparaître plusieurs fois, d’autres être absents d’une époque. Pour les PLM sans ce rééchantillonnage, `LengthGroupedSampler` rapproche des longueurs similaires afin de réduire le padding effectif, tout en conservant les exemples. Les lots valent 16 pour les PLM et 64 pour les autres réseaux ; `num_workers=0` limite les processus Windows.

**Résultat attendu à l’exécution.** Un `DataLoader` prêt à être parcouru par `train_epoch` est renvoyé.

**À retenir.** Le groupement par longueur n’est pas une suppression de données. La validation n’est jamais suréchantillonnée : elle garde sa distribution naturelle.

**Optimisation CPU ajoutée.** BatchSampler fournit maintenant des listes d’indices directement à TensorDataset : la lecture est vectorisée, sans collate Python tweet par tweet. L’ordre des lots et l’état RNG ont été comparés sur données synthétiques, y compris suréchantillonnage et regroupement par longueur. Workers=0 et pin_memory=False sont adaptés au CPU.

**Lecture complémentaire, dans l’ordre des opérations.**

Pour `oversample`, `WeightedRandomSampler` tire avec remise des exemples du train avec poids inverse à l'effectif de leur classe. Sinon un PLM utilise `LengthGroupedSampler`, les autres un échantillonnage aléatoire. Les lots sont indexés de façon vectorisée, `num_workers=0` évite des processus supplémentaires et `pin_memory` est activé seulement pour CUDA.

**Définitions importantes :** `make_training_loader`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def make_training_loader(xtrain,y,sy,seed,imbalance,plm):
    dataset=TensorDataset(*xtrain,y,sy)
    generator=torch.Generator().manual_seed(seed)
    if imbalance=='oversample':
        weights=(1/np.bincount(y.numpy()))[y.numpy()]
        sampler=WeightedRandomSampler(weights,len(y),replacement=True,generator=generator)
    elif plm:
        from transformers.trainer_pt_utils import LengthGroupedSampler
        sampler=LengthGroupedSampler(batch_size=BATCH,lengths=xtrain[1].sum(1).tolist(),generator=generator)
    else: sampler=RandomSampler(dataset,generator=generator)
    # TensorDataset accepte une liste d'indices : une extraction par tenseur, sans boucle Python par tweet.
    batches=BatchSampler(sampler,BATCH if plm else 64,drop_last=False)
    return DataLoader(dataset,batch_size=None,sampler=batches,generator=generator,num_workers=0,pin_memory=DEVICE.type=='cuda')
```

<a id="tp2-050"></a>

#### Cellule de code 050 — Orchestrer entraînement, validation et checkpoint

**Objectif.** Appliquer une procédure commune et traçable aux réseaux C, D2 et E.

**Comprendre le code.** La fonction complète la configuration, enregistre les identifiants utilisés et vérifie la compatibilité d’un checkpoint terminé. Sinon, elle prépare AdamW, pertes et éventuel warmup PLM de 10 %, puis alterne époque train et inférence val. À chaque époque, le seuil est choisi sur la validation autorisée et le F1 décide si les poids sont sauvegardés. La patience déclenche un arrêt après absence de progrès. Pour AraVec, les embeddings sont dégelés après les deux premières époques. Le meilleur état est rechargé avant l’export.

**Résultat attendu à l’exécution.** Un appel renvoie la clé d’expérience et le meilleur modèle ; historique, probabilités et métadonnées sont écrits. Les métadonnées finales sont publiées par remplacement atomique.

**À retenir.** Les labels test ne participent ni à l’arrêt ni au checkpoint. Avec `val_mask`, seule une partie de val choisit le checkpoint, utile pour F2. Un run interrompu avant ses métadonnées finales n’est pas considéré terminé. Le meilleur checkpoint peut précéder la dernière époque parcourue.

**Optimisation CPU ajoutée.** Le vecteur val de l’époque ayant gagné est conservé, évitant un deuxième forward identique après le rechargement. Un cache signé permet les reprises sans nouvelle validation. Les poids sont écrits dans un fichier pending puis publiés en fin de run ; les anciens résultats authentifiés ne sont pas recalculés artificiellement.

**Mise à jour qualité / affichage.** AdamW applique le weight decay **0,01 uniformément**, comme point de départ du TP, sans exception pour les biais ou normalisations. Les embeddings statiques initialement figés restent dans l'optimiseur pour leur dégel. Les architectures, pertes, données et seuils de validation sont conservés. Le budget PLM de cinq époques maximum appartient à la plage 3–5 du sujet. Budget et politique d'optimisation entrent dans la signature ; les anciens checkpoints incompatibles, métadonnées, prédictions et journaux sont archivés dans `results/history/` avant remplacement. Aucun entraînement n'a été lancé pour ce retour au réglage du TP.

**Lecture complémentaire, dans l’ordre des opérations.**

Le choix de reprise valide signature, contexte et métadonnées ; sinon les anciens artefacts sont archivés avant un nouveau run. Chaque époque évalue val et son seuil ; seul un meilleur F1 sauvegarde le checkpoint pending. L'arrêt anticipé sélectionne la meilleure époque, pas nécessairement la dernière ; `val_mask` permet de sélectionner sans dialecte égyptien pour F2. Définir cette fonction n'est pas lancer les appels d'apprentissage qui suivent.

**Définitions importantes :** `archive_previous_run`, `train_neural`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def archive_previous_run(path,meta):
    if not path.exists() and not meta.exists(): return
    from shutil import copy2
    folder=ROOT/'results/history'/f'{path.stem}_{time.time_ns()}'
    folder.mkdir(parents=True)
    for name in ['results.csv','registry.json','learning_curves.csv']:
        source=ROOT/'results'/name
        if source.exists(): copy2(source,folder/name)
    predictions=ROOT/'predictions'/f'{path.stem}_val.npy'
    for source in [path,meta,predictions,predictions.with_suffix('.validation.json')]:
        if not source.resolve().is_relative_to(ROOT.resolve()): raise ValueError('Archive hors du projet.')
        if source.exists(): source.replace(folder/source.name)
    print('Ancien run conservé dans',folder,flush=True)

def train_neural(model,xtrain,xval,name,seed=42,config=None,epochs=STATIC_EPOCHS,
                 lr=1e-3,imbalance='weighted',labels=None,sentiments=None,lam=0,freeze_epochs=0,plm=False,val_mask=None):
    config=dict(config or {},seed=seed,epochs=epochs,lr=lr,imbalance=imbalance,lam=lam,
                train_ids=(config or {}).get('train_ids',train.id.tolist()))
    if plm: config.update(dynamic_padding=True,length_grouped=True,optimizer_policy='adamw_uniform_wd_0p01_v1')
    path=ROOT/f'models/{name}_seed{seed}.pt'; meta=path.with_suffix('.json'); sig=cache_signature(config)
    model=model.to(DEVICE)
    if plm and DEVICE.type!='cpu':
        config.update(device_type=DEVICE.type,precision=PLM_PRECISION,batch=BATCH)
    p=None; best_path=path.with_suffix('.pending.pt')
    trained_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if freeze_epochs: trained_params+=model.embedding.weight.numel()
    selection_mask=np.ones(len(val),dtype=bool) if val_mask is None else np.asarray(val_mask)
    config['selection_val_ids']=val.id[selection_mask].tolist()
    sig=cache_signature(config)
    if path.exists() and compatible_metadata(meta) and json.loads(meta.read_text())['signature']==sig:
        payload=torch.load(path,map_location=DEVICE,weights_only=True); model.load_state_dict(payload); del payload
        saved=json.loads(meta.read_text()); seconds=saved['seconds']; history=saved['history']
    else:
        archive_previous_run(path,meta)
        seed_all(seed); y=YTRAIN if labels is None else labels; sy=STRAIN if sentiments is None else sentiments
        loader=make_training_loader(xtrain,y,sy,seed,imbalance,plm)
        opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
        scaler=torch.amp.GradScaler('cuda') if plm and DEVICE.type=='cuda' else None
        total=epochs*len(loader); scheduler=get_linear_schedule_with_warmup(opt,int(.1*total),total) if plm else None
        weights=torch.tensor(len(y)/(2*np.bincount(y.numpy())),dtype=torch.float32,device=DEVICE) if imbalance=='weighted' else None
        loss_fn=nn.CrossEntropyLoss(weight=weights); sent_loss=nn.CrossEntropyLoss()
        start=time.perf_counter(); best=-1; stale=0; history=[]; best_p=None
        for epoch in range(epochs):
            if freeze_epochs and epoch==freeze_epochs:
                model.embedding.weight.requires_grad_(True)
            mean_loss=train_epoch(model,loader,opt,loss_fn,sent_loss,lam,scheduler,plm,name,seed,epoch,scaler)
            p=neural_predict(model,xval,batch=BATCH if plm else 64)
            t=best_threshold(val.y[selection_mask],p[selection_mask]); f=float(f1_score(val.y[selection_mask],p[selection_mask]>=t,zero_division=0))
            history.append({'epoch':epoch+1,'loss':mean_loss,'val_f1':f,'threshold':t})
            print(name,seed,'époque',epoch+1,'perte',round(mean_loss,4),'F1 val',round(f,4),flush=True)
            if f>best+1e-6:
                best=f; stale=0; best_p=p.copy(); torch.save(model.state_dict(),best_path)
            else: stale+=1
            if stale>=PATIENCE and epoch>=freeze_epochs: break
        seconds=time.perf_counter()-start; best_path.replace(path); p=best_p
        draft=meta.with_suffix('.json.tmp')
        draft.write_text(json.dumps({'signature':sig,'seconds':seconds,'history':history,'config':config,'runtime_context':CACHE_CONTEXT,'checkpoint_stat':checkpoint_stat(path),
            'performance':{'threads':torch.get_num_threads(),'inter_op':torch.get_num_interop_threads(),'loader':'vectorized_batches_v1','precision':PLM_PRECISION if plm else 'float32','device':str(DEVICE)}},indent=2))
        draft.replace(meta)
        model.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=True))
    p=validation_scores(model,xval,path,sig,candidate=p,plm=plm)
    HISTORY[:]=[h for h in HISTORY if (h['model'],h['seed'])!=(name,seed)]
    HISTORY.extend([dict(model=name,seed=seed,**h) for h in history])
    pd.DataFrame(HISTORY).to_csv(ROOT/'results/learning_curves.csv',index=False)
    key=register(name,seed,p,seconds,trained_params,config.get('preprocessing','conservative'),
                 'plm' if plm else config.get('kind','static'),path,config,
                 extra={'epochs_completed':len(history),'n_train':len(xtrain[0])})
    return key,model
```

<a id="tp2-051"></a>

#### Cellule de code 051 — Comparer les réseaux à embeddings aléatoires

**Objectif.** Créer la référence C sans connaissance lexicale préentraînée externe.

**Comprendre le code.** La boucle construit BiLSTM, BiLSTM avec attention et mini-Transformer avec la même graine et le budget commun. Une quatrième expérience enlève les positions du Transformer. Chaque entraînement est confié à la même fonction de validation, puis sa clé est conservée. Les objets modèles devenus inutiles sont supprimés pour limiter leur coexistence en mémoire.

**Résultat attendu à l’exécution.** Quatre expériences C sont ajoutées ou reprises depuis des checkpoints compatibles ; un tableau affiche les métriques de validation.

**À retenir.** Cette cellule déclenche de vrais apprentissages si les caches ne conviennent pas. Les différences C1/C2/C3 doivent être lues à budget comparable ; l’ablation sans positions porte sur l’ordre, pas sur une simple réduction de la taille du vocabulaire.

**Lecture complémentaire, dans l’ordre des opérations.**

Trois appels entraînent C1/C2/C3 avec embeddings aléatoires, puis un quatrième entraîne C3 sans positions. Les autres paramètres sont conservés pour isoler l'ablation positionnelle. Comparer une seule graine donne une observation locale, pas une preuve générale de supériorité architecturale.

**Code de référence.**

```python
static_keys=[]

for arch,label in [('bilstm','C1_BiLSTM'),('attention','C2_attention'),('transformer','C3_Transformer')]:
    seed_all(42)
    key,model=train_neural(StaticNet(arch),[XW_TRAIN],[XW_VAL],label,config={'arch':arch,'embedding':'random','position':True})
    static_keys.append(key); del model; gc.collect()

seed_all(42)

key,model=train_neural(StaticNet('transformer',position=False),[XW_TRAIN],[XW_VAL],'C3_sans_position',
                      config={'arch':'transformer','embedding':'random','position':False})

static_keys.append(key)

del model

gc.collect()

display(pd.DataFrame(RESULTS).loc[lambda d:d.model.str.startswith('C')])
```

<a id="tp2-052"></a>

#### Cellule de code 052 — Intégrer la normalisation officielle AraVec

**Objectif.** Rechercher les vecteurs préentraînés avec les conventions graphiques du vocabulaire AraVec.

**Comprendre le code.** La fonction reproduite du dépôt AraVec normalise plusieurs lettres et signes, retire des diacritiques et limite certains allongements. Elle est conservée dans le notebook pour que cette étape ne soit pas cachée dans un script auxiliaire. Elle sera appliquée à chaque token avant de consulter les vecteurs, pas utilisée pour remplacer tout le prétraitement conservateur du tweet.

**Résultat attendu à l’exécution.** `clean_str` est définie localement ; aucune archive Word2Vec n’est chargée ici.

**À retenir.** Les normalisations d’AraVec peuvent détruire des indices si on les applique aveuglément au texte entier. Leur usage est ici limité à la recherche d’un vecteur compatible ; les tokens originaux conservateurs restent les unités du modèle.

**Lecture complémentaire, dans l’ordre des opérations.**

`clean_str` reprend la normalisation officielle AraVec pour rechercher un token dans son vocabulaire. Il ne faut pas remplacer tout le nettoyage conservateur par cette fonction sans analyser les marqueurs retirés. Son usage principal arrive dans la construction de la matrice d'embeddings.

**Définitions importantes :** `clean_str`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def clean_str(text):
    search = ["أ","إ","آ","ة","_","-","/",".","،"," و "," يا ",'"',"ـ","'","ى","\\",'\n', '\t','&quot;','?','؟','!']
    replace = ["ا","ا","ا","ه"," "," ","","",""," و"," يا","","","","ي","",' ', ' ',' ',' ? ',' ؟ ',' ! ']
    
    #remove tashkeel
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(p_tashkeel,"", text)
    
    #remove longation
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(p_longation, subst, text)
    
    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')
    
    for i in range(0, len(search)):
        text = text.replace(search[i], replace[i])
    
    #trim    
    text = text.strip()

    return text
```

<a id="tp2-053"></a>

#### Cellule de code 053 — Identifier le cache réduit des embeddings AraVec

**Objectif.** Réutiliser uniquement la matrice nécessaire au vocabulaire train courant.

**Comprendre le code.** `matrix_path` pointe vers les vecteurs réduits. Une signature est calculée sur le dictionnaire `VOCAB`, trié lors de la sérialisation pour obtenir un résultat stable. Le fichier voisin contiendra cette signature. Vérifier le vocabulaire est nécessaire car chaque ligne de la matrice doit rester alignée avec le même identifiant de token.

**Résultat attendu à l’exécution.** Les chemins et l’empreinte du vocabulaire sont disponibles pour la construction ou la reprise de la matrice.

**À retenir.** Une matrice de bonne dimension peut pourtant être incorrecte si ses lignes sont dans un autre ordre. Le contrôle ne se limite donc pas au nombre de mots.

**Lecture complémentaire, dans l’ordre des opérations.**

La matrice réduite ne contient que les lignes utiles au vocabulaire du projet. Le hash de `VOCAB` évite de reprendre une matrice dont l'ordre des lignes ne correspond plus aux identifiants. L'existence de cette matrice ne prouve pas qu'un classifieur C a déjà été entraîné.

**Code de référence.**

```python
matrix_path=ROOT/'models/aravec_train_matrix.npy'

vocab_signature=hashlib.sha256(json.dumps(VOCAB,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

matrix_meta=matrix_path.with_suffix('.cache.json')
```

<a id="tp2-054"></a>

#### Cellule de code 054 — Construire la matrice AraVec et mesurer sa couverture

**Objectif.** Initialiser les embeddings connus depuis Word2Vec tout en donnant une représentation aux autres tokens.

**Comprendre le code.** Un cache au vocabulaire compatible est relu. Sinon, les vecteurs AraVec sont chargés, puis une matrice de forme `(taille_vocabulaire, 100)` est initialisée avec une graine fixe. PAD reste nul. Pour chaque token, `clean_str` fournit la clé de recherche ; un vecteur trouvé remplace l’initialisation aléatoire. La couverture est calculée en types distincts et en occurrences du train, puis les gros vecteurs sources sont libérés.

**Résultat attendu à l’exécution.** `ARAVEC` contient les seules lignes utiles au modèle et un tableau rapporte les couvertures sauvegardées.

**À retenir.** La couverture AraVec et l’OOV de `VOCAB` sont différents : un token peut être connu du train tout en n’ayant pas de vecteur externe. Une forte couverture en types ne garantit pas la même couverture des occurrences fréquentes.

**Lecture complémentaire, dans l’ordre des opérations.**

Les tokens trouvés reçoivent leur vecteur AraVec ; les autres gardent une initialisation aléatoire contrôlée et la ligne PAD reste nulle. Le taux par types et le taux pondéré par occurrences train ont des dénominateurs différents. Charger le grand modèle AraVec et extraire sa matrice réduite ne met à jour aucun poids C.

**Code de référence.**

```python
if matrix_path.exists() and matrix_meta.exists() and json.loads(matrix_meta.read_text())['vocab_signature']==vocab_signature:
    ARAVEC=np.load(matrix_path); coverage=json.loads((ROOT/'models/aravec_coverage.json').read_text())
else:
    import scipy.linalg
    # Compatibilité gensim 4.3 avec scipy récent, sans modifier le paquet installé.
    if not hasattr(scipy.linalg,'triu'): scipy.linalg.triu=np.triu
    from gensim.models import Word2Vec
    if IN_COLAB: prepare_aravec(ROOT)
    candidates=list((ROOT/'models/aravec').rglob('*.mdl'))
    assert candidates,'AraVec manquant dans models/aravec.'
    vectors=Word2Vec.load(str(candidates[0])).wv
    rng=np.random.default_rng(42); ARAVEC=rng.normal(0,.05,(len(VOCAB),100)).astype('float32'); ARAVEC[0]=0
    found=[]
    for token,i in VOCAB.items():
        normalized=clean_str(token)
        if i>1 and normalized in vectors: ARAVEC[i]=vectors[normalized]; found.append(token)
    coverage={'vocab_types':len(VOCAB)-2,'types_found':len(found),
        'type_coverage':len(found)/max(len(VOCAB)-2,1),
        'train_token_coverage':sum(counts[t] for t in found)/sum(counts.values())}
    np.save(matrix_path,ARAVEC); matrix_meta.write_text(json.dumps({'vocab_signature':vocab_signature})); (ROOT/'models/aravec_coverage.json').write_text(json.dumps(coverage,indent=2))
    del vectors; gc.collect()

display(pd.Series(coverage).to_frame('Couverture AraVec'))
```

<a id="tp2-055"></a>

#### Cellule de code 055 — Comparer C1/C2/C3 avec AraVec

**Objectif.** Mesurer ce que change l’initialisation préentraînée à architectures et budget communs.

**Comprendre le code.** Les trois réseaux reçoivent la même matrice AraVec de départ. `freeze=True` empêche d’abord les mises à jour des embeddings ; `freeze_epochs=2` demande leur dégel après deux époques dans la boucle commune. BiLSTM, attention et Transformer sont chacun évalués sur val avec un seuil optimisé et leur meilleur checkpoint est conservé.

**Résultat attendu à l’exécution.** Trois expériences AraVec sont ajoutées à `static_keys` et au journal, ou reprises si elles sont compatibles.

**À retenir.** Le meilleur checkpoint peut se situer pendant la phase figée si le dégel n’améliore pas val. Il faut lire l’historique avant d’affirmer que le modèle retenu bénéficie d’un ajustement des embeddings. Cette cellule peut entraîner plusieurs réseaux.

**Lecture complémentaire, dans l’ordre des opérations.**

Chaque architecture est réinitialisée avec la graine 42 et reçoit la même matrice AraVec. `freeze_epochs=2` autorise l'apprentissage des autres couches pendant que l'embedding reste figé. La sélection sur val peut garder un checkpoint antérieur au dégel.

**Code de référence.**

```python
for arch,label in [('bilstm','C1_AraVec'),('attention','C2_AraVec'),('transformer','C3_AraVec')]:
    seed_all(42)
    key,model=train_neural(StaticNet(arch,ARAVEC,freeze=True),[XW_TRAIN],[XW_VAL],label,
        config={'arch':arch,'embedding':'aravec','position':True,'freeze_epochs':2},freeze_epochs=2)
    static_keys.append(key); del model; gc.collect()
```

<a id="tp2-056"></a>

#### Cellule de code 056 — Isoler l’effet du traitement du déséquilibre

**Objectif.** Comparer trois stratégies sur une architecture C sélectionnée uniquement sur val.

**Comprendre le code.** Le meilleur C1/C2/C3 admissible est retenu hors ablation sans positions. La configuration gagnante sert de référence pondérée ; elle est réentraînée avec suréchantillonnage puis perte standard. Les entrées, l’architecture et le budget restent comparables. Pour chaque méthode, on calcule le F1 à 0,5 puis au seuil choisi sur val afin de séparer effet d’apprentissage et effet de décision.

**Résultat attendu à l’exécution.** `imbalance.csv` présente les trois lignes avec F1 à 0,5, F1 au seuil val et valeur du seuil.

**À retenir.** Ces comparaisons réutilisent val pour sélectionner et analyser ; elles peuvent donc la surajuster. Le test indépendant final reste indispensable. Une pondération élevée ne signifie pas que les scores deviennent automatiquement des probabilités calibrées.

**Lecture complémentaire, dans l’ordre des opérations.**

Le meilleur C1/C2/C3 est choisi sur val en excluant l'ablation sans positions pour ce choix initial. Les variantes sur-échantillonnée et standard gardent sa configuration de base. La table sépare F1 au seuil 0,5 et au seuil val, pour ne pas attribuer au seul entraînement un gain qui vient du seuil.

**Code de référence.**

```python
eligible=pd.DataFrame(RESULTS).loc[lambda d:d.model.str.match(r'C[123]_') & ~d.model.str.contains('sans_position')]

best_c_row=eligible.sort_values('val_f1_sarcastique',ascending=False).iloc[0]

best_c_key=f'{best_c_row.model}_seed42'

BEST_C_CFG=REGISTRY[best_c_key]['config'].copy()

imbalance_keys=[best_c_key]

for imbalance in ['oversample','standard']:
    seed_all(42); cfg=dict(BEST_C_CFG,imbalance=imbalance)
    emb=ARAVEC if cfg['embedding']=='aravec' else None
    key,model=train_neural(StaticNet(cfg['arch'],emb,position=cfg.get('position',True),freeze=emb is not None),
        [XW_TRAIN],[XW_VAL],f'C_meilleur_{imbalance}',config=cfg,imbalance=imbalance,freeze_epochs=cfg.get('freeze_epochs',0))
    imbalance_keys.append(key); static_keys.append(key); del model; gc.collect()

imbalance_table=[]

for key in imbalance_keys:
    p=np.load(ROOT/f'predictions/{key}_val.npy'); rec=REGISTRY[key]
    imbalance_table.append({'model':key,'F1_seuil_05':scores(val.y,p,.5)['f1_sarcastique'],
        'F1_seuil_val':scores(val.y,p,rec['threshold'])['f1_sarcastique'],'seuil':rec['threshold']})

display(pd.DataFrame(imbalance_table))

pd.DataFrame(imbalance_table).to_csv(ROOT/'results/imbalance.csv',index=False)
```

### Partie D : encodeur gelé

<a id="tp2-057"></a>

#### Cellule de code 057 — Configurer le préprocesseur officiel AraBERT Twitter

**Objectif.** Respecter les conventions du PLM sans effacer les répétitions utiles au sarcasme.

**Comprendre le code.** `ArabertPreprocessor` est instancié pour le modèle Twitter choisi. `remove_non_digit_repetition=False` désactive la réduction des répétitions non numériques. Deux assertions vérifient sur un exemple concret que le rire ههههه et l’allongement سلاااام sont préservés. Ce sont des contrôles de comportement, pas des mesures de performance du classifieur.

**Résultat attendu à l’exécution.** Le préprocesseur `ARA_PREP` est disponible et un message confirme la conservation testée des répétitions.

**À retenir.** Un préprocesseur officiel peut avoir des options qui ne conviennent pas à la tâche. Les tester vaut mieux que supposer qu’un comportement par défaut préserve tous les marqueurs.

**Lecture complémentaire, dans l’ordre des opérations.**

AraBERT Twitter reçoit son préprocesseur officiel avec `remove_non_digit_repetition=False`. Les assertions vérifient que les répétitions attendues survivent. Un changement de ce préprocesseur rendrait incohérent un checkpoint entraîné avec l'ancienne convention.

**Code de référence.**

```python
from arabert.preprocess import ArabertPreprocessor

ARA_PREP=ArabertPreprocessor(model_name=ARA_NAME,remove_non_digit_repetition=False)

assert 'ههههه' in ARA_PREP.preprocess('يا سلاااام 😂 ههههه !!!')

assert 'سلاااام' in ARA_PREP.preprocess('يا سلاااام 😂 ههههه !!!')

print('AraBERT : préprocesseur officiel, conservation explicite des répétitions.')
```

<a id="tp2-058"></a>

#### Cellule de code 058 — Adapter les textes à chaque encodeur

**Objectif.** Éviter d’imposer exactement la même normalisation à deux PLM préentraînés différemment.

**Comprendre le code.** Pour AraBERT, le nettoyage d’export précède le préprocesseur officiel. Quand l’ablation sans liens est demandée, liens et mentions sont supprimés avant ce préprocesseur. Pour MARBERT, la préparation minimale `plm` garde davantage la graphie, avec gestion cohérente des URL/mentions. La sortie reste une liste de textes ; aucun label auxiliaire n’est ajouté.

**Résultat attendu à l’exécution.** Une fonction centralise la préparation des entrées PLM pour D, E et les réentraînements F.

**À retenir.** Une comparaison équitable ne signifie pas un nettoyage identique si les modèles attendent des conventions différentes. Il faut documenter ces conventions et les conserver entre apprentissage et inférence.

**Lecture complémentaire, dans l’ordre des opérations.**

La fonction choisit la préparation à partir du nom exact du PLM. Pour l'ablation, liens et mentions sont supprimés avant la préparation officielle AraBERT. Aucun sentiment ou dialecte observé n'est injecté dans les textes.

**Définitions importantes :** `plm_texts`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def plm_texts(texts,name,drop_links=False):
    if name==ARA_NAME:
        # Le préprocesseur officiel gère URL/mentions et conserve les emojis Twitter.
        cleaned=[export_clean(s) for s in texts]
        if drop_links: cleaned=[MENTION_RE.sub(' ',URL_RE.sub(' ',s)) for s in cleaned]
        return [ARA_PREP.preprocess(s) for s in cleaned]
    return [prepare_text(s,'plm',drop_links) for s in texts]
```

<a id="tp2-059"></a>

#### Cellule de code 059 — Réutiliser les tokenizers en mémoire

**Objectif.** Éviter de relire inutilement les vocabulaires et configurations de tokenisation.

**Comprendre le code.** `TOKENIZERS` est indexé par nom de modèle. Au premier appel, `AutoTokenizer.from_pretrained` ouvre les ressources locales ; aux appels suivants, le même objet est rendu. Cette optimisation est particulièrement utile lorsque la latence sera mesurée tweet par tweet.

**Résultat attendu à l’exécution.** Un appel renvoie le tokenizer correspondant au checkpoint demandé.

**À retenir.** Le tokenizer est celui du modèle, pas le vocabulaire appris pour C. Il est préentraîné à l’extérieur du TP : on ne lui ajoute pas des tokens appris sur les labels test.

**Lecture complémentaire, dans l’ordre des opérations.**

Un tokenizer déjà chargé est réutilisé dans `TOKENIZERS`. Cela évite de relire ses fichiers à chaque run, sans réapprendre son vocabulaire. Changer d'encodeur implique de choisir son propre tokenizer, pas celui du premier modèle disponible.

**Définitions importantes :** `tokenizer_for`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def tokenizer_for(name):
    if name not in TOKENIZERS:
        TOKENIZERS[name]=AutoTokenizer.from_pretrained(REVISIONS[name]['path'],local_files_only=True)
    return TOKENIZERS[name]
```

<a id="tp2-060"></a>

#### Cellule de code 060 — Transformer les tweets en entrées BERT

**Objectif.** Produire les identifiants de sous-mots et les masques utilisés par les encodeurs.

**Comprendre le code.** Après préparation adaptée, le tokenizer ajoute ses tokens spéciaux, tronque à `MAX_LEN` et complète jusqu’à cette longueur. `input_ids` contient les indices du vocabulaire PLM ; `attention_mask` vaut 1 aux positions valides et 0 au padding. Les deux tenseurs sont alignés et de forme `(N, 64)` avec la configuration actuelle.

**Résultat attendu à l’exécution.** La fonction renvoie une liste de deux tenseurs, utilisée par l’extraction gelée et le fine-tuning.

**À retenir.** Les 64 positions incluent les tokens spéciaux. La tokenisation peut déjà être stockée en longueur fixe tandis que le forward E retirera les colonnes PAD inutiles à chaque batch : ces deux mécanismes ne se contredisent pas.

**Optimisation CPU ajoutée.** Un cache disque et un cache RAM borné à quatre encodages réutilisent exactement les identifiants et masques. La clé dépend des textes, du PLM, de la longueur, du prétraitement et de la version des tokenizers. Le cache ne construit aucun nouveau vocabulaire sur val/test.

**Lecture complémentaire, dans l’ordre des opérations.**

Le hash sert de nom au fichier tokenisé, avec contenu, encodeur et conventions. Le cache disque contient des entiers et masques, pas des états BERT entraînés. Le cache RAM est borné à quatre encodages ; supprimer l'entrée la plus ancienne ne supprime pas son fichier disque.

**Définitions importantes :** `plm_encode`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def plm_encode(texts,name,drop_links=False):
    tag=['tokenized_v1',name,drop_links,__import__('tokenizers').__version__]
    signature=inference_signature(texts,tag)
    if signature in TOKENIZED: return TOKENIZED[signature]
    folder=ROOT/'models/tokenized'; folder.mkdir(exist_ok=True)
    path=folder/f'{signature}.pt'
    if path.exists():
        payload=torch.load(path,map_location='cpu',weights_only=True)
        inputs=[payload['input_ids'],payload['attention_mask']]
        assert all(x.shape==(len(texts),MAX_LEN) and x.dtype==torch.long for x in inputs)
    else:
        tokenizer=tokenizer_for(name)
        enc=tokenizer(plm_texts(texts,name,drop_links),truncation=True,max_length=MAX_LEN,padding='max_length',return_tensors='pt')
        inputs=[enc['input_ids'],enc['attention_mask']]
        draft=path.with_suffix('.pt.tmp')
        torch.save(dict(zip(['input_ids','attention_mask'],inputs)),draft); draft.replace(path)
    if len(TOKENIZED)>=4: TOKENIZED.pop(next(iter(TOKENIZED)))
    TOKENIZED[signature]=inputs
    return inputs
```

<a id="tp2-061"></a>

#### Cellule de code 061 — Extraire les états gelés et les stocker sur disque

**Objectif.** Réutiliser les représentations AraBERT sans recalculer l’encodeur à chaque apprentissage de tête.

**Comprendre le code.** Un cache signé est ouvert en mode mémoire mappée s’il correspond aux textes. Sinon, AraBERT est placé en évaluation et parcouru par lots sous `inference_mode`. Les états `last_hidden_state` sont sauvegardés en float16 dans un tableau `(N, 64, hidden_size)`, avec les masques à part. Un fichier partiel est remplacé seulement à la fin de l’extraction ; le temps d’extraction est enregistré séparément.

**Résultat attendu à l’exécution.** La fonction renvoie les états et les masques ; l’encodage sera effectivement calculé seulement si le cache manque ou est incompatible.

**À retenir.** `eval()` seul ne gèle pas les gradients : ici, l’encodeur ne reçoit ni rétropropagation ni mise à jour d’optimiseur. Le stockage float16 économise le disque et la RAM ; les têtes reconvertissent en float32 pour le CPU.

**Optimisation CPU ajoutée.** Les tweets sont regroupés par longueur pour l’extraction, les colonnes de padding terminal inutiles ne sont pas calculées et les lignes sont remises dans leur ordre officiel. Le fichier garde la forme N×64×768. Le masque rend sans effet les positions PAD mises à zéro. Les caches antérieurs compatibles sont conservés.

**Lecture complémentaire, dans l’ordre des opérations.**

`open_memmap` écrit progressivement `(N,64,H)` en float16 au lieu de garder tout le tableau en RAM. Les lots sont triés par longueur, puis replacés à leurs indices d'origine ; les colonnes non calculées sont remplies de zéro et masquées. Le fichier partiel ne devient officiel qu'après écriture et flush.

**Définitions importantes :** `frozen_features`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def frozen_features(texts,split):
    path=ROOT/f'models/frozen_{split}.npy'; maskpath=ROOT/f'models/frozen_{split}_mask.npy'
    if maskpath.exists() and valid_cache(path,texts,'frozen_arabert'): return np.load(path,mmap_mode='r'),np.load(maskpath)
    inputs=plm_encode(texts,ARA_NAME); model=frozen_encoder()
    temp=path.with_name(path.stem+'_partial.npy')
    array=np.lib.format.open_memmap(temp,mode='w+',dtype='float16',shape=(len(texts),MAX_LEN,model.config.hidden_size))
    lengths=inputs[1].sum(1).numpy(); order=np.argsort(-lengths,kind='stable')
    start=time.perf_counter()
    with torch.inference_mode():
        for j in range(0,len(texts),BATCH):
            index=order[j:j+BATCH]; length=int(lengths[index].max())
            ids,mask=[t[index,:length].to(DEVICE) for t in inputs]
            states=model(input_ids=ids,attention_mask=mask).last_hidden_state.cpu().numpy().astype('float16')
            array[index]=0  # Seul le padding absent du calcul est nul ; les tokens valides restent intacts.
            array[index,:length]=states
            if (j//BATCH+1)%50==0: print('Extraction AraBERT',split,min(j+BATCH,len(texts)),'/',len(texts),flush=True)
    array.flush(); del array; temp.replace(path); np.save(maskpath,inputs[1].numpy())
    mark_cache(path,texts,'frozen_arabert')
    (ROOT/f'results/frozen_{split}_time.json').write_text(json.dumps({'seconds':time.perf_counter()-start}))
    return np.load(path,mmap_mode='r'),inputs[1].numpy()
```

<a id="tp2-062"></a>

#### Cellule de code 062 — Obtenir les représentations de train et val pour D

**Objectif.** Préparer une seule base de caractéristiques commune à D1 et D2.

**Comprendre le code.** `FT` et `FV` contiennent les états des tokens du train et de val ; `MT` et `MV` sont leurs masques. Les appels utilisent des noms de partition distincts et les signatures des textes. Les mêmes états pourront alimenter CLS, moyenne et BiLSTM attention sans repasser l’encodeur.

**Résultat attendu à l’exécution.** Des tableaux de formes `(10842, 64, 768)` et `(1255, 64, 768)` sont attendus pour les données actuelles et AraBERT base, accompagnés de leurs masques.

**À retenir.** Cette cellule peut être coûteuse lors de la première extraction, même sans entraînement. Le coût doit être rapporté séparément du temps très court d’une régression logistique sur ces états.

**Lecture complémentaire, dans l’ordre des opérations.**

L'extraction train/val appelle un encodeur gelé et peut prendre du temps même sans aucun backward. Les objets retournés sont des memmaps et masques, réutilisés par D1/D2. Le temps de cette extraction doit être ajouté si l'on évalue le coût complet de préparation de D.

**Code de référence.**

```python
FT,MT=frozen_features(train.tweet.tolist(),'train')

FV,MV=frozen_features(val.tweet.tolist(),'val')
```

<a id="tp2-063"></a>

#### Cellule de code 063 — Comparer résumé CLS et moyenne masquée

**Objectif.** Réduire une séquence d’états BERT à un vecteur par tweet pour D1.

**Comprendre le code.** Le mode CLS prend l’état de la première position. Le mode mean multiplie les états par le masque, somme sur les positions puis divise par le nombre de positions valides. Les tokens spéciaux valides sont inclus : c’est une convention fixe. Le traitement en blocs de 64 tweets évite la conversion simultanée de tout le corpus en float32.

**Exemple ou repère.** $\bar h=\sum_i m_i h_i / \sum_i m_i$. Le dénominateur dépend du masque, pas de la longueur maximale 64.

**Résultat attendu à l’exécution.** Le tableau renvoyé a la forme `(N, 768)`, quelle que soit la longueur réelle des tweets.

**À retenir.** Inclure les positions PAD dans une moyenne fausserait la représentation selon la longueur. Le découpage par blocs limite les temporaires RAM sans changer la formule par tweet.

**Lecture complémentaire, dans l’ordre des opérations.**

Pour CLS, on prend la position 0 ; pour mean, on somme seulement les positions valides et on divise par leur nombre. Le code travaille par blocs de 64 pour limiter les copies float32 de la memmap float16. Les masques incluent les tokens spéciaux valides, et non uniquement les mots ordinaires.

**Définitions importantes :** `pooled_features`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def pooled_features(h,m,mode):
    if mode=='cls': return np.asarray(h[:,0],dtype=np.float32)
    result=np.empty((len(h),h.shape[-1]),dtype=np.float32)
    for start in range(0,len(h),64):
        stop=start+64
        block=np.asarray(h[start:stop],dtype=np.float32)
        mask=m[start:stop,:,None]
        result[start:stop]=(block*mask).sum(1)/m[start:stop].sum(1,keepdims=True)
    return result
```

<a id="tp2-064"></a>

#### Cellule de code 064 — Apprendre les deux classifieurs linéaires D1

**Objectif.** Comparer CLS et moyenne avec exactement la même famille de tête.

**Comprendre le code.** Pour chaque pooling, une régression logistique pondérée est ajustée sur les représentations du train. Elle produit ensuite des probabilités val ; le seuil est choisi par `register` et le modèle est sauvegardé en `joblib`. Le nombre de paramètres rapporté correspond ici aux coefficients de la tête, pas à l’encodeur gelé.

**Résultat attendu à l’exécution.** Les clés D1 CLS et D1 mean sont ajoutées à `d_keys`, avec leurs métriques et durées de tête.

**À retenir.** Un encodeur gelé reste nécessaire pour prédire un nouveau tweet. Il faut ajouter sa taille et son temps d’extraction lorsqu’on compare le coût global à TF-IDF ou au fine-tuning.

**Lecture complémentaire, dans l’ordre des opérations.**

Les deux régressions logistiques apprennent sur les vecteurs gelés de train, pas sur les labels val. Le seuil est ensuite choisi sur val par `register`. Les 769 paramètres de tête correspondent à 768 coefficients et un biais ; ils n'incluent pas les poids AraBERT conservés pour l'inférence.

**Code de référence.**

```python
d_keys=[]

for pooling in ['cls','mean']:
    path=ROOT/f'models/D1_{pooling}_seed42.joblib'; start=time.perf_counter()
    clf=LogisticRegression(C=1,class_weight='balanced',max_iter=1500,random_state=42)
    clf.fit(pooled_features(FT,MT,pooling),train.y); elapsed=time.perf_counter()-start; joblib.dump(clf,path)
    p=clf.predict_proba(pooled_features(FV,MV,pooling))[:,1]
    d_keys.append(register(f'D1_{pooling}',42,p,elapsed,clf.coef_.size+1,'arabert_official','frozen_lr',path,{'pooling':pooling}))
```

<a id="tp2-065"></a>

#### Cellule de code 065 — Définir une tête BiLSTM attention sur états BERT

**Objectif.** Conserver la séquence de représentations contextualisées au lieu de la réduire immédiatement à un seul vecteur.

**Comprendre le code.** `FrozenHead` reçoit des états de dimension 768, les convertit en float32 et utilise le masque pour compacter les séquences. Le BiLSTM produit 128 valeurs par position ; `AttentionPool` en fait un résumé ; dropout et couche linéaire donnent deux logits. L’option `return_attention` renvoie aussi les poids pour la visualisation.

**Résultat attendu à l’exécution.** Une classe de tête profonde est définie, avec entrées `(B, L, 768)` et masque `(B, L)`.

**À retenir.** La tête possède des paramètres entraînables, mais les états BERT fournis restent fixes dans D2. Ce n’est donc pas un fine-tuning de l’encodeur, contrairement à E.

**Lecture complémentaire, dans l’ordre des opérations.**

La tête reçoit des vecteurs BERT de dimension 768 et les fait passer par un BiLSTM 64×2, puis par le même mécanisme d'attention que C2. Seule cette tête est entraînée ; l'encodeur a déjà fourni les représentations fixes. Le masque continue à contrôler les PAD malgré le changement de représentation.

**Définitions importantes :** `FrozenHead`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
class FrozenHead(nn.Module):
    def __init__(self,dim=768):
        super().__init__(); self.encoder=nn.LSTM(dim,64,batch_first=True,bidirectional=True)
        self.attention=AttentionPool(128); self.drop=nn.Dropout(.25); self.head=nn.Linear(128,2)
    def forward(self,h,mask,return_attention=False):
        h=h.float(); mask=mask.bool()
        packed=nn.utils.rnn.pack_padded_sequence(h,mask.sum(1).cpu(),batch_first=True,enforce_sorted=False)
        packed,_=self.encoder(packed); h,_=nn.utils.rnn.pad_packed_sequence(packed,batch_first=True,total_length=mask.shape[1])
        c,a=self.attention(h,mask); out=self.head(self.drop(c))
        return (out,a) if return_attention else out
```

<a id="tp2-066"></a>

#### Cellule de code 066 — Apprendre D2 et sélectionner cinq tweets à visualiser

**Objectif.** Évaluer la tête profonde puis préparer une inspection de ses poids attentionnels.

**Comprendre le code.** Les représentations stockées sont copiées dans des tenseurs puis transmises à la boucle commune. Après sélection du meilleur checkpoint, une inférence val collecte scores et attention. Les cinq premières annotations positives de val sont choisies, sans regarder les résultats test. Le tokenizer AraBERT reconstruit les sous-mots associés aux positions.

**Résultat attendu à l’exécution.** D2 est enregistré ; `att`, `chosen`, `val_ids` et une liste de lignes attentionnelles sont prêts.

**À retenir.** Les copies des états peuvent occuper environ un gigaoctet pour le train, même avec le stockage float16. Ces cinq tweets servent à illustrer une pondération, pas à constituer un échantillon représentatif de toutes les erreurs.

**Lecture complémentaire, dans l’ordre des opérations.**

Le réseau D2 est réellement entraîné par l'appel à `train_neural`. L'appel `attention=True` renvoie ensuite scores et poids pour l'inspection de validation. Les cinq tweets sélectionnés sont les premiers positifs de val, pas cinq preuves de prédictions forcément correctes.

**Code de référence.**

```python
seed_all(42)

DXT=[torch.from_numpy(np.array(FT)),torch.tensor(MT)]

DXV=[torch.from_numpy(np.array(FV)),torch.tensor(MV)]

d2_key,d2_model=train_neural(FrozenHead(),DXT,DXV,'D2_BiLSTM_attention',
    config={'kind':'frozen_head','preprocessing':'arabert_official'})

d_keys.append(d2_key)

p_att,att=neural_predict(d2_model,DXV,attention=True)

chosen=np.flatnonzero(val.y.to_numpy()==1)[:5]

tokenizer_ara=AutoTokenizer.from_pretrained(REVISIONS[ARA_NAME]['path'],local_files_only=True)

val_ids=plm_encode(val.tweet.tolist(),ARA_NAME)[0]

attention_rows=[]
```

<a id="tp2-067"></a>

#### Cellule de code 067 — Afficher et exporter l’attention sur cinq tweets

**Objectif.** Relier les poids numériques de D2 aux sous-mots effectivement lus.

**Comprendre le code.** Pour chaque tweet, le masque limite l’affichage aux positions valides. Les sous-mots sont colorés avec une intensité relative à l’attention maximale du tweet ; le survol donne le poids. Un CSV conserve identifiant, token et poids. Cinq graphiques par position sont aussi sauvegardés. Les gros objets de tête et tenseurs temporaires sont ensuite supprimés.

**Résultat attendu à l’exécution.** Des textes colorés, un graphique à cinq panneaux et `attention_five_tweets.csv` deviennent consultables.

**À retenir.** La couleur est normalisée séparément pour chaque tweet : deux couleurs aussi foncées ne garantissent pas deux poids absolus identiques. Un pic sur CLS/SEP ou un mot thématique doit aussi être discuté, sans le transformer en explication causale.

**Lecture complémentaire, dans l’ordre des opérations.**

Les tokens affichés proviennent des mêmes identifiants AraBERT que ceux des états gelés. Chaque poids est associé à une position, y compris parfois CLS/SEP et sous-mots. Le CSV et la carte HTML conservent les valeurs ; les barres seules ne donnent pas le sens linguistique du token.

**Code de référence.**

```python
for i in chosen:
    length=int(MV[i].sum()); tokens=tokenizer_ara.convert_ids_to_tokens(val_ids[i,:length])
    weights=att[i,:length]
    cells=' '.join(f'<span title="{w:.4f}" style="background:rgba(35,110,180,{float(w/weights.max()):.3f});padding:4px;display:inline-block">{html.escape(t)}</span>' for t,w in zip(tokens,weights))
    display(HTML(f'<div dir="rtl" style="line-height:2.5">{cells}</div>'))
    attention_rows.extend({'val_id':int(val.iloc[i].id),'token':t,'attention':float(w)} for t,w in zip(tokens,weights))

pd.DataFrame(attention_rows).to_csv(ROOT/'results/attention_five_tweets.csv',index=False,encoding='utf-8-sig')

fig,axes=plt.subplots(5,1,figsize=(8,8))

for ax,i in zip(axes,chosen):
    length=int(MV[i].sum()); ax.bar(np.arange(length),att[i,:length],color='#275D88')
    ax.set(title=f'Tweet validation id {int(val.iloc[i].id)} — attention par position',ylabel='Poids',xlabel='Position token')

savefig(fig,'attention_five_tweets')

del d2_model,DXT,DXV
```

### Partie E : fine-tuning complet

<a id="tp2-068"></a>

#### Cellule de code 068 — Demander le nettoyage des objets Python inutilisés

**Objectif.** Réduire la coexistence d’objets après l’étape D2 avant les calculs plus lourds.

**Comprendre le code.** `gc.collect()` demande au ramasse-miettes de rechercher des objets devenus inaccessibles, notamment ceux supprimés dans la cellule précédente. C’est une opération de gestion de mémoire ; elle n’entraîne pas de modèle et ne recalcule pas les scores.

**Résultat attendu à l’exécution.** La cellule peut afficher un entier technique lié aux objets récupérés, ou ne donner aucune information utile à l’analyse scientifique.

**À retenir.** Cet entier n’est ni un score ni un volume en octets. Le nettoyage Python ne garantit pas que toute la mémoire soit immédiatement rendue au système : certaines bibliothèques conservent leurs allocations.

**Optimisation CPU ajoutée.** Le cache d’encodeur gelé est également vidé avant E pour garder de la marge sous 16 Go. Les encodeurs entraînables ne sont pas partagés entre les expériences.

**Lecture complémentaire, dans l’ordre des opérations.**

`del` retire des références et `gc.collect()` demande le nettoyage d'objets Python inutilisés. Cela ne supprime pas les checkpoints sur disque et ne garantit pas que toute la mémoire réservée par CUDA soit immédiatement rendue au système. La libération des encodeurs en cache est aussi gérée par le chargeur.

**Code de référence.**

```python
FROZEN_MODELS.clear()  # Ne pas conserver l’encodeur gelé pendant E.
gc.collect()
```

<a id="tp2-069"></a>

#### Cellule de code 069 — Définir le fine-tuning mono et multi-tâche

**Objectif.** Partager un encodeur complet tout en distinguant la tête sarcasme et l’éventuelle tête sentiment.

**Comprendre le code.** `FineTuneNet` charge l’encodeur demandé et ajoute dropout puis une tête à deux logits. En multi-tâche, une deuxième tête fournit trois logits sentiment. Le résumé est exclusivement `last_hidden_state[:, 0]`. Le pooler présent mais inutilisé est explicitement non entraînable. Le forward retire les colonnes terminales entièrement PAD selon la longueur maximale valide du lot, tout en gardant les tokens et leurs positions.

**Résultat attendu à l’exécution.** Le mono-tâche renverra `(B, 2)` ; le multi-tâche une paire `(B, 2)` et `(B, 3)`.

**À retenir.** Les couches utiles de l’encodeur restent entraînables : cette architecture ne remplace pas le fine-tuning par une extraction gelée. Le paramètre `encoder` permet des tests avec un modèle jouet sans charger un PLM complet.

**Lecture complémentaire, dans l’ordre des opérations.**

Le CLS brut passe par dropout 0,1 puis la tête sarcasme ; la tête sentiment est présente uniquement si `multitask=True`. Le pooler inutilisé est gelé, mais les couches utiles de l'encodeur restent ajustables. Le découpage des colonnes PAD s'arrête à la longueur maximale valide du lot.

**Définitions importantes :** `FineTuneNet`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
class FineTuneNet(nn.Module):
    def __init__(self,name,multitask=False,encoder=None):
        super().__init__(); self.enc=load_hf(name) if encoder is None else encoder; h=self.enc.config.hidden_size
        # La tête utilise le CLS brut, pas le pooler ; ses poids ne reçoivent aucun gradient.
        if getattr(self.enc,'pooler',None) is not None: self.enc.pooler.requires_grad_(False)
        self.drop=nn.Dropout(.1); self.sarc=nn.Linear(h,2)
        self.sent=nn.Linear(h,3) if multitask else None
    def forward(self,input_ids,attention_mask):
        length=int(attention_mask.sum(1).max())
        input_ids=input_ids[:,:length]; attention_mask=attention_mask[:,:length]
        cls=self.drop(self.enc(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state[:,0])
        logits=self.sarc(cls)
        return (logits,self.sent(cls)) if self.sent is not None else logits
```

<a id="tp2-070"></a>

#### Cellule de code 070 — Préparer le suivi des expériences E

**Objectif.** Séparer les références des runs E et les lambda sélectionnés pour les répétitions.

**Comprendre le code.** `e_keys` accueillera les clés de chaque entraînement mono ou multi-tâche. `E_BEST_LAMBDA` associera à AraBERT/MARBERT le lambda retenu après la grille à graine 42. Ces structures servent à sélectionner des configurations et à retrouver leurs résultats dans le registre.

**Résultat attendu à l’exécution.** Une liste et un dictionnaire initialement vides sont définis.

**À retenir.** Relancer cette cellule isolément effacerait ce suivi en mémoire. Ce n’est pas une commande de suppression des checkpoints, mais cela casserait les dépendances des cellules E suivantes ; respecter l’ordre du notebook.

**Lecture complémentaire, dans l’ordre des opérations.**

`e_keys` et `E_BEST_LAMBDA` constituent un suivi en mémoire, pas un chargement automatique du registre disque. Les réinitialiser après la grille ferait perdre les références nécessaires aux répétitions. Cela ne retire pas les fichiers déjà créés, mais peut casser la suite du notebook.

**Code de référence.**

```python
e_keys=[]

E_BEST_LAMBDA={}
```

<a id="tp2-071"></a>

#### Cellule de code 071 — Définir la grille mono/multi pour un encodeur

**Objectif.** Comparer l’absence de tâche auxiliaire et trois poids de sentiment selon le protocole.

**Comprendre le code.** La fonction tokenise train et val pour un encodeur, mesure combien de textes dépassent la longueur choisie puis boucle sur `lam = 0, 0.3, 0.5, 1`. Zéro correspond au mono-tâche ; les autres valeurs activent la tête sentiment. Chaque modèle part du checkpoint préentraîné avec la même graine 42, les mêmes budgets et un taux initial de 2e-5. La boucle commune choisit le checkpoint et le seuil sur val.

**Résultat attendu à l’exécution.** La fonction est seulement définie ici ; son appel lancera quatre configurations pour le PLM indiqué.

**À retenir.** La troncature est mesurée avec le tokenizer de chaque encodeur. Le meilleur lambda est un hyperparamètre sélectionné sur val ; aucun lambda ne doit être retiré simplement parce qu’un autre semble déjà prometteur.

**Optimisation CPU ajoutée.** Le tokenizer déjà chargé et les encodages déjà disponibles sont réutilisés. Les quatre lambda, tous les exemples et la mesure de troncature restent dans la grille.

**Lecture complémentaire, dans l’ordre des opérations.**

La grille compte lambda=0 (E1) et trois lambda positifs (E2). La longueur avant troncature est auditée sur les textes train pour documenter l'information perdue. Chaque appel part de poids pré-entraînés et d'une nouvelle tête, à graine contrôlée ; ce ne sont pas quatre continuations du même modèle.

**Définitions importantes :** `run_plm_grid`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def run_plm_grid(name,short):
    xt=plm_encode(train.tweet.tolist(),name); xv=plm_encode(val.tweet.tolist(),name)
    tok=tokenizer_for(name)
    lengths=[len(ids) for ids in tok(plm_texts(train.tweet.tolist(),name),add_special_tokens=True)['input_ids']]
    print(short,'train tronqué à 64 :',sum(n>MAX_LEN for n in lengths),'/',len(lengths),flush=True)
    for lam in [0,.3,.5,1.]:
        seed_all(42); label=f'E1_{short}_mono' if lam==0 else f'E2_{short}_lambda{str(lam).replace(".","p")}'
        cfg={'model_name':name,'multitask':lam>0,'preprocessing':'arabert_official' if name==ARA_NAME else 'plm_minimal'}
        key,model=train_neural(FineTuneNet(name,lam>0),xt,xv,label,seed=42,config=cfg,
            epochs=PLM_EPOCHS,lr=2e-5,lam=lam,plm=True)
        e_keys.append(key); del model; gc.collect()
    del xt,xv; gc.collect()
```

<a id="tp2-072"></a>

#### Cellule de code 072 — Définir les répétitions finales d’un encodeur

**Objectif.** Observer la variabilité du mono-tâche et du meilleur multi-tâche sur trois graines.

**Comprendre le code.** Parmi les variantes multi-tâches de cet encodeur à graine 42, la fonction choisit celle au meilleur F1 val. Elle mémorise son lambda puis relance le mono et ce multi-tâche avec 43 et 44. Ces répétitions complètent la graine 42 déjà calculée. Les hyperparamètres sont conservés ; les initialisations/tirages changent de manière contrôlée.

**Résultat attendu à l’exécution.** Un appel ajoute quatre runs : deux graines pour deux configurations finales de cet encodeur.

**À retenir.** La moyenne sur trois graines ne consiste pas à choisir la meilleure graine. Les deux autres lambda restent des essais de sélection à graine unique et ne doivent pas être présentés comme trois répétitions du modèle final.

**Lecture complémentaire, dans l’ordre des opérations.**

Le meilleur lambda est celui de graine 42 sur validation, puis il est répété aux graines 43/44 avec E1. On ne choisit pas la meilleure graine d'après le test. `E_BEST_LAMBDA` mémorise un choix par encodeur, car les deux PLM peuvent préférer des valeurs différentes.

**Définitions importantes :** `repeat_plm_finals`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def repeat_plm_finals(name,short):
    xt=plm_encode(train.tweet.tolist(),name); xv=plm_encode(val.tweet.tolist(),name)
    multi=[k for k in e_keys if REGISTRY[k]['name'].startswith(f'E2_{short}')]
    chosen_key=max(multi,key=lambda k:next(r['val_f1_sarcastique'] for r in RESULTS if r['model']==REGISTRY[k]['name'] and r['seed']==42))
    chosen_lam=REGISTRY[chosen_key]['config']['lam']; E_BEST_LAMBDA[short]=chosen_lam
    for lam in [0,chosen_lam]:
        label=f'E1_{short}_mono' if lam==0 else REGISTRY[chosen_key]['name']
        for seed in [43,44]:
            seed_all(seed); cfg={'model_name':name,'multitask':lam>0,'preprocessing':'arabert_official' if name==ARA_NAME else 'plm_minimal'}
            key,model=train_neural(FineTuneNet(name,lam>0),xt,xv,label,seed=seed,config=cfg,
                epochs=PLM_EPOCHS,lr=2e-5,lam=lam,plm=True)
            e_keys.append(key); del model; gc.collect()
    del xt,xv; gc.collect()
```

<a id="tp2-073"></a>

#### Cellule de code 073 — Exécuter la grille AraBERT Twitter

**Objectif.** Réaliser les quatre comparaisons E requises pour AraBERT à graine 42.

**Comprendre le code.** Cette ligne appelle la grille définie précédemment avec `ARA_NAME`. Elle peut charger les quatre checkpoints historiques compatibles ; sinon, elle lance leurs entraînements complets. Même avec des poids déjà disponibles, la validation et les contrôles de reprise peuvent prendre du temps.

**Résultat attendu à l’exécution.** Des messages de troncature, progression/validation et enregistrement des quatre variantes apparaîtront lors de votre exécution.

**À retenir.** C’est une cellule de calcul réel, pas une simple définition. Une attente longue sur CPU n’autorise pas à inventer la sortie ; les logs et les fichiers terminés doivent faire foi.

**Lecture complémentaire, dans l’ordre des opérations.**

L'appel démarre les quatre configurations AraBERT de la grille, sauf reprise de checkpoints compatibles. C'est une cellule coûteuse d'apprentissage, bien qu'elle n'ait qu'une ligne. La longueur d'un bloc de code n'indique donc pas sa durée.

**Code de référence.**

```python
run_plm_grid(ARA_NAME,"AraBERT")
```

**Petit extrait de sortie déjà enregistrée dans ce fichier (non recalculée pour le guide).**

```text
AraBERT train tronqué à 64 : 76 / 10842
```

<a id="tp2-074"></a>

#### Cellule de code 074 — Répéter les deux configurations finales AraBERT

**Objectif.** Compléter les graines 43/44 après la grille AraBERT.

**Comprendre le code.** `repeat_plm_finals` choisit le meilleur lambda AraBERT sur val à graine 42, puis ajuste le mono-tâche et ce multi-tâche pour les deux autres graines. Les poids de départ proviennent du préentraînement pour chaque nouveau run ; il ne s’agit pas de continuer le modèle entraîné avec la graine 42.

**Résultat attendu à l’exécution.** Deux groupes AraBERT complets à trois graines seront disponibles si les quatre runs se terminent.

**À retenir.** Cette étape peut durer plusieurs heures sur CPU. Un checkpoint partiel sans métadonnées finales ne remplace pas un run terminé ; le journal doit distinguer strictement ces deux situations.

**Lecture complémentaire, dans l’ordre des opérations.**

Cet appel ajoute les quatre runs AraBERT de répétition : E1 à 43/44 et le meilleur E2 à 43/44. Avec la graine 42 déjà disponible, chaque configuration finale dispose de trois graines. Les lambda non sélectionnés restent à une seule graine.

**Code de référence.**

```python
repeat_plm_finals(ARA_NAME,"AraBERT")
```

**Petit extrait de sortie déjà enregistrée dans ce fichier (non recalculée pour le guide).**

```text
E1_AraBERT_mono 44 époque 1 batch 50 / 339 perte 0.6745
E1_AraBERT_mono 44 époque 1 batch 100 / 339 perte 0.6112
E1_AraBERT_mono 44 époque 1 batch 150 / 339 perte 0.5665
E1_AraBERT_mono 44 époque 1 batch 200 / 339 perte 0.5411
E1_AraBERT_mono 44 époque 1 batch 250 / 339 perte 0.524
E1_AraBERT_mono 44 époque 1 batch 300 / 339 perte 0.5082
E1_AraBERT_mono 44 époque 1 perte 0.4939 F1 val 0.6776
E1_AraBERT_mono 44 époque 2 batch 50 / 339 perte 0.3161
E1_AraBERT_mono 44 époque 2 batch 100 / 339 perte 0.3329
E1_AraBERT_mono 44 époque 2 batch 150 / 339 perte 0.3384
E1_AraBERT_mono 44 époque 2 batch 200 / 339 perte 0.3288
E1_AraBERT_mono 44 époque 2 batch 250 / 339 perte 0.3336
E1_AraBERT_mono 44 époque 2 batch 300 / 339 perte 0.3324
E1_AraBERT_mono 44 époque 2 perte 0.3321 F1 val 0.6878
E1_AraBERT_mono 44 époque 3 batch 50 / 339 perte 0.2266
E1_AraBERT_mono 44 époque 3 batch 100 / 339 perte 0.2284
E1_AraBERT_mono 44 époque 3 batch 150 / 339 perte 0.2299
E1_AraBERT_mono 44 époque 3 batch 200 / 339 perte 0.2307
E1_AraBERT_mono 44 époque 3 batch 250 / 339 perte 0.2244
E1_AraBERT_mono 44 époque 3 batch 300 / 339 perte 0.2253
E1_AraBERT_mono 44 époque 3 perte 0.2256 F1 val 0.6734
E1_AraBERT_mono 44 époque 4 batch 50 / 339 perte 0.143
E1_AraBERT_mono 44 époque 4 batch 100 / 339 perte 0.143
E1_AraBERT_mono 44 époque 4 batch 150 / 339 perte 0.142
E1_AraBERT_mono 44 époque 4 batch 200 / 339 perte 0.1369
E1_AraBERT_mono 44 époque 4 batch 250 / 339 perte 0.1354
E1_AraBERT_mono 44 époque 4 batch 300 / 339 perte 0.1313
E1_AraBERT_mono 44 époque 4 perte 0.1313 F1 val 0.6729
E1_AraBERT_mono_seed44 F1 val = 0.6878 seuil = 0.8212 temps = 188.5 s
E2_AraBERT_lambda0p5 44 époque 1 batch 50 / 339 perte 1.2849
E2_AraBERT_lambda0p5 44 époque 1 batch 100 / 339 perte 1.1502
E2_AraBERT_lambda0p5 44 époque 1 batch 150 / 339 perte 1.0603
E2_AraBERT_lambda0p5 44 époque 1 batch 200 / 339 perte 0.9954
E2_AraBERT_lambda0p5 44 époque 1 batch 250 / 339 perte 0.9479
E2_AraBERT_lambda0p5 44 époque 1 batch 300 / 339 perte 0.9105
E2_AraBERT_lambda0p5 44 époque 1 perte 0.8836 F1 val 0.6833
E2_AraBERT_lambda0p5 44 époque 2 batch 50 / 339 perte 0.6004
E2_AraBERT_lambda0p5 44 époque 2 batch 100 / 339 perte 0.6
```

<a id="tp2-075"></a>

#### Cellule de code 075 — Exécuter la grille MARBERTv2

**Objectif.** Soumettre le deuxième encodeur imposé au même protocole mono/multi-tâche.

**Comprendre le code.** La grille utilise le tokenizer et le prétraitement minimal de MARBERT, puis évalue les quatre lambda, dont zéro pour le mono. Les durées ne sont pas déduites de celles d’AraBERT : elles sont mesurées. Les réglages de budget, sélection val et seuil restent communs pour rendre la comparaison lisible.

**Résultat attendu à l’exécution.** Quatre expériences MARBERT à graine 42 et leur taux de troncature seront produits ou repris si compatibles.

**À retenir.** Le vocabulaire et la tokenisation des deux PLM diffèrent. Une même longueur maximale n’implique pas exactement le même nombre de mots conservés ; lire les statistiques de troncature avant d’interpréter un écart de qualité.

**Lecture complémentaire, dans l’ordre des opérations.**

Même grille que pour AraBERT, mais avec MARBERT et son tokenizer/prétraitement propres. On ne recharge pas un checkpoint AraBERT dans une architecture MARBERT. Le nom court du run sert à distinguer les résultats, pas à changer les labels.

**Code de référence.**

```python
run_plm_grid(MAR_NAME,"MARBERT")
```

<a id="tp2-076"></a>

#### Cellule de code 076 — Répéter MARBERT et récapituler E

**Objectif.** Achever les groupes MARBERT puis afficher l’ensemble des expériences de fine-tuning.

**Comprendre le code.** La première instruction ajoute les graines 43/44 du mono MARBERT et de son meilleur lambda multi-tâche. Le tableau suivant filtre les lignes E et expose modèle, graine, F1 val, durée, paramètres entraînables et nombre d’époques effectivement parcourues. On peut ainsi vérifier que les configurations finales sont complètes avant le test.

**Résultat attendu à l’exécution.** Les deux encodeurs auront au total 16 runs E prévus par cette stratégie de grille puis répétitions, si tout s’est terminé.

**À retenir.** La comparaison des encodeurs doit utiliser les groupes finaux à trois graines, pas le score maximal d’un essai isolé. Ce tableau reste une évaluation de validation, même s’il résume beaucoup de calculs.

**Lecture complémentaire, dans l’ordre des opérations.**

Les quatre répétitions MARBERT complètent les 16 runs E. `epochs_completed` compte les époques parcourues, pas l'époque du checkpoint gagnant. La table permet de lire simultanément F1 val, durée et paramètres, sans utiliser les labels test pour cette comparaison.

**Code de référence.**

```python
repeat_plm_finals(MAR_NAME,"MARBERT")

display(pd.DataFrame(RESULTS).loc[lambda d:d.model.str.startswith('E')][['model','seed','val_f1_sarcastique','train_seconds','trainable_parameters','epochs_completed']])
```

### Sélection finale avant le test

<a id="tp2-077"></a>

#### Cellule de code 077 — Retrouver une ligne de résultat à partir de sa clé

**Objectif.** Passer du registre d’un modèle à sa mesure enregistrée sans multiplier les recherches manuelles.

**Comprendre le code.** `REGISTRY` fournit le nom et la graine associés à une clé. La fonction cherche ensuite dans `RESULTS` la ligne portant ces deux valeurs. Elle est utilisée pour comparer les F1 val, compléter les métriques test et agréger les répétitions.

**Résultat attendu à l’exécution.** `row_for_key` renverra la ligne correspondante sous forme de dictionnaire lors de ses appels.

**À retenir.** Le nom seul ne suffit pas à identifier un run : plusieurs graines partagent une architecture. Garder cette association explicite évite de mélanger seuils, probabilités et mesures de deux expériences.

**Lecture complémentaire, dans l’ordre des opérations.**

Le registre transforme une clé en nom/graine et la fonction retrouve la ligne correspondante. Deux graines du même modèle ont potentiellement des checkpoints et seuils différents. Cette fonction renvoie la ligne existante : ses mises à jour ultérieures modifient le journal en mémoire.

**Définitions importantes :** `row_for_key`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def row_for_key(key):
    rec=REGISTRY[key]; return next(r for r in RESULTS if r['model']==rec['name'] and r['seed']==rec['seed'])
```

<a id="tp2-078"></a>

#### Cellule de code 078 — Choisir les configurations finales sans consulter le test

**Objectif.** Fixer les représentants des familles A, C, D, E1 et E2 à partir de val.

**Comprendre le code.** Pour A/C/D, le code retient le meilleur F1 val parmi leurs candidats enregistrés. Pour E1, les encodeurs sont comparés par moyenne val des graines. Pour E2, seules les configurations disposant de trois lignes sont admissibles à la comparaison des moyennes : les lambda rejetés à graine unique ne peuvent donc pas battre artificiellement un groupe complet. `FINAL_GROUPS` rassemble les clés finales.

**Résultat attendu à l’exécution.** Un dictionnaire de cinq familles est préparé. A/C/D contiennent encore leur graine 42 ; E1/E2 référencent déjà les trois graines.

**À retenir.** La baseline balanced obligatoire reste documentée même si une autre variante A est retenue. Le choix d’un modèle final ne doit jamais être refait après lecture de son F1 test.

**Lecture complémentaire, dans l’ordre des opérations.**

A/C/D sont choisis sur leur F1 val, E1 sur sa moyenne val et E2 seulement parmi les configurations disposant de trois lignes. `FINAL_GROUPS` identifie les cinq familles finales entraînées. Ce choix est verrouillé avant l'évaluation test, ce qui évite une sélection opportuniste du meilleur score test.

**Code de référence.**

```python
best_a_key=max(baseline_keys,key=lambda k:row_for_key(k)['val_f1_sarcastique'])

best_c_key=max(static_keys,key=lambda k:row_for_key(k)['val_f1_sarcastique'])

best_d_key=max(d_keys,key=lambda k:row_for_key(k)['val_f1_sarcastique'])

edf=pd.DataFrame(RESULTS).loc[lambda d:d.model.str.startswith('E')]

best_e1_name=edf.loc[edf.model.str.startswith('E1')].groupby('model').val_f1_sarcastique.mean().idxmax()

e2_averages=edf.loc[edf.model.str.startswith('E2')].groupby('model').val_f1_sarcastique.agg(['mean','count'])

best_e2_name=e2_averages.loc[e2_averages['count'].eq(3),'mean'].idxmax()

FINAL_GROUPS={'A':[best_a_key],'C':[best_c_key],'D':[best_d_key],
    'E1':[f'{best_e1_name}_seed{s}' for s in SEEDS], 'E2':[f'{best_e2_name}_seed{s}' for s in SEEDS]}
```

<a id="tp2-079"></a>

#### Cellule de code 079 — Compléter les trois graines de A, C et D

**Objectif.** Donner aux cinq familles entraînées des répétitions finales comparables.

**Comprendre le code.** La configuration choisie à graine 42 est reconstruite avec 43 puis 44. Selon sa famille, le code utilise le pipeline TF-IDF, le réseau statique, la tête linéaire gelée ou BiLSTM attention gelé. Pour D, les représentations AraBERT déjà extraites sont réutilisées. Le meilleur prompt NLI est ensuite choisi à partir de son F1 val.

**Résultat attendu à l’exécution.** A/C/D obtiennent chacun trois clés. Le prompt NLI final est déterminé avant toute ouverture des labels test.

**À retenir.** Une méthode quasi déterministe peut avoir un écart-type nul entre graines ; cela ne supprime pas l’incertitude liée à l’échantillon test. Les graines évaluent surtout une variabilité d’apprentissage, pas toute l’incertitude scientifique.

**Lecture complémentaire, dans l’ordre des opérations.**

Cette cellule complète les graines 43/44 de A, C et D, en conservant les configurations retenues. D peut être une régression gelée ou une tête D2 selon le choix val. B1 choisit son prompt sur val mais reste un seul passage, sans entraînement multi-graines sur le corpus.

**Code de référence.**

```python
for family,key in [('A',best_a_key),('C',best_c_key),('D',best_d_key)]:
    rec=REGISTRY[key]; cfg=rec['config'].copy()
    for seed in [43,44]:
        seed_all(seed)
        if family=='A':
            repeated=run_tfidf(rec['name'],cfg['mode'],cfg['imbalance'],seed,cfg['drop_links'])
        elif family=='C':
            emb=ARAVEC if cfg['embedding']=='aravec' else None
            repeated,model=train_neural(StaticNet(cfg['arch'],emb,position=cfg.get('position',True),freeze=emb is not None),
                [XW_TRAIN],[XW_VAL],rec['name'],seed=seed,config=cfg,
                imbalance=cfg.get('imbalance','weighted'),freeze_epochs=cfg.get('freeze_epochs',0))
            del model; gc.collect()
        elif rec['kind']=='frozen_lr':
            pooling=cfg['pooling']; clf=LogisticRegression(C=1,class_weight='balanced',max_iter=1500,random_state=seed)
            start=time.perf_counter(); clf.fit(pooled_features(FT,MT,pooling),train.y); seconds=time.perf_counter()-start
            path=ROOT/f'models/{rec["name"]}_seed{seed}.joblib'; joblib.dump(clf,path)
            repeated=register(rec['name'],seed,clf.predict_proba(pooled_features(FV,MV,pooling))[:,1],seconds,clf.coef_.size+1,
                'arabert_official','frozen_lr',path,cfg)
        else:
            repeated,model=train_neural(FrozenHead(),[torch.from_numpy(np.array(FT)),torch.tensor(MT)],
                [torch.from_numpy(np.array(FV)),torch.tensor(MV)],rec['name'],seed=seed,config=cfg)
            del model; gc.collect()
        FINAL_GROUPS[family].append(repeated)

best_nli_key=max(nli_keys,key=lambda k:row_for_key(k)['val_f1_sarcastique'])
```

<a id="tp2-080"></a>

#### Cellule de code 080 — Verrouiller modèles et seuils avant le test

**Objectif.** Créer une trace explicite de ce qui a été choisi avant de connaître les réponses du test officiel.

**Comprendre le code.** `FINAL_KEYS` réunit les 15 runs des cinq familles entraînées et les deux méthodes directes B1/B2. Le manifeste sauvegarde horodatage, groupes, configurations, seuils, signatures des données et budgets. Un tableau récapitule les métriques val utilisées pour cette sélection.

**Résultat attendu à l’exécution.** `results/final_manifest.json` est écrit et doit lister 17 références finales avec leurs décisions fixées.

**À retenir.** Un horodatage documente la procédure, sans constituer une preuve cryptographique de non-consultation du test. La discipline expérimentale est de conserver ces choix lors des cellules F/G suivantes.

**Lecture complémentaire, dans l’ordre des opérations.**

Le manifeste stocke les modèles et seuils retenus avant l'ouverture des labels test. C'est une trace du verrouillage du protocole, pas seulement une table finale de scores. Vérifier trois clés par famille ne garantit pas à lui seul que les fichiers distants ont été rapatriés localement.

**Code de référence.**

```python
FINAL_KEYS=[k for keys in FINAL_GROUPS.values() for k in keys]+[best_nli_key,b2_key]

manifest={'locked_at_utc':datetime.now(timezone.utc).isoformat(),'groups':FINAL_GROUPS,
    'models':[REGISTRY[k] for k in FINAL_KEYS],'data_sha256':DATA_SIG,'seeds':SEEDS,
    'hyperparameters':{'max_len':MAX_LEN,'plm_epochs':PLM_EPOCHS,'static_epochs':STATIC_EPOCHS,'batch_plm':BATCH}}

(ROOT/'results/final_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')

display(pd.DataFrame([row_for_key(k) for k in FINAL_KEYS])[['model','seed','val_f1_sarcastique','threshold','train_seconds']])

print('Manifestes et seuils verrouillés AVANT ouverture des labels test.')
```

### Partie F : évaluation et généralisation

<a id="tp2-081"></a>

#### Cellule de code 081 — Ouvrir le test officiel en partie F

**Objectif.** Évaluer la généralisation après le verrouillage, sans réutiliser le test comme validation.

**Comprendre le code.** Le CSV test est enfin lu avec ses annotations et reçoit ses identifiants officiels. L’absence de collision avec train est revérifiée. Les distributions globales, par dialecte et par sentiment sont comparées entre partitions. Les corrélations URL/mention/artefact sont complétées pour test. Le train officiel et le test sont réunis pour rechercher des clés contradictoires a posteriori.

**Résultat attendu à l’exécution.** Les tableaux décrivent le prior et les sous-groupes test réellement observés. Aucun modèle n’est entraîné dans cette cellule.

**À retenir.** Ces informations sont descriptives. Leur découverte ne permet pas de modifier ensuite seuils, prompts ou hyperparamètres pour améliorer le score test annoncé ; cela invaliderait son rôle d’évaluation indépendante.

**Lecture complémentaire, dans l’ordre des opérations.**

Le CSV test est maintenant lu avec ses labels, et les statistiques supervisées test deviennent autorisées en F. Les taux globaux et par groupe exposent le changement de distribution. Les identifiants restent ceux du test officiel ; on ne filtre pas les exemples difficiles hors de l'évaluation globale.

**Code de référence.**

```python
test=pd.read_csv(DATA_DIR/'testing_data.csv')

test['id']=np.arange(len(test))

test['y']=test.sarcasm.astype(int)

test['key']=test.tweet.map(duplicate_key)

assert len(test)==3000 and set(train.key).isdisjoint(test.key)

display(pd.DataFrame({n:{'n':len(f),'sarcastiques':int(f.y.sum()),'taux':f.y.mean()} for n,f in [('train',train),('val',val),('test',test)]}).T)

for column in ['dialect','sentiment']:
    display(pd.concat({n:f.groupby(column).y.agg(['size','sum','mean']) for n,f in [('train',train),('val',val),('test',test)]}))

for name,frame in [('train',train),('test',test)]:
    flags=text_features(frame.tweet)
    display(pd.DataFrame([{'split':name,'indice':c,'presence':present,'n':int(flags[c].eq(present).sum()),
        'taux_sarcasme':frame.y[flags[c].eq(present)].mean()} for c in ['url','mention','artefact_guillemets'] for present in [False,True]]))

combined=pd.concat([full.assign(split='train_officiel'),test.assign(split='test')],ignore_index=True)

combined['key']=combined.tweet.map(duplicate_key)
```

<a id="tp2-082"></a>

#### Cellule de code 082 — Compléter l’audit post-test et extraire ses états gelés

**Objectif.** Relier les contradictions, la couverture linguistique et les représentations nécessaires aux prédictions D.

**Comprendre le code.** Les groupes portant plusieurs labels sont exportés à partir du corpus combiné. Les effectifs `magreb` de train et test rendent visible la faiblesse éventuelle de couverture pour un usage marocain. Les OOV et troncatures sont calculés avec le vocabulaire train déjà fixé. `frozen_features` fournit ensuite les états et masques du test, sans mettre à jour l’encodeur.

**Résultat attendu à l’exécution.** Un CSV de contradictions et des statistiques post-test sont produits ; `FX` et `MX` pourront être partagés entre les modèles D.

**À retenir.** Observer des inconnus dans test n’autorise pas à agrandir rétroactivement le vocabulaire C. Le groupe `magreb` ne garantit pas une représentation suffisante de la darija marocaine ni de l’Arabizi.

**Lecture complémentaire, dans l’ordre des opérations.**

Les contradictions sont désormais recherchées dans l'ensemble autorisé après ouverture du test, pour l'analyse. La couverture lexicale et le nombre `magreb` aident à interpréter les limites. Les représentations gelées test sont calculées pour D, sans entraîner l'encodeur ni ajuster une tête sur ces labels.

**Code de référence.**

```python
all_contradictions=set(combined.groupby('key').y.nunique().loc[lambda s:s>1].index)

combined.loc[combined.key.isin(all_contradictions)].to_csv(ROOT/'results/contradictions_all_post_test.csv',index=False,encoding='utf-8-sig')

display(pd.DataFrame({'split':['train','test'],'magreb':[int(train.dialect.eq('magreb').sum()),int(test.dialect.eq('magreb').sum())]}))

display(pd.DataFrame({n:token_stats(f.tweet) for n,f in [('train',train),('val',val),('test',test)]}).T)

FX,MX=frozen_features(test.tweet.tolist(),'test')
```

<a id="tp2-083"></a>

#### Cellule de code 083 — Recharger un modèle enregistré

**Objectif.** Reconstruire la bonne architecture avant d’y replacer les poids de son checkpoint.

**Comprendre le code.** TF-IDF et les têtes linéaires sont relus avec `joblib`. Pour les réseaux, le champ `kind` détermine si l’on construit `StaticNet`, `FrozenHead` ou `FineTuneNet`. La configuration restitue les variantes nécessaires, puis `load_state_dict` charge les tenseurs sauvegardés. Le modèle est envoyé sur le périphérique courant et mis en évaluation.

**Résultat attendu à l’exécution.** La fonction renverra un modèle prêt pour l’inférence. Les méthodes B1/B2 ont leur propre procédure de chargement et peuvent renvoyer ici `None`.

**À retenir.** Un fichier de poids ne suffit pas sans la structure et les conventions d’entrée correspondantes. Un état incompatible doit provoquer une erreur plutôt qu’être chargé partiellement pour fabriquer un résultat.

**Lecture complémentaire, dans l’ordre des opérations.**

Le type `kind` détermine le constructeur avant chargement du `state_dict`. Les poids ne contiennent pas à eux seuls la définition de toutes les architectures : la configuration doit correspondre. `weights_only=True` limite le chargement aux structures de poids attendues pour les checkpoints PyTorch.

**Définitions importantes :** `load_registered`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def load_registered(rec):
    cfg=rec['config']; kind=rec['kind']
    if kind in ['tfidf','frozen_lr']: return joblib.load(ROOT/rec['path'])
    if kind=='static': model=StaticNet(cfg['arch'],position=cfg.get('position',True))
    elif kind=='frozen_head': model=FrozenHead()
    elif kind=='plm': model=FineTuneNet(cfg['model_name'],cfg['multitask'])
    else: return None
    model.load_state_dict(torch.load(ROOT/rec['path'],map_location=DEVICE,weights_only=True)); return model.to(DEVICE).eval()
```

<a id="tp2-084"></a>

#### Cellule de code 084 — Uniformiser la prédiction entre les familles

**Objectif.** Appliquer à chaque texte le pipeline compatible avec la configuration enregistrée.

**Comprendre le code.** Le registre choisit nettoyage, encodage et modèle : TF-IDF reçoit des chaînes, C des identifiants de mots, D des états gelés et E les tenseurs du tokenizer PLM. Les variantes sans liens retirent ceux-ci avant traitement. Les états test déjà extraits sont réutilisés pour D. Si un modèle est fourni, il est conservé pour éviter son rechargement ; sinon, l’objet local est libéré en fin d’appel.

**Résultat attendu à l’exécution.** Un vecteur de scores aligné sur les textes est renvoyé. Pour B2, ce vecteur représente exceptionnellement la règle binaire, tandis que P(négatif) est conservée séparément pour l’AP.

**À retenir.** Cette fonction ne sélectionne pas un nouveau seuil. Modifier l’entrée au moment d’une ablation doit être explicite et cohérent, pas un changement silencieux du pipeline final.

**Lecture complémentaire, dans l’ordre des opérations.**

Le répartiteur utilise le pipeline adapté à chaque famille et applique les ablations prévues dans sa configuration. Si un modèle est déjà fourni, il peut être réutilisé au lieu d'être rechargé. Les conventions B2 et NLI restent spécifiques ; le score de B2 destiné à la décision n'est pas sa probabilité continue d'AP.

**Définitions importantes :** `predict_registered`. Définir ces objets ne les appelle pas encore.

**Code de référence.**

```python
def predict_registered(rec,texts,split_tag=None,model=None):
    kind=rec['kind']; cfg=rec['config']; own=model is None
    if model is None: model=load_registered(rec)
    if cfg.get('drop_links') and kind in ['static','frozen_lr','frozen_head']:
        texts=[MENTION_RE.sub(' ',URL_RE.sub(' ',s)) for s in texts]
        split_tag=None
    if kind=='tfidf':
        p=model.predict_proba([prepare_text(s,cfg['mode'],cfg['drop_links']) for s in texts])[:,1]
    elif kind=='static': p=neural_predict(model,[word_encode(texts)])
    elif kind in ['frozen_lr','frozen_head']:
        if split_tag=='test': h,m=FX,MX
        else: h,m=frozen_features(texts,split_tag or hashlib.sha256('\n'.join(texts).encode()).hexdigest()[:12])
        if kind=='frozen_lr': p=model.predict_proba(pooled_features(h,m,cfg['pooling']))[:,1]
        else: p=neural_predict(model,[torch.tensor(np.asarray(h)),torch.tensor(m)])
    elif kind=='plm': p=neural_predict(model,plm_encode(texts,cfg['model_name'],cfg.get('drop_links',False)),batch=BATCH)
    elif kind=='nli': p=cached_nli(texts,cfg['language'],split_tag or 'custom')[0]
    elif kind=='sentiment': p=sentiment_predict(texts,split_tag or 'custom')[1].astype(float)
    else: raise ValueError(kind)
    if own: del model; gc.collect()
    return p
```

<a id="tp2-085"></a>

#### Cellule de code 085 — Évaluer les modèles finaux et exporter leurs prédictions

**Objectif.** Produire des scores test vérifiables ligne par ligne par l’enseignant.

**Comprendre le code.** Chaque clé finale est évaluée avec son seuil fixé sur val. Le CSV exporte `id`, `y_true`, `y_pred` et `probability`. Pour B2, la décision reste la règle argmax négatif, mais la probabilité exportée est P(négatif) et l’AP utilise ce score continu. Le temps global de pipeline et la taille de l’artefact sont enregistrés ; les champs test sont ajoutés aux lignes de résultats.

**Résultat attendu à l’exécution.** Dix-sept CSV finaux et un tableau F1/macro-F1/AP test sont attendus après calcul. Une figure à deux panneaux est préparée pour les courbes suivantes.

**À retenir.** Le temps global peut inclure chargement et lecture de caches ; il ne remplace pas la latence CPU batch 1 mesurée en G. Les scores exportés servent à recalculer les métriques, pas à retoucher les décisions après coup.

**Mise à jour qualité / affichage.** La figure diagnostic n'est plus créée ici : elle attend la cellule suivante, qui possède les données nécessaires pour la remplir immédiatement.

**Lecture complémentaire, dans l’ordre des opérations.**

Un CSV est écrit par modèle final avec `id,y_true,y_pred,probability`. B2 exporte sa probabilité sentiment et conserve sa décision argmax, qui ne doit pas être recalculée par un seuil de probabilité 0,5. Les tableaux de résultats sauvegardés dans le notebook ne sont pas ces CSV : ils restent sur le serveur d'exécution tant qu'ils ne sont pas téléchargés.

**Code de référence.**

```python
TEST_P={}

TEST_ROWS=[]

for key in FINAL_KEYS:
    rec=REGISTRY[key]; path=ROOT/f'predictions/{key}.csv'
    start=time.perf_counter(); p=predict_registered(rec,test.tweet.tolist(),'test'); elapsed=time.perf_counter()-start
    TEST_P[key]=p; pred=(p>=rec['threshold']).astype(int)
    export_p=sentiment_predict(test.tweet.tolist(),'test')[0] if rec['kind']=='sentiment' else p
    pd.DataFrame({'id':test.id,'y_true':test.y,'y_pred':pred,'probability':np.asarray(export_p,dtype=np.float64)}).to_csv(path,index=False)
    row=row_for_key(key); row.update({'test_'+k:v for k,v in scores(test.y,p,rec['threshold']).items()})
    # AP B2 doit utiliser la probabilité de sentiment, pas seulement la décision binaire.
    if rec['kind']=='sentiment': row['test_pr_auc_ap']=float(average_precision_score(test.y,sentiment_predict(test.tweet.tolist(),'test')[0]))
    row['test_inference_total_seconds']=elapsed; row['test_pipeline_ms_per_tweet']=1000*elapsed/len(test)
    row['artifact_size_mb']=(ROOT/rec['path']).stat().st_size/1e6 if (ROOT/rec['path']).is_file() else sum(f.stat().st_size for f in (ROOT/rec['path']).glob('*') if f.is_file())/1e6
    TEST_ROWS.append(row.copy()); pd.DataFrame(RESULTS).to_csv(ROOT/'results/results.csv',index=False)
    print(key,'F1 test',round(row['test_f1_sarcastique'],4),flush=True)

summary=pd.DataFrame(TEST_ROWS)

display(summary[['model','seed','threshold','test_f1_sarcastique','test_macro_f1','test_pr_auc_ap']])
```

<a id="tp2-086"></a>

#### Cellule de code 086 — Tracer l’effet du seuil sur le test à titre diagnostique

**Objectif.** Étudier comment un changement de distribution peut déplacer le compromis précision/rappel.

**Comprendre le code.** Pour la graine 42 de chaque famille, une grille de 201 seuils entre 0 et 1 fournit F1=f(seuil). Un point marque le seuil utilisable, déjà choisi sur val. La courbe PR est également construite. Le meilleur seuil calculé directement sur les labels test est stocké séparément avec son F1, sous l’étiquette posthoc.

**Résultat attendu à l’exécution.** Les courbes sont alimentées et `threshold_rows` compare seuil val et seuil test a posteriori.

**À retenir.** Le maximum test est une description optimiste obtenue en connaissant les réponses. Il n’est jamais autorisé à remplacer le seuil val pour annoncer la performance finale ou choisir le modèle de production.

**Mise à jour qualité / affichage.** Les deux axes sont maintenant créés, tracés, annotés, sauvegardés et affichés ici, dans une seule cellule. Les seuils posthoc restent un diagnostic test : ils ne servent pas à sélectionner ou régler le modèle.

**Lecture complémentaire, dans l’ordre des opérations.**

Les courbes F1(seuil) et précision-rappel utilisent les scores test du modèle déjà fixé. Les points val servent de référence visuelle ; l'optimum test est calculé seulement pour constater un décalage. Le code ne remplace pas automatiquement les seuils du registre par cet optimum.

**Code de référence.**

```python
fig,axes=plt.subplots(1,2,figsize=(11,4))

threshold_rows=[]

for family,keys in FINAL_GROUPS.items():
    key=keys[0]; p=TEST_P[key]; t=REGISTRY[key]['threshold']; grid=np.linspace(0,1,201)
    f=[f1_score(test.y,p>=g,zero_division=0) for g in grid]
    axes[0].plot(grid,f,label=family); axes[0].scatter([t],[f1_score(test.y,p>=t)],s=20)
    prec,rec,_=precision_recall_curve(test.y,p); axes[1].plot(rec,prec,label=family)
    post=best_threshold(test.y,p)
    threshold_rows.append({'famille':family,'seuil_val':t,'seuil_test_posthoc':post,
        'f1_test_seuil_val':scores(test.y,p,t)['f1_sarcastique'],'f1_test_posthoc':scores(test.y,p,post)['f1_sarcastique']})

axes[0].set(title='Test diagnostic a posteriori uniquement',xlabel='Seuil',ylabel='F1 sarcastique')

axes[0].legend()

axes[1].set(title='Courbes précision rappel test',xlabel='Rappel',ylabel='Précision')

axes[1].legend()

savefig(fig,'threshold_and_pr_test')
```

<a id="tp2-087"></a>

#### Cellule de code 087 — Présenter et sauvegarder le diagnostic de seuil

**Objectif.** Rendre explicite la différence entre réglage utilisable et observation a posteriori.

**Comprendre le code.** La cellule ajoute titres, axes et légendes aux courbes précédentes. L’image `threshold_and_pr_test.png` et le tableau `threshold_shift.csv` sont sauvegardés, puis les valeurs sont affichées. La courbe précision-rappel expose plusieurs compromis sans en choisir un nouveau pour le modèle final.

**Résultat attendu à l’exécution.** Une figure à deux panneaux et un tableau lisible documentent le décalage de seuil.

**À retenir.** Un grand écart entre seuils peut signaler prior, calibration ou distribution différents. Il ne suffit pas à attribuer tout l’écart au seul prior, car vocabulaire, dialectes et autres caractéristiques peuvent aussi changer.

**Mise à jour qualité / affichage.** Cette cellule exporte et affiche seulement le tableau du décalage de seuil ; les courbes complètes sont affichées en cellule 086.

**Lecture complémentaire, dans l’ordre des opérations.**

Le tableau `threshold_rows` contient des colonnes qui distinguent seuil val et seuil test posthoc. Il ne faut pas annoncer le F1 posthoc comme performance d'un système dont le seuil aurait été fixé avant le test. Le CSV rend cette distinction consultable hors du noyau.

**Code de référence.**

```python
pd.DataFrame(threshold_rows).to_csv(ROOT/'results/threshold_shift.csv',index=False)

display(pd.DataFrame(threshold_rows))
```

<a id="tp2-088"></a>

#### Cellule de code 088 — Évaluer les sous-groupes et retirer les indices des vrais positifs

**Objectif.** Compléter le score global par des mesures de robustesse ciblées, sans changer le seuil.

**Comprendre le code.** Pour chaque famille à graine 42, les scores sont calculés avec/sans URL et par dialecte, en affichant effectifs et positifs. F4 sélectionne ensuite les vrais positifs du modèle original : les textes sont modifiés par `erase_cues`, puis repassés dans le même modèle. La proportion encore détectée est comparée au taux initial de 1 sur ce même ensemble conditionnel.

**Résultat attendu à l’exécution.** `subgroups.csv` et `cues_ablation.csv` rapportent les sous-groupes et la chute de détection.

**À retenir.** Cette chute n’est ni une baisse de F1 sur tout le test ni un rappel global. Le sous-ensemble varie selon le modèle ; les différences entre familles doivent donc être lues avec leurs effectifs de vrais positifs initiaux.

**Lecture complémentaire, dans l’ordre des opérations.**

Les sous-groupes URL et dialectes sont évalués à seuil inchangé. Pour F4, `tp_mask` ne garde que les vrais positifs initiaux ; la chute est `1 - détection_après`, pas une variation de F1 sur tout le test. Ne pas confondre le nombre de vrais positifs sélectionnés avec le nombre total de sarcastiques réels.

**Code de référence.**

```python
subgroups=[]

cue_rows=[]

url_flags=text_features(test.tweet).url.to_numpy()

for family,keys in FINAL_GROUPS.items():
    key=keys[0]; rec=REGISTRY[key]; p=TEST_P[key]; t=rec['threshold']
    for group in [False,True]:
        mask=url_flags==group
        subgroups.append({'famille':family,'URL':group,'n':int(mask.sum()),'positifs':int(test.y[mask].sum()),**scores(test.y[mask],p[mask],t)})
    for dialect in sorted(test.dialect.unique()):
        mask=test.dialect.eq(dialect).to_numpy()
        subgroups.append({'famille':family,'dialecte':dialect,'n':int(mask.sum()),'positifs':int(test.y[mask].sum()),**scores(test.y[mask],p[mask],t)})
    mask=(test.y.to_numpy()==1)&(p>=t); texts=[erase_cues(s) for s in test.tweet[mask]]
    if texts:
        modified=predict_registered(rec,texts,f'cues_{family}')
        after=float(np.mean(modified>=t))
        cue_rows.append({'famille':family,'n_initialement_detectes':int(mask.sum()),'taux_detection_avant':1.,'taux_detection_apres':after,'chute':1-after})

pd.DataFrame(subgroups).to_csv(ROOT/'results/subgroups.csv',index=False)

pd.DataFrame(cue_rows).to_csv(ROOT/'results/cues_ablation.csv',index=False)

display(pd.DataFrame(subgroups))

display(pd.DataFrame(cue_rows))
```

<a id="tp2-089"></a>

#### Cellule de code 089 — Fixer le meilleur modèle E pour l’ablation dialectale

**Objectif.** Choisir la référence de réentraînement sans dialecte égyptien à partir de validation.

**Comprendre le code.** La moyenne F1 val des groupes E1 et E2 est comparée. La graine 42 de la famille gagnante fournit encodeur, architecture mono/multi et lambda de référence. Sa configuration est copiée afin de construire des variantes sans modifier les paramètres enregistrés du modèle original. Une liste vide accueillera les diagnostics de réentraînement.

**Résultat attendu à l’exécution.** `erec`, `ecfg`, `ename` et `robust_rows` préparent les cellules F2/F3 suivantes.

**À retenir.** Le modèle de référence est choisi sur val, pas selon son résultat sur les tweets egypt test. La sélection principale a cependant vu des exemples egypt : ce point limitera la revendication de dialecte entièrement non vu.

**Lecture complémentaire, dans l’ordre des opérations.**

Le meilleur E entre mono et multi est choisi par moyenne val sur les graines déjà prévues. Sa configuration est copiée pour les variantes F. L'architecture n'est donc pas choisie en regardant lequel fonctionne le mieux sur le dialecte test après coup.

**Code de référence.**

```python
best_e_family=max(['E1','E2'],key=lambda f:np.mean([row_for_key(k)['val_f1_sarcastique'] for k in FINAL_GROUPS[f]]))

best_e_key=FINAL_GROUPS[best_e_family][0]

erec=REGISTRY[best_e_key]

ecfg=erec['config'].copy()

ename=ecfg['model_name']

robust_rows=[]
```

<a id="tp2-090"></a>

#### Cellule de code 090 — Réentraîner A, C et D sans URL ni mention

**Objectif.** Mesurer l’effet des raccourcis après adaptation du modèle, au lieu de seulement supprimer des éléments à l’inférence.

**Comprendre le code.** Les configurations A/C/D retenues sont reconstruites à graine 42. Les liens et mentions sont retirés des textes de train, val et test. A apprend un nouveau TF-IDF ; C garde son architecture et son vocabulaire de référence pour cette ablation ; D utilise de nouvelles extractions gelées sur les textes modifiés puis apprend une nouvelle tête. Chaque variante choisit son seuil sur val et est comparée à son original sur le même test.

**Résultat attendu à l’exécution.** Trois lignes de robustesse et leurs fichiers de prédictions sont ajoutés, sans remplacer les résultats des modèles finaux.

**À retenir.** Cette cellule lance des entraînements et éventuellement des extractions, même si les encodeurs D sont gelés. Une différence combine l’effet du retrait des indices et celui du réapprentissage ; ce n’est pas la même question que F4.

**Lecture complémentaire, dans l’ordre des opérations.**

A/C/D sont réappris sur train sans URL/mention ; val et test sont prétraités selon la même ablation. Pour D, les états doivent être réextraits puisque changer le texte change les représentations gelées. Comparer ce résultat à une simple suppression à l'inférence mélangerait deux questions différentes.

**Code de référence.**

```python
for family in ['A','C','D']:
    reference_key=FINAL_GROUPS[family][0]; reference=REGISTRY[reference_key]; cfg=reference['config'].copy()
    seed_all(42); name=f'F_{family}_sans_url_mention'
    if family=='A':
        key=run_tfidf(name,cfg['mode'],cfg['imbalance'],42,True)
        p=predict_registered(REGISTRY[key],test.tweet.tolist(),'test')
    elif family=='C':
        xt=[word_encode([prepare_text(s,drop_links=True) for s in train.tweet])]
        xv=[word_encode([prepare_text(s,drop_links=True) for s in val.tweet])]
        emb=ARAVEC if cfg['embedding']=='aravec' else None
        cfg['drop_links']=True
        key,model=train_neural(StaticNet(cfg['arch'],emb,position=cfg.get('position',True),freeze=emb is not None),xt,xv,name,
            config=cfg,imbalance=cfg.get('imbalance','weighted'),freeze_epochs=cfg.get('freeze_epochs',0))
        p=neural_predict(model,[word_encode([prepare_text(s,drop_links=True) for s in test.tweet])]); del model; gc.collect()
    else:
        # Extraction sans URL/mention ; mêmes poids gelés, nouvelle tête entraînée sur train seulement.
        ht,mt=frozen_features([MENTION_RE.sub(' ',URL_RE.sub(' ',s)) for s in train.tweet],'train_without_links')
        hv,mv=frozen_features([MENTION_RE.sub(' ',URL_RE.sub(' ',s)) for s in val.tweet],'val_without_links')
        hx,mx=frozen_features([MENTION_RE.sub(' ',URL_RE.sub(' ',s)) for s in test.tweet],'test_without_links')
        if reference['kind']=='frozen_lr':
            pooling=cfg['pooling']; model=LogisticRegression(C=1,class_weight='balanced',max_iter=1500,random_state=42)
            start=time.perf_counter(); model.fit(pooled_features(ht,mt,pooling),train.y); seconds=time.perf_counter()-start
            path=ROOT/f'models/{name}_seed42.joblib'; joblib.dump(model,path)
            key=register(name,42,model.predict_proba(pooled_features(hv,mv,pooling))[:,1],seconds,model.coef_.size+1,
                'arabert_official_without_links','frozen_lr',path,dict(cfg,drop_links=True))
            p=model.predict_proba(pooled_features(hx,mx,pooling))[:,1]
        else:
            key,model=train_neural(FrozenHead(),[torch.tensor(np.asarray(ht)),torch.tensor(mt)],
                [torch.tensor(np.asarray(hv)),torch.tensor(mv)],name,config=dict(cfg,drop_links=True))
            p=neural_predict(model,[torch.tensor(np.asarray(hx)),torch.tensor(mx)]); del model; gc.collect()
    threshold=REGISTRY[key]['threshold']; rr={'diagnostic':f'{family}_sans_url_mention','n_train':len(train),'n_test':len(test),
        'threshold':threshold,**scores(test.y,p,threshold),'f1_reference':scores(test.y,TEST_P[reference_key],reference['threshold'])['f1_sarcastique']}
    robust_rows.append(rr); row_for_key(key).update({'test_'+k:v for k,v in scores(test.y,p,threshold).items()})
    pd.DataFrame({'id':test.id,'y_true':test.y,'y_pred':(p>=threshold).astype(int),'probability':np.asarray(p,dtype=np.float64)}).to_csv(ROOT/f'predictions/{key}.csv',index=False)
```

<a id="tp2-091"></a>

#### Cellule de code 091 — Réentraîner E sans egypt et sans liens

**Objectif.** Tester deux difficultés de généralisation avec des variantes séparées du fine-tuning complet.

**Comprendre le code.** Trois runs sont prévus : meilleur E sans egypt, E1 sans URL/mention et E2 sans URL/mention. Pour sans egypt, train et validation de sélection excluent ce dialecte ; le seuil est donc choisi uniquement sur val non egypt. Son score est comparé sur le sous-ensemble egypt test à la référence entraînée sur tout train. Les variantes sans liens gardent tout train, retirent URL/mention partout et évaluent le test complet. Les CSV conservent toutes les lignes, avec le sous-ensemble évalué précisé dans le journal.

**Résultat attendu à l’exécution.** Trois diagnostics supplémentaires, dont un score ciblé egypt, sont exportés avec leurs probabilités.

**À retenir.** L’architecture et le lambda de sans egypt héritent du protocole principal : ce n’est pas une sélection entièrement aveugle à ce dialecte. Le préentraînement externe peut aussi en contenir. Ces runs ne remplacent pas les modèles finaux verrouillés.

**Lecture complémentaire, dans l’ordre des opérations.**

F2 retire `egypt` du train et aussi de la validation utilisée pour choisir checkpoint/seuil. La prédiction peut être faite sur les 3 000 tweets, mais la métrique F2 est calculée uniquement sur les tweets egypt. Les deux autres variantes E1/E2 sans liens utilisent tout le train et constituent des runs distincts.

**Code de référence.**

```python
for diagnostic in ['sans_egypt','E1_sans_url_mention','E2_sans_url_mention']:
    reference_e=erec if diagnostic=='sans_egypt' else REGISTRY[FINAL_GROUPS[diagnostic[:2]][0]]
    ecfg=reference_e['config'].copy(); ename=ecfg['model_name']
    subset=train.loc[~train.dialect.eq('egypt')] if diagnostic=='sans_egypt' else train
    drop=diagnostic!='sans_egypt'; cfg=dict(ecfg,train_ids=subset.id.tolist(),drop_links=drop)
    xt=plm_encode(subset.tweet.tolist(),ename,drop); xv=plm_encode(val.tweet.tolist(),ename,drop)
    seed_all(42)
    key,model=train_neural(FineTuneNet(ename,ecfg['multitask']),xt,xv,f'F_{diagnostic}',config=cfg,
        epochs=PLM_EPOCHS,lr=2e-5,lam=ecfg['lam'],plm=True,
        labels=torch.tensor(subset.y.to_numpy(),dtype=torch.long),sentiments=torch.tensor(subset.sentiment.map(SENT_MAP).to_numpy(),dtype=torch.long),
        val_mask=(~val.dialect.eq('egypt').to_numpy()) if diagnostic=='sans_egypt' else None)
    pv=neural_predict(model,xv,batch=BATCH)
    vm=~val.dialect.eq('egypt').to_numpy() if diagnostic=='sans_egypt' else np.ones(len(val),dtype=bool)
    threshold=best_threshold(val.y[vm],pv[vm])
    REGISTRY[key]['threshold']=threshold
    row_for_key(key).update({'val_'+k:v for k,v in scores(val.y,pv,threshold).items()})
    pp=neural_predict(model,plm_encode(test.tweet.tolist(),ename,drop),batch=BATCH)
    tm=test.dialect.eq('egypt').to_numpy() if diagnostic=='sans_egypt' else np.ones(len(test),dtype=bool)
    rr={'diagnostic':diagnostic,'n_train':len(subset),'n_test':int(tm.sum()),'threshold':threshold,
        **scores(test.y[tm],pp[tm],threshold),
        'f1_reference':scores(test.y[tm],TEST_P[reference_e['key']][tm],reference_e['threshold'])['f1_sarcastique']}
    robust_rows.append(rr)
    row_for_key(key).update({'test_'+k:v for k,v in scores(test.y[tm],pp[tm],threshold).items()})
    row_for_key(key).update({'threshold':threshold,'test_subset':'egypt' if diagnostic=='sans_egypt' else 'all','n_test_evaluated':int(tm.sum())})
    pd.DataFrame({'id':test.id,'y_true':test.y,'y_pred':(pp>=threshold).astype(int),'probability':np.asarray(pp,dtype=np.float64)}).to_csv(ROOT/f'predictions/{key}.csv',index=False)
    del model,xt,xv; gc.collect()
```

<a id="tp2-092"></a>

#### Cellule de code 092 — Rassembler les résultats des réentraînements de robustesse

**Objectif.** Lire les effets des ablations avec les mêmes colonnes de comparaison.

**Comprendre le code.** `robust_rows` contient les six diagnostics des cellules précédentes : cinq variantes sans liens et le meilleur E sans egypt. La table précise méthode, effectifs, seuil, métriques et F1 de référence sur le sous-ensemble correspondant. Elle est sauvegardée dans `retraining_robustness.csv` puis affichée.

**Résultat attendu à l’exécution.** Un tableau unique permet d’identifier gain, stabilité ou dégradation selon le diagnostic.

**À retenir.** Comparer les scores sur le même périmètre : un F1 egypt ne se compare pas directement à un F1 de tout le test. Les différences sont des résultats d’ablation du corpus observé, pas une garantie de comportement sur de nouvelles sources.

**Lecture complémentaire, dans l’ordre des opérations.**

Le tableau rassemble le F1 de référence et celui du réentraînement avec les effectifs réellement évalués. Une différence de score doit être interprétée sur le même sous-groupe, notamment pour `sans_egypt`. Une ligne sur 306 tweets ne se compare pas directement à une ligne globale sur 3 000.

**Code de référence.**

```python
pd.DataFrame(robust_rows).to_csv(ROOT/'results/retraining_robustness.csv',index=False)

display(pd.DataFrame(robust_rows))
```

<a id="tp2-093"></a>

#### Cellule de code 093 — Calculer les intervalles bootstrap à 95 %

**Objectif.** Quantifier l’incertitude due aux tweets du test pour un modèle et un seuil fixés.

**Comprendre le code.** Pour chacun des 15 runs des cinq familles entraînées, 1 000 jeux d’indices sont tirés avec remise, chacun de la taille du test. Le F1 est recalculé sur chaque rééchantillon en gardant les prédictions originales. Les percentiles 2,5 et 97,5 fournissent les bornes empiriques. Aucun réseau n’est réentraîné dans cette étape.

**Exemple ou repère.** Dans un rééchantillon, le même tweet peut apparaître plusieurs fois et un autre ne pas apparaître : c’est le principe du tirage avec remise, pas une modification du jeu test officiel exporté.

**Résultat attendu à l’exécution.** `bootstrap.csv` contient un F1 et un intervalle par modèle/graine, avec `resamples=1000`.

**À retenir.** Cet intervalle est conditionnel au modèle entraîné. Il diffère de l’écart-type entre graines et suppose des tweets suffisamment indépendants ; doublons ou groupes de provenance peuvent rendre l’incertitude trop optimiste.

**Lecture complémentaire, dans l’ordre des opérations.**

À chaque tirage bootstrap, les mêmes indices sélectionnent labels et décisions, pour conserver l'appariement. On recalcule TP/FP/FN et F1 sur cet échantillon ; les percentiles 2,5 et 97,5 forment l'intervalle affiché. Aucun gradient ni nouveau checkpoint n'est calculé pendant ces 1 000 tirages.

**Code de référence.**

```python
from scipy.stats import binomtest

rng=np.random.default_rng(42)

ytest=test.y.to_numpy()

bootstrap_rows=[]

for family,keys in FINAL_GROUPS.items():
    for key in keys:
        pred=TEST_P[key]>=REGISTRY[key]['threshold']; vals=[]
        for b in range(1000):
            idx=rng.integers(0,len(test),len(test)); yy=ytest[idx]; pp=pred[idx]
            tp=np.sum((yy==1)&pp); fp=np.sum((yy==0)&pp); fn=np.sum((yy==1)&~pp)
            vals.append(2*tp/max(2*tp+fp+fn,1))
        lo,hi=np.percentile(vals,[2.5,97.5])
        bootstrap_rows.append({'family':family,'model':key,'f1':scores(ytest,TEST_P[key],REGISTRY[key]['threshold'])['f1_sarcastique'],'ci_low':lo,'ci_high':hi,'resamples':1000})

bootstrap=pd.DataFrame(bootstrap_rows)

bootstrap.to_csv(ROOT/'results/bootstrap.csv',index=False)

display(bootstrap)
```

<a id="tp2-094"></a>

#### Cellule de code 094 — Comparer deux modèles avec McNemar exact

**Objectif.** Tester si leurs désaccords de correction sont équilibrés sur les mêmes tweets.

**Comprendre le code.** Les deux familles sont retenues selon leur moyenne F1 val, puis comparées à graine 42. Les vecteurs `c1` et `c2` indiquent si chaque décision est correcte. `b` compte les tweets réussis seulement par le premier, `c` ceux réussis seulement par le second. Sous l’hypothèse d’égalité, la répartition des désaccords est testée avec une binomiale de paramètre 0,5. Sans désaccord, la p-valeur est fixée à 1.

**Résultat attendu à l’exécution.** Le dictionnaire `mcnemar` contient les deux modèles, les effectifs discordants et la p-valeur exacte bilatérale.

**À retenir.** McNemar compare des taux d’erreur appariés, pas directement le F1 sarcastique. Une petite p-valeur n’indique ni la taille du gain ni son utilité pratique ; lire aussi les métriques et intervalles.

**Lecture complémentaire, dans l’ordre des opérations.**

Les deux familles comparées sont les mieux classées sur validation, avec leur graine représentative 42. `b` compte modèle 1 correct et modèle 2 faux ; `c` l'inverse. Le test binomial exact sous p=0,5 évalue l'asymétrie de ces désaccords ; une p-value élevée n'établit pas l'équivalence des modèles.

**Code de référence.**

```python
family_rank=sorted(FINAL_GROUPS,key=lambda f:np.mean([row_for_key(k)['val_f1_sarcastique'] for k in FINAL_GROUPS[f]]),reverse=True)

k1,k2=[FINAL_GROUPS[f][0] for f in family_rank[:2]]

c1=(TEST_P[k1]>=REGISTRY[k1]['threshold'])==ytest

c2=(TEST_P[k2]>=REGISTRY[k2]['threshold'])==ytest

b=int(np.sum(c1&~c2))

c=int(np.sum(~c1&c2))

pvalue=binomtest(b,b+c,.5).pvalue if b+c else 1.

mcnemar={'model_1':k1,'model_2':k2,'b_correct_1_only':b,'c_correct_2_only':c,'pvalue_exact':pvalue}
```

<a id="tp2-095"></a>

#### Cellule de code 095 — Exporter le test statistique

**Objectif.** Rendre vérifiables les effectifs et le calcul de significativité.

**Comprendre le code.** Le dictionnaire construit auparavant est sauvegardé dans `mcnemar.json` et présenté sous forme de tableau vertical. Les noms des modèles et les comptes `b`/`c` accompagnent la p-valeur afin d’éviter un résultat statistique isolé de sa comparaison réelle.

**Résultat attendu à l’exécution.** Un fichier JSON et une table de McNemar sont disponibles dans les livrables.

**À retenir.** Ne pas résumer la conclusion à significatif/non significatif. La sélection sur val, les hypothèses d’indépendance et la portée d’un test sur les erreurs doivent rester expliquées.

**Lecture complémentaire, dans l’ordre des opérations.**

Le JSON conserve les deux clés, b/c et la p-value pour retrouver le calcul. La table n'est pas un score moyen des trois graines : c'est une comparaison appariée des deux checkpoints indiqués. Il faut donner leurs noms quand on commente la significativité.

**Code de référence.**

```python
(ROOT/'results/mcnemar.json').write_text(json.dumps(mcnemar,indent=2))

display(pd.Series(mcnemar).to_frame('McNemar'))
```

<a id="tp2-096"></a>

#### Cellule de code 096 — Prélever les erreurs du modèle retenu

**Objectif.** Préparer une analyse qualitative de faux négatifs et de fausses alertes.

**Comprendre le code.** La première famille du classement val fournit son run graine 42 et son seuil déjà fixé. Le code prend jusqu’à 15 FN et 15 FP dans l’ordre du test. Chaque ligne conserve texte, annotation, score, dialecte, sentiment et présence dans un groupe de clés contradictoires. Une catégorie provisoire est proposée à partir de marqueurs observables ; un champ rappelle la relecture humaine nécessaire.

**Résultat attendu à l’exécution.** Une table `errors` est construite. On attend 30 lignes si au moins 15 erreurs de chaque type existent ; sinon, le nombre réel doit être rapporté.

**À retenir.** Le prélèvement des premières erreurs n’est pas aléatoire ni représentatif de tous les échecs. Un marqueur, un dialecte ou une contradiction de clé ne suffit pas à certifier la cause d’une erreur ou un bruit d’annotation.

**Lecture complémentaire, dans l’ordre des opérations.**

La famille retenue est choisie sur validation, puis on prélève les 15 premières positions FN et les 15 premières FP. Ce prélèvement n'est pas un échantillon aléatoire représentatif de toutes les erreurs. `category_provisional` est une aide automatique ; la relecture humaine reste nécessaire pour séparer contexte, dialecte et bruit possible.

**Code de référence.**

```python
chosen_final_family=family_rank[0]

CHOSEN_KEY=FINAL_GROUPS[chosen_final_family][0]

chosen_rec=REGISTRY[CHOSEN_KEY]

chosen_p=TEST_P[CHOSEN_KEY]

chosen_pred=chosen_p>=chosen_rec['threshold']

errors=[]

for kind,mask in [('FN',(ytest==1)&~chosen_pred),('FP',(ytest==0)&chosen_pred)]:
    ids=np.flatnonzero(mask)[:15]
    for i in ids:
        r=test.iloc[i]; cues=bool(re.search(r'ه{2,}|[!?؟]{2,}',r.tweet) or emoji.get_emoji_regexp().search(r.tweet))
        contradictory=r.key in all_contradictions
        category='labels contradictoires observés' if contradictory else ('marqueur mal interprété possible' if cues else ('dialecte et contexte à examiner' if r.dialect!='msa' else 'ironie sans marqueur ou contexte requis'))
        errors.append({'id':int(r.id),'type':kind,'tweet':r.tweet,'y_true':int(r.y),'probability':float(chosen_p[i]),
            'dialect':r.dialect,'sentiment':r.sentiment,'category_provisional':category,'contradictory_key':contradictory,
            'human_review':'À vérifier sur le texte et son contexte'})

errors=pd.DataFrame(errors)
```

<a id="tp2-097"></a>

#### Cellule de code 097 — Afficher les erreurs et leurs catégories provisoires

**Objectif.** Fournir les éléments nécessaires à une relecture humaine traçable.

**Comprendre le code.** Le CSV `errors_30.csv` est écrit avec les textes arabes, puis affiché. Un regroupement par type d’erreur et catégorie fournit un premier comptage. Le message final donne l’effectif réel et le nombre de clés contradictoires dans cette sélection, au lieu de supposer automatiquement que 30 erreurs ont été obtenues.

**Résultat attendu à l’exécution.** Les erreurs sont lisibles dans le notebook et exportées pour annoter `human_review` avec des justifications.

**À retenir.** Les comptes automatiques décrivent des hypothèses, pas des catégories validées. Examiner contexte requis, ironie sans marqueur, marqueur mal interprété, dialecte et bruit possible ; ne jamais déduire un pourcentage certain de bruit de ces seules règles.

**Lecture complémentaire, dans l’ordre des opérations.**

Les colonnes associent texte, vrai label, score, dialecte et catégorie provisoire. Une clé contradictoire repérée peut signaler un problème, mais son absence ne prouve pas la qualité certaine du label. Les fréquences affichées résument la sélection de trente erreurs, pas une estimation validée du taux de bruit du corpus.

**Code de référence.**

```python
errors.to_csv(ROOT/'results/errors_30.csv',index=False,encoding='utf-8-sig')

display(errors)

display(errors.groupby(['type','category_provisional']).size().rename('n').to_frame())

print('Nombre réel d’erreurs affichées :',len(errors),'— contradiction certaine dans cette sélection :',int(errors.contradictory_key.sum()))
```

**Petit extrait de sortie déjà enregistrée dans ce fichier (non recalculée pour le guide).**

```text
Nombre réel d’erreurs affichées : 30 — contradiction certaine dans cette sélection : 0
```

### Partie G : comparaison et décision

<a id="tp2-098"></a>

#### Cellule de code 098 — Imposer le CPU pour la comparaison de latence

**Objectif.** Mesurer le coût d’inférence sur le matériel demandé, même si l’apprentissage avait utilisé CUDA ailleurs.

**Comprendre le code.** `DEVICE` est explicitement replacé sur CPU. Les fonctions de chargement et de prédiction qui consultent cette variable utiliseront ce périphérique pour G. La liste `comparison` accueillera ensuite les lignes agrégées des familles.

**Résultat attendu à l’exécution.** La configuration de benchmark CPU est active et la table de comparaison est initialisée.

**À retenir.** Cette affectation n’accélère pas les entraînements précédents et ne modifie pas leurs résultats. Elle évite simplement d’étiqueter par erreur une mesure GPU comme une latence CPU.

**Après CUDA.** Ce retour explicite au CPU est intentionnel : G compare les latences CPU, et non les latences T4. Les poids entraînés sont chargés avec `map_location=DEVICE`.

**Lecture complémentaire, dans l’ordre des opérations.**

Après l'entraînement GPU, `DEVICE` est remplacé par CPU pour mesurer la latence demandée en G. Cela ne convertit pas l'historique d'entraînement en entraînement CPU. Les étapes suivantes rechargent les modèles sur ce périphérique pour la mesure.

**Code de référence.**

```python
DEVICE=torch.device('cpu')  # G mesure toujours la latence CPU, même après entraînement CUDA.
comparison=[]
```

<a id="tp2-099"></a>

#### Cellule de code 099 — Mesurer qualité, coût et latence des familles entraînées

**Objectif.** Comparer les cinq familles finales en tenant compte du système complet utilisé pour prédire.

**Comprendre le code.** Les F1 et AP test sont regroupés sur trois graines ; l’écart-type utilise `ddof=1`. Pour la latence, la graine 42 traite 20 tweets un par un, après échauffement et sur trois répétitions ; la médiane des durées moyennes par tweet est retenue. Les poids sont déjà chargés, mais préparation/tokenisation font partie du traitement. Pour D, l’encodeur AraBERT est exécuté et sa taille est ajoutée à celle de la tête.

**Résultat attendu à l’exécution.** Une ligne par famille rassemble moyenne/dispersion, durée train, millisecondes par tweet, taille en Mo décimaux et paramètres de la partie entraînable.

**À retenir.** La taille sur disque n’est pas la RAM maximale. La latence dépend des textes, de leur longueur et de la charge du CPU ; 20 tweets ne constituent pas un benchmark universel. La moyenne des durées train par run n’est pas le coût total de toutes les recherches de configurations.

**Lecture complémentaire, dans l’ordre des opérations.**

Pour chaque famille, les métriques sont agrégées sur trois graines, mais la latence est mesurée avec un checkpoint représentatif. Un passage de chauffe est écarté, puis une médiane de passages sur vingt tweets est retenue. Pour D, la taille inclut l'encodeur, pas seulement la tête ; l'extraction train gelée reste un coût distinct.

**Code de référence.**

```python
for family,keys in FINAL_GROUPS.items():
    rows=[row_for_key(k) for k in keys]
    f=[r['test_f1_sarcastique'] for r in rows]
    key=keys[0]; rec=REGISTRY[key]
    latency=float('nan')
    if rec['kind'] in ['tfidf','static','plm']:
        model=load_registered(rec); sample=test.tweet.head(20).tolist()
        predict_registered(rec,sample[:1],model=model)
        durations=[]
        for repeat in range(3):
            start=time.perf_counter()
            for text in sample: predict_registered(rec,[text],model=model)
            durations.append((time.perf_counter()-start)*1000/len(sample))
        latency=float(np.median(durations)); del model; gc.collect()
    elif rec['kind'] in ['frozen_lr','frozen_head']:
        encoder=load_hf(ARA_NAME).eval(); head=load_registered(rec); tokenizer=tokenizer_ara
        sample=test.tweet.head(20).tolist(); durations=[]
        with torch.inference_mode():
            for repeat in range(4):
                start=time.perf_counter()
                for text in sample:
                    inp=plm_encode([text],ARA_NAME); ids,mask=[x.to(DEVICE) for x in inp]
                    length=int(mask.sum()); ids=ids[:,:length]; mask=mask[:,:length]
                    h=encoder(input_ids=ids,attention_mask=mask).last_hidden_state
                    if rec['kind']=='frozen_lr': head.predict_proba(pooled_features(h.cpu().numpy(),mask.cpu().numpy(),rec['config']['pooling']))
                    else: head(h,mask)
                if repeat: durations.append((time.perf_counter()-start)*1000/len(sample))
        latency=float(np.median(durations)); del encoder,head; gc.collect()
    size=rows[0]['artifact_size_mb']
    if rec['kind'] in ['frozen_lr','frozen_head']:
        encoder_files=list(Path(REVISIONS[ARA_NAME]['path']).glob('*.safetensors')) or list(Path(REVISIONS[ARA_NAME]['path']).glob('pytorch_model.bin'))
        size+=sum(f.stat().st_size for f in encoder_files)/1e6
    comparison.append({'family':family,'model':rec['name'],'n_seeds':len(keys),
        'val_f1_mean':np.mean([r['val_f1_sarcastique'] for r in rows]),'test_f1_mean':np.mean(f),'test_f1_std':np.std(f,ddof=1),
        'test_ap_mean':np.mean([r['test_pr_auc_ap'] for r in rows]),'train_seconds_mean':np.mean([r['train_seconds'] for r in rows]),
        'cpu_batch1_latency_ms':latency,'complete_weights_mb':size,'threshold_seed42':rec['threshold'],
        'trainable_parameters':rows[0]['trainable_parameters']})
```

<a id="tp2-100"></a>

#### Cellule de code 100 — Mesurer le coût des deux méthodes directes

**Objectif.** Ne pas confondre absence d’entraînement sur ArSarcasm et inférence gratuite.

**Comprendre le code.** NLI et sentiment sont chargés sur CPU puis testés sur les mêmes 20 tweets, avec un passage d’échauffement et trois passages mesurés. Pour NLI, un tweet implique deux paires avec les hypothèses sarcastique/sérieuse ; sentiment calcule une classification. Les lignes indiquent une seule exécution et aucune durée d’apprentissage sur le corpus du TP.

**Résultat attendu à l’exécution.** Les lignes B1 et B2 sont ajoutées à la comparaison avec leur latence et taille d’artefact.

**À retenir.** Leur écart-type entre graines est laissé non défini : une seule exécution n’est pas trois répétitions. Pour un artefact stocké en dossier, la taille peut inclure les fichiers annexes ; garder cette convention en lisant les Mo annoncés.

**Lecture complémentaire, dans l’ordre des opérations.**

B1/B2 ne coûtent aucun ajustement sur ce corpus, mais doivent encore exécuter de grands encodeurs à chaque inférence. B1 évalue deux hypothèses par tweet, ce qui augmente le calcul. La comparaison doit garder la même machine CPU que les familles entraînées.

**Code de référence.**

```python
for family,key in [('B1',best_nli_key),('B2',b2_key)]:
    rec=REGISTRY[key]; row=row_for_key(key); name=NLI_NAME if family=='B1' else SA_NAME
    model=load_hf(name,True).eval(); tok=AutoTokenizer.from_pretrained(REVISIONS[name]['path'],local_files_only=True)
    sample=test.tweet.head(20).tolist(); durations=[]
    with torch.inference_mode():
        for repeat in range(4):
            start=time.perf_counter()
            for text in sample:
                if family=='B1':
                    ids=tok(export_clean(text),truncation=True,max_length=MAX_LEN,add_special_tokens=False)['input_ids']
                    premise=tok.decode(ids,skip_special_tokens=True)
                    enc=tok([premise,premise],list(PROMPTS[rec['config']['language']]),padding=True,truncation='only_first',max_length=MAX_LEN+24,return_tensors='pt').to(DEVICE)
                    logits=model(**enc).logits
                    entail=next(i for i,label in model.config.id2label.items() if 'entail' in label.lower())
                    logits[:,entail].softmax(0)
                else:
                    enc=tok(export_clean(text),padding=True,truncation=True,max_length=MAX_LEN,return_tensors='pt').to(DEVICE)
                    model(**enc).logits.argmax(-1)
            if repeat: durations.append((time.perf_counter()-start)*1000/len(sample))
    comparison.append({'family':family,'model':rec['name'],'n_seeds':1,'val_f1_mean':row['val_f1_sarcastique'],
        'test_f1_mean':row['test_f1_sarcastique'],'test_f1_std':float('nan'),'test_ap_mean':row['test_pr_auc_ap'],
        'train_seconds_mean':0.,'cpu_batch1_latency_ms':float(np.median(durations)),
        'complete_weights_mb':row['artifact_size_mb'],'threshold_seed42':rec['threshold'],'trainable_parameters':0})
    del model; gc.collect()
```

<a id="tp2-101"></a>

#### Cellule de code 101 — Exporter la comparaison et visualiser l’apprentissage

**Objectif.** Associer les résultats finaux aux trajectoires de validation des modèles retenus.

**Comprendre le code.** La liste des lignes est convertie en table, exportée dans `final_comparison.csv` et affichée. L’historique rassemble ensuite les F1 val par époque pour C/D/E1/E2 et leurs graines. Une tête linéaire sans époques est identifiée explicitement. Le graphique des courbes est sauvegardé. La matrice de confusion du modèle déjà choisi est calculée sur ses prédictions test.

**Résultat attendu à l’exécution.** Un tableau de synthèse, quatre panneaux d’apprentissage et une matrice de confusion préparée pour affichage sont produits.

**À retenir.** Une courbe train/val n’est pas une preuve de robustesse externe. Le meilleur checkpoint peut être antérieur à la dernière époque ; lire les courbes avec les règles d’arrêt plutôt que supposer que plus d’époques améliorent toujours la qualité.

**Mise à jour qualité / affichage.** Les courbes d'apprentissage sont entièrement affichées ici. La matrice de confusion est créée dans la cellule suivante pour recevoir son titre avant son affichage.

**Lecture complémentaire, dans l’ordre des opérations.**

La comparaison est exportée puis les trajectoires F1 val sont tracées par famille et graine. Une courbe par époque montre le choix du checkpoint et le surapprentissage possible ; elle ne correspond pas à une trajectoire F1 test. Une tête linéaire sans historique d'époques n'a pas à recevoir une courbe artificielle.

**Code de référence.**

```python
comparison=pd.DataFrame(comparison)

comparison.to_csv(ROOT/'results/final_comparison.csv',index=False)

display(comparison.round(4))

history_frame=pd.DataFrame(HISTORY)

fig,axes=plt.subplots(2,2,figsize=(10,7))

for ax,family in zip(axes.flat,['C','D','E1','E2']):
    model_name=REGISTRY[FINAL_GROUPS[family][0]]['name']
    curves=history_frame.loc[history_frame.model.eq(model_name)]
    for seed,curve in curves.groupby('seed'):
        ax.plot(curve.epoch,curve.val_f1,marker='o',label=f'Graine {seed}')
    if len(curves): ax.legend(fontsize=8)
    else:
        message='Classifieur linéaire sans époques' if REGISTRY[FINAL_GROUPS[family][0]]['kind'] in ['tfidf','frozen_lr'] else 'Historique indisponible : récupérer learning_curves.csv'
        ax.text(.5,.5,message,ha='center',wrap=True,transform=ax.transAxes)
    ax.set(title=f'{family} — {model_name}',xlabel='Époque',ylabel='F1 validation')

savefig(fig,'learning_curves_final')
```

<a id="tp2-102"></a>

#### Cellule de code 102 — Présenter les erreurs globales et préparer la décision

**Objectif.** Rendre visibles les compromis des modèles au seuil déjà retenu.

**Comprendre le code.** La matrice reçoit son titre avec famille et seuil val, puis est sauvegardée. Le diagramme en barres montre la moyenne F1 test ; les barres d’erreur des familles entraînées représentent l’écart-type sur trois graines. Pour B1/B2, la valeur manquante est tracée sans barre, pas interprétée comme une variance connue nulle. Le dictionnaire `decision` réunit le choix fait sur val et ses métriques test.

**Résultat attendu à l’exécution.** Deux figures sont enregistrées et une décision traçable est préparée avec un horodatage de fin.

**À retenir.** La figure test expose le choix de validation ; elle ne doit pas conduire à déplacer après coup ce choix vers la barre la plus haute. Une recommandation réelle doit aussi considérer latence, taille et résultats F, notamment pour des tweets marocains.

**Mise à jour qualité / affichage.** La matrice de confusion et le graphique comparatif sont chacun créés, remplis et explicitement affichés dans cette cellule. Aucun axe vide n'est créé dans une cellule antérieure.

**Lecture complémentaire, dans l’ordre des opérations.**

La matrice de confusion vient du modèle sélectionné et de son seuil val verrouillé. Le barplot utilise les moyennes test et l'écart-type des graines, pas des intervalles bootstrap. `decision` conserve la clé exacte et les métriques du checkpoint représentatif ; ces dernières peuvent différer de la moyenne de famille.

**Code de référence.**

```python
fig,ax=plt.subplots(figsize=(6,5))

ConfusionMatrixDisplay.from_predictions(ytest,chosen_pred,display_labels=['Non sarcastique','Sarcastique'],cmap='Blues',ax=ax,colorbar=False)

ax.set_title(f'Modèle retenu sur val : {chosen_final_family} — seuil {chosen_rec["threshold"]:.3f}')

savefig(fig,'chosen_confusion')

fig,ax=plt.subplots(figsize=(8,4))

ax.bar(comparison.family,comparison.test_f1_mean,yerr=comparison.test_f1_std.fillna(0),capsize=5,color='#275D88')

ax.set(ylabel='F1 sarcastique test',title='Entraînés : 3 graines ; zero-shot : 1 exécution',ylim=(0,1))

savefig(fig,'final_comparison')

decision={'chosen_key':CHOSEN_KEY,'family':chosen_final_family,'criterion':'moyenne F1 validation sur trois graines',
          'threshold':chosen_rec['threshold'],'test_metrics':scores(ytest,chosen_p,chosen_rec['threshold']),
          'completed_at_utc':datetime.now(timezone.utc).isoformat()}
```

<a id="tp2-103"></a>

#### Cellule de code 103 — Sauvegarder la décision et le journal final

**Objectif.** Conserver une synthèse lisible sans dépendre de l’état du noyau Python.

**Comprendre le code.** `decision.json` contient le modèle, le seuil, le critère de choix et les mesures test. `results.csv` est réécrit avec les résultats et diagnostics réellement ajoutés pendant l’exécution. Les messages indiquent la référence retenue et où trouver les prédictions.

**Résultat attendu à l’exécution.** La décision et le journal final sont présents sous `ROOT/results/` ; le notebook affiche les métriques du modèle choisi.

**À retenir.** Ces messages n’ont de valeur que si les cellules précédentes se sont terminées sans erreur. Ils ne remplacent ni le contrôle des 17 fichiers de prédictions ni la relecture qualitative des erreurs et du compte rendu.

**Lecture complémentaire, dans l’ordre des opérations.**

`decision.json` décrit un checkpoint choisi, alors que `results.csv` contient l'ensemble des runs. Le message d'export témoigne d'une écriture dans la session d'exécution ; il ne vérifie pas que les CSV ont été téléchargés sur Windows. Une nouvelle sauvegarde de notebook ne remplace pas ces fichiers pour un recalcul indépendant.

**Code de référence.**

```python
(ROOT/'results/decision.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')

pd.DataFrame(RESULTS).to_csv(ROOT/'results/results.csv',index=False)

print('Modèle retenu :',CHOSEN_KEY,'seuil',chosen_rec['threshold'])

print('Scores test :',decision['test_metrics'])

print('Toutes les prédictions finales exportées dans predictions/.')
```

**Petit extrait de sortie déjà enregistrée dans ce fichier (non recalculée pour le guide).**

```text
Modèle retenu : E2_AraBERT_lambda0p5_seed42 seuil 0.5790433287620544
Scores test : {'f1_sarcastique': 0.5775, 'precision': 0.5930680359435173, 'recall': 0.5627283800243605, 'macro_f1': 0.7119318181818182, 'pr_auc_ap': 0.6175355658283671}
Toutes les prédictions finales exportées dans predictions/.
```

<a id="tp2-104"></a>

#### Cellule de code 104 — Consigner les versions et vérifier les livrables calculés

**Objectif.** Terminer par des assertions qui bloquent une déclaration de fin incohérente.

**Comprendre le code.** Les versions des bibliothèques principales sont enregistrées sans réécrire les dépendances figées. Le dictionnaire `checks` vérifie les clés disjointes entre partitions, les 3 000 lignes test, trois clés par famille finale et la présence des fichiers de prédictions. Si un contrôle est faux, l’assertion interrompt la cellule. Un JSON conserve les contrôles et le chemin Python.

**Résultat attendu à l’exécution.** Un tableau de booléens et `validation_checks.json` documentent la cohérence de l’exécution scientifique.

**À retenir.** La présence d’un fichier ne garantit pas son contenu : le contrôle final externe recalcule aussi les métriques et vérifie les compteurs du notebook. Après sauvegarde, utiliser les commandes de synchronisation et de rapport indiquées dans la dernière section ; ne pas annoncer une exécution complète avant ces vérifications.

**Lecture complémentaire, dans l’ordre des opérations.**

Les versions effectives sont enregistrées, puis les assertions vérifient partitions disjointes, taille test, trois graines et présence des exports. Une assertion vraie dans Colab concerne ce système de fichiers au moment de l'exécution. Elle ne garantit pas la disponibilité des mêmes ressources dans un clone Git ou sur le PC local.

**Code de référence.**

```python
import importlib.metadata as metadata

packages=['torch','transformers','huggingface-hub','safetensors','tokenizers','scikit-learn','numpy','pandas','scipy','gensim','arabert','pyarabic','emoji','sentencepiece','nbformat','ipykernel','matplotlib','joblib','reportlab','pymupdf']

versions={p:metadata.version(p) for p in packages}

(ROOT/'results/runtime_versions.json').write_text(json.dumps(versions,indent=2))

checks={'train_val_disjoint':set(train.key).isdisjoint(val.key),'train_test_disjoint':set(train.key).isdisjoint(test.key),
    'val_test_disjoint':set(val.key).isdisjoint(test.key),
    'test_official_count':len(test)==3000,'three_seeds_per_final_family':all(len(keys)==3 for keys in FINAL_GROUPS.values()),
    'final_prediction_files':all((ROOT/f'predictions/{key}.csv').exists() for key in FINAL_KEYS)}

assert all(checks.values()),checks

(ROOT/'results/validation_checks.json').write_text(json.dumps({'checks':checks,'versions':versions,'python':sys.executable},indent=2))

display(pd.Series(checks).to_frame('Vérifié'))

print('Calcul scientifique terminé. Régénérer le PDF : python -m src.report ; puis python -m tests.verify_final')
```

**Petit extrait de sortie déjà enregistrée dans ce fichier (non recalculée pour le guide).**

```text
Calcul scientifique terminé. Régénérer le PDF : python -m src.report ; puis python -m tests.verify_final
```

<a id="cellules-testeur"></a>

## 7. Guide des 6 cellules du testeur de phrases

Ce notebook dépend des ressources issues de l’apprentissage, mais peut être utilisé dans un noyau séparé une fois ces ressources disponibles. Il importe `PhraseTester` au lieu de relancer les 104 cellules du TP.

<a id="testeur-01"></a>

### Testeur — cellule 1 : 1 — Préparer les dépendances

Le bloc Colab est identique à celui du notebook principal, avec les versions compatibles Python 3.13. Si une installation est nécessaire, son arrêt « Installation terminée » demande de redémarrer le noyau puis de recommencer. En Conda local, ce bloc n’installe rien. Aucun poids n’est téléchargé par ce notebook.

**Explication détaillée.**

Détecte Colab, prépare les dépendances et fournit le petit paquet d'inférence intégré. L'archive inclut du code, pas de poids entraînés. Après une installation réussie, le noyau doit être redémarré avant de continuer ; dans Conda local, cette préparation automatique ne s'applique pas.

**Code de référence.**

```python
from pathlib import Path
import os, sys, hashlib, base64, io, zipfile
IN_COLAB=bool(os.environ.get('COLAB_RELEASE_TAG')) or Path('/content').is_dir()
if IN_COLAB:
    print('TP2_COLAB_BOOTSTRAP_V2 — préparation compatible Python 3.13',flush=True)
    from importlib.metadata import version, PackageNotFoundError
    import subprocess
    # Copie autonome des dépendances, synchronisée avec requirements-colab.txt.
    COLAB_REQUIREMENTS = ['transformers==4.48.0', 'huggingface-hub==0.29.1', 'safetensors==0.5.3', 'tokenizers==0.21.0', 'gensim==4.4.0', 'arabert==1.0.1', 'pyarabic==0.6.15', 'emoji==1.4.2', 'sentencepiece==0.2.1', 'nbformat==5.9.2', 'reportlab==5.0.1', 'pymupdf==1.28.2', 'requests==2.32.3', 'pyflakes==3.0.1']
    missing=[]
    for specification in COLAB_REQUIREMENTS:
        package,wanted=specification.split('==',1)
        try: installed=version(package)
        except PackageNotFoundError: installed=None
        if installed!=wanted: missing.append(specification)
    if missing:
        print('Préparation Colab — Python',sys.version.split()[0],':',', '.join(missing),flush=True)
        # Ces extensions doivent avoir une wheel : éviter une compilation C++/Rust.
        command=[sys.executable,'-m','pip','install','--disable-pip-version-check',
                 '--progress-bar','off',
                 '--only-binary=safetensors,tokenizers,gensim,sentencepiece,pymupdf',*missing]
        installation=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                    text=True,errors='replace')
        print(installation.stdout,flush=True)
        if installation.returncode:
            raise RuntimeError(f"Échec de pip (code {installation.returncode}). "
                               "Copier les lignes ERROR affichées ci-dessus pour le diagnostic. "
                               "Ne pas poursuivre les cellules scientifiques.")
        raise RuntimeError('Installation terminée : redémarre le noyau Colab puis relance depuis la première cellule.')
os.environ['USE_TF']='0'
os.environ['USE_FLAX']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'
```

<a id="testeur-02"></a>

### Testeur — cellule 2 : 2 — Retrouver les checkpoints et préparer le lecteur

Le dossier est déterminé à partir de `TP2_ROOT`, de la racine courante ou du dossier Colab de l’entraînement. Le petit paquet intégré fournit uniquement le code manquant ; il ne remplace pas les checkpoints. Il est vérifié par SHA256, extrait sous le projet et n’écrase pas les sources présentes. `PhraseTester` lit le registre et charge les poids seulement à la première prédiction. Les modules sont lisibles dans `src/predict.py`, `src/models.py` et `src/preprocess.py`.

**Explication détaillée.**

Retrouve `ROOT` et exige `results/registry.json`, puis importe le lecteur et crée `tester`. Le registre, `results.csv` et les checkpoints doivent provenir de la même expérience. Un registre local CPU associé à des poids Colab différents n'est pas une configuration cohérente, même si leurs noms se ressemblent.

**Code de référence.**

```python
ROOT=Path(os.environ.get('TP2_ROOT',Path.cwd())).resolve()
if ROOT.name=='notebooks': ROOT=ROOT.parent
if IN_COLAB and not (ROOT/'results/registry.json').is_file() and 'TP2_ROOT' not in os.environ:
    candidates=[p for p in ROOT.iterdir() if p.is_dir() and (p/'results/registry.json').is_file()]
    ROOT=candidates[0] if len(candidates)==1 else Path('/content/TP2_ArSarcasme')
if not (ROOT/'results/registry.json').is_file():
    raise FileNotFoundError(f"Aucun registre dans {ROOT}. Choisir le même dossier distant que l'entraînement "
                            "ou récupérer results/registry.json, results/results.csv et les poids dans models/.")
# SOURCE_ARCHIVE : archive intégrée abrégée pour la lecture (8,863 caractères sur cette ligne).
SOURCE_ARCHIVE = "<archive base64 : voir le notebook original>"
payload=base64.b64decode(SOURCE_ARCHIVE)
assert hashlib.sha256(payload).hexdigest()=='f8daa2833def488ab17bcd6ede3aa800c6d54016e7c3a49c87607702e0fee5d0'
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for entry in archive.infolist():
        target=(ROOT/entry.filename).resolve()
        if not target.is_relative_to(ROOT): raise ValueError('Chemin intégré invalide.')
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(archive.read(entry))
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.predict import PhraseTester
from IPython.display import display
tester=PhraseTester(ROOT)
print('Dossier :',ROOT,'— appareil :',tester.device)
```

<a id="testeur-03"></a>

### Testeur — cellule 3 : 3 — Choisir le modèle

Le tableau affiche la clé du modèle, sa famille, sa graine, son F1 de validation et la présence de son fichier. Le choix initial reprend `decision.json` s’il est disponible, sinon le meilleur score de validation disponible hors ablations F. Pour comparer une autre architecture, copier sa clé exacte dans `MODELE` entre guillemets. `tester.refresh()` relit les fichiers après de nouveaux entraînements.

**Explication détaillée.**

`list_models()` expose clé, type, graine, F1 validation et présence du fichier. `default_key()` privilégie la clé de `decision.json` si elle existe et est disponible ; sinon il prend le meilleur F1 val disponible hors variantes F. Changer `MODELE` ne déclenche pas d'entraînement et n'ajuste pas le seuil sur les phrases.

**Code de référence.**

```python
display(tester.list_models())
MODELE=tester.default_key()
# Exemple de choix manuel : MODELE = 'E1_AraBERT_mono_seed42'
print('Modèle choisi :',MODELE)
```

**Petit extrait de sortie déjà enregistrée dans ce fichier (non recalculée pour le guide).**

```text
Modèle choisi : E2_AraBERT_lambda0p5_seed42
```

<a id="testeur-04"></a>

### Testeur — cellule 4 : 4 — Insérer vos phrases

Écrire une phrase arabe par ligne entre les triples guillemets. Emojis, négations, dialecte et répétitions peuvent être conservés : le lecteur applique le même prétraitement que la famille entraînée. Comme pendant l’entraînement, la longueur est limitée à 64 tokens ; les textes plus longs sont tronqués. Les exemples sont des entrées de démonstration, sans labels de référence ni mesure de qualité.

**Explication détaillée.**

Une ligne non vide devient un exemple ; les triples guillemets permettent plusieurs phrases dans une seule chaîne. `splitlines()` coupe les lignes et `strip()` enlève uniquement les espaces extérieurs. Ne pas normaliser manuellement les phrases avec une autre convention avant de les donner au lecteur ; il applique celle du checkpoint.

**Code de référence.**

```python
PHRASES = """يا سلام، الإنترنت انقطع مرة أخرى، خدمة رائعة فعلاً 😂
الجو جميل اليوم وسأذهب للمشي.
شكراً على التأخير ساعتين، فعلاً تحترمون وقت الناس!
لم يعجبني هذا الفيلم."""
phrases=[s.strip() for s in PHRASES.splitlines() if s.strip()]
print(len(phrases),'phrases prêtes à tester.')
```

**Petit extrait de sortie déjà enregistrée dans ce fichier (non recalculée pour le guide).**

```text
4 phrases prêtes à tester.
```

<a id="testeur-05"></a>

### Testeur — cellule 5 : 5 — Afficher les prédictions

Le lecteur recharge le checkpoint exact et son seuil enregistré, puis utilise `eval()` et `inference_mode()`. Il affiche le score, la règle de décision et la prédiction. **B2 est particulier** : son score est P(sentiment négatif), mais sa décision dépend de l’argmax parmi les trois sentiments, comme dans le TP ; elle n’est pas obtenue en comparant cette probabilité à 0,5. Les autres scores sont comparés au seuil de validation sauvegardé. Aucun seuil n’est réglé sur les phrases saisies.

**Explication détaillée.**

Le premier appel charge les ressources du checkpoint, les appels suivants réutilisent le même modèle si la clé et le fichier n'ont pas changé. Pour A/C/D/E/B1, la décision compare le score au seuil val ; pour B2, elle suit l'argmax négatif. Le tableau donne le score et sa nature, pas une certification de la justesse de l'étiquette.

**Code de référence.**

```python
resultats=tester.predict(phrases,key=MODELE)
display(resultats)
```

<a id="testeur-06"></a>

### Testeur — cellule 6 : 6 — Saisie interactive facultative

Exécuter cette cellule pour taper une nouvelle phrase à la demande. La même instance du modèle est réutilisée. Une entrée vide est ignorée. Pour revenir à la liste, modifier la cellule 4 et relancer la cellule 5.

**Explication détaillée.**

`input()` attend une saisie interactive et la cellule reste occupée tant que l'utilisateur n'a pas répondu. Une entrée vide affiche un message et ne lance pas de prédiction. Pour tester plusieurs autres phrases, revenir à la liste de la cellule 4 puis relancer la cellule 5 ; le modèle compatible reste réutilisable.

**Code de référence.**

```python
phrase=input('Saisir une phrase arabe : ').strip()
if phrase:
    display(tester.predict(phrase,key=MODELE))
else:
    print('Aucune phrase saisie.')
```

### Ce qui se passe dans `PhraseTester.predict`

1. Convertir la saisie en liste et refuser toute phrase vide, type incorrect ou taille de lot non positive.
2. Retrouver la clé dans le registre ; vérifier que le chemin reste dans le projet et que le checkpoint existe.
3. Recharger uniquement si le modèle a changé, à l'aide d'une empreinte clé/taille/date ; libérer l'ancien modèle sinon.
4. Reconstruire la bonne architecture, charger les poids ajustés, le tokenizer et les éventuels vocabulaires/préprocesseurs.
5. Exécuter `eval()` et `inference_mode()` ; conserver les conventions numériques nécessaires, dont la quantification des états D.
6. Appliquer le seuil enregistré, sauf B2 qui utilise l'argmax sentiment ; construire le tableau final.

| Colonne affichée | Comment la lire |
|---|---|
| `phrase` | Texte original saisi, pour retrouver l'exemple |
| `modele` | Clé complète, comprenant architecture/configuration et graine |
| `score` | Score sarcasme, ou probabilité de sentiment négatif pour B2 |
| `nature_score` | Rappelle la nature du score pour éviter une mauvaise interprétation |
| `seuil_val` | Seuil choisi lors de la validation ; vide/NaN pour la décision argmax B2 |
| `regle` | Règle effectivement appliquée, pas une règle supposée |
| `prediction` | Classe obtenue, qui peut être erronée hors distribution |

Changer seulement la phrase ne fait pas changer le seuil. Pour choisir manuellement un modèle, recopier **une clé existante** du tableau, par exemple `MODELE='E1_AraBERT_mono_seed42'`. Le nom d'un dossier de poids pré-entraînés n'est pas une clé de modèle ajusté.


<a id="variantes"></a>

## 8. Anciennes versions : correspondance de toutes les cellules

Ces correspondances sont des **renvois de lecture**, pas une garantie que les versions exécutent exactement le même protocole. Une cellule identique partage l’explication du principal. Une variante affiche aussi son propre code et ses notes pour rendre les différences vérifiables. La sélection est fondée sur le code normalisé, jamais sur les valeurs de performance.

### Version `TP2_Sarcasme_Arabe_T4_Execution.ipynb`

L’archive conserve le plafond PLM de **trois époques**. Le notebook de travail utilise **cinq** au maximum ; ne pas mélanger leurs checkpoints, seuils et scores.

| Cellule de cette version | Renvoi au principal | État du code |
|---|---|---|
| 001 | [Cellule 001](#tp2-001) | Identique |
| 002 | [Cellule 002](#tp2-002) | [Variante expliquée](#archive-002) |
| 003 | [Cellule 003](#tp2-003) | Identique |
| 004 | [Cellule 004](#tp2-004) | [Variante expliquée](#archive-004) |
| 005 | [Cellule 005](#tp2-005) | Identique |
| 006 | [Cellule 006](#tp2-006) | Identique |
| 007 | [Cellule 007](#tp2-007) | Identique |
| 008 | [Cellule 008](#tp2-008) | Identique |
| 009 | [Cellule 009](#tp2-009) | Identique |
| 010 | [Cellule 010](#tp2-010) | Identique |
| 011 | [Cellule 011](#tp2-011) | Identique |
| 012 | [Cellule 012](#tp2-012) | Identique |
| 013 | [Cellule 013](#tp2-013) | Identique |
| 014 | [Cellule 014](#tp2-014) | Identique |
| 015 | [Cellule 015](#tp2-015) | Identique |
| 016 | [Cellule 016](#tp2-016) | Identique |
| 017 | [Cellule 017](#tp2-017) | Identique |
| 018 | [Cellule 018](#tp2-018) | [Variante expliquée](#archive-018) |
| 019 | [Cellule 019](#tp2-019) | [Variante expliquée](#archive-019) |
| 020 | [Cellule 020](#tp2-020) | Identique |
| 021 | [Cellule 021](#tp2-021) | Identique |
| 022 | [Cellule 022](#tp2-022) | Identique |
| 023 | [Cellule 023](#tp2-023) | Identique |
| 024 | [Cellule 024](#tp2-024) | Identique |
| 025 | [Cellule 025](#tp2-025) | [Variante expliquée](#archive-025) |
| 026 | [Cellule 026](#tp2-026) | Identique |
| 027 | [Cellule 027](#tp2-027) | Identique |
| 028 | [Cellule 028](#tp2-028) | Identique |
| 029 | [Cellule 029](#tp2-029) | Identique |
| 030 | [Cellule 030](#tp2-030) | Identique |
| 031 | [Cellule 031](#tp2-031) | Identique |
| 032 | [Cellule 032](#tp2-032) | Identique |
| 033 | [Cellule 033](#tp2-033) | Identique |
| 034 | [Cellule 034](#tp2-034) | Identique |
| 035 | [Cellule 035](#tp2-035) | Identique |
| 036 | [Cellule 036](#tp2-036) | Identique |
| 037 | [Cellule 037](#tp2-037) | Identique |
| 038 | [Cellule 038](#tp2-038) | Identique |
| 039 | [Cellule 039](#tp2-039) | Identique |
| 040 | [Cellule 040](#tp2-040) | Identique |
| 041 | [Cellule 041](#tp2-041) | Identique |
| 042 | [Cellule 042](#tp2-042) | Identique |
| 043 | [Cellule 043](#tp2-043) | Identique |
| 044 | [Cellule 044](#tp2-044) | Identique |
| 045 | [Cellule 045](#tp2-045) | Identique |
| 046 | [Cellule 046](#tp2-046) | Identique |
| 047 | [Cellule 047](#tp2-047) | Identique |
| 048 | [Cellule 048](#tp2-048) | Identique |
| 049 | [Cellule 049](#tp2-049) | Identique |
| 050 | [Cellule 050](#tp2-050) | [Variante expliquée](#archive-050) |
| 051 | [Cellule 051](#tp2-051) | Identique |
| 052 | [Cellule 052](#tp2-052) | Identique |
| 053 | [Cellule 053](#tp2-053) | Identique |
| 054 | [Cellule 054](#tp2-054) | Identique |
| 055 | [Cellule 055](#tp2-055) | Identique |
| 056 | [Cellule 056](#tp2-056) | Identique |
| 057 | [Cellule 057](#tp2-057) | Identique |
| 058 | [Cellule 058](#tp2-058) | Identique |
| 059 | [Cellule 059](#tp2-059) | Identique |
| 060 | [Cellule 060](#tp2-060) | Identique |
| 061 | [Cellule 061](#tp2-061) | Identique |
| 062 | [Cellule 062](#tp2-062) | Identique |
| 063 | [Cellule 063](#tp2-063) | Identique |
| 064 | [Cellule 064](#tp2-064) | Identique |
| 065 | [Cellule 065](#tp2-065) | Identique |
| 066 | [Cellule 066](#tp2-066) | Identique |
| 067 | [Cellule 067](#tp2-067) | Identique |
| 068 | [Cellule 068](#tp2-068) | Identique |
| 069 | [Cellule 069](#tp2-069) | Identique |
| 070 | [Cellule 070](#tp2-070) | Identique |
| 071 | [Cellule 071](#tp2-071) | Identique |
| 072 | [Cellule 072](#tp2-072) | Identique |
| 073 | [Cellule 073](#tp2-073) | Identique |
| 074 | [Cellule 074](#tp2-074) | Identique |
| 075 | [Cellule 075](#tp2-075) | Identique |
| 076 | [Cellule 076](#tp2-076) | Identique |
| 077 | [Cellule 077](#tp2-077) | Identique |
| 078 | [Cellule 078](#tp2-078) | Identique |
| 079 | [Cellule 079](#tp2-079) | Identique |
| 080 | [Cellule 080](#tp2-080) | Identique |
| 081 | [Cellule 081](#tp2-081) | Identique |
| 082 | [Cellule 082](#tp2-082) | Identique |
| 083 | [Cellule 083](#tp2-083) | Identique |
| 084 | [Cellule 084](#tp2-084) | Identique |
| 085 | [Cellule 085](#tp2-085) | [Variante expliquée](#archive-085) |
| 086 | [Cellule 086](#tp2-086) | [Variante expliquée](#archive-086) |
| 087 | [Cellule 087](#tp2-087) | [Variante expliquée](#archive-087) |
| 088 | [Cellule 088](#tp2-088) | Identique |
| 089 | [Cellule 089](#tp2-089) | Identique |
| 090 | [Cellule 090](#tp2-090) | Identique |
| 091 | [Cellule 091](#tp2-091) | Identique |
| 092 | [Cellule 092](#tp2-092) | Identique |
| 093 | [Cellule 093](#tp2-093) | Identique |
| 094 | [Cellule 094](#tp2-094) | Identique |
| 095 | [Cellule 095](#tp2-095) | Identique |
| 096 | [Cellule 096](#tp2-096) | Identique |
| 097 | [Cellule 097](#tp2-097) | Identique |
| 098 | [Cellule 098](#tp2-098) | Identique |
| 099 | [Cellule 099](#tp2-099) | Identique |
| 100 | [Cellule 100](#tp2-100) | Identique |
| 101 | [Cellule 101](#tp2-101) | [Variante expliquée](#archive-101) |
| 102 | [Cellule 102](#tp2-102) | [Variante expliquée](#archive-102) |
| 103 | [Cellule 103](#tp2-103) | Identique |
| 104 | [Cellule 104](#tp2-104) | Identique |



<a id="archive-002"></a>

#### TP2_Sarcasme_Arabe_T4_Execution.ipynb — variante de la cellule 002

**Rôle :** consulter l’explication de la [cellule principale 002](#tp2-002). Le bloc ci-dessous est celui de cette version ; il permet de vérifier précisément ce qui change.

**Rappel de la version principale actuelle, à comparer au code historique ci-dessous :** La priorité des chemins est `TP2_ROOT`, puis le répertoire courant, avec remontée si celui-ci s'appelle `notebooks`. Dans Colab, l'archive intégrée fournit des sources, pas des checkpoints entraînés ; son SHA256 et le confinement des chemins sont contrôlés avant extraction. `prepare_data(ROOT)` peut télécharger les CSV officiels. Les assertions vérifient la présence des fichiers ; elles ne calculent aucun F1.

**Fiche présente dans cette version (repères historiques).**
**Objectif.** Respecter l'organisation du dépôt demandée dans le TP et utiliser uniquement ses ressources locales.

**Comprendre le code.** Les métriques évaluent les prédictions ; `Pipeline` et `FeatureUnion` construisent la baseline ; les classes `Auto*` ouvrent les poids locaux. `ROOT` désigne la racine du dépôt. Si le noyau démarre dans `notebooks/`, le bloc remonte d'un niveau ; s'il démarre à la racine, il y reste. `DATA_DIR = ROOT/'data/raw'` indique l'emplacement des deux CSV officiels. Les autres fonctions construisent leurs chemins sous `ROOT`, par exemple `models/hf/` ou `results/results.csv`. Les assertions détectent des données ou poids manquants avant les calculs.

**Résultat attendu à l'exécution.** Les chemins du dépôt sont prêts sans chargement de modèle, téléchargement ni apprentissage. Les chemins de checkpoints enregistrés sont relatifs à `ROOT`, ce qui permet de déplacer l'ensemble du dépôt.

**À retenir.** Ouvrir VS Code à la racine contenant `src/`, `notebooks/`, `data/` et `requirements.txt`. Le notebook reste la source pédagogique visible ; les modules exportés servent aux tests et à la reproductibilité. Aucune ressource n'est recherchée dans un dossier voisin.

**Sur Colab.** Les chemins Windows ne sont pas accessibles au serveur. Transférer le projet complet via « Upload to Colab », ou transférer le code puis préparer les ressources avec les scripts officiels. Si le dossier courant est `/content`, un unique projet enfant est détecté ; sinon définir `TP2_ROOT` avant les imports. Les assertions s'arrêtent avant tout apprentissage si données ou poids manquent.

**Session Colab vide.** Le noyau distant ne voit pas le disque Windows. Un petit paquet intégré fournit les scripts de préparation et les manifestes exacts, sans poids ni données. L'archive est contrôlée par SHA256 et extraite seulement sous la racine choisie, sans écraser les fichiers existants. `prepare_data` télécharge les deux CSV officiels et contrôle leurs SHA256, sans consulter les labels test. Avant B, `prepare_models` récupère les quatre révisions Hugging Face épinglées et vérifie leurs poids. AraVec est préparé à son étape C. Aucun de ces téléchargements n'entraîne de modèle. Les sources de ces opérations sont lisibles dans `src/colab_resources.py`.

```python
from sklearn.metrics import (f1_score, precision_score, recall_score, accuracy_score,
    average_precision_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve)

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import FeatureUnion, Pipeline

from sklearn.linear_model import LogisticRegression

import joblib

import emoji

warnings.filterwarnings('ignore',message='This pattern is interpreted as a regular expression, and has match groups')

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from transformers import get_linear_schedule_with_warmup, set_seed

# TP2_LAYOUT
ROOT = Path(os.environ.get('TP2_ROOT',Path.cwd())).resolve()
if ROOT.name == 'notebooks': ROOT = ROOT.parent
if IN_COLAB and not (ROOT/'data/raw/training_data.csv').is_file() and 'TP2_ROOT' not in os.environ:
    candidates=[p for p in ROOT.iterdir() if p.is_dir() and (p/'src/preprocess.py').is_file()]
    ROOT=candidates[0] if len(candidates)==1 else Path('/content/TP2_ArSarcasme')
if IN_COLAB:
    import base64, io, zipfile
    # BOOTSTRAP_ARCHIVE : archive intégrée abrégée pour la lecture (7,630 caractères sur cette ligne).
    BOOTSTRAP_ARCHIVE = "<archive base64 : voir le notebook original>"
    payload=base64.b64decode(BOOTSTRAP_ARCHIVE)
    assert hashlib.sha256(payload).hexdigest()=='e06f5dd5e5204f08e44fe86fda703ed311dfb836bf553d63173d8207c3acff0c'
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for info in archive.infolist():
            target=(ROOT/info.filename).resolve()
            assert target.is_relative_to(ROOT), 'Chemin de préparation invalide.'
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(archive.read(info))
    if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
    from src.colab_resources import prepare_data, prepare_models, prepare_aravec
    prepare_data(ROOT)
DATA_DIR = ROOT/'data/raw'
assert (DATA_DIR/'training_data.csv').is_file(), 'CSV manquant dans data/raw/. Sur Colab utiliser le notebook T4 depuis la première cellule.'
assert (DATA_DIR/'testing_data.csv').is_file(), 'CSV officiel manquant dans data/raw/.'
print('Dossier du projet :',ROOT)
```

<a id="archive-004"></a>

#### TP2_Sarcasme_Arabe_T4_Execution.ipynb — variante de la cellule 004

**Rôle :** consulter l’explication de la [cellule principale 004](#tp2-004). Le bloc ci-dessous est celui de cette version ; il permet de vérifier précisément ce qui change.

**Rappel de la version principale actuelle, à comparer au code historique ci-dessous :** Le code courant fixe `STATIC_EPOCHS=8`, `PLM_EPOCHS=5`, `PATIENCE=2`. Ce sont des plafonds ; l'arrêt anticipé peut réduire les époques réalisées. `CACHE_CONTEXT` décrit notamment matériel, versions et batch ; il sert à refuser la réutilisation silencieuse d'un run incompatible. Les chemins de révision sont replacés sous la racine de cette session.

**Valeurs effectives dans cette variante :** `STATIC_EPOCHS, PLM_EPOCHS, PATIENCE = 8, 3, 2`. Ces valeurs, et non celles du rappel courant, gouvernent les appels de cette variante.

**Fiche présente dans cette version (repères historiques).**
**Objectif.** Fixer les règles communes de durée et empêcher la reprise de checkpoints provenant d’un autre contexte.

**Comprendre le code.** Les petits réseaux ont au maximum huit époques ; les PLM en ont trois ; `PATIENCE=2` autorise deux validations sans amélioration avant arrêt. `RUN_VERSION` identifie la logique expérimentale. `CACHE_CONTEXT` rassemble longueur, tailles de lots, versions et précision. Le manifeste `REVISIONS` précise les versions des quatre PLM ; leurs chemins sont recalculés à partir du dossier actuel.

**Résultat attendu à l’exécution.** Un contexte reproductible et un dictionnaire de ressources locales sont disponibles.

**À retenir.** Le cache économise des calculs, mais ne crée pas une nouvelle expérience. Une modification scientifique doit être accompagnée d’un changement de version et d’une réévaluation des caches concernés.

```python
STATIC_EPOCHS, PLM_EPOCHS, PATIENCE = 8, 3, 2

RUN_VERSION = 'tp2-v1'

CACHE_CONTEXT={'max_len':MAX_LEN,'batch_plm':BATCH,'batch_static':64,
    'torch':torch.__version__,'transformers':__import__('transformers').__version__,
    'precision':'float32','protocol':'conservative-tp2-v1'}

if DEVICE.type=='cuda':
    CACHE_CONTEXT.update(device_type='cuda',plm_precision=PLM_PRECISION,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),numpy=np.__version__,pandas=pd.__version__,
        sklearn=__import__('sklearn').__version__)

REVISIONS = json.loads((ROOT/'models/revisions.json').read_text()) if (ROOT/'models/revisions.json').exists() else {}

for model_name,entry in REVISIONS.items():
    entry['path']=str(ROOT/'models/hf'/model_name.replace('/','--'))
```

<a id="archive-018"></a>

#### TP2_Sarcasme_Arabe_T4_Execution.ipynb — variante de la cellule 018

**Rôle :** consulter l’explication de la [cellule principale 018](#tp2-018). Le bloc ci-dessous est celui de cette version ; il permet de vérifier précisément ce qui change.

**Rappel de la version principale actuelle, à comparer au code historique ci-dessous :** Les assertions testent des propriétés du nettoyage sur des exemples contrôlés. Les agrégations `size/sum/mean` signifient effectif, nombre de sarcastiques et proportion. Les données test étiquetées ne sont pas encore introduites dans ces descriptions supervisées.

**Fiche présente dans cette version (repères historiques).**
**Objectif.** Vérifier des propriétés essentielles du nettoyage avant d’interpréter les distributions.

**Comprendre le code.** Les assertions contrôlent la conservation de la négation لا, l’idempotence du nettoyage et le remplacement des liens/mentions. L’idempotence signifie que nettoyer une seconde fois ne doit pas continuer à déformer le texte. Pour train et val, `groupby` rapporte taille, nombre de sarcastiques et proportion par dialecte puis sentiment. Une figure vide est préparée pour la cellule suivante.

**Résultat attendu à l’exécution.** Les contrôles passent ou arrêtent l’exécution avec une assertion ; les tables décrivent les partitions de développement.

**À retenir.** Un taux calculé sur un très petit sous-groupe est instable. La colonne `size` compte autant que la colonne `mean` pour interpréter les différences entre dialectes.

```python
assert 'لا' in prepare_text('لَا أحب هذا')

assert prepare_text('مرحبا   بالعالم') == prepare_text(prepare_text('مرحبا   بالعالم'))

assert 'http' not in prepare_text(examples[0]) and '@ami' not in prepare_text(examples[0])

print('Contrôles du prétraitement et de séparation : OK.')

for name,frame in [('train',train),('val',val)]:
    print(name, 'n =',len(frame),'sarcasme =',round(frame.y.mean(),4))
    display(frame.groupby('dialect').y.agg(['size','sum','mean']))
    display(frame.groupby('sentiment').y.agg(['size','sum','mean']))

fig,ax=plt.subplots(figsize=(7,3))
```

<a id="archive-019"></a>

#### TP2_Sarcasme_Arabe_T4_Execution.ipynb — variante de la cellule 019

**Rôle :** consulter l’explication de la [cellule principale 019](#tp2-019). Le bloc ci-dessous est celui de cette version ; il permet de vérifier précisément ce qui change.

**Rappel de la version principale actuelle, à comparer au code historique ci-dessous :** Le graphique compare les taux de sarcasme par dialecte de train et val. Les effectifs doivent être lus avec les taux : 0 % sur cinq exemples ne représente pas la même information que 0 % sur des milliers. Création, remplissage et affichage sont réunis pour éviter des axes vides en sortie inline.

**Fiche présente dans cette version (repères historiques).**
**Objectif.** Lire rapidement les différences de proportion sarcastique dans le train nettoyé.

**Comprendre le code.** La moyenne de `y` est calculée pour chaque dialecte puis dessinée sur la figure préparée auparavant. Les étiquettes précisent qu’il s’agit d’une proportion, et non d’un nombre de tweets. `tight_layout` ajuste les marges ; `savefig` conserve une version PNG avant l’affichage.

**Résultat attendu à l’exécution.** Un diagramme en barres apparaît et `figures/prior_train.png` est créé dans le dépôt local.

**À retenir.** Ce graphique montre la distribution des annotations du corpus, pas un taux d’erreur du modèle ni une propriété universelle d’un dialecte. Revenir aux effectifs du tableau précédent pour juger sa stabilité.

```python
train.groupby('dialect').y.mean().plot.bar(ax=ax,color='#275D88')

ax.set(ylabel='Proportion sarcastique',title='Train nettoyé par dialecte',xlabel='Dialecte')

plt.tight_layout()

fig.savefig(ROOT/'figures/prior_train.png')

plt.show()
```

<a id="archive-025"></a>

#### TP2_Sarcasme_Arabe_T4_Execution.ipynb — variante de la cellule 025

**Rôle :** consulter l’explication de la [cellule principale 025](#tp2-025). Le bloc ci-dessous est celui de cette version ; il permet de vérifier précisément ce qui change.

**Rappel de la version principale actuelle, à comparer au code historique ci-dessous :** `ax.has_data()` ou des textes vérifient qu'au moins un axe possède du contenu. `display(fig)` affiche explicitement cette figure puis `plt.close(fig)` libère son état Matplotlib. Cela évite de laisser le backend choisir une autre figure courante ; le contrôle ne prouve toutefois pas que tous les sous-axes d'une figure sont sémantiquement complets.

**Fiche présente dans cette version (repères historiques).**
**Objectif.** Conserver les graphiques consultables dans le notebook et dans les livrables.

**Comprendre le code.** `savefig` ajuste la disposition, construit un chemin sous `ROOT/figures/`, enregistre le PNG et affiche la figure. `bbox_inches="tight"` réduit les marges superflues et limite les étiquettes coupées. Le nom est fourni par la cellule qui produit le graphique.

**Résultat attendu à l’exécution.** Une fonction utilitaire est définie. Aucun fichier graphique n’est créé tant qu’elle n’est pas appelée avec une figure.

**À retenir.** Une image sauvegardée n’apporte pas une mesure nouvelle : elle représente les tableaux ou calculs réellement produits. Une légende précise doit indiquer partition, métrique et seuil concernés.

```python
def savefig(fig,name):
    fig.tight_layout(); fig.savefig(ROOT/f'figures/{name}.png',bbox_inches='tight'); plt.show()
```

<a id="archive-050"></a>

#### TP2_Sarcasme_Arabe_T4_Execution.ipynb — variante de la cellule 050

**Rôle :** consulter l’explication de la [cellule principale 050](#tp2-050). Le bloc ci-dessous est celui de cette version ; il permet de vérifier précisément ce qui change.

**Rappel de la version principale actuelle, à comparer au code historique ci-dessous :** Le choix de reprise valide signature, contexte et métadonnées ; sinon les anciens artefacts sont archivés avant un nouveau run. Chaque époque évalue val et son seuil ; seul un meilleur F1 sauvegarde le checkpoint pending. L'arrêt anticipé sélectionne la meilleure époque, pas nécessairement la dernière ; `val_mask` permet de sélectionner sans dialecte égyptien pour F2. Définir cette fonction n'est pas lancer les appels d'apprentissage qui suivent.

**Fiche présente dans cette version (repères historiques).**
**Objectif.** Appliquer une procédure commune et traçable aux réseaux C, D2 et E.

**Comprendre le code.** La fonction complète la configuration, enregistre les identifiants utilisés et vérifie la compatibilité d’un checkpoint terminé. Sinon, elle prépare AdamW, pertes et éventuel warmup PLM de 10 %, puis alterne époque train et inférence val. À chaque époque, le seuil est choisi sur la validation autorisée et le F1 décide si les poids sont sauvegardés. La patience déclenche un arrêt après absence de progrès. Pour AraVec, les embeddings sont dégelés après les deux premières époques. Le meilleur état est rechargé avant l’export.

**Résultat attendu à l’exécution.** Un appel renvoie la clé d’expérience et le meilleur modèle ; historique, probabilités et métadonnées sont écrits. Les métadonnées finales sont publiées par remplacement atomique.

**À retenir.** Les labels test ne participent ni à l’arrêt ni au checkpoint. Avec `val_mask`, seule une partie de val choisit le checkpoint, utile pour F2. Un run interrompu avant ses métadonnées finales n’est pas considéré terminé. Le meilleur checkpoint peut précéder la dernière époque parcourue.

**Optimisation CPU ajoutée.** Le vecteur val de l’époque ayant gagné est conservé, évitant un deuxième forward identique après le rechargement. Un cache signé permet les reprises sans nouvelle validation. Les poids sont écrits dans un fichier pending puis publiés en fin de run ; les anciens résultats authentifiés ne sont pas recalculés artificiellement.

```python
def train_neural(model,xtrain,xval,name,seed=42,config=None,epochs=STATIC_EPOCHS,
                 lr=1e-3,imbalance='weighted',labels=None,sentiments=None,lam=0,freeze_epochs=0,plm=False,val_mask=None):
    config=dict(config or {},seed=seed,epochs=epochs,lr=lr,imbalance=imbalance,lam=lam,
                train_ids=(config or {}).get('train_ids',train.id.tolist()))
    if plm: config.update(dynamic_padding=True,length_grouped=True)
    path=ROOT/f'models/{name}_seed{seed}.pt'; meta=path.with_suffix('.json'); sig=cache_signature(config)
    model=model.to(DEVICE)
    if plm and DEVICE.type!='cpu':
        config.update(device_type=DEVICE.type,precision=PLM_PRECISION,batch=BATCH)
    p=None; best_path=path.with_suffix('.pending.pt')
    trained_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if freeze_epochs: trained_params+=model.embedding.weight.numel()
    selection_mask=np.ones(len(val),dtype=bool) if val_mask is None else np.asarray(val_mask)
    config['selection_val_ids']=val.id[selection_mask].tolist()
    sig=cache_signature(config)
    if path.exists() and compatible_metadata(meta) and json.loads(meta.read_text())['signature']==sig:
        payload=torch.load(path,map_location=DEVICE,weights_only=True); model.load_state_dict(payload); del payload
        saved=json.loads(meta.read_text()); seconds=saved['seconds']; history=saved['history']
    else:
        seed_all(seed); y=YTRAIN if labels is None else labels; sy=STRAIN if sentiments is None else sentiments
        loader=make_training_loader(xtrain,y,sy,seed,imbalance,plm)
        opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
        scaler=torch.amp.GradScaler('cuda') if plm and DEVICE.type=='cuda' else None
        total=epochs*len(loader); scheduler=get_linear_schedule_with_warmup(opt,int(.1*total),total) if plm else None
        weights=torch.tensor(len(y)/(2*np.bincount(y.numpy())),dtype=torch.float32,device=DEVICE) if imbalance=='weighted' else None
        loss_fn=nn.CrossEntropyLoss(weight=weights); sent_loss=nn.CrossEntropyLoss()
        start=time.perf_counter(); best=-1; stale=0; history=[]; best_p=None
        for epoch in range(epochs):
            if freeze_epochs and epoch==freeze_epochs:
                model.embedding.weight.requires_grad_(True)
            mean_loss=train_epoch(model,loader,opt,loss_fn,sent_loss,lam,scheduler,plm,name,seed,epoch,scaler)
            p=neural_predict(model,xval,batch=BATCH if plm else 64)
            t=best_threshold(val.y[selection_mask],p[selection_mask]); f=float(f1_score(val.y[selection_mask],p[selection_mask]>=t,zero_division=0))
            history.append({'epoch':epoch+1,'loss':mean_loss,'val_f1':f,'threshold':t})
            print(name,seed,'époque',epoch+1,'perte',round(mean_loss,4),'F1 val',round(f,4),flush=True)
            if f>best+1e-6:
                best=f; stale=0; best_p=p.copy(); torch.save(model.state_dict(),best_path)
            else: stale+=1
            if stale>=PATIENCE and epoch>=freeze_epochs: break
        seconds=time.perf_counter()-start; best_path.replace(path); p=best_p
        draft=meta.with_suffix('.json.tmp')
        draft.write_text(json.dumps({'signature':sig,'seconds':seconds,'history':history,'config':config,'runtime_context':CACHE_CONTEXT,'checkpoint_stat':checkpoint_stat(path),
            'performance':{'threads':torch.get_num_threads(),'inter_op':torch.get_num_interop_threads(),'loader':'vectorized_batches_v1','precision':PLM_PRECISION if plm else 'float32','device':str(DEVICE)}},indent=2))
        draft.replace(meta)
        model.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=True))
    p=validation_scores(model,xval,path,sig,candidate=p,plm=plm)
    HISTORY.extend([dict(model=name,seed=seed,**h) for h in history])
    pd.DataFrame(HISTORY).to_csv(ROOT/'results/learning_curves.csv',index=False)
    key=register(name,seed,p,seconds,trained_params,config.get('preprocessing','conservative'),
                 'plm' if plm else config.get('kind','static'),path,config,
                 extra={'epochs_completed':len(history),'n_train':len(xtrain[0])})
    return key,model
```

<a id="archive-085"></a>

#### TP2_Sarcasme_Arabe_T4_Execution.ipynb — variante de la cellule 085

**Rôle :** consulter l’explication de la [cellule principale 085](#tp2-085). Le bloc ci-dessous est celui de cette version ; il permet de vérifier précisément ce qui change.

**Rappel de la version principale actuelle, à comparer au code historique ci-dessous :** Un CSV est écrit par modèle final avec `id,y_true,y_pred,probability`. B2 exporte sa probabilité sentiment et conserve sa décision argmax, qui ne doit pas être recalculée par un seuil de probabilité 0,5. Les tableaux de résultats sauvegardés dans le notebook ne sont pas ces CSV : ils restent sur le serveur d'exécution tant qu'ils ne sont pas téléchargés.

**Fiche présente dans cette version (repères historiques).**
**Objectif.** Produire des scores test vérifiables ligne par ligne par l’enseignant.

**Comprendre le code.** Chaque clé finale est évaluée avec son seuil fixé sur val. Le CSV exporte `id`, `y_true`, `y_pred` et `probability`. Pour B2, la décision reste la règle argmax négatif, mais la probabilité exportée est P(négatif) et l’AP utilise ce score continu. Le temps global de pipeline et la taille de l’artefact sont enregistrés ; les champs test sont ajoutés aux lignes de résultats.

**Résultat attendu à l’exécution.** Dix-sept CSV finaux et un tableau F1/macro-F1/AP test sont attendus après calcul. Une figure à deux panneaux est préparée pour les courbes suivantes.

**À retenir.** Le temps global peut inclure chargement et lecture de caches ; il ne remplace pas la latence CPU batch 1 mesurée en G. Les scores exportés servent à recalculer les métriques, pas à retoucher les décisions après coup.

```python
TEST_P={}

TEST_ROWS=[]

for key in FINAL_KEYS:
    rec=REGISTRY[key]; path=ROOT/f'predictions/{key}.csv'
    start=time.perf_counter(); p=predict_registered(rec,test.tweet.tolist(),'test'); elapsed=time.perf_counter()-start
    TEST_P[key]=p; pred=(p>=rec['threshold']).astype(int)
    export_p=sentiment_predict(test.tweet.tolist(),'test')[0] if rec['kind']=='sentiment' else p
    pd.DataFrame({'id':test.id,'y_true':test.y,'y_pred':pred,'probability':np.asarray(export_p,dtype=np.float64)}).to_csv(path,index=False)
    row=row_for_key(key); row.update({'test_'+k:v for k,v in scores(test.y,p,rec['threshold']).items()})
    # AP B2 doit utiliser la probabilité de sentiment, pas seulement la décision binaire.
    if rec['kind']=='sentiment': row['test_pr_auc_ap']=float(average_precision_score(test.y,sentiment_predict(test.tweet.tolist(),'test')[0]))
    row['test_inference_total_seconds']=elapsed; row['test_pipeline_ms_per_tweet']=1000*elapsed/len(test)
    row['artifact_size_mb']=(ROOT/rec['path']).stat().st_size/1e6 if (ROOT/rec['path']).is_file() else sum(f.stat().st_size for f in (ROOT/rec['path']).glob('*') if f.is_file())/1e6
    TEST_ROWS.append(row.copy()); pd.DataFrame(RESULTS).to_csv(ROOT/'results/results.csv',index=False)
    print(key,'F1 test',round(row['test_f1_sarcastique'],4),flush=True)

summary=pd.DataFrame(TEST_ROWS)

display(summary[['model','seed','threshold','test_f1_sarcastique','test_macro_f1','test_pr_auc_ap']])

fig,axes=plt.subplots(1,2,figsize=(11,4))
```

<a id="archive-086"></a>

#### TP2_Sarcasme_Arabe_T4_Execution.ipynb — variante de la cellule 086

**Rôle :** consulter l’explication de la [cellule principale 086](#tp2-086). Le bloc ci-dessous est celui de cette version ; il permet de vérifier précisément ce qui change.

**Rappel de la version principale actuelle, à comparer au code historique ci-dessous :** Les courbes F1(seuil) et précision-rappel utilisent les scores test du modèle déjà fixé. Les points val servent de référence visuelle ; l'optimum test est calculé seulement pour constater un décalage. Le code ne remplace pas automatiquement les seuils du registre par cet optimum.

**Fiche présente dans cette version (repères historiques).**
**Objectif.** Étudier comment un changement de distribution peut déplacer le compromis précision/rappel.

**Comprendre le code.** Pour la graine 42 de chaque famille, une grille de 201 seuils entre 0 et 1 fournit F1=f(seuil). Un point marque le seuil utilisable, déjà choisi sur val. La courbe PR est également construite. Le meilleur seuil calculé directement sur les labels test est stocké séparément avec son F1, sous l’étiquette posthoc.

**Résultat attendu à l’exécution.** Les courbes sont alimentées et `threshold_rows` compare seuil val et seuil test a posteriori.

**À retenir.** Le maximum test est une description optimiste obtenue en connaissant les réponses. Il n’est jamais autorisé à remplacer le seuil val pour annoncer la performance finale ou choisir le modèle de production.

```python
threshold_rows=[]

for family,keys in FINAL_GROUPS.items():
    key=keys[0]; p=TEST_P[key]; t=REGISTRY[key]['threshold']; grid=np.linspace(0,1,201)
    f=[f1_score(test.y,p>=g,zero_division=0) for g in grid]
    axes[0].plot(grid,f,label=family); axes[0].scatter([t],[f1_score(test.y,p>=t)],s=20)
    prec,rec,_=precision_recall_curve(test.y,p); axes[1].plot(rec,prec,label=family)
    post=best_threshold(test.y,p)
    threshold_rows.append({'famille':family,'seuil_val':t,'seuil_test_posthoc':post,
        'f1_test_seuil_val':scores(test.y,p,t)['f1_sarcastique'],'f1_test_posthoc':scores(test.y,p,post)['f1_sarcastique']})
```

<a id="archive-087"></a>

#### TP2_Sarcasme_Arabe_T4_Execution.ipynb — variante de la cellule 087

**Rôle :** consulter l’explication de la [cellule principale 087](#tp2-087). Le bloc ci-dessous est celui de cette version ; il permet de vérifier précisément ce qui change.

**Rappel de la version principale actuelle, à comparer au code historique ci-dessous :** Le tableau `threshold_rows` contient des colonnes qui distinguent seuil val et seuil test posthoc. Il ne faut pas annoncer le F1 posthoc comme performance d'un système dont le seuil aurait été fixé avant le test. Le CSV rend cette distinction consultable hors du noyau.

**Fiche présente dans cette version (repères historiques).**
**Objectif.** Rendre explicite la différence entre réglage utilisable et observation a posteriori.

**Comprendre le code.** La cellule ajoute titres, axes et légendes aux courbes précédentes. L’image `threshold_and_pr_test.png` et le tableau `threshold_shift.csv` sont sauvegardés, puis les valeurs sont affichées. La courbe précision-rappel expose plusieurs compromis sans en choisir un nouveau pour le modèle final.

**Résultat attendu à l’exécution.** Une figure à deux panneaux et un tableau lisible documentent le décalage de seuil.

**À retenir.** Un grand écart entre seuils peut signaler prior, calibration ou distribution différents. Il ne suffit pas à attribuer tout l’écart au seul prior, car vocabulaire, dialectes et autres caractéristiques peuvent aussi changer.

```python
axes[0].set(title='Test diagnostic a posteriori uniquement',xlabel='Seuil',ylabel='F1 sarcastique')

axes[0].legend()

axes[1].set(title='Courbes précision rappel test',xlabel='Rappel',ylabel='Précision')

axes[1].legend()

savefig(fig,'threshold_and_pr_test')

pd.DataFrame(threshold_rows).to_csv(ROOT/'results/threshold_shift.csv',index=False)

display(pd.DataFrame(threshold_rows))
```

<a id="archive-101"></a>

#### TP2_Sarcasme_Arabe_T4_Execution.ipynb — variante de la cellule 101

**Rôle :** consulter l’explication de la [cellule principale 101](#tp2-101). Le bloc ci-dessous est celui de cette version ; il permet de vérifier précisément ce qui change.

**Rappel de la version principale actuelle, à comparer au code historique ci-dessous :** La comparaison est exportée puis les trajectoires F1 val sont tracées par famille et graine. Une courbe par époque montre le choix du checkpoint et le surapprentissage possible ; elle ne correspond pas à une trajectoire F1 test. Une tête linéaire sans historique d'époques n'a pas à recevoir une courbe artificielle.

**Fiche présente dans cette version (repères historiques).**
**Objectif.** Associer les résultats finaux aux trajectoires de validation des modèles retenus.

**Comprendre le code.** La liste des lignes est convertie en table, exportée dans `final_comparison.csv` et affichée. L’historique rassemble ensuite les F1 val par époque pour C/D/E1/E2 et leurs graines. Une tête linéaire sans époques est identifiée explicitement. Le graphique des courbes est sauvegardé. La matrice de confusion du modèle déjà choisi est calculée sur ses prédictions test.

**Résultat attendu à l’exécution.** Un tableau de synthèse, quatre panneaux d’apprentissage et une matrice de confusion préparée pour affichage sont produits.

**À retenir.** Une courbe train/val n’est pas une preuve de robustesse externe. Le meilleur checkpoint peut être antérieur à la dernière époque ; lire les courbes avec les règles d’arrêt plutôt que supposer que plus d’époques améliorent toujours la qualité.

```python
comparison=pd.DataFrame(comparison)

comparison.to_csv(ROOT/'results/final_comparison.csv',index=False)

display(comparison.round(4))

history_frame=pd.DataFrame(HISTORY)

fig,axes=plt.subplots(2,2,figsize=(10,7))

for ax,family in zip(axes.flat,['C','D','E1','E2']):
    model_name=REGISTRY[FINAL_GROUPS[family][0]]['name']
    curves=history_frame.loc[history_frame.model.eq(model_name)]
    for seed,curve in curves.groupby('seed'):
        ax.plot(curve.epoch,curve.val_f1,marker='o',label=f'Graine {seed}')
    if len(curves): ax.legend(fontsize=8)
    else: ax.text(.5,.5,'Classifieur linéaire sans époques',ha='center',transform=ax.transAxes)
    ax.set(title=f'{family} — {model_name}',xlabel='Époque',ylabel='F1 validation')

savefig(fig,'learning_curves_final')

fig,ax=plt.subplots(figsize=(6,5))

ConfusionMatrixDisplay.from_predictions(ytest,chosen_pred,display_labels=['Non sarcastique','Sarcastique'],cmap='Blues',ax=ax,colorbar=False)
```

<a id="archive-102"></a>

#### TP2_Sarcasme_Arabe_T4_Execution.ipynb — variante de la cellule 102

**Rôle :** consulter l’explication de la [cellule principale 102](#tp2-102). Le bloc ci-dessous est celui de cette version ; il permet de vérifier précisément ce qui change.

**Rappel de la version principale actuelle, à comparer au code historique ci-dessous :** La matrice de confusion vient du modèle sélectionné et de son seuil val verrouillé. Le barplot utilise les moyennes test et l'écart-type des graines, pas des intervalles bootstrap. `decision` conserve la clé exacte et les métriques du checkpoint représentatif ; ces dernières peuvent différer de la moyenne de famille.

**Fiche présente dans cette version (repères historiques).**
**Objectif.** Rendre visibles les compromis des modèles au seuil déjà retenu.

**Comprendre le code.** La matrice reçoit son titre avec famille et seuil val, puis est sauvegardée. Le diagramme en barres montre la moyenne F1 test ; les barres d’erreur des familles entraînées représentent l’écart-type sur trois graines. Pour B1/B2, la valeur manquante est tracée sans barre, pas interprétée comme une variance connue nulle. Le dictionnaire `decision` réunit le choix fait sur val et ses métriques test.

**Résultat attendu à l’exécution.** Deux figures sont enregistrées et une décision traçable est préparée avec un horodatage de fin.

**À retenir.** La figure test expose le choix de validation ; elle ne doit pas conduire à déplacer après coup ce choix vers la barre la plus haute. Une recommandation réelle doit aussi considérer latence, taille et résultats F, notamment pour des tweets marocains.

```python
ax.set_title(f'Modèle retenu sur val : {chosen_final_family} — seuil {chosen_rec["threshold"]:.3f}')

savefig(fig,'chosen_confusion')

fig,ax=plt.subplots(figsize=(8,4))

ax.bar(comparison.family,comparison.test_f1_mean,yerr=comparison.test_f1_std.fillna(0),capsize=5,color='#275D88')

ax.set(ylabel='F1 sarcastique test',title='Entraînés : 3 graines ; zero-shot : 1 exécution',ylim=(0,1))

savefig(fig,'final_comparison')

decision={'chosen_key':CHOSEN_KEY,'family':chosen_final_family,'criterion':'moyenne F1 validation sur trois graines',
          'threshold':chosen_rec['threshold'],'test_metrics':scores(ytest,chosen_p,chosen_rec['threshold']),
          'completed_at_utc':datetime.now(timezone.utc).isoformat()}
```

### Version `TP2_Sarcasme_Arabe.ipynb`

Ancien fichier local : une cellule supplémentaire de diagnostic CUDA avant les imports décale la numérotation. Son code est documenté ici pour comprendre un ancien onglet ; ce n’est pas une consigne de revenir à cette version.

| Cellule de cette version | Renvoi au principal | État du code |
|---|---|---|
| 001 | Ajout spécifique | [Variante expliquée](#ancien-001) |
| 002 | [Cellule 001](#tp2-001) | [Variante expliquée](#ancien-002) |
| 003 | [Cellule 002](#tp2-002) | [Variante expliquée](#ancien-003) |
| 004 | [Cellule 003](#tp2-003) | [Variante expliquée](#ancien-004) |
| 005 | [Cellule 004](#tp2-004) | [Variante expliquée](#ancien-005) |
| 006 | [Cellule 005](#tp2-005) | Identique |
| 007 | [Cellule 006](#tp2-006) | [Variante expliquée](#ancien-007) |
| 008 | [Cellule 007](#tp2-007) | Identique |
| 009 | [Cellule 008](#tp2-008) | Identique |
| 010 | [Cellule 009](#tp2-009) | Identique |
| 011 | [Cellule 010](#tp2-010) | Identique |
| 012 | [Cellule 011](#tp2-011) | Identique |
| 013 | [Cellule 012](#tp2-012) | Identique |
| 014 | [Cellule 013](#tp2-013) | Identique |
| 015 | [Cellule 014](#tp2-014) | Identique |
| 016 | [Cellule 015](#tp2-015) | Identique |
| 017 | [Cellule 016](#tp2-016) | Identique |
| 018 | [Cellule 017](#tp2-017) | Identique |
| 019 | [Cellule 018](#tp2-018) | [Variante expliquée](#ancien-019) |
| 020 | [Cellule 019](#tp2-019) | [Variante expliquée](#ancien-020) |
| 021 | [Cellule 020](#tp2-020) | Identique |
| 022 | [Cellule 021](#tp2-021) | Identique |
| 023 | [Cellule 022](#tp2-022) | Identique |
| 024 | [Cellule 023](#tp2-023) | Identique |
| 025 | [Cellule 024](#tp2-024) | Identique |
| 026 | [Cellule 025](#tp2-025) | [Variante expliquée](#ancien-026) |
| 027 | [Cellule 026](#tp2-026) | Identique |
| 028 | [Cellule 027](#tp2-027) | Identique |
| 029 | [Cellule 028](#tp2-028) | Identique |
| 030 | [Cellule 029](#tp2-029) | Identique |
| 031 | [Cellule 030](#tp2-030) | [Variante expliquée](#ancien-031) |
| 032 | [Cellule 031](#tp2-031) | Identique |
| 033 | [Cellule 032](#tp2-032) | [Variante expliquée](#ancien-033) |
| 034 | [Cellule 033](#tp2-033) | Identique |
| 035 | [Cellule 034](#tp2-034) | Identique |
| 036 | [Cellule 035](#tp2-035) | Identique |
| 037 | [Cellule 036](#tp2-036) | Identique |
| 038 | [Cellule 037](#tp2-037) | Identique |
| 039 | [Cellule 038](#tp2-038) | Identique |
| 040 | [Cellule 039](#tp2-039) | Identique |
| 041 | [Cellule 040](#tp2-040) | Identique |
| 042 | [Cellule 041](#tp2-041) | Identique |
| 043 | [Cellule 042](#tp2-042) | Identique |
| 044 | [Cellule 043](#tp2-043) | Identique |
| 045 | [Cellule 044](#tp2-044) | Identique |
| 046 | [Cellule 045](#tp2-045) | Identique |
| 047 | [Cellule 046](#tp2-046) | Identique |
| 048 | [Cellule 047](#tp2-047) | [Variante expliquée](#ancien-048) |
| 049 | [Cellule 048](#tp2-048) | [Variante expliquée](#ancien-049) |
| 050 | [Cellule 049](#tp2-049) | [Variante expliquée](#ancien-050) |
| 051 | [Cellule 050](#tp2-050) | [Variante expliquée](#ancien-051) |
| 052 | [Cellule 051](#tp2-051) | Identique |
| 053 | [Cellule 052](#tp2-052) | Identique |
| 054 | [Cellule 053](#tp2-053) | Identique |
| 055 | [Cellule 054](#tp2-054) | [Variante expliquée](#ancien-055) |
| 056 | [Cellule 055](#tp2-055) | Identique |
| 057 | [Cellule 056](#tp2-056) | Identique |
| 058 | [Cellule 057](#tp2-057) | Identique |
| 059 | [Cellule 058](#tp2-058) | Identique |
| 060 | [Cellule 059](#tp2-059) | Identique |
| 061 | [Cellule 060](#tp2-060) | Identique |
| 062 | [Cellule 061](#tp2-061) | Identique |
| 063 | [Cellule 062](#tp2-062) | Identique |
| 064 | [Cellule 063](#tp2-063) | Identique |
| 065 | [Cellule 064](#tp2-064) | Identique |
| 066 | [Cellule 065](#tp2-065) | Identique |
| 067 | [Cellule 066](#tp2-066) | Identique |
| 068 | [Cellule 067](#tp2-067) | Identique |
| 069 | [Cellule 068](#tp2-068) | Identique |
| 070 | [Cellule 069](#tp2-069) | Identique |
| 071 | [Cellule 070](#tp2-070) | Identique |
| 072 | [Cellule 071](#tp2-071) | Identique |
| 073 | [Cellule 072](#tp2-072) | Identique |
| 074 | [Cellule 073](#tp2-073) | Identique |
| 075 | [Cellule 074](#tp2-074) | Identique |
| 076 | [Cellule 075](#tp2-075) | Identique |
| 077 | [Cellule 076](#tp2-076) | Identique |
| 078 | [Cellule 077](#tp2-077) | Identique |
| 079 | [Cellule 078](#tp2-078) | Identique |
| 080 | [Cellule 079](#tp2-079) | Identique |
| 081 | [Cellule 080](#tp2-080) | Identique |
| 082 | [Cellule 081](#tp2-081) | Identique |
| 083 | [Cellule 082](#tp2-082) | Identique |
| 084 | [Cellule 083](#tp2-083) | Identique |
| 085 | [Cellule 084](#tp2-084) | Identique |
| 086 | [Cellule 085](#tp2-085) | [Variante expliquée](#ancien-086) |
| 087 | [Cellule 086](#tp2-086) | [Variante expliquée](#ancien-087) |
| 088 | [Cellule 087](#tp2-087) | [Variante expliquée](#ancien-088) |
| 089 | [Cellule 088](#tp2-088) | Identique |
| 090 | [Cellule 089](#tp2-089) | Identique |
| 091 | [Cellule 090](#tp2-090) | Identique |
| 092 | [Cellule 091](#tp2-091) | Identique |
| 093 | [Cellule 092](#tp2-092) | Identique |
| 094 | [Cellule 093](#tp2-093) | Identique |
| 095 | [Cellule 094](#tp2-094) | Identique |
| 096 | [Cellule 095](#tp2-095) | Identique |
| 097 | [Cellule 096](#tp2-096) | Identique |
| 098 | [Cellule 097](#tp2-097) | Identique |
| 099 | [Cellule 098](#tp2-098) | Identique |
| 100 | [Cellule 099](#tp2-099) | Identique |
| 101 | [Cellule 100](#tp2-100) | Identique |
| 102 | [Cellule 101](#tp2-101) | [Variante expliquée](#ancien-102) |
| 103 | [Cellule 102](#tp2-102) | [Variante expliquée](#ancien-103) |
| 104 | [Cellule 103](#tp2-103) | Identique |
| 105 | [Cellule 104](#tp2-104) | Identique |



<a id="ancien-001"></a>

#### TP2_Sarcasme_Arabe.ipynb — variante de la cellule 001

**Diagnostic CUDA ajouté en tête de cette ancienne version.** `import torch` rend PyTorch disponible ; l’assertion exige un GPU CUDA visible ; `get_device_name(0)` affiche son nom. Cette cellule ne lance aucun entraînement. Elle échoue volontairement sur le Conda CPU local. La préparation T4 courante distingue plus complètement Colab et CPU local dans les cellules 001–006.

```python
import torch

assert torch.cuda.is_available(), "GPU CUDA non détecté"
print(torch.cuda.get_device_name(0))
```

<a id="ancien-002"></a>

#### TP2_Sarcasme_Arabe.ipynb — variante de la cellule 002

**Rôle :** consulter l’explication de la [cellule principale 001](#tp2-001). Le bloc ci-dessous est celui de cette version ; il permet de vérifier précisément ce qui change.

**Rappel de la version principale actuelle, à comparer au code historique ci-dessous :** Le bloc `IN_COLAB` compare les versions installées à la liste figée ; il ne touche pas au Conda local. Un arrêt « Installation terminée » demande de redémarrer le noyau, alors qu'un arrêt « Échec de pip » demande de lire les erreurs d'installation. Les variables OMP/MKL sont fixées avant les imports numériques : six threads localement, au plus deux dans Colab. Le code actuel supporte bien le GPU dans les cellules suivantes ; l'ancienne mention « aucun réglage GPU » dans la fiche d'origine ne décrit que le réglage des threads CPU.

```python
from pathlib import Path

import os, sys, re, json, time, random, hashlib, gc, math, copy, platform, warnings

CPU_THREADS=6  # 12 testés : gain forward limité, backward non mesuré.
os.environ['OMP_NUM_THREADS']=str(CPU_THREADS)
os.environ['MKL_NUM_THREADS']=str(CPU_THREADS)
os.environ['USE_TF']='0'
os.environ['USE_FLAX']='0'
from collections import Counter

from datetime import datetime, timezone

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from IPython.display import display, Markdown, HTML

import torch

from torch import nn

from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler, RandomSampler, BatchSampler

from sklearn.model_selection import train_test_split
```

<a id="ancien-003"></a>

#### TP2_Sarcasme_Arabe.ipynb — variante de la cellule 003

**Rôle :** consulter l’explication de la [cellule principale 002](#tp2-002). Le bloc ci-dessous est celui de cette version ; il permet de vérifier précisément ce qui change.

**Rappel de la version principale actuelle, à comparer au code historique ci-dessous :** La priorité des chemins est `TP2_ROOT`, puis le répertoire courant, avec remontée si celui-ci s'appelle `notebooks`. Dans Colab, l'archive intégrée fournit des sources, pas des checkpoints entraînés ; son SHA256 et le confinement des chemins sont contrôlés avant extraction. `prepare_data(ROOT)` peut télécharger les CSV officiels. Les assertions vérifient la présence des fichiers ; elles ne calculent aucun F1.

**Fiche présente dans cette version (repères historiques).**
**Objectif.** Respecter l'organisation du dépôt demandée dans le TP et utiliser uniquement ses ressources locales.

**Comprendre le code.** Les métriques évaluent les prédictions ; `Pipeline` et `FeatureUnion` construisent la baseline ; les classes `Auto*` ouvrent les poids locaux. `ROOT` désigne la racine du dépôt. Si le noyau démarre dans `notebooks/`, le bloc remonte d'un niveau ; s'il démarre à la racine, il y reste. `DATA_DIR = ROOT/'data/raw'` indique l'emplacement des deux CSV officiels. Les autres fonctions construisent leurs chemins sous `ROOT`, par exemple `models/hf/` ou `results/results.csv`. Les assertions détectent des données ou poids manquants avant les calculs.

**Résultat attendu à l'exécution.** Les chemins du dépôt sont prêts sans chargement de modèle, téléchargement ni apprentissage. Les chemins de checkpoints enregistrés sont relatifs à `ROOT`, ce qui permet de déplacer l'ensemble du dépôt.

**À retenir.** Ouvrir VS Code à la racine contenant `src/`, `notebooks/`, `data/` et `requirements.txt`. Le notebook reste la source pédagogique visible ; les modules exportés servent aux tests et à la reproductibilité. Aucune ressource n'est recherchée dans un dossier voisin.

```python
from sklearn.metrics import (f1_score, precision_score, recall_score, accuracy_score,
    average_precision_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve)

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import FeatureUnion, Pipeline

from sklearn.linear_model import LogisticRegression

import joblib

import emoji

warnings.filterwarnings('ignore',message='This pattern is interpreted as a regular expression, and has match groups')

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from transformers import get_linear_schedule_with_warmup, set_seed

# TP2_LAYOUT
ROOT = Path.cwd().resolve()
if ROOT.name == 'notebooks': ROOT = ROOT.parent
DATA_DIR = ROOT/'data/raw'
assert (DATA_DIR/'training_data.csv').is_file(), 'Ouvrir VS Code à la racine du dépôt TP2.'
assert (DATA_DIR/'testing_data.csv').is_file(), 'CSV officiel manquant dans data/raw/.'
assert (ROOT/'models/assets_complete.json').is_file(), 'Conserver les poids locaux dans models/.'
```

<a id="ancien-004"></a>

#### TP2_Sarcasme_Arabe.ipynb — variante de la cellule 004

**Rôle :** consulter l’explication de la [cellule principale 003](#tp2-003). Le bloc ci-dessous est celui de cette version ; il permet de vérifier précisément ce qui change.

**Rappel de la version principale actuelle, à comparer au code historique ci-dessous :** `select_device(True)` refuse une session Colab sans CUDA, pour éviter un entraînement CPU involontaire. Sur T4 : batch PLM 32 et précision AMP float16 ; sur CPU local : batch 16 et float32. `MAX_LEN=64` compte des positions, pas 64 caractères. Les dictionnaires de cache initialisés ici sont partagés par les fonctions suivantes.

**Fiche présente dans cette version (repères historiques).**
**Objectif.** Définir un budget compatible avec les 16 Go de RAM installés et organiser les emplacements de sortie.

**Comprendre le code.** `mkdir(..., exist_ok=True)` prépare les dossiers du dépôt local sans effacer leur contenu. `DEVICE` impose ici le CPU demandé. Six threads correspondent au réglage retenu sur le Ryzen 5 5600G. `SEEDS` fixe les trois répétitions finales, `MAX_LEN=64` limite les séquences et `BATCH=16` fixe le nombre de tweets par lot PLM. `TOKENIZERS` est un cache en mémoire.

**Résultat attendu à l’exécution.** Les variables de configuration sont définies ; cette cellule ne lance aucun entraînement.

**À retenir.** La présence d’un GPU Radeon intégré ne suffit pas à fournir CUDA. La longueur 64 est un choix de budget aligné sur le préentraînement Twitter d’AraBERT ; son effet sera mesuré par les taux de troncature, pas supposé négligeable.

**Optimisation CPU ajoutée.** Le périphérique est maintenant explicitement CPU. Intra-op=6, inter-op=1 ; le réglage inter-op demande un noyau neuf. Les autres budgets, graines et tailles de lots restent identiques.

```python
assert (DATA_DIR / 'training_data.csv').exists(), 'Conserver les deux CSV dans data/raw/.'

os.environ['USE_TF'] = '0'

os.environ['USE_FLAX'] = '0'

for folder in ['results','predictions','figures','models','data/processed','logs']:
    (ROOT / folder).mkdir(exist_ok=True, parents=True)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

TOKENIZERS = {}
TOKENIZED = {}  # Cache borné : quatre jeux de tenseurs CPU.
FROZEN_MODELS = {}  # Un seul encodeur gelé ; vidé avant tout autre chargement.

os.environ['HF_HUB_DISABLE_XET'] = '1'

torch.set_num_threads(CPU_THREADS)
if torch.get_num_interop_threads()!=1:
    try: torch.set_num_interop_threads(1)
    except RuntimeError: print('Pour appliquer inter-op=1, redémarrer le noyau avant Run All.')

DEVICE = torch.device('cpu')

SEEDS = [42, 43, 44]

MAX_LEN = 64

BATCH = 16
```

<a id="ancien-005"></a>

#### TP2_Sarcasme_Arabe.ipynb — variante de la cellule 005

**Rôle :** consulter l’explication de la [cellule principale 004](#tp2-004). Le bloc ci-dessous est celui de cette version ; il permet de vérifier précisément ce qui change.

**Rappel de la version principale actuelle, à comparer au code historique ci-dessous :** Le code courant fixe `STATIC_EPOCHS=8`, `PLM_EPOCHS=5`, `PATIENCE=2`. Ce sont des plafonds ; l'arrêt anticipé peut réduire les époques réalisées. `CACHE_CONTEXT` décrit notamment matériel, versions et batch ; il sert à refuser la réutilisation silencieuse d'un run incompatible. Les chemins de révision sont replacés sous la racine de cette session.

**Valeurs effectives dans cette variante :** `STATIC_EPOCHS, PLM_EPOCHS, PATIENCE = 8, 3, 2`. Ces valeurs, et non celles du rappel courant, gouvernent les appels de cette variante.

**Fiche présente dans cette version (repères historiques).**
**Objectif.** Fixer les règles communes de durée et empêcher la reprise de checkpoints provenant d’un autre contexte.

**Comprendre le code.** Les petits réseaux ont au maximum huit époques ; les PLM en ont trois ; `PATIENCE=2` autorise deux validations sans amélioration avant arrêt. `RUN_VERSION` identifie la logique expérimentale. `CACHE_CONTEXT` rassemble longueur, tailles de lots, versions et précision. Le manifeste `REVISIONS` précise les versions des quatre PLM ; leurs chemins sont recalculés à partir du dossier actuel.

**Résultat attendu à l’exécution.** Un contexte reproductible et un dictionnaire de ressources locales sont disponibles.

**À retenir.** Le cache économise des calculs, mais ne crée pas une nouvelle expérience. Une modification scientifique doit être accompagnée d’un changement de version et d’une réévaluation des caches concernés.

```python
STATIC_EPOCHS, PLM_EPOCHS, PATIENCE = 8, 3, 2

RUN_VERSION = 'tp2-v1'

CACHE_CONTEXT={'max_len':MAX_LEN,'batch_plm':BATCH,'batch_static':64,
    'torch':torch.__version__,'transformers':__import__('transformers').__version__,
    'precision':'float32','protocol':'conservative-tp2-v1'}

REVISIONS = json.loads((ROOT/'models/revisions.json').read_text()) if (ROOT/'models/revisions.json').exists() else {}

for model_name,entry in REVISIONS.items():
    entry['path']=str(ROOT/'models/hf'/model_name.replace('/','--'))
```

<a id="ancien-007"></a>

#### TP2_Sarcasme_Arabe.ipynb — variante de la cellule 007

**Rôle :** consulter l’explication de la [cellule principale 006](#tp2-006). Le bloc ci-dessous est celui de cette version ; il permet de vérifier précisément ce qui change.

**Rappel de la version principale actuelle, à comparer au code historique ci-dessous :** `sys.executable` révèle le Python du noyau réellement actif. Le code accepte Conda local ou Colab et affiche les informations GPU si CUDA est choisi. Les options pandas changent seulement l'affichage, pas les exemples utilisés ; voir une table tronquée n'est pas une réduction du dataset.

**Fiche présente dans cette version (repères historiques).**
**Objectif.** Appliquer la graine de référence et rendre l’environnement réellement utilisé visible dans les sorties.

**Comprendre le code.** `seed_all(42)` initialise cette session. Les options pandas permettent de lire des tweets assez longs et les options matplotlib harmonisent les graphiques. `sys.executable` affiche le véritable interpréteur du noyau, ce qui est plus fiable que le nom du terminal ouvert. L’assertion finale contrôle la présence du chemin Anaconda attendu.

**Résultat attendu à l’exécution.** Un récapitulatif indique Python, sa version, PyTorch, le périphérique et le nombre de threads.

**À retenir.** Vérifier ce récapitulatif avant les calculs longs. Un terminal Anaconda ouvert ailleurs ne force pas automatiquement le notebook à utiliser cet interpréteur.

```python
seed_all(42)

pd.set_option('display.max_colwidth', 220)

pd.set_option('display.max_rows', 65)

plt.rcParams.update({'figure.dpi':120, 'axes.spines.top':False, 'axes.spines.right':False})

print('Python :',sys.executable, '\nVersion :',sys.version, '\nPyTorch :',torch.__version__,
      '\nPériphérique :',DEVICE, '\nCœurs utilisés :',torch.get_num_threads())

assert 'anaconda3' in sys.executable.lower(), 'Le noyau demandé est celui de Conda.'
```

<a id="ancien-019"></a>

#### TP2_Sarcasme_Arabe.ipynb — variante de la cellule 019

**Rôle :** consulter l’explication de la [cellule principale 018](#tp2-018). Le bloc ci-dessous est celui de cette version ; il permet de vérifier précisément ce qui change.

**Rappel de la version principale actuelle, à comparer au code historique ci-dessous :** Les assertions testent des propriétés du nettoyage sur des exemples contrôlés. Les agrégations `size/sum/mean` signifient effectif, nombre de sarcastiques et proportion. Les données test étiquetées ne sont pas encore introduites dans ces descriptions supervisées.

**Fiche présente dans cette version (repères historiques).**
**Objectif.** Vérifier des propriétés essentielles du nettoyage avant d’interpréter les distributions.

**Comprendre le code.** Les assertions contrôlent la conservation de la négation لا, l’idempotence du nettoyage et le remplacement des liens/mentions. L’idempotence signifie que nettoyer une seconde fois ne doit pas continuer à déformer le texte. Pour train et val, `groupby` rapporte taille, nombre de sarcastiques et proportion par dialecte puis sentiment. Une figure vide est préparée pour la cellule suivante.

**Résultat attendu à l’exécution.** Les contrôles passent ou arrêtent l’exécution avec une assertion ; les tables décrivent les partitions de développement.

**À retenir.** Un taux calculé sur un très petit sous-groupe est instable. La colonne `size` compte autant que la colonne `mean` pour interpréter les différences entre dialectes.

```python
assert 'لا' in prepare_text('لَا أحب هذا')

assert prepare_text('مرحبا   بالعالم') == prepare_text(prepare_text('مرحبا   بالعالم'))

assert 'http' not in prepare_text(examples[0]) and '@ami' not in prepare_text(examples[0])

print('Contrôles du prétraitement et de séparation : OK.')

for name,frame in [('train',train),('val',val)]:
    print(name, 'n =',len(frame),'sarcasme =',round(frame.y.mean(),4))
    display(frame.groupby('dialect').y.agg(['size','sum','mean']))
    display(frame.groupby('sentiment').y.agg(['size','sum','mean']))

fig,ax=plt.subplots(figsize=(7,3))
```

<a id="ancien-020"></a>

#### TP2_Sarcasme_Arabe.ipynb — variante de la cellule 020

**Rôle :** consulter l’explication de la [cellule principale 019](#tp2-019). Le bloc ci-dessous est celui de cette version ; il permet de vérifier précisément ce qui change.

**Rappel de la version principale actuelle, à comparer au code historique ci-dessous :** Le graphique compare les taux de sarcasme par dialecte de train et val. Les effectifs doivent être lus avec les taux : 0 % sur cinq exemples ne représente pas la même information que 0 % sur des milliers. Création, remplissage et affichage sont réunis pour éviter des axes vides en sortie inline.

**Fiche présente dans cette version (repères historiques).**
**Objectif.** Lire rapidement les différences de proportion sarcastique dans le train nettoyé.

**Comprendre le code.** La moyenne de `y` est calculée pour chaque dialecte puis dessinée sur la figure préparée auparavant. Les étiquettes précisent qu’il s’agit d’une proportion, et non d’un nombre de tweets. `tight_layout` ajuste les marges ; `savefig` conserve une version PNG avant l’affichage.

**Résultat attendu à l’exécution.** Un diagramme en barres apparaît et `figures/prior_train.png` est créé dans le dépôt local.

**À retenir.** Ce graphique montre la distribution des annotations du corpus, pas un taux d’erreur du modèle ni une propriété universelle d’un dialecte. Revenir aux effectifs du tableau précédent pour juger sa stabilité.

```python
train.groupby('dialect').y.mean().plot.bar(ax=ax,color='#275D88')

ax.set(ylabel='Proportion sarcastique',title='Train nettoyé par dialecte',xlabel='Dialecte')

plt.tight_layout()

fig.savefig(ROOT/'figures/prior_train.png')

plt.show()
```

<a id="ancien-026"></a>

#### TP2_Sarcasme_Arabe.ipynb — variante de la cellule 026

**Rôle :** consulter l’explication de la [cellule principale 025](#tp2-025). Le bloc ci-dessous est celui de cette version ; il permet de vérifier précisément ce qui change.

**Rappel de la version principale actuelle, à comparer au code historique ci-dessous :** `ax.has_data()` ou des textes vérifient qu'au moins un axe possède du contenu. `display(fig)` affiche explicitement cette figure puis `plt.close(fig)` libère son état Matplotlib. Cela évite de laisser le backend choisir une autre figure courante ; le contrôle ne prouve toutefois pas que tous les sous-axes d'une figure sont sémantiquement complets.

**Fiche présente dans cette version (repères historiques).**
**Objectif.** Conserver les graphiques consultables dans le notebook et dans les livrables.

**Comprendre le code.** `savefig` ajuste la disposition, construit un chemin sous `ROOT/figures/`, enregistre le PNG et affiche la figure. `bbox_inches="tight"` réduit les marges superflues et limite les étiquettes coupées. Le nom est fourni par la cellule qui produit le graphique.

**Résultat attendu à l’exécution.** Une fonction utilitaire est définie. Aucun fichier graphique n’est créé tant qu’elle n’est pas appelée avec une figure.

**À retenir.** Une image sauvegardée n’apporte pas une mesure nouvelle : elle représente les tableaux ou calculs réellement produits. Une légende précise doit indiquer partition, métrique et seuil concernés.

```python
def savefig(fig,name):
    fig.tight_layout(); fig.savefig(ROOT/f'figures/{name}.png',bbox_inches='tight'); plt.show()
```

<a id="ancien-031"></a>

#### TP2_Sarcasme_Arabe.ipynb — variante de la cellule 031

**Rôle :** consulter l’explication de la [cellule principale 030](#tp2-030). Le bloc ci-dessous est celui de cette version ; il permet de vérifier précisément ce qui change.

**Rappel de la version principale actuelle, à comparer au code historique ci-dessous :** Cette cellule peut télécharger les poids nécessaires dans Colab, avant d'annoncer leurs noms. AraBERT et MARBERT sont les encodeurs de D/E ; mDeBERTa sert au NLI ; CAMeLBERT sert au sentiment. Le couple d'hypothèses conserve l'ordre sarcastique puis sérieux dans chaque langue.

**Fiche présente dans cette version (repères historiques).**
**Objectif.** Fixer les checkpoints et les formulations qui seront comparés sans entraînement sur le sarcasme.

**Comprendre le code.** Les noms identifient exactement NLI multilingue, CAMeLBERT sentiment, AraBERT Twitter et MARBERTv2. Le manifeste local doit être présent ; cette assertion évite une attente ou un téléchargement implicite. `PROMPTS` contient, en arabe/français/anglais, une hypothèse sarcastique puis une hypothèse sérieuse. Les chemins locaux des modèles sont recalculés après lecture des révisions.

**Résultat attendu à l’exécution.** Les trois paires d’hypothèses sont prêtes et les modèles sont associés à leurs ressources locales.

**À retenir.** L’ordre des deux hypothèses compte : la première représente ici le sarcasme. Le choix ultérieur du meilleur prompt se fera sur val ; aucune formulation n’est sélectionnée à partir du test.

```python
NLI_NAME='MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'

assert (ROOT/'models/assets_complete.json').exists(), 'Poids locaux manquants dans models/hf.'

REVISIONS.update(json.loads((ROOT/'models/revisions.json').read_text()))

SA_NAME='CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment'

ARA_NAME='aubmindlab/bert-base-arabertv02-twitter'

MAR_NAME='UBC-NLP/MARBERTv2'

PROMPTS={'ar':('هذه التغريدة ساخرة.','هذه التغريدة جادة.'),
         'fr':('Ce tweet est sarcastique.','Ce tweet est sérieux.'),
         'en':('This tweet is sarcastic.','This tweet is serious.')}
for model_name,entry in REVISIONS.items():
    entry['path']=str(ROOT/'models/hf'/model_name.replace('/','--'))
```

<a id="ancien-033"></a>

#### TP2_Sarcasme_Arabe.ipynb — variante de la cellule 033

**Rôle :** consulter l’explication de la [cellule principale 032](#tp2-032). Le bloc ci-dessous est celui de cette version ; il permet de vérifier précisément ce qui change.

**Rappel de la version principale actuelle, à comparer au code historique ci-dessous :** La signature d'inférence dépend des textes, révisions, longueur et fichier de prétraitement ; elle ne dépend pas seulement du nom `val`. Sur GPU, le contexte CUDA entre aussi dans la signature. Cette identité aide à éviter le mélange de scores issus de textes ou conventions différents.

**Fiche présente dans cette version (repères historiques).**
**Objectif.** Empêcher la réutilisation de scores calculés sur d’autres textes, prompts ou versions.

**Comprendre le code.** L’empreinte couvre la liste ordonnée des textes, un tag décrivant l’usage, la longueur maximale, la version logique, PyTorch, les révisions PLM et le hash du prétraitement exporté. Les textes sont ordonnés : une permutation doit aussi être détectée puisque les scores sauvegardés sont alignés ligne par ligne.

**Résultat attendu à l’exécution.** Un appel retourne une signature SHA256 pour une inférence précise.

**À retenir.** Le cache d’inférence est distinct du checkpoint d’apprentissage. Des poids identiques ne suffisent pas si le texte d’entrée, l’ordre ou la préparation a changé.

```python
def inference_signature(texts,tag):
    payload={'texts':list(texts),'tag':tag,'max_len':MAX_LEN,'run_version':RUN_VERSION,'torch':torch.__version__,
             'revisions':{k:v['revision'] for k,v in REVISIONS.items()},
             'preprocess_sha256':hashlib.sha256((ROOT/'src/preprocess.py').read_bytes()).hexdigest()}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
```

<a id="ancien-048"></a>

#### TP2_Sarcasme_Arabe.ipynb — variante de la cellule 048

**Rôle :** consulter l’explication de la [cellule principale 047](#tp2-047). Le bloc ci-dessous est celui de cette version ; il permet de vérifier précisément ce qui change.

**Rappel de la version principale actuelle, à comparer au code historique ci-dessous :** `eval()` désactive le dropout ; `inference_mode()` évite les gradients. Le regroupement par longueur réduit le padding PLM, mais `probs[index]=...` restaure l'ordre officiel avant métriques/export. Dans le multi-tâche, seul le premier tenseur de logits (sarcasme) fournit le score principal.

**Fiche présente dans cette version (repères historiques).**
**Objectif.** Convertir les logits d’un modèle en scores sans modifier ses poids.

**Comprendre le code.** `model.eval()` désactive les comportements d’apprentissage tels que le dropout ; `torch.inference_mode()` évite le suivi des gradients. Les entrées sont découpées en lots et envoyées sur le périphérique courant. Pour un modèle multi-tâche, seule la première sortie, sarcasme, est utilisée ici. Le softmax donne un score de classe 1 ; les poids attentionnels sont également collectés si demandés.

**Résultat attendu à l’exécution.** La fonction renvoie un vecteur de longueur nombre de tweets, ou une paire scores/attention.

**À retenir.** `eval()` et l’absence de gradients ont des rôles différents : le premier règle le comportement des couches, la seconde évite le coût de la rétropropagation. Aucun seuil n’est choisi à l’intérieur de cette fonction.

**Optimisation CPU ajoutée.** L’inférence PLM regroupe les longueurs similaires, puis réécrit chaque score à sa position d’origine. La taille des lots et tous les textes sont conservés ; C/D2 gardent leur parcours habituel. Des arrondis float32 peuvent varier selon la taille effective des calculs.

```python
def neural_predict(model,inputs,batch=64,attention=False):
    model.eval(); n=len(inputs[0]); probs=np.empty(n,dtype=np.float32); att=[]
    grouped=hasattr(model,'enc') and len(inputs)==2 and not attention
    order=np.argsort(-inputs[1].sum(1).cpu().numpy(),kind='stable') if grouped else np.arange(n)
    with torch.inference_mode():
        for start in range(0,n,batch):
            index=order[start:start+batch]
            selection=index if grouped else slice(start,start+batch)
            args=[x[selection].to(DEVICE) for x in inputs]
            out=model(*args,return_attention=True) if attention else model(*args)
            if attention: out,a=out; att.extend(a.cpu().numpy())
            if isinstance(out,tuple): out=out[0]
            probs[index]=out.softmax(-1)[:,1].cpu().numpy()
    return (probs,np.asarray(att)) if attention else probs
```

<a id="ancien-049"></a>

#### TP2_Sarcasme_Arabe.ipynb — variante de la cellule 049

**Rôle :** consulter l’explication de la [cellule principale 048](#tp2-048). Le bloc ci-dessous est celui de cette version ; il permet de vérifier précisément ce qui change.

**Rappel de la version principale actuelle, à comparer au code historique ci-dessous :** Cette cellule définit `optimizer_step` et `train_epoch` ; elle n'entraîne rien tant qu'elles ne sont pas appelées. En AMP, `unscale_` précède le clipping et le scheduler ne progresse pas après un overflow avec pas sauté. La perte affichée est une moyenne des pertes de lots, pas une métrique F1.

**Fiche présente dans cette version (repères historiques).**
**Objectif.** Réunir la mise à jour des poids effectuée sur tous les lots du train.

**Comprendre le code.** Chaque lot contient les entrées `xx`, la cible sarcasme `yy` et la cible sentiment `ss`. Les gradients précédents sont remis à zéro ; le modèle calcule ses logits ; la perte sarcasme est calculée. Si `lam` est non nul, la perte sentiment est ajoutée. `backward` calcule les gradients, le clipping borne leur norme à 1 et AdamW met à jour les paramètres. Le scheduler ajuste ensuite le taux d’apprentissage. Les messages de progression PLM sont espacés de 50 lots.

**Exemple ou repère.** En multi-tâche : $L=L_{sarc}+\lambda L_{sent}$. Les sentiments sont fournis à la perte, pas concaténés aux tokens du tweet.

**Résultat attendu à l’exécution.** Lors d’un appel, une époque réelle est parcourue et la moyenne des pertes de lots est renvoyée.

**À retenir.** Une perte train qui baisse ne garantit pas une meilleure généralisation. Les choix de checkpoint et d’arrêt se feront avec le F1 val dans la fonction suivante.

**Optimisation CPU ajoutée.** La somme des pertes est maintenue au fil des lots au lieu de refaire une moyenne de toute la liste à chaque message. Le backward, le clipping, AdamW, le scheduler et l’ordre des lots restent identiques. Aucun gain global d’apprentissage n’a été mesuré.

```python
def train_epoch(model,loader,opt,loss_fn,sent_loss,lam,scheduler,plm,name,seed,epoch):
    model.train(); total_loss=0.; steps=0
    for step,batch in enumerate(loader):
        *xx,yy,ss=[t.to(DEVICE) for t in batch]
        opt.zero_grad(set_to_none=True); out=model(*xx)
        logits=out[0] if isinstance(out,tuple) else out
        loss=loss_fn(logits,yy)
        if lam: loss=loss+lam*sent_loss(out[1],ss)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if scheduler: scheduler.step()
        total_loss+=loss.item(); steps+=1
        if plm and steps%50==0:
            print(name,seed,'époque',epoch+1,'batch',steps,'/',len(loader),'perte',round(total_loss/steps,4),flush=True)
    return total_loss/max(steps,1)
```

<a id="ancien-050"></a>

#### TP2_Sarcasme_Arabe.ipynb — variante de la cellule 050

**Rôle :** consulter l’explication de la [cellule principale 049](#tp2-049). Le bloc ci-dessous est celui de cette version ; il permet de vérifier précisément ce qui change.

**Rappel de la version principale actuelle, à comparer au code historique ci-dessous :** Pour `oversample`, `WeightedRandomSampler` tire avec remise des exemples du train avec poids inverse à l'effectif de leur classe. Sinon un PLM utilise `LengthGroupedSampler`, les autres un échantillonnage aléatoire. Les lots sont indexés de façon vectorisée, `num_workers=0` évite des processus supplémentaires et `pin_memory` est activé seulement pour CUDA.

**Fiche présente dans cette version (repères historiques).**
**Objectif.** Fournir les mêmes exemples d’apprentissage au modèle avec une stratégie de tirage explicite.

**Comprendre le code.** `TensorDataset` aligne entrées et cibles. En suréchantillonnage, `WeightedRandomSampler` tire avec remise selon l’inverse des fréquences de classes : certains tweets peuvent apparaître plusieurs fois, d’autres être absents d’une époque. Pour les PLM sans ce rééchantillonnage, `LengthGroupedSampler` rapproche des longueurs similaires afin de réduire le padding effectif, tout en conservant les exemples. Les lots valent 16 pour les PLM et 64 pour les autres réseaux ; `num_workers=0` limite les processus Windows.

**Résultat attendu à l’exécution.** Un `DataLoader` prêt à être parcouru par `train_epoch` est renvoyé.

**À retenir.** Le groupement par longueur n’est pas une suppression de données. La validation n’est jamais suréchantillonnée : elle garde sa distribution naturelle.

**Optimisation CPU ajoutée.** BatchSampler fournit maintenant des listes d’indices directement à TensorDataset : la lecture est vectorisée, sans collate Python tweet par tweet. L’ordre des lots et l’état RNG ont été comparés sur données synthétiques, y compris suréchantillonnage et regroupement par longueur. Workers=0 et pin_memory=False sont adaptés au CPU.

```python
def make_training_loader(xtrain,y,sy,seed,imbalance,plm):
    dataset=TensorDataset(*xtrain,y,sy)
    generator=torch.Generator().manual_seed(seed)
    if imbalance=='oversample':
        weights=(1/np.bincount(y.numpy()))[y.numpy()]
        sampler=WeightedRandomSampler(weights,len(y),replacement=True,generator=generator)
    elif plm:
        from transformers.trainer_pt_utils import LengthGroupedSampler
        sampler=LengthGroupedSampler(batch_size=BATCH,lengths=xtrain[1].sum(1).tolist(),generator=generator)
    else: sampler=RandomSampler(dataset,generator=generator)
    # TensorDataset accepte une liste d'indices : une extraction par tenseur, sans boucle Python par tweet.
    batches=BatchSampler(sampler,BATCH if plm else 64,drop_last=False)
    return DataLoader(dataset,batch_size=None,sampler=batches,generator=generator,num_workers=0,pin_memory=False)
```

<a id="ancien-051"></a>

#### TP2_Sarcasme_Arabe.ipynb — variante de la cellule 051

**Rôle :** consulter l’explication de la [cellule principale 050](#tp2-050). Le bloc ci-dessous est celui de cette version ; il permet de vérifier précisément ce qui change.

**Rappel de la version principale actuelle, à comparer au code historique ci-dessous :** Le choix de reprise valide signature, contexte et métadonnées ; sinon les anciens artefacts sont archivés avant un nouveau run. Chaque époque évalue val et son seuil ; seul un meilleur F1 sauvegarde le checkpoint pending. L'arrêt anticipé sélectionne la meilleure époque, pas nécessairement la dernière ; `val_mask` permet de sélectionner sans dialecte égyptien pour F2. Définir cette fonction n'est pas lancer les appels d'apprentissage qui suivent.

**Fiche présente dans cette version (repères historiques).**
**Objectif.** Appliquer une procédure commune et traçable aux réseaux C, D2 et E.

**Comprendre le code.** La fonction complète la configuration, enregistre les identifiants utilisés et vérifie la compatibilité d’un checkpoint terminé. Sinon, elle prépare AdamW, pertes et éventuel warmup PLM de 10 %, puis alterne époque train et inférence val. À chaque époque, le seuil est choisi sur la validation autorisée et le F1 décide si les poids sont sauvegardés. La patience déclenche un arrêt après absence de progrès. Pour AraVec, les embeddings sont dégelés après les deux premières époques. Le meilleur état est rechargé avant l’export.

**Résultat attendu à l’exécution.** Un appel renvoie la clé d’expérience et le meilleur modèle ; historique, probabilités et métadonnées sont écrits. Les métadonnées finales sont publiées par remplacement atomique.

**À retenir.** Les labels test ne participent ni à l’arrêt ni au checkpoint. Avec `val_mask`, seule une partie de val choisit le checkpoint, utile pour F2. Un run interrompu avant ses métadonnées finales n’est pas considéré terminé. Le meilleur checkpoint peut précéder la dernière époque parcourue.

**Optimisation CPU ajoutée.** Le vecteur val de l’époque ayant gagné est conservé, évitant un deuxième forward identique après le rechargement. Un cache signé permet les reprises sans nouvelle validation. Les poids sont écrits dans un fichier pending puis publiés en fin de run ; les anciens résultats authentifiés ne sont pas recalculés artificiellement.

```python
def train_neural(model,xtrain,xval,name,seed=42,config=None,epochs=STATIC_EPOCHS,
                 lr=1e-3,imbalance='weighted',labels=None,sentiments=None,lam=0,freeze_epochs=0,plm=False,val_mask=None):
    config=dict(config or {},seed=seed,epochs=epochs,lr=lr,imbalance=imbalance,lam=lam,
                train_ids=(config or {}).get('train_ids',train.id.tolist()))
    if plm: config.update(dynamic_padding=True,length_grouped=True)
    path=ROOT/f'models/{name}_seed{seed}.pt'; meta=path.with_suffix('.json'); sig=cache_signature(config)
    model=model.to(DEVICE)
    if plm and DEVICE.type!='cpu':
        config.update(device_type=DEVICE.type,precision='float32',batch=BATCH)
    p=None; best_path=path.with_suffix('.pending.pt')
    trained_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if freeze_epochs: trained_params+=model.embedding.weight.numel()
    selection_mask=np.ones(len(val),dtype=bool) if val_mask is None else np.asarray(val_mask)
    config['selection_val_ids']=val.id[selection_mask].tolist()
    sig=cache_signature(config)
    if path.exists() and compatible_metadata(meta) and json.loads(meta.read_text())['signature']==sig:
        payload=torch.load(path,map_location=DEVICE,weights_only=True); model.load_state_dict(payload); del payload
        saved=json.loads(meta.read_text()); seconds=saved['seconds']; history=saved['history']
    else:
        seed_all(seed); y=YTRAIN if labels is None else labels; sy=STRAIN if sentiments is None else sentiments
        loader=make_training_loader(xtrain,y,sy,seed,imbalance,plm)
        opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
        total=epochs*len(loader); scheduler=get_linear_schedule_with_warmup(opt,int(.1*total),total) if plm else None
        weights=torch.tensor(len(y)/(2*np.bincount(y.numpy())),dtype=torch.float32,device=DEVICE) if imbalance=='weighted' else None
        loss_fn=nn.CrossEntropyLoss(weight=weights); sent_loss=nn.CrossEntropyLoss()
        start=time.perf_counter(); best=-1; stale=0; history=[]; best_p=None
        for epoch in range(epochs):
            if freeze_epochs and epoch==freeze_epochs:
                model.embedding.weight.requires_grad_(True)
            mean_loss=train_epoch(model,loader,opt,loss_fn,sent_loss,lam,scheduler,plm,name,seed,epoch)
            p=neural_predict(model,xval,batch=BATCH if plm else 64)
            t=best_threshold(val.y[selection_mask],p[selection_mask]); f=float(f1_score(val.y[selection_mask],p[selection_mask]>=t,zero_division=0))
            history.append({'epoch':epoch+1,'loss':mean_loss,'val_f1':f,'threshold':t})
            print(name,seed,'époque',epoch+1,'perte',round(mean_loss,4),'F1 val',round(f,4),flush=True)
            if f>best+1e-6:
                best=f; stale=0; best_p=p.copy(); torch.save(model.state_dict(),best_path)
            else: stale+=1
            if stale>=PATIENCE and epoch>=freeze_epochs: break
        seconds=time.perf_counter()-start; best_path.replace(path); p=best_p
        draft=meta.with_suffix('.json.tmp')
        draft.write_text(json.dumps({'signature':sig,'seconds':seconds,'history':history,'config':config,'runtime_context':CACHE_CONTEXT,'checkpoint_stat':checkpoint_stat(path),
            'performance':{'threads':torch.get_num_threads(),'inter_op':torch.get_num_interop_threads(),'loader':'vectorized_batches_v1','precision':'float32'}},indent=2))
        draft.replace(meta)
        model.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=True))
    p=validation_scores(model,xval,path,sig,candidate=p,plm=plm)
    HISTORY.extend([dict(model=name,seed=seed,**h) for h in history])
    pd.DataFrame(HISTORY).to_csv(ROOT/'results/learning_curves.csv',index=False)
    key=register(name,seed,p,seconds,trained_params,config.get('preprocessing','conservative'),
                 'plm' if plm else config.get('kind','static'),path,config,
                 extra={'epochs_completed':len(history),'n_train':len(xtrain[0])})
    return key,model
```

<a id="ancien-055"></a>

#### TP2_Sarcasme_Arabe.ipynb — variante de la cellule 055

**Rôle :** consulter l’explication de la [cellule principale 054](#tp2-054). Le bloc ci-dessous est celui de cette version ; il permet de vérifier précisément ce qui change.

**Rappel de la version principale actuelle, à comparer au code historique ci-dessous :** Les tokens trouvés reçoivent leur vecteur AraVec ; les autres gardent une initialisation aléatoire contrôlée et la ligne PAD reste nulle. Le taux par types et le taux pondéré par occurrences train ont des dénominateurs différents. Charger le grand modèle AraVec et extraire sa matrice réduite ne met à jour aucun poids C.

**Fiche présente dans cette version (repères historiques).**
**Objectif.** Initialiser les embeddings connus depuis Word2Vec tout en donnant une représentation aux autres tokens.

**Comprendre le code.** Un cache au vocabulaire compatible est relu. Sinon, les vecteurs AraVec sont chargés, puis une matrice de forme `(taille_vocabulaire, 100)` est initialisée avec une graine fixe. PAD reste nul. Pour chaque token, `clean_str` fournit la clé de recherche ; un vecteur trouvé remplace l’initialisation aléatoire. La couverture est calculée en types distincts et en occurrences du train, puis les gros vecteurs sources sont libérés.

**Résultat attendu à l’exécution.** `ARAVEC` contient les seules lignes utiles au modèle et un tableau rapporte les couvertures sauvegardées.

**À retenir.** La couverture AraVec et l’OOV de `VOCAB` sont différents : un token peut être connu du train tout en n’ayant pas de vecteur externe. Une forte couverture en types ne garantit pas la même couverture des occurrences fréquentes.

```python
if matrix_path.exists() and matrix_meta.exists() and json.loads(matrix_meta.read_text())['vocab_signature']==vocab_signature:
    ARAVEC=np.load(matrix_path); coverage=json.loads((ROOT/'models/aravec_coverage.json').read_text())
else:
    import scipy.linalg
    # Compatibilité gensim 4.3 avec scipy récent, sans modifier le paquet installé.
    if not hasattr(scipy.linalg,'triu'): scipy.linalg.triu=np.triu
    from gensim.models import Word2Vec
    candidates=list((ROOT/'models/aravec').rglob('*.mdl'))
    assert candidates,'AraVec manquant dans models/aravec.'
    vectors=Word2Vec.load(str(candidates[0])).wv
    rng=np.random.default_rng(42); ARAVEC=rng.normal(0,.05,(len(VOCAB),100)).astype('float32'); ARAVEC[0]=0
    found=[]
    for token,i in VOCAB.items():
        normalized=clean_str(token)
        if i>1 and normalized in vectors: ARAVEC[i]=vectors[normalized]; found.append(token)
    coverage={'vocab_types':len(VOCAB)-2,'types_found':len(found),
        'type_coverage':len(found)/max(len(VOCAB)-2,1),
        'train_token_coverage':sum(counts[t] for t in found)/sum(counts.values())}
    np.save(matrix_path,ARAVEC); matrix_meta.write_text(json.dumps({'vocab_signature':vocab_signature})); (ROOT/'models/aravec_coverage.json').write_text(json.dumps(coverage,indent=2))
    del vectors; gc.collect()

display(pd.Series(coverage).to_frame('Couverture AraVec'))
```

<a id="ancien-086"></a>

#### TP2_Sarcasme_Arabe.ipynb — variante de la cellule 086

**Rôle :** consulter l’explication de la [cellule principale 085](#tp2-085). Le bloc ci-dessous est celui de cette version ; il permet de vérifier précisément ce qui change.

**Rappel de la version principale actuelle, à comparer au code historique ci-dessous :** Un CSV est écrit par modèle final avec `id,y_true,y_pred,probability`. B2 exporte sa probabilité sentiment et conserve sa décision argmax, qui ne doit pas être recalculée par un seuil de probabilité 0,5. Les tableaux de résultats sauvegardés dans le notebook ne sont pas ces CSV : ils restent sur le serveur d'exécution tant qu'ils ne sont pas téléchargés.

**Fiche présente dans cette version (repères historiques).**
**Objectif.** Produire des scores test vérifiables ligne par ligne par l’enseignant.

**Comprendre le code.** Chaque clé finale est évaluée avec son seuil fixé sur val. Le CSV exporte `id`, `y_true`, `y_pred` et `probability`. Pour B2, la décision reste la règle argmax négatif, mais la probabilité exportée est P(négatif) et l’AP utilise ce score continu. Le temps global de pipeline et la taille de l’artefact sont enregistrés ; les champs test sont ajoutés aux lignes de résultats.

**Résultat attendu à l’exécution.** Dix-sept CSV finaux et un tableau F1/macro-F1/AP test sont attendus après calcul. Une figure à deux panneaux est préparée pour les courbes suivantes.

**À retenir.** Le temps global peut inclure chargement et lecture de caches ; il ne remplace pas la latence CPU batch 1 mesurée en G. Les scores exportés servent à recalculer les métriques, pas à retoucher les décisions après coup.

```python
TEST_P={}

TEST_ROWS=[]

for key in FINAL_KEYS:
    rec=REGISTRY[key]; path=ROOT/f'predictions/{key}.csv'
    start=time.perf_counter(); p=predict_registered(rec,test.tweet.tolist(),'test'); elapsed=time.perf_counter()-start
    TEST_P[key]=p; pred=(p>=rec['threshold']).astype(int)
    export_p=sentiment_predict(test.tweet.tolist(),'test')[0] if rec['kind']=='sentiment' else p
    pd.DataFrame({'id':test.id,'y_true':test.y,'y_pred':pred,'probability':np.asarray(export_p,dtype=np.float64)}).to_csv(path,index=False)
    row=row_for_key(key); row.update({'test_'+k:v for k,v in scores(test.y,p,rec['threshold']).items()})
    # AP B2 doit utiliser la probabilité de sentiment, pas seulement la décision binaire.
    if rec['kind']=='sentiment': row['test_pr_auc_ap']=float(average_precision_score(test.y,sentiment_predict(test.tweet.tolist(),'test')[0]))
    row['test_inference_total_seconds']=elapsed; row['test_pipeline_ms_per_tweet']=1000*elapsed/len(test)
    row['artifact_size_mb']=(ROOT/rec['path']).stat().st_size/1e6 if (ROOT/rec['path']).is_file() else sum(f.stat().st_size for f in (ROOT/rec['path']).glob('*') if f.is_file())/1e6
    TEST_ROWS.append(row.copy()); pd.DataFrame(RESULTS).to_csv(ROOT/'results/results.csv',index=False)
    print(key,'F1 test',round(row['test_f1_sarcastique'],4),flush=True)

summary=pd.DataFrame(TEST_ROWS)

display(summary[['model','seed','threshold','test_f1_sarcastique','test_macro_f1','test_pr_auc_ap']])

fig,axes=plt.subplots(1,2,figsize=(11,4))
```

<a id="ancien-087"></a>

#### TP2_Sarcasme_Arabe.ipynb — variante de la cellule 087

**Rôle :** consulter l’explication de la [cellule principale 086](#tp2-086). Le bloc ci-dessous est celui de cette version ; il permet de vérifier précisément ce qui change.

**Rappel de la version principale actuelle, à comparer au code historique ci-dessous :** Les courbes F1(seuil) et précision-rappel utilisent les scores test du modèle déjà fixé. Les points val servent de référence visuelle ; l'optimum test est calculé seulement pour constater un décalage. Le code ne remplace pas automatiquement les seuils du registre par cet optimum.

**Fiche présente dans cette version (repères historiques).**
**Objectif.** Étudier comment un changement de distribution peut déplacer le compromis précision/rappel.

**Comprendre le code.** Pour la graine 42 de chaque famille, une grille de 201 seuils entre 0 et 1 fournit F1=f(seuil). Un point marque le seuil utilisable, déjà choisi sur val. La courbe PR est également construite. Le meilleur seuil calculé directement sur les labels test est stocké séparément avec son F1, sous l’étiquette posthoc.

**Résultat attendu à l’exécution.** Les courbes sont alimentées et `threshold_rows` compare seuil val et seuil test a posteriori.

**À retenir.** Le maximum test est une description optimiste obtenue en connaissant les réponses. Il n’est jamais autorisé à remplacer le seuil val pour annoncer la performance finale ou choisir le modèle de production.

```python
threshold_rows=[]

for family,keys in FINAL_GROUPS.items():
    key=keys[0]; p=TEST_P[key]; t=REGISTRY[key]['threshold']; grid=np.linspace(0,1,201)
    f=[f1_score(test.y,p>=g,zero_division=0) for g in grid]
    axes[0].plot(grid,f,label=family); axes[0].scatter([t],[f1_score(test.y,p>=t)],s=20)
    prec,rec,_=precision_recall_curve(test.y,p); axes[1].plot(rec,prec,label=family)
    post=best_threshold(test.y,p)
    threshold_rows.append({'famille':family,'seuil_val':t,'seuil_test_posthoc':post,
        'f1_test_seuil_val':scores(test.y,p,t)['f1_sarcastique'],'f1_test_posthoc':scores(test.y,p,post)['f1_sarcastique']})
```

<a id="ancien-088"></a>

#### TP2_Sarcasme_Arabe.ipynb — variante de la cellule 088

**Rôle :** consulter l’explication de la [cellule principale 087](#tp2-087). Le bloc ci-dessous est celui de cette version ; il permet de vérifier précisément ce qui change.

**Rappel de la version principale actuelle, à comparer au code historique ci-dessous :** Le tableau `threshold_rows` contient des colonnes qui distinguent seuil val et seuil test posthoc. Il ne faut pas annoncer le F1 posthoc comme performance d'un système dont le seuil aurait été fixé avant le test. Le CSV rend cette distinction consultable hors du noyau.

**Fiche présente dans cette version (repères historiques).**
**Objectif.** Rendre explicite la différence entre réglage utilisable et observation a posteriori.

**Comprendre le code.** La cellule ajoute titres, axes et légendes aux courbes précédentes. L’image `threshold_and_pr_test.png` et le tableau `threshold_shift.csv` sont sauvegardés, puis les valeurs sont affichées. La courbe précision-rappel expose plusieurs compromis sans en choisir un nouveau pour le modèle final.

**Résultat attendu à l’exécution.** Une figure à deux panneaux et un tableau lisible documentent le décalage de seuil.

**À retenir.** Un grand écart entre seuils peut signaler prior, calibration ou distribution différents. Il ne suffit pas à attribuer tout l’écart au seul prior, car vocabulaire, dialectes et autres caractéristiques peuvent aussi changer.

```python
axes[0].set(title='Test diagnostic a posteriori uniquement',xlabel='Seuil',ylabel='F1 sarcastique')

axes[0].legend()

axes[1].set(title='Courbes précision rappel test',xlabel='Rappel',ylabel='Précision')

axes[1].legend()

savefig(fig,'threshold_and_pr_test')

pd.DataFrame(threshold_rows).to_csv(ROOT/'results/threshold_shift.csv',index=False)

display(pd.DataFrame(threshold_rows))
```

<a id="ancien-102"></a>

#### TP2_Sarcasme_Arabe.ipynb — variante de la cellule 102

**Rôle :** consulter l’explication de la [cellule principale 101](#tp2-101). Le bloc ci-dessous est celui de cette version ; il permet de vérifier précisément ce qui change.

**Rappel de la version principale actuelle, à comparer au code historique ci-dessous :** La comparaison est exportée puis les trajectoires F1 val sont tracées par famille et graine. Une courbe par époque montre le choix du checkpoint et le surapprentissage possible ; elle ne correspond pas à une trajectoire F1 test. Une tête linéaire sans historique d'époques n'a pas à recevoir une courbe artificielle.

**Fiche présente dans cette version (repères historiques).**
**Objectif.** Associer les résultats finaux aux trajectoires de validation des modèles retenus.

**Comprendre le code.** La liste des lignes est convertie en table, exportée dans `final_comparison.csv` et affichée. L’historique rassemble ensuite les F1 val par époque pour C/D/E1/E2 et leurs graines. Une tête linéaire sans époques est identifiée explicitement. Le graphique des courbes est sauvegardé. La matrice de confusion du modèle déjà choisi est calculée sur ses prédictions test.

**Résultat attendu à l’exécution.** Un tableau de synthèse, quatre panneaux d’apprentissage et une matrice de confusion préparée pour affichage sont produits.

**À retenir.** Une courbe train/val n’est pas une preuve de robustesse externe. Le meilleur checkpoint peut être antérieur à la dernière époque ; lire les courbes avec les règles d’arrêt plutôt que supposer que plus d’époques améliorent toujours la qualité.

```python
comparison=pd.DataFrame(comparison)

comparison.to_csv(ROOT/'results/final_comparison.csv',index=False)

display(comparison.round(4))

history_frame=pd.DataFrame(HISTORY)

fig,axes=plt.subplots(2,2,figsize=(10,7))

for ax,family in zip(axes.flat,['C','D','E1','E2']):
    model_name=REGISTRY[FINAL_GROUPS[family][0]]['name']
    curves=history_frame.loc[history_frame.model.eq(model_name)]
    for seed,curve in curves.groupby('seed'):
        ax.plot(curve.epoch,curve.val_f1,marker='o',label=f'Graine {seed}')
    if len(curves): ax.legend(fontsize=8)
    else: ax.text(.5,.5,'Classifieur linéaire sans époques',ha='center',transform=ax.transAxes)
    ax.set(title=f'{family} — {model_name}',xlabel='Époque',ylabel='F1 validation')

savefig(fig,'learning_curves_final')

fig,ax=plt.subplots(figsize=(6,5))

ConfusionMatrixDisplay.from_predictions(ytest,chosen_pred,display_labels=['Non sarcastique','Sarcastique'],cmap='Blues',ax=ax,colorbar=False)
```

<a id="ancien-103"></a>

#### TP2_Sarcasme_Arabe.ipynb — variante de la cellule 103

**Rôle :** consulter l’explication de la [cellule principale 102](#tp2-102). Le bloc ci-dessous est celui de cette version ; il permet de vérifier précisément ce qui change.

**Rappel de la version principale actuelle, à comparer au code historique ci-dessous :** La matrice de confusion vient du modèle sélectionné et de son seuil val verrouillé. Le barplot utilise les moyennes test et l'écart-type des graines, pas des intervalles bootstrap. `decision` conserve la clé exacte et les métriques du checkpoint représentatif ; ces dernières peuvent différer de la moyenne de famille.

**Fiche présente dans cette version (repères historiques).**
**Objectif.** Rendre visibles les compromis des modèles au seuil déjà retenu.

**Comprendre le code.** La matrice reçoit son titre avec famille et seuil val, puis est sauvegardée. Le diagramme en barres montre la moyenne F1 test ; les barres d’erreur des familles entraînées représentent l’écart-type sur trois graines. Pour B1/B2, la valeur manquante est tracée sans barre, pas interprétée comme une variance connue nulle. Le dictionnaire `decision` réunit le choix fait sur val et ses métriques test.

**Résultat attendu à l’exécution.** Deux figures sont enregistrées et une décision traçable est préparée avec un horodatage de fin.

**À retenir.** La figure test expose le choix de validation ; elle ne doit pas conduire à déplacer après coup ce choix vers la barre la plus haute. Une recommandation réelle doit aussi considérer latence, taille et résultats F, notamment pour des tweets marocains.

```python
ax.set_title(f'Modèle retenu sur val : {chosen_final_family} — seuil {chosen_rec["threshold"]:.3f}')

savefig(fig,'chosen_confusion')

fig,ax=plt.subplots(figsize=(8,4))

ax.bar(comparison.family,comparison.test_f1_mean,yerr=comparison.test_f1_std.fillna(0),capsize=5,color='#275D88')

ax.set(ylabel='F1 sarcastique test',title='Entraînés : 3 graines ; zero-shot : 1 exécution',ylim=(0,1))

savefig(fig,'final_comparison')

decision={'chosen_key':CHOSEN_KEY,'family':chosen_final_family,'criterion':'moyenne F1 validation sur trois graines',
          'threshold':chosen_rec['threshold'],'test_metrics':scores(ytest,chosen_p,chosen_rec['threshold']),
          'completed_at_utc':datetime.now(timezone.utc).isoformat()}
```

<a id="diagnostic"></a>
## 9. Lire les résultats et diagnostiquer les problèmes

### 9.1 « Tous mes scores de phrases sont faibles »

Une phrase peut sembler clairement ironique à un humain et rester mal classée. Il faut distinguer quatre questions : le lecteur charge-t-il le checkpoint voulu ? Le prétraitement est-il cohérent ? Le score est-il correctement interprété et comparé au bon seuil ? Le modèle généralise-t-il aux formulations saisies ? Une mauvaise sortie ne permet pas de répondre automatiquement « non » à la première question.

Vérifier la clé et le seuil affichés, la présence des poids ajustés et les ressources de la même session. Tester des exemples de validation avec leurs probabilités sauvegardées peut diagnostiquer un chargement ; sélectionner quelques exemples extrêmes ne mesure pas la qualité réelle. Pour conclure sur la généralisation, il faut un nouveau lot annoté indépendant, suffisamment diversifié.

Les phrases de démonstration sont généralement plus courtes, sur d'autres sujets et parfois dans un autre dialecte que les tweets d'entraînement. Le sarcasme dépend du contexte, du contraste implicite et de connaissances absentes du texte. Un modèle qui exploite des mots de sentiment et raccourcis du corpus peut donc rater ces exemples malgré un F1 test non nul. **Ne pas déplacer le seuil pour réussir ces quatre phrases puis présenter cela comme une amélioration validée.**

### 9.2 « Le modèle a fait cinq époques, pourquoi n'est-il pas meilleur ? »

Plus d'époques peut diminuer la perte train tout en accentuant le surapprentissage. La validation décide du meilleur checkpoint ; la cinquième époque n'est pas forcément retenue. Pour comparer les deux versions, lire les mêmes agrégats sur les trois graines et conserver l'identité des seuils. Comparer uniquement la meilleure ligne test de chaque version serait une sélection supplémentaire sur test.

Le fichier du compte rendu a été construit précédemment depuis l'archive à trois époques. Le présent guide lit aussi les notebooks sauvegardés plus récemment ; leurs sorties peuvent donc différer. Il ne met pas à jour silencieusement les métriques, figures ou conclusions du rapport précédent.

### 9.3 Erreurs fréquentes

| Message / symptôme | Ce qu'il signifie | Action raisonnable |
|---|---|---|
| `ModuleNotFoundError: emoji` | Paquet absent du Python du noyau | Exécuter la préparation compatible puis redémarrer le bon noyau ; ne pas installer dans un autre Python local par erreur |
| `Installation terminée : redémarre le noyau` | Arrêt volontaire après installation réussie | Redémarrer puis reprendre depuis la première cellule |
| `Échec de pip` ou `CalledProcessError` | Installation réellement échouée | Lire les lignes `ERROR`, vérifier versions/Python du noyau avant la suite |
| `CSV manquant` / assertion de racine | Mauvais dossier d'exécution ou données non préparées | Lire `ROOT` affiché et vérifier les données sur le **serveur** courant |
| `Colab sans CUDA` | Serveur CPU sélectionné ou GPU indisponible | Sélectionner une session GPU T4 ; changer seulement son alias ne crée pas de GPU |
| `Aucun registre` / `Checkpoint absent` | Poids ajustés absents ou mauvais projet | Revenir à la session d'entraînement ou récupérer ensemble registre et checkpoints |
| Warning BERT pooler nouvellement initialisé | Poids inutilisés du pooler possibles | Vérifier l'architecture et le chargement ; ici la tête lit le CLS brut, donc ce warning seul n'explique pas les scores |
| Figure avec axes 0–1 et aucun contenu | Objet affiché avant son remplissage, ou aucune donnée tracée | Utiliser la cellule complète qui crée, remplit, affiche et sauvegarde la même figure ; ne pas inventer de points |
| La dernière cellule du testeur reste occupée | `input()` attend votre réponse | Entrer une phrase dans la saisie VS Code ; une ligne vide saute la prédiction |
| Les CSV existent dans Colab mais pas dans Windows | Stockages distincts | Télécharger les artefacts ; Ctrl+S sauvegarde le notebook, pas les sous-dossiers distants |

### 9.4 Ce que l'on peut expliquer à l'oral

1. « Je commence par une séparation fixe et un audit, pour empêcher qu'un même tweet apparaisse en apprentissage et évaluation. »
2. « Je garde une baseline TF-IDF peu coûteuse, puis je compare apprentissage statique, encodeur gelé et fine-tuning. »
3. « Je traite explicitement le déséquilibre et j'optimise les seuils sur validation, car l'accuracy majoritaire est trompeuse. »
4. « En multi-tâche, le sentiment est une seconde cible : il n'est jamais une information fournie à la prédiction de la phrase. »
5. « Je distingue score du corpus, incertitude, robustesse dialectale et réussite sur de nouvelles phrases ; mes essais montrent que ce dernier point reste une limite. »
6. « Je conserve le checkpoint et le seuil qui lui appartient ; changer de lambda ou de graine revient à changer de modèle. »

### 9.5 Glossaire rapide

| Terme | Sens dans ce projet |
|---|---|
| Token | Unité produite par la segmentation : mot/signe pour C ou sous-mot pour BERT |
| Embedding | Vecteur numérique associé à une unité ; statique au départ pour C, contextualisé par l'encodeur pour D/E |
| Logit | Score brut avant normalisation, qui peut être négatif ou supérieur à 1 |
| Forward | Calcul des sorties du réseau pour les entrées données |
| Backward | Calcul des gradients de la perte par rapport aux paramètres |
| Batch | Lot d'exemples traité lors d'un même passage |
| Époque | Parcours de l'ensemble des lots d'apprentissage définis |
| Learning rate | Taille contrôlant l'amplitude des pas de l'optimiseur |
| Weight decay | Régularisation des poids, distincte du poids de classe de la perte |
| Dropout | Désactivation aléatoire d'activations pendant train ; désactivée par eval |
| Checkpoint | Poids sauvegardés d'un état du modèle ; ne contient pas à lui seul toutes les ressources nécessaires |
| Early stopping | Arrêt après absence d'amélioration de la métrique val, patience 2 ici |
| Cache | Résultat antérieur réutilisable seulement si ses métadonnées restent compatibles |
| Gelé | Poids de l'encodeur non ajustés, mais forward toujours nécessaire pour produire ses états |
| Fine-tuning | Ajustement des couches utiles du modèle pré-entraîné sur la tâche |
| Seuil | Valeur transformant un score en décision ; différente de lambda |
| Lambda | Coefficient de la perte sentiment dans E2 ; différent du learning rate |
| Prior | Proportion de la classe sarcastique dans une population |
| OOV | Token hors du vocabulaire appris ; dépend de la tokenisation et du dénominateur |
| Calibration | Accord entre un score probabiliste et la fréquence réelle des événements ; non garantie par softmax |


<a id="couverture"></a>

## 10. Vérification de la couverture du guide

Inventaire des fichiers locaux lus pour produire ce document. « Exécutées » décrit les compteurs sauvegardés au moment de la lecture ; aucune cellule n’a été exécutée pour générer le guide. Le hash porte sur la concaténation exacte des sources des cellules de code, **pas sur les sorties**.

| Notebook | Cellules de code couvertes | Avec compteur sauvegardé | Erreurs enregistrées |
|---|---|---|---|
| `TP2_Sarcasme_Arabe_T4.ipynb` | 104 | 104 | 0 |
| `Tester_Phrases.ipynb` | 6 | 6 | 0 |
| `TP2_Sarcasme_Arabe_T4_Execution.ipynb` | 104 | 104 | 0 |
| `TP2_Sarcasme_Arabe.ipynb` | 105 | 3 | 1 |


### Empreintes de code utilisées

```text
TP2_Sarcasme_Arabe_T4.ipynb
  SHA256 fbecae8439222c0458fb09e78e072e9d27a2af61586377afebe207cbd0571755
Tester_Phrases.ipynb
  SHA256 ac028d2bcd2a168a9b750c33cab8933cc4255824b355a8e60cad1470f7531114
TP2_Sarcasme_Arabe_T4_Execution.ipynb
  SHA256 f5cd3b07cb2059bd07fb8ffc03dfecca2b39c4fa8f71ff7438c32a757a5ba4f3
TP2_Sarcasme_Arabe.ipynb
  SHA256 1b9afa15b7d3efef6fbafa1b3c80867668cee890a9bdc8dd27789560009b7a35
```

Les notes sont relues contre le code courant. Les extraits d’archives base64 sont les seules lignes abrégées pour la lisibilité ; les calculs, architectures et appels d’expérience restent visibles. Les cellules des variantes sans différence renvoient aux fiches principales, au lieu de dupliquer une explication identique.

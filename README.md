# TP2 — Détection du sarcasme en arabe

**Travail individuel : Omar Elkhali — Filière BDIOT.**

Dépôt public : **https://github.com/OmarElkhali/TP2-ArSarcasm-v2**.

## Consulter le compte rendu et les résultats

- [Compte rendu PDF](Compte_Rendu_TP2.pdf) : **six pages de corps + quatre pages d'annexes**, réponses Q1–Q12, comparaison des modèles, robustesse, incertitude et essais sur phrases.
- [Notebook T4 exécuté, avec ses sorties](notebooks/TP2_Sarcasme_Arabe_T4_Execution.ipynb) : copie préservée de l'exécution **104/104 cellules, sans erreur enregistrée**, avec **trois époques PLM maximum** (plage du TP : 3–5).
- [Notebook de travail](notebooks/TP2_Sarcasme_Arabe_T4.ipynb) : code courant, plafond **cinq époques** ; les anciens scores ne sont pas attribués à ces nouveaux réglages non exécutés.
- [Testeur de phrases](notebooks/Tester_Phrases.ipynb) : utilise les checkpoints entraînés de votre session. Ses sorties locales ne sont pas enregistrées ; les valeurs de la capture fournie sont transcrites dans [phrases_capture.csv](results/report_data/phrases_capture.csv), avec leur provenance.
- [Tous les tableaux extraits](results/report_data/) et [version texte du rapport](results/compte_rendu.md). Les 24 lignes de `results/results.csv` sont l'historique **CPU/validation**, séparé des [17 évaluations test T4](results/execution_t4_precedente_evaluation_test.csv).

Le meilleur E2 AraBERT atteint **F1 test moyen 0,6096 ± 0,0131** sur trois graines. Les deux phrases ironiques de la démonstration sont pourtant classées non sarcastiques. Le rapport expose cette limite, sans présenter les quatre phrases comme un nouveau jeu de test ni garantir un gain du multi-tâche.

**Pièces encore nécessaires à une remise entièrement vérifiable :** récupérer depuis Colab les **17 CSV de prédictions test individuelles**, absents de cette copie locale, et finaliser la relecture contextuelle des 30 erreurs (catégories actuellement provisoires). Les sorties sauvegardées attestent les expériences ; elles ne remplacent pas ces exports pour recalculer les scores.

Ce dépôt reprend la structure « Livrables & reproductibilité » du TP. Les données officielles et poids sont exclus de Git ; les scripts de préparation les téléchargent et vérifient leurs manifestes. Les checkpoints entraînés doivent être sauvegardés séparément.

```text
TP2 ARABE SARCASME LLM/
├── .git/
├── README.md
├── requirements.txt
├── Compte_Rendu_TP2.pdf
├── src/
│   ├── data.py
│   ├── preprocess.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── report.py
│   ├── sync.py
│   └── assets/
├── notebooks/
│   └── TP2_Sarcasme_Arabe_T4.ipynb
├── tests/
├── data/
│   ├── raw/                    # CSV officiels et manifeste SHA256
│   └── processed/              # Partitions après exclusions
├── models/                     # PLM, AraVec, checkpoints et caches
├── results/
│   ├── results.csv
│   ├── matrice_conformite.md
│   └── history/                # Sauvegardes et checkpoint partiel interrompu
├── predictions/
├── figures/
└── logs/
```

## Exécution dans VS Code

Ouvrir **la racine de ce dépôt**, puis [notebooks/TP2_Sarcasme_Arabe_T4.ipynb](notebooks/TP2_Sarcasme_Arabe_T4.ipynb). Fermer les anciens onglets nommés `TP2_Sarcasme_Arabe.ipynb` pour ne pas réenregistrer l'ancien code. Les sorties antérieures sont préservées dans `results/history/`. En local sélectionner **Python Conda TP2 (base Anaconda)** ; dans Colab sélectionner le **Python 3 du serveur GPU T4**. Redémarrer le noyau avant votre exécution. Le notebook accepte un dossier courant à la racine ou dans `notebooks/`.

Les bibliothèques proviennent du Conda existant. Les données, tokenizers et poids sont déjà locaux ; aucun autre dossier de projet n'est nécessaire. Les poids Hugging Face sont chargés avec `local_files_only=True`. Les checkpoints et les chemins enregistrés sont relatifs au dépôt.

Avant votre exécution, contrôles légers sans entraînement :

```powershell
python -m tests.verify_layout
python -m tests.check_cpu_optimisations
```

Le notebook garde les **104 cellules de code avec leurs explications détaillées**. Tout le code pédagogique et les expériences A–G y sont visibles ; `src.sync` exporte les fonctions dans les modules demandés par le professeur.

Si le noyau manque dans VS Code, l'enregistrer depuis l'environnement Conda actif :

```powershell
python -m ipykernel install --user --name tp2-conda --display-name 'Python Conda TP2 (base Anaconda)'
```

## Exécution avec la T4 Colab dans VS Code

Dans le sélecteur de noyau, choisir **Colab → New Colab Server → GPU → T4**, puis **Latest** et **Python 3**. Éviter Auto Connect s'il reprend votre ancienne session CPU. Le nom donné au serveur est une étiquette ; la cellule de diagnostic doit afficher un GPU réel. [Guide officiel](https://github.com/googlecolab/colab-vscode/wiki/User-Guide).

Le notebook utilise maintenant CUDA quand il est disponible. **Dans Colab, l'absence de CUDA arrête le notebook** ; aucun entraînement ne démarre silencieusement sur CPU. Le CPU Conda local reste utilisable. Sur GPU, le batch PLM vaut **32** et les fine-tunings utilisent **AMP float16 + GradScaler** ; C, les têtes D, le zero-shot et l'extraction gelée restent en float32. Les poids maîtres restent float32. G mesure toujours la latence sur le CPU de la machine d'exécution, donc sur le CPU Colab si vous faites G dans Colab.

Le serveur Colab ne voit pas votre disque Windows. **La version T4 peut maintenant démarrer sans transférer le dossier local** : un petit paquet intégré fournit les scripts de préparation et les manifestes, sans données ni poids. Après la préparation des paquets Python, la cellule des chemins télécharge les deux CSV officiels et vérifie leurs SHA256. Avant B, les quatre PLM sont téléchargés aux révisions exactes et leurs poids sont vérifiés ; AraVec est préparé à son utilisation en C. Aucun label test n'est consulté par cette préparation. Les téléchargements volumineux peuvent prendre plusieurs minutes ; ils ne sont pas des entraînements.

Dans une session vide, la racine distante devient **`/content/TP2_ArSarcasme`**. Les ressources vérifiées sont réutilisées dans cette session. Pour retrouver une ancienne copie persistante, définir explicitement `TP2_ROOT` avant les imports. Le code de préparation est lisible dans `src/colab_resources.py` ; l'archive intégrée ne contient que des sources et manifestes, avec contrôle SHA256 et des chemins d'extraction bornés au projet.

Une autre possibilité est de transférer le dossier complet avec **Upload to Colab**, puis de travailler dans ce dossier distant. Cela permet de récupérer vos checkpoints antérieurs plutôt que de repartir des poids préentraînés. Les signatures vérifient si les caches CPU/GPU sont compatibles.

Depuis un **terminal Colab**, dans le dossier du projet distant (adapter ce chemin à l'emplacement transféré) :

```bash
cd "/content/TP2 ARABE SARCASME LLM"
python -m pip install -r requirements-colab.txt
# Seulement si les données et poids n'ont pas été transférés :
python -m src.assets.prepare
python -m src.assets.aravec
```

`requirements-colab.txt` complète Colab sans demander de remplacement de son PyTorch CUDA ni de ses bibliothèques numériques préinstallées. Il utilise Gensim 4.4.0 pour les runtimes récents ; l'environnement local reste inchangé. Les versions effectives sont enregistrées à la fin du notebook. **Redémarrer le noyau après installation**, puis utiliser le notebook dans `notebooks/`.

La première cellule prépare aussi automatiquement ces compléments dans Colab, avant les imports. Elle détecte les paquets absents ou les versions différentes, puis lance pip avec le Python du noyau courant. Après une installation, son message **« Installation terminée : redémarre le noyau Colab »** est un arrêt volontaire : redémarrer, puis relancer depuis la première cellule. Si tout est déjà conforme, elle poursuit sans réinstallation. La liste est intégrée à cette cellule et contrôlée par un test contre `requirements-colab.txt`, afin de fonctionner même avant le transfert de ce fichier. Aucune installation automatique n'est faite dans Conda local.

Pour le runtime Python 3.13, les compléments utilisent **safetensors 0.5.3** et **SentencePiece 0.2.1**, dont les distributions binaires Linux sont compatibles avec ce Python. Transformers et les modèles du TP restent inchangés. La cellule affiche désormais toute la sortie de pip et son code de retour en cas d'échec ; copier ses lignes `ERROR:` pour diagnostiquer un autre problème. La disponibilité des distributions est vérifiée, mais l'installation dans votre session Colab doit encore être exécutée par vous. Sources : [safetensors sur PyPI](https://pypi.org/project/safetensors/0.5.3/), [SentencePiece sur PyPI](https://pypi.org/project/sentencepiece/0.2.1/).

**Si l'ancienne erreur revient avec `check_call`, `safetensors==0.4.3` ou `sentencepiece==0.2.0`**, l'éditeur exécute encore l'ancienne cellule. Après une correction externe, fermer l'onglet sans enregistrer sa copie obsolète, puis rouvrir le fichier depuis l'explorateur du projet. La première cellule corrigée affiche `TP2_COLAB_BOOTSTRAP_V2`. Redémarrer uniquement le noyau ne recharge pas le code de l'éditeur. Les sorties de l'ancienne version ont été conservées dans `results/history/` avant correction.

À la racine distante ou dans son sous-dossier `notebooks/`, les chemins fonctionnent directement. Pour désigner une copie particulière, définir avant les imports la variable d'environnement `TP2_ROOT` avec son chemin exact. Le notebook affiche la racine réellement utilisée, afin d'éviter de confondre disque local et stockage Colab.

Les anciens checkpoints CPU ne sont pas considérés comme des runs CUDA/AMP terminés : précision, batch et matériel participent à la compatibilité des caches. Le passage à CUDA/FP16 peut modifier les arrondis et les scores ; il faut mesurer les résultats GPU et conserver la copie CPU locale. Aucun entraînement T4 n'a été testé par l'assistant.

Les sorties des cellules apparaissent dans votre notebook VS Code : **Ctrl+S** les enregistre localement. Les CSV, figures et poids restent sur Colab. Sauvegarder les checkpoints sur un stockage persistant pour la reprise après perte de session. Pour récupérer les sorties, on peut créer une archive depuis le terminal Colab :

```bash
python -m zipfile -c /content/TP2_sorties.zip results predictions figures
```

Télécharger cette archive depuis la vue Colab. Les poids `.pt` et leurs métadonnées dans `models/` se récupèrent séparément si vous souhaitez reprendre les calculs localement. Les limites de session et la disponibilité GPU ne sont pas garanties par le projet.

Contrôle supplémentaire local, sans GPU, sans backward et sans entraînement : `python -m tests.check_gpu_config`. Il simule la sélection CUDA, vérifie l'ordre AMP/clipping et le scheduler avec des doublures. Il ne remplace pas un essai sur une vraie T4.

## Réglages pour rechercher de meilleurs scores

Le notebook principal prépare un **plafond de 5 époques PLM**, dans la plage 3–5 du TP, avec patience 2 et sélection du meilleur checkpoint sur validation. AdamW utilise uniformément le **weight decay 0,01**, sans exception pour les biais ou normalisations. Le réglage T4 conserve batch 32, lr 2e-5, warmup 10 % et FP16. Ces réglages ne garantissent pas un gain : les prochaines métriques doivent être mesurées. Architectures, données, pertes, trois lambda, graines et expériences obligatoires restent identiques. Les anciens checkpoints incompatibles et leurs prédictions sont archivés dans `results/history/` avant remplacement. Les signatures enregistrent la politique d'optimisation et le budget.

La précédente exécution T4 complète et ses 104 cellules exécutées sont conservées dans le dossier indiqué par `results/quality_and_plots_changes.json`. Les sorties de la version modifiée ont été retirées pour ne pas attribuer ces anciens scores aux nouveaux réglages. Les résultats historiques de `results/results.csv` restent inchangés.

## Tester les modèles avec des phrases

Ouvrir [notebooks/Tester_Phrases.ipynb](notebooks/Tester_Phrases.ipynb), puis choisir le Conda local ou le **même serveur Colab** que celui qui contient les checkpoints. Exécuter les cellules de préparation, choisir une clé dans `MODELE`, écrire les phrases dans `PHRASES` (une par ligne), puis lancer la cellule de prédiction. La dernière cellule permet aussi de saisir une phrase avec `input()`. Une nouvelle phrase ne déclenche aucun entraînement et réutilise le modèle chargé.

Le lecteur [src/predict.py](src/predict.py) couvre TF-IDF, C statiques, D gelés, PLM mono/multitâche, NLI et sentiment. Il reprend vocabulaire, prétraitement et seuil du checkpoint choisi ; aucun seuil n'est réglé sur les phrases saisies. B2 conserve sa décision par argmax du sentiment négatif, indépendamment de la probabilité continue affichée. Les scores ne sont pas des probabilités de justesse garanties.

Le notebook de saisie contient un petit paquet de sources pour Colab, sans données ni poids. Il faut disposer du registre et des checkpoints **entraînés**, pas seulement des poids préentraînés. Sur une nouvelle session, récupérer `results/registry.json`, `results/results.csv`, les fichiers `.pt` et `.json` ou `.joblib` dans `models/`, `models/vocab.json` pour C et les ressources `models/hf/` nécessaires pour B/D/E. Conserver ces chemins sous la même racine, ou définir `TP2_ROOT` explicitement.

## Correction des graphiques vides

Les figures du prior, des seuils/PR et de la matrice de confusion étaient réparties entre plusieurs cellules. Le backend inline pouvait afficher puis fermer des axes encore vides en fin de cellule. La création, le remplissage et l'affichage sont maintenant regroupés. `savefig` affiche explicitement l'objet figure avec `display(fig)` puis le ferme, et refuse une figure entièrement vide. Voir le fonctionnement du [backend inline Matplotlib](https://matplotlib.org/stable/users/explain/figure/interactive.html).

Si le noyau de l'exécution précédente contient encore les variables, **aucun entraînement n'est nécessaire pour réafficher les graphes** : recharger le code corrigé depuis le disque puis exécuter seulement 019, 025, 067, 086–087 et 101–102 (numéros des fiches de code), selon les figures voulues. Sinon, récupérer les PNG déjà produits dans `figures/` sur Colab. Un noyau vide ne peut pas reconstruire une courbe PR réelle sans les prédictions et labels de l'évaluation ; ne pas les remplacer par des données simulées.

Vérification légère : `python -m tests.check_quality_and_phrases`. Le contrôle utilise une vraie inférence TF-IDF locale, des doublures pour les autres familles et des données synthétiques pour les figures. Il ne lance ni PLM préentraîné, ni backward, ni entraînement. Les tests de configuration GPU, des chemins et du paquet Colab restent disponibles.

## Conformité scientifique et calcul

Architectures conservées : C1 BiLSTM avec max-pooling masqué ; C2 BiLSTM avec attention implémentée ; C3 Transformer à deux couches, dimension 128, quatre têtes, positions sinusoïdales et ablation sans positions ; D1 AraBERT gelé CLS/mean + régression logistique ; D2 états gelés + BiLSTM/attention ; E1/E2 AraBERT et MARBERT mono/multi-tâche. Les embeddings aléatoires/AraVec, trois lambda et trois graines finales restent prévus, ainsi que les tests F1–F6 et la comparaison G. Voir [matrice de conformité](results/matrice_conformite.md).

Machine locale : **Ryzen 5 5600G, 16 Go installés**, CPU float32, six threads intra-op, inter-op 1, batch PLM 16. Sur T4 : batch PLM 32 et AMP float16 pour E/F, comme point de départ GPU du sujet. Architectures, pertes, datasets et expériences sont conservés. Aucun remplacement par un modèle plus petit ni réduction des exemples. Lots vectorisés, caches et padding dynamique évitent des calculs inutiles.

Les **24 résultats de validation CPU historiques** et les caches compatibles sont conservés. L'exécution T4 complète est publiée séparément avec ses sorties ; le notebook de travail modifié n'est pas présenté comme exécuté intégralement. Le compte rendu distingue ces versions et les exports test encore absents localement. Le checkpoint CPU interrompu a été conservé localement dans `results/history/interrupted_kernel/`, sans être déclaré terminé.

Les anciens budgets CPU ne décrivent pas la durée de l'exécution T4. Les temps d'ajustement mesurés sont dans le rapport ; ils excluent téléchargements, extraction gelée et préparation, et ne constituent pas un temps total. Le gain de 1,86× du DataLoader concerne seulement un microbenchmark synthétique. Détails historiques : [optimisation CPU](results/optimisation_cpu.md).

## Après votre exécution

Sauvegarder le notebook, puis depuis la racine :

```powershell
python -m src.report
python -m tests.verify_final
```

Le générateur reconstruit le PDF depuis **le notebook T4 exécuté préservé**, l'historique CPU et la transcription de capture, sans charger de modèle ni entraîner. Il extrait tous les tableaux dans `results/report_data/`. La lecture HTML utilise `lxml` (figé dans les dépendances Conda). Le contrôle final refuse de déclarer le TP terminé tant que les sorties obligatoires manquent ; il est attendu qu'il refuse la copie locale dépourvue des CSV test. Le PDF comprend six pages de corps et quatre annexes. Une nouvelle exécution devra être sauvegardée et identifiée avant d'actualiser sa provenance dans le rapport.

Après une modification du code du notebook : `python -m src.sync` réexporte ses fonctions sans exécuter les cellules.

## Reproductibilité et Git

`requirements.txt` fige les versions. Split officiel, graine 42 et stratification sont conservés ; transformations apprises sur train uniquement ; test scellé jusqu'en F ; seuils et choix faits sur validation. Les chemins des prédictions finales sont `predictions/<modèle>_seed<k>.csv`.

Le dépôt est public sous [OmarElkhali/TP2-ArSarcasm-v2](https://github.com/OmarElkhali/TP2-ArSarcasm-v2). Données et poids volumineux sont exclus du suivi Git. Pour préparer une nouvelle installation, ces commandes téléchargent les ressources officielles, sans entraînement :

```powershell
python -m src.assets.prepare
python -m src.assets.aravec
```

Identité, filière et lien Git sont renseignés. Les 17 exports test et la relecture humaine restent à compléter avant une remise entièrement vérifiable. Aucun entraînement ni mise à niveau de Conda n'a été effectué pour produire le rapport.

# Optimisation CPU du TP2 — Ryzen 5 5600G

Ce rapport décrit les mesures de l'intervention CPU antérieure. Le notebook accepte désormais aussi CUDA/AMP sur Colab T4 ; les réglages GPU, distincts des benchmarks CPU ci-dessous, figurent dans le README et `gpu_adaptation_audit.json`. Aucun temps d'entraînement T4 n'a été mesuré.

Vérification du 17 octobre 2026. Le notebook est dans notebooks/ ; les ressources et sorties suivent la structure officielle du dépôt TP. **Aucun entraînement, backward, pas d'optimiseur ou Run All n'a été exécuté.** Les tests n'ont chargé aucun poids de PLM préentraîné. Les données complètes n'ont servi qu'à compter des longueurs dans des masques déjà disponibles ; aucune nouvelle inférence de corpus n'a été effectuée.

## Machine et environnement conservés

AMD Ryzen 5 5600G : six cœurs physiques, douze threads logiques, AVX2 ; 16 Go de RAM installés, environ 13,9 Gio utilisables après réservation matérielle. Calcul exclusivement CPU float32. Anaconda existant : Python 3.11.5, PyTorch 2.6.0+cpu, Transformers 4.48.0, oneDNN actif. Aucune mise à niveau de paquet ni installation de compilateur.

La skill Ponytail installée a été appliquée : privilégier les mécanismes natifs existants et limiter les caches en mémoire. La recherche dans les skills locales n'a pas trouvé de skill spécialisée PyTorch/Transformers. Les décisions techniques reposent sur les sources officielles et le code des versions réellement installées.

## Modifications appliquées

| Poste | Modification | Portée et contrôle |
|---|---|---|
| Threads | `OMP_NUM_THREADS=6`, `MKL_NUM_THREADS=6` avant les imports numériques ; intra-op 6, inter-op 1 | Évite la sursouscription. Redémarrer le noyau avant Run All si inter-op a déjà été initialisé. |
| DataLoader | `BatchSampler` et `TensorDataset` indexé par listes, `batch_size=None` | Une extraction par tenseur remplace les appels Python par tweet et leur collation. Les quatre stratégies conservent exactement les lots, l'ordre et l'état du générateur sur deux époques synthétiques. |
| Processus | `num_workers=0`, `pin_memory=False` | Les données sont déjà des tenseurs en RAM ; pas de processus Windows supplémentaires ni de copies destinées au transfert CUDA. Aucun gain multi-processus n'est revendiqué. |
| Tokenisation | Réutilisation des tokenizers ; cache des tenseurs sur disque et FIFO de quatre entrées en RAM | Signature des textes, modèle, prétraitement, versions et longueur ; mêmes IDs et masques que la tokenisation précédente. Cache réutilisé pour grilles et graines. |
| Validation | Conservation des probabilités de la meilleure époque ; cache lié au checkpoint et à la signature du run | Supprime la seconde validation après restauration du meilleur checkpoint et sa répétition lors de la reprise d'un run terminé. Contrôle SHA256 des probabilités et identité du checkpoint par taille/date nanoseconde. |
| Inférence PLM | Tri stable par longueur, lots tronqués au plus long texte du lot, remise dans l'ordre initial | Tous les tokens valides et masques sont conservés. Aucun seuil ni ordre des exemples exportés n'est modifié. |
| Extraction gelée D | Même tri et padding dynamique ; un seul encodeur gelé réutilisé entre extractions | Le cache fixe N×64×768 reste compatible ; positions hors masque mises à zéro. Un autre chargement PLM libère ce cache d'encodeur. |
| Boucle | Accumulation scalaire des pertes, F1 seul pour sélectionner l'époque | Évite listes et métriques redondantes ; ordre backward, clipping, AdamW et scheduler conservé. |
| Checkpoints | Meilleur état écrit dans un fichier provisoire puis remplacé à la fin du run | Les runs terminés restent disponibles si un nouvel essai est interrompu. Un run partiel n'est pas déclaré terminé. |

Le regroupement par longueur et le padding dynamique **étaient déjà présents pour l'entraînement E**. Ils ne sont pas présentés comme un nouveau gain de cette intervention. `inference_mode`, `zero_grad(set_to_none=True)` et le chargement local des poids étaient aussi déjà utilisés. Les poids des modèles entraînables restent initialisés séparément pour chaque expérience : partager un encodeur fine-tuné entre graines changerait le protocole.

## Mesures légères réelles

Source reproductible : `python -m tests.bench_cpu_light`. Résultats bruts : [cpu_microbench.json](cpu_microbench.json). Mesures courtes et sensibles à la charge système, réalisées sur des tenseurs synthétiques, sans backward. Une couche BERT aléatoire de dimension 768, 12 têtes, intermédiaire 3072, batch 16 ; longueurs 24 et 48, deux passages en ordre inversé.

| Threads intra-op | Moyenne du forward d'une couche, ms |
|---:|---:|
| 1 | 90,505 |
| 2 | 53,268 |
| 3 | 55,401 |
| 4 | 42,097 |
| 6 | 31,379 |
| 12 | 28,517 |

Douze threads sont les plus rapides sur ce proxy, environ 9,1 % devant six. Le réglage par défaut reste **six**, déjà utilisé par les entraînements historiques : le gain du proxy est limité, varie selon la longueur, et aucune mesure de backward ne permet de conclure que douze accélèrent l'entraînement. Ce choix ne signifie pas que six sont toujours les plus rapides. Aucun autotuning ne relance de modèle à l'ouverture du notebook.

Pour parcourir 4 096 exemples synthétiques, le DataLoader précédent prend 50,057 ms et le nouveau 26,857 ms : **accélération de 1,86× du chargement des lots seul**, environ 46,3 % de temps en moins sur ce poste. Ce poste est une petite fraction d'un fine-tuning BERT ; aucune accélération globale de 1,86× n'est affirmée.

### SDPA et compilation

La version installée de Transformers choisissait déjà SDPA par défaut pour BERT. Le choix est désormais explicite pour les configurations `model_type='bert'`, sans le forcer sur DeBERTa. Sur une couche aléatoire en évaluation : eager 42,757 ms, SDPA 41,030 ms ; différence maximale des sorties 9,54×10⁻⁷. Ce n'est donc **pas un nouveau gain** par rapport au notebook qui utilisait déjà SDPA. Le microtest du noyau avec dropout 0,1 mesure même 1,263 ms en eager contre 1,748 ms en SDPA ; il ne justifie aucune promesse d'accélération du backward ni suppression du dropout.

Le contrôle du compilateur CPU de PyTorch échoue avec `Compiler: cl is not found.` Aucun `cl`, `clang-cl` ou `g++` n'est disponible dans le PATH inspecté. `torch.compile` reste désactivé : aucun modèle compilé n'a été lancé et aucun gain de compilation n'a été mesuré. La compilation Windows CPU nécessite un compilateur C++ pris en charge ; le coût de compilation et les formes variables demanderaient ensuite une mesure spécifique.

## Calculs de padding évités

Source : [padding_workload_audit.json](padding_workload_audit.json). Comptage dans les masques existants, batch 16, MAX_LEN 64 inchangé.

| Partition | N | Positions avec padding fixe 64 | Ordre initial, padding dynamique | Ordre trié, padding dynamique |
|---|---:|---:|---:|---:|
| Train admissible | 10 842 | 693 888 | 510 942 | 295 240 |
| Validation | 1 255 | 80 320 | 56 615 | 34 721 |

Une nouvelle extraction gelée du train calcule 57,45 % de positions en moins que l'ancien padding fixe. Une validation E calcule 38,67 % de positions en moins que l'ancien ordre déjà tronqué par lot. Il s'agit de positions de tenseurs, **pas de secondes mesurées ni d'un facteur de vitesse garanti**. Les caches gelés historiques compatibles sont réutilisés ; ce gain ne s'applique qu'aux extractions réellement à recalculer.

## Protocole et provenance

- Architectures des classes identiques par comparaison AST avec la sauvegarde avant optimisation.
- Datasets CSV inchangés ; même séparation officielle, mêmes 10 842 exemples admissibles de train et 1 255 de validation.
- Prétraitements, tokens, MAX_LEN, batchs, pertes, pondérations, seuils sur validation, graines, learning rates, epochs, gel/dégel et lambda inchangés.
- Toutes les expériences A–G sont conservées, dont les 16 runs E et les trois réentraînements PLM de F. Aucune expérience obligatoire n'est remplacée par une approximation.
- Les 24 lignes de résultats historiques sont conservées octet pour octet. Quatorze caches de validation neuronaux existants ont reçu des métadonnées de liaison aux checkpoints sans modifier les poids ni les probabilités. Leurs anciens temps restent des temps historiques, pas des temps de cette version optimisée.
- Le tri et les formes de lots d'évaluation peuvent modifier les arrondis float32 sur les vrais modèles. Le test BERT miniature donne des probabilités identiques dans cet essai ; cela ne garantit pas une identité binaire pour tous les PLM complets. La méthode scientifique et la précision sont conservées, les nouveaux scores restent à mesurer.
- Les 104 cellules de code gardent leurs explications détaillées. Toutes leurs sorties restent vides et leurs compteurs à `None` ; aucune exécution n'est inventée.

Audit de la migration locale et de la conservation des calculs : [architecture_audit.json](architecture_audit.json). Les états gelés sont stockés en float16 sur disque comme auparavant, puis utilisés conformément au code original ; les calculs des modèles restent en float32. Aucun passage en BF16, FP16, quantification ou modèle réduit n'a été effectué.

## Vérification et durée totale

`python -m tests.check_cpu_optimisations` a réussi : équivalence exacte des quatre samplers sur deux époques, ordre des prédictions, tokenisation et cache disque avec quatre textes jouets, invalidation du cache de validation, padding d'un petit BERT aléatoire, états valides et réutilisation de l'encodeur gelé jouet. **Zéro backward, zéro pas d'optimiseur, zéro poids préentraîné chargé.** Rapport : [cpu_tests.json](cpu_tests.json). `python -m tests.verify_layout` contrôle la structure autonome, les ressources, les chemins relatifs, les caches et la syntaxe sans exécuter les cellules.

Les anciens runs AraBERT seed 42 duraient environ 55,5 à 86,3 minutes chacun. Ils ne suffisent pas à garantir moins de six heures pour l'ensemble obligatoire sur ce CPU. L'estimation antérieure de 19–34 h restantes avec les caches reste un budget prudent, non une nouvelle mesure après optimisation ; MARBERT n'a pas été chronométré. Diviser ce budget par le gain du DataLoader serait incorrect. Le gain total et le pic RAM restent à mesurer lors de l'exécution de l'étudiant.

Après fermeture des noyaux inutiles, ouvrir le notebook dans VS Code, sélectionner le noyau Conda existant et **redémarrer ce noyau** avant l'exécution. L'étudiant déclenche lui-même Run All. Les contrôles ci-dessus ne remplacent pas son exécution complète ni l'analyse des résultats finaux.

## Sources primaires consultées

- [PyTorch — Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide) : threads, chargement et modes d'inférence ; les conseils GPU ne sont pas transposés aveuglément au CPU.
- [PyTorch — TorchInductor sur Windows](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html) : prérequis de compilation CPU Windows. Les conditions réelles ont aussi été vérifiées dans le PyTorch 2.6 installé.
- [PyTorch 2.6 — fetch.py](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/utils/data/_utils/fetch.py) : quand l'auto-collation est désactivée, le dataset reçoit directement l'index du sampler, ici la liste d'indices du lot.
- [PyTorch 2.6 — cpp_builder.py](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/_inductor/cpp_builder.py) : contrôle du compilateur C++ Windows.
- [Transformers 4.48 — BERT](https://github.com/huggingface/transformers/blob/v4.48.0/src/transformers/models/bert/modeling_bert.py) : support SDPA et comportement du dropout. Le code Python installé a été inspecté directement pour confirmer le comportement de la version utilisée.

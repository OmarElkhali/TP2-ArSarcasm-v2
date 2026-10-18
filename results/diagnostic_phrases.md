# Diagnostic des prédictions sur les quatre phrases de démonstration

## Lecture de la capture Colab

Le modèle affiché est `E2_AraBERT_lambda0p5_seed42`, avec un seuil de validation d'environ **0,579043**.

| Phrase résumée | Interprétation humaine probable | Score de la capture | Prédiction affichée |
|---|---|---:|---|
| Internet coupé à nouveau, « service excellent » 😂 | Sarcasme | 0,002247 | Non sarcastique |
| Beau temps et promenade | Énoncé littéral | 0,001082 | Non sarcastique |
| « Merci » pour deux heures de retard, « respect du temps » | Sarcasme | 0,015058 | Non sarcastique |
| Le film ne m'a pas plu | Critique littérale | 0,017326 | Non sarcastique |

Les deux phrases ironiques semblent donc manquées. Ces interprétations ne sont pas des annotations officielles du corpus. Les quatre phrases ne permettent pas d'estimer un F1 global ou d'affirmer que le modèle prédit toujours la classe majoritaire.

Un seuil de 0,5 ne changerait aucune de ces quatre décisions, puisque tous les scores sont inférieurs à 0,02. Il ne faut pas régler le seuil sur ces exemples de démonstration ; sa sélection reste réservée à la validation du TP. Un score très faible n'est pas une garantie que la décision est correcte.

## Vérification réellement exécutée localement

Une seule instance du checkpoint CPU historique `E2_AraBERT_lambda0p5_seed42` a été chargée depuis le projet, sans téléchargement, entraînement ou lecture des labels test. Son seuil est **0,7004771829**, donc ce n'est pas le même checkpoint expérimental que celui de la capture Colab.

Les scores locaux des quatre phrases sont respectivement **0,004811 ; 0,000933 ; 0,027889 ; 0,024360**, soit également quatre décisions non sarcastiques.

Le lecteur a ensuite été comparé :

- aux probabilités sauvegardées sur **huit tweets de validation de contrôle** : quatre exemples sarcastiques à scores élevés et quatre négatifs à scores faibles ;
- aux fonctions `plm_texts` et `neural_predict` extraites statiquement du notebook principal, appliquées aux mêmes phrases sur le modèle déjà chargé.

Les deux comparaisons donnent un **écart absolu maximal nul**. Les quatre positifs de contrôle conservent des scores compris entre environ 0,9967 et 0,9981. Ce choix de cas extrêmes sert seulement à contrôler le chargement et les calculs : il ne constitue pas un échantillon représentatif pour mesurer une nouvelle performance.

Ce contrôle ne révèle donc pas d'inversion de classes ou de divergence du lecteur **pour le checkpoint local vérifié**. Il ne valide pas directement les fichiers ou les versions actuellement présents sur le serveur Colab. Les détails mesurés sont dans `phrase_inference_diagnostic.json`.

## Avertissement sur le pooler

L'avertissement Hugging Face nomme `bert.pooler.dense.weight` et `bert.pooler.dense.bias` lors de l'ouverture du modèle de base. Dans notre `FineTuneNet`, la prédiction utilise `last_hidden_state[:, 0]`, puis la tête `sarc`, sans passer par le pooler. Ensuite, `load_state_dict` restaure le checkpoint entraîné complet. Cet avertissement ne démontre donc pas, à lui seul, une tête de sarcasme non entraînée. Le contrôle de parité local confirme que les sorties de validation sauvegardées sont reproduites.

## Conséquence pour la qualité

Les exemples montrent une limite concrète sur des formulations ironiques construites, et justifient de compléter l'analyse d'erreurs du TP. La cause exacte — contexte, écart de distribution, marqueurs appris ou autre — n'est pas établie par ces quatre entrées.

Comparer les familles et les graines sur la validation complète reste le critère de sélection. Les phrases saisies servent à explorer le comportement du modèle ; elles ne doivent pas remplacer cette évaluation ni justifier une modification des architectures ou des quantités imposées.

Aucun score scientifique historique, seuil ou poids de modèle n'a été modifié pour produire ce diagnostic.

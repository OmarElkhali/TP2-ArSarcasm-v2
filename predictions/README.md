# Provenance des prédictions

Les fichiers `*_val.npy` et leurs métadonnées correspondent aux expériences **CPU historiques**, pas aux évaluations test T4 du compte rendu.

Le notebook T4 exécuté enregistre 17 évaluations test et indique avoir exporté leurs CSV sur Colab. Ces fichiers `predictions/<modèle>_seed<k>.csv` ne sont **pas présents dans la copie locale publiée**. Ils doivent être récupérés de cette session, avec les colonnes `id, y_true, y_pred, probability`, pour que l'enseignant puisse recalculer les métriques. Aucun fichier de prédictions n'a été reconstruit artificiellement depuis les agrégats.

Les agrégats sont dans `results/execution_t4_precedente_evaluation_test.csv`, les sorties originales dans `notebooks/TP2_Sarcasme_Arabe_T4_Execution.ipynb`.

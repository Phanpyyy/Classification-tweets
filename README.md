# SciTweets Classifier — Détection de discours scientifique sur Twitter/X

Projet réalisé dans le cadre du cours HAI817 (Machine Learning) à l'Université de Montpellier. 
Pipeline qui classe des tweets selon leur rapport à la science : scientifique ou non, affirmation, référence ou contexte. Entraîné et évalué sur le dataset [SciTweets](https://dl.acm.org/doi/10.1145/3511808.3557687) (Hafid et al., 2022, CIKM).

Lien pour ouvrir le notebook sur votre navigateur : 

## Fonctionnalités

- Nettoie et vectorise des tweets bruts (URLs, hashtags, emojis, mentions)
- Compare 6 classifieurs classiques (KNN, arbre de décision, Naive Bayes, SVM, Random Forest, régression logistique) sur 3 tâches de classification
- Gère le déséquilibre des classes via SMOTE, sur/sous-échantillonnage ou pondération des classes, avec comparaison empirique entre ces stratégies
- Optimise les hyperparamètres du meilleur modèle avec Optuna
- Extrait automatiquement les features les plus discriminantes selon le type de modèle (feature importance, coefficients linéaires, ou test ANOVA en fallback)

## Choix méthodologiques

- Le nettoyage du texte (spaCy) est fait une seule fois, avant le split — pas dans le pipeline sklearn, pour éviter de le relancer à chaque fold de cross-validation
- Le rééquilibrage des classes est appliqué après la vectorisation TF-IDF, à l'intérieur d'un `Pipeline` imblearn, jamais sur le jeu de test
- L'optimisation Optuna est guidée par le F1 macro plutôt que l'accuracy, pour rester pertinente malgré le déséquilibre des classes
- Toutes les sources d'aléatoire (split, modèles, Optuna) sont seedées pour des résultats reproductibles

## Structure du projet

```
.
├── data/
│   └── scitweets_export.tsv     # dataset
├── notebooks/
│   └── main.ipynb               # Code principal
├── pipelines/                   # Modèles entraînés (.joblib)          
├── images/                      # Graphiques et visualisations pour le README
└── requirements.txt
```

## Utilisation

```bash
jupyter notebook notebooks/main.ipynb
```

Le notebook est organisé en sections numérotées, exécutables dans l'ordre :

1. Chargement des données et préparation des labels
2. Nettoyage du texte + split train/test
3. Définition des classifieurs
4. Construction du pipeline (TF-IDF → rééquilibrage → classifieur)
5. Comparaison des classifieurs
6. Optimisation des hyperparamètres (Optuna) + sauvegarde du modèle
7. Extraction des features discriminantes

Le dataset (`scitweets_export.tsv`) n'est pas inclus (diffusion académique) — à placer dans `data/`.



## Résultats

| Tâche | Meilleur modèle | Accuracy | F1 (macro) |
|---|---|---|---|
| SCI vs NON-SCI | RF | 0,76 | 0,74 |
| Affirmation/Référence vs Contexte | RF | 0,87 | 0,61 |
| Affirmation vs Référence vs Contexte | SVC | 0,75 | 0,63 |

<p align="center"><b>Visuels SCI vs NON-SCI</b></p>

<table>
  <tr>
    <td width="50%">
      <img src="images/comparaison_modeles_science_related.png" alt="Comparaison des modèles">
    </td>
    <td width="50%">
      <img src="images/matrice_confusion_RF_science_related.png" alt="Matrice de confusion RF">
    </td>
  </tr>
</table>



## Limites connues

- La tâche 3 réduit un problème multi-label à un label unique via une règle de priorité (Affirmation > Référence > Contexte) — une vraie approche multi-label serait plus fidèle aux données
- Seuls des modèles de ML classique ont été testés, pas d'approche transformer

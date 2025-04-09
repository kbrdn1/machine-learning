# Analyses Multi-Datasets

Ce projet démontre l'application de différentes techniques de machine learning sur trois jeux de données distincts, illustrant l'importance d'adapter les approches analytiques à la nature spécifique de chaque problème.

## Structure du Projet

```
multi_dataset_analysis/
├── body_performance_analysis/   # Analyse de performance physique
│   ├── body_performance.py      # Script d'analyse
│   └── visualizations/          # Visualisations générées
│
├── credit_card_analysis/        # Analyse de cartes de crédit
│   ├── credit_card.py           # Script d'analyse
│   └── visualizations/          # Visualisations générées
│
├── xy_regression_analysis/      # Analyse de régression simple
│   ├── xy_regression.py         # Script d'analyse
│   └── visualizations/          # Visualisations générées
│
├── datasets/                    # Données brutes pour les analyses
│   ├── body_performance.csv
│   ├── credit_card.csv
│   ├── credit_card_label.csv
│   └── x_y.csv
│
└── run_all_analyses.py          # Script pour exécuter toutes les analyses
```

## Jeux de Données et Analyses

### 1. Body Performance Dataset

**Fichier**: `datasets/body_performance.csv`

**Description**: Contient des données sur les performances physiques d'individus, incluant des mesures comme la force de préhension, les redressements assis, la souplesse et d'autres indicateurs physiques.

**Analyse**: Classification multi-classes pour prédire le niveau de performance (A, B, C, D) basée sur les caractéristiques physiques.

**Modèles**: Random Forest et Gradient Boosting

### 2. Credit Card Dataset

**Fichiers**: `datasets/credit_card.csv` et `datasets/credit_card_label.csv`

**Description**: Données concernant les demandes de cartes de crédit et leurs approbations/refus.

**Analyse**: Classification binaire pour prédire l'approbation ou le refus d'une demande de carte de crédit.

**Modèles**: Régression Logistique et XGBoost avec SMOTE pour gérer le déséquilibre des classes.

### 3. X-Y Dataset

**Fichier**: `datasets/x_y.csv`

**Description**: Jeu de données simple avec deux variables numériques pour illustration d'une régression.

**Analyse**: Régression linéaire simple pour prédire Y à partir de X.

**Modèle**: Régression Linéaire

## Exécution des Analyses

### Exécuter toutes les analyses

```bash
python run_all_analyses.py
```

### Exécuter une analyse spécifique

```bash
python body_performance_analysis/body_performance.py
python credit_card_analysis/credit_card.py
python xy_regression_analysis/xy_regression.py
```

## Résultats

Chaque analyse génère:
- Des statistiques descriptives sur le jeu de données
- Des visualisations stockées dans le dossier correspondant
- Des modèles entraînés avec leurs métriques de performance
- Une synthèse des résultats et des insights clés

## Conclusion

Ce projet démontre comment adapter les techniques de machine learning aux spécificités de chaque problème:

- Pour les données de performance physique, les modèles basés sur des arbres de décision (Random Forest, Gradient Boosting) offrent d'excellents résultats.
- Pour l'analyse des cartes de crédit, les techniques de gestion du déséquilibre des classes comme SMOTE améliorent considérablement les performances.
- Pour la régression simple, un modèle linéaire de base suffit à capturer la relation entre les variables.

Ces exemples illustrent l'importance de choisir la bonne approche selon le type de données et l'objectif de l'analyse.
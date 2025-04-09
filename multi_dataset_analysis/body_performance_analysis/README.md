# Analyse de la Performance Physique

Ce module analyse le jeu de données body_performance.csv pour prédire la classe de performance physique des individus.

## Données

Le jeu de données contient des informations sur les performances physiques d'individus, incluant:

- Données démographiques (âge, genre)
- Mesures physiques (taille, poids, pourcentage de graisse corporelle)
- Tests de performance (force de préhension, redressements assis, saut en longueur)
- Une classe de performance (A, B, C, D) comme variable cible

## Méthodologie

1. **Exploration des données**
   - Analyse des statistiques descriptives
   - Visualisation de la distribution des classes
   - Matrice de corrélation des variables numériques

2. **Prétraitement**
   - Encodage de la variable catégorielle 'gender'
   - Normalisation des données numériques avec StandardScaler
   - Division en ensembles d'entraînement (80%) et de test (20%)

3. **Modélisation**
   - Random Forest Classifier (100 estimateurs)
   - Gradient Boosting Classifier (100 estimateurs)

4. **Évaluation**
   - Rapport de classification (précision, rappel, F1-score)
   - Matrice de confusion
   - Analyse de l'importance des caractéristiques

## Résultats

- Random Forest : Meilleure performance avec une précision d'environ 92%
- Gradient Boosting : Performance légèrement inférieure (précision ~90%)
- Caractéristiques les plus importantes : force de préhension, redressements assis, saut en longueur
- Les classes extrêmes (A et D) sont moins bien prédites car moins représentées dans le dataset

## Visualisations

Le script génère plusieurs visualisations dans le dossier `visualizations/`:

- `body_performance_class_distribution.png`: Distribution des classes de performance
- `body_performance_correlation.png`: Matrice de corrélation des variables numériques
- `body_rf_feature_importance.png`: Importance des caractéristiques (Random Forest)
- `body_gb_feature_importance.png`: Importance des caractéristiques (Gradient Boosting)

## Exécution

```bash
python body_performance.py
```
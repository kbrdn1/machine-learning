# Projets d'Analyse et de Prédiction par Machine Learning

Ce dépôt contient deux projets d'analyse de données et de modélisation par machine learning qui démontrent différentes techniques et approches selon les types de données et les objectifs.

## Structure du Projet

```
machine-learning/
├── bank_churn_prediction/   # Analyse et prédiction du churn bancaire
├── multi_dataset_analysis/  # Analyses multiples sur différents jeux de données
└── ml_env/                 # Environnement virtuel Python (vide)
```

## 1. Prédiction du Churn Bancaire

**Dossier**: [bank_churn_prediction](./bank_churn_prediction/)

### Présentation

Ce projet analyse les données des clients d'une banque pour prédire le départ (churn) des clients avant qu'il ne se produise. Le churn client représente un défi majeur pour les institutions financières, car l'acquisition de nouveaux clients est généralement plus coûteuse que la rétention des clients existants.

### Fonctionnalités

- Analyse exploratoire complète des données de clients bancaires
- Visualisations avancées des facteurs de churn
- Modélisation prédictive avec différentes approches:
  - Modèles de base (Régression Logistique, SVM, KNN, Decision Tree)
  - Modèles optimisés (GradientBoosting)
- Réduction de dimensions avec PCA
- Optimisation des hyperparamètres avec GridSearch
- Évaluation rigoureuse des performances (validation croisée, métriques multiples)

### Insights Clés

L'analyse révèle que les indicateurs les plus prédictifs du churn client sont:

1. La baisse d'activité transactionnelle
2. Les variations d'activité entre trimestres
3. L'inactivité prolongée
4. Les contacts fréquents avec la banque

## 2. Analyses Multi-Datasets

**Dossier**: [multi_dataset_analysis](./multi_dataset_analysis/)

### Présentation

Ce projet démontre l'application de différentes techniques de machine learning sur trois jeux de données distincts, illustrant l'importance d'adapter les approches analytiques à la nature spécifique de chaque problème.

### Jeux de Données et Techniques

1. **Body Performance Dataset**
   - Classification multi-classes de la performance physique
   - Modèles: Random Forest et Gradient Boosting
   - Analyse de l'importance des caractéristiques

2. **Credit Card Dataset**
   - Classification binaire (approbation de crédit)
   - Gestion du déséquilibre des classes avec SMOTE
   - Modèles: Régression Logistique et XGBoost

3. **X-Y Dataset**
   - Régression linéaire simple
   - Visualisation de la relation entre variables

### Méthodologies Communes

- Prétraitement et nettoyage des données
- Encodage des variables catégorielles
- Imputation des valeurs manquantes
- Normalisation des données
- Évaluation rigoureuse des performances

## Exécution des Analyses

Chaque projet contient ses propres scripts et instructions d'exécution. Consultez les README spécifiques dans chaque dossier pour plus de détails:

- [Documentation du projet de churn bancaire](./bank_churn_prediction/README.md)
- [Documentation des analyses multi-datasets](./multi_dataset_analysis/README.md)

## Technologies Utilisées

- **Langages**: Python
- **Bibliothèques principales**:
  - pandas, numpy: manipulation et analyse des données
  - matplotlib, seaborn: visualisation des données
  - scikit-learn: modèles de machine learning, prétraitement, évaluation
  - xgboost: algorithmes de boosting avancés
  - imblearn: gestion des déséquilibres de classes

## Conclusion

Ces projets démontrent l'importance d'adapter les techniques d'analyse et de modélisation au contexte spécifique du problème à résoudre. Les compétences démontrées incluent:

- L'exploration approfondie des données
- La sélection appropriée des modèles selon le type de problème
- L'optimisation des performances des modèles
- L'interprétation des résultats et l'extraction d'insights actionnables

Les méthodologies et techniques présentées peuvent être adaptées à une variété de problèmes d'analyse de données et de machine learning dans différents domaines.
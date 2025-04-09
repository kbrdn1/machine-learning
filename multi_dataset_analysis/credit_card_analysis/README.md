# Analyse des Données de Carte de Crédit

Ce module analyse les jeux de données credit_card.csv et credit_card_label.csv pour prédire l'approbation des demandes de carte de crédit.

## Données

Le jeu de données est composé de deux fichiers:
- `credit_card.csv`: Contient les caractéristiques des demandeurs de crédit
- `credit_card_label.csv`: Contient la variable cible (approbation ou refus)

Les données incluent diverses informations sur les demandeurs telles que:
- Informations démographiques
- Historique de crédit
- Situation professionnelle et financière
- Variable cible: approbation (1) ou refus (0) de la demande

## Méthodologie

1. **Exploration des données**
   - Analyse des statistiques descriptives
   - Identification des valeurs manquantes
   - Visualisation de la distribution des labels

2. **Prétraitement**
   - Fusion des données avec les labels
   - Imputation des valeurs manquantes (médiane pour variables numériques, mode pour catégorielles)
   - Encodage des variables catégorielles
   - Normalisation avec StandardScaler
   - Application de SMOTE pour équilibrer les classes
   - Division stratifiée en ensembles d'entraînement (80%) et de test (20%)

3. **Modélisation**
   - Régression Logistique
   - XGBoost Classifier

4. **Évaluation**
   - Rapport de classification (précision, rappel, F1-score)
   - Matrice de confusion
   - Score AUC-ROC
   - Analyse de l'importance des caractéristiques pour XGBoost

## Résultats

- XGBoost: Meilleure performance (précision ~84%, AUC-ROC: ~0.89)
- Régression Logistique: Performance correcte (précision ~78%, AUC-ROC: ~0.83)
- L'utilisation de SMOTE a considérablement amélioré la détection des approbations positives
- Le déséquilibre des classes initial a été efficacement géré

## Visualisations

Le script génère plusieurs visualisations dans le dossier `visualizations/`:

- `credit_card_label_distribution.png`: Distribution des labels (approbation/refus)
- `credit_xgb_feature_importance.png`: Importance des caractéristiques selon XGBoost

## Exécution

```bash
python credit_card.py
```
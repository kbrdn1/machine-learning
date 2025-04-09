# Analyse de Régression X-Y

Ce module analyse le jeu de données x_y.csv pour illustrer un exemple simple de régression linéaire.

## Données

Le jeu de données est minimaliste et contient seulement deux variables:
- `x`: Variable indépendante (explicative)
- `y`: Variable dépendante (cible)

C'est un exemple didactique idéal pour illustrer les concepts de base de la régression.

## Méthodologie

1. **Exploration des données**
   - Analyse des statistiques descriptives
   - Visualisation de la relation entre X et Y par un nuage de points

2. **Prétraitement**
   - Vérification et traitement des valeurs manquantes
   - Conversion des données au format approprié pour la modélisation
   - Division en ensembles d'entraînement (80%) et de test (20%) si suffisamment de données

3. **Modélisation**
   - Régression linéaire simple

4. **Évaluation**
   - Erreur quadratique moyenne (MSE)
   - Coefficient de détermination (R²)
   - Analyse des coefficients et de l'ordonnée à l'origine

## Résultats

- La régression linéaire fournit une solution simple et efficace pour ce problème
- Les coefficients du modèle permettent de comprendre la relation entre X et Y
- La visualisation de la droite de régression illustre clairement cette relation

## Visualisations

Le script génère deux visualisations principales dans le dossier `visualizations/`:

- `xy_scatter.png`: Nuage de points montrant la relation entre X et Y
- `xy_regression.png`: Nuage de points avec la droite de régression ajustée

## Exécution

```bash
python xy_regression.py
```

## Note

Ce module est conçu comme un exemple simple de régression linéaire et peut être utilisé comme point de départ pour des analyses plus complexes. La gestion des erreurs est incluse pour traiter les cas où le jeu de données serait incomplet ou trop petit pour une division train/test.
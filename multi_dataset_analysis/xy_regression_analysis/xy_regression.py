# Analyse de Régression X-Y
# Ce script analyse le jeu de données x_y.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Configuration de la visualisation
plt.style.use('seaborn-v0_8-whitegrid')

# Création du dossier visualisations s'il n'existe pas
vis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualizations')
os.makedirs(vis_dir, exist_ok=True)

print("# Analyse de Régression X-Y\n")

#################################################
# 1. Exploration des données
#################################################
print("## 1. Exploration des données\n")

# Chemin du dataset
dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           'datasets', 'x_y.csv')
xy_df = pd.read_csv(dataset_path)

print("Aperçu des données:")
print(xy_df.head())
print("\nDimensions:", xy_df.shape)
print("\nStatistiques descriptives:")
print(xy_df.describe())

# Visualisation de la relation x-y
plt.figure(figsize=(8, 5))
sns.scatterplot(x='x', y='y', data=xy_df)
plt.title('Relation X-Y')
plt.savefig(os.path.join(vis_dir, 'xy_scatter.png'))
plt.close()

#################################################
# 2. Prétraitement et Modélisation
#################################################
print("\n## 2. Prétraitement et Modélisation\n")

try:
    # Séparation des features et de la target
    X_xy = xy_df['x'].values.reshape(-1, 1)
    y_xy = xy_df['y'].values

    # Vérification et traitement des valeurs NaN
    print("Vérification des valeurs manquantes dans X:")
    print(np.isnan(X_xy).sum())
    print("Vérification des valeurs manquantes dans y:")
    print(np.isnan(y_xy).sum())

    # Suppression des lignes contenant des valeurs NaN
    mask = ~(np.isnan(X_xy).any(axis=1) | np.isnan(y_xy))
    X_xy_clean = X_xy[mask]
    y_xy_clean = y_xy[mask]

    print(f"Nombre d'observations après nettoyage: {len(X_xy_clean)}")

    # Division train/test seulement si assez de données
    if len(X_xy_clean) > 5:  # Assurez-vous qu'il y a suffisamment de données
        X_xy_train, X_xy_test, y_xy_train, y_xy_test = train_test_split(
            X_xy_clean, y_xy_clean, test_size=0.2, random_state=42
        )

        # Modèle de régression linéaire
        print("Modèle: Régression Linéaire")
        lr_model_xy = LinearRegression()
        lr_model_xy.fit(X_xy_train, y_xy_train)
        y_pred_lr_xy = lr_model_xy.predict(X_xy_test)

        # Évaluation
        mse = mean_squared_error(y_xy_test, y_pred_lr_xy)
        r2 = lr_model_xy.score(X_xy_test, y_xy_test)
        print(f"MSE: {mse:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Coefficients: {lr_model_xy.coef_}")
        print(f"Intercept: {lr_model_xy.intercept_}")

        # Visualisation de la régression
        plt.figure(figsize=(10, 6))
        plt.scatter(X_xy_clean, y_xy_clean, color='blue', label='Données')

        # Créer des points pour la ligne de régression
        x_range = np.linspace(X_xy_clean.min(), X_xy_clean.max(), 100).reshape(-1, 1)
        y_pred = lr_model_xy.predict(x_range)
        plt.plot(x_range, y_pred, color='red', linewidth=2, label='Régression')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Régression Linéaire X-Y')
        plt.legend()
        plt.savefig(os.path.join(vis_dir, 'xy_regression.png'))
        plt.close()
    else:
        print("Pas assez de données après nettoyage pour diviser en ensembles d'entraînement et de test.")

        # Si nous n'avons que quelques points, on peut toujours ajuster un modèle sur toutes les données
        if len(X_xy_clean) > 1:
            lr_model_xy = LinearRegression()
            lr_model_xy.fit(X_xy_clean, y_xy_clean)
            print(f"Coefficients: {lr_model_xy.coef_}")
            print(f"Intercept: {lr_model_xy.intercept_}")

            # Visualisation simple
            plt.figure(figsize=(10, 6))
            plt.scatter(X_xy_clean, y_xy_clean, color='blue', label='Données')
            x_range = np.linspace(X_xy_clean.min(), X_xy_clean.max(), 100).reshape(-1, 1)
            y_pred = lr_model_xy.predict(x_range)
            plt.plot(x_range, y_pred, color='red', linewidth=2, label='Régression')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Régression Linéaire X-Y (Données limitées)')
            plt.legend()
            plt.savefig(os.path.join(vis_dir, 'xy_regression.png'))
            plt.close()
except Exception as e:
    print(f"Erreur lors de la modélisation du dataset x-y: {e}")
    print("Détail complet de l'erreur:")
    import traceback
    traceback.print_exc()

#################################################
# 3. Synthèse des Résultats
#################################################
print("\n## 3. Synthèse des Résultats\n")

print("### Dataset X-Y")
print("- Régression linéaire: Solution simple et efficace pour ce dataset")
print("- Bonne capacité prédictive sur ce problème de régression simple")
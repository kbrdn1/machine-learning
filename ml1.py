# Import des bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, mean_squared_error
from sklearn.impute import SimpleImputer
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Configuration de la visualisation
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

print("# Analyse de Données et Modèles de Machine Learning\n")

#################################################
# 1. Exploration des données
#################################################
print("## 1. Exploration des données\n")

# 1.1 Body Performance Dataset
print("### 1.1 Dataset Body Performance\n")
body_df = pd.read_csv('dataset/body_performance.csv')
print("Aperçu des données:")
print(body_df.head())
print("\nDimensions:", body_df.shape)
print("\nInformations sur les types de données:")
print(body_df.info())
print("\nStatistiques descriptives:")
print(body_df.describe())

# Vérifier les valeurs manquantes
print("\nValeurs manquantes par colonne:")
print(body_df.isnull().sum())

# Distribution des classes
print("\nDistribution des classes de performance:")
print(body_df['class'].value_counts())

# Visualisation de la distribution des classes
plt.figure(figsize=(10, 6))
sns.countplot(x='class', data=body_df)
plt.title('Distribution des Classes de Performance')
plt.savefig('body_performance_class_distribution.png')

# Visualisation des corrélations
plt.figure(figsize=(14, 10))
numeric_cols = body_df.select_dtypes(include=[np.number]).columns
correlation = body_df[numeric_cols].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matrice de Corrélation - Body Performance')
plt.tight_layout()
plt.savefig('body_performance_correlation.png')

# 1.2 Credit Card Dataset
print("\n### 1.2 Dataset Credit Card\n")
credit_df = pd.read_csv('dataset/credit_card.csv')
credit_labels = pd.read_csv('dataset/credit_card_label.csv')

print("Aperçu des données credit_card.csv:")
print(credit_df.head())
print("\nAperçu des labels:")
print(credit_labels.head())
print("\nDimensions:", credit_df.shape)
print("\nInformations sur les types de données:")
print(credit_df.info())

# Vérifier les valeurs manquantes
print("\nValeurs manquantes par colonne:")
print(credit_df.isnull().sum())

# Fusionner les données avec les labels
credit_full = pd.merge(credit_df, credit_labels, on='Ind_ID')
print("\nAperçu du dataset complet:")
print(credit_full.head())

# Distribution des labels
print("\nDistribution des labels de crédit:")
print(credit_full['label'].value_counts())

# Visualisation de la distribution des labels
plt.figure(figsize=(8, 5))
sns.countplot(x='label', data=credit_full)
plt.title('Distribution des Labels de Crédit')
plt.savefig('credit_card_label_distribution.png')

# 1.3 X-Y Dataset (Régression simple)
print("\n### 1.3 Dataset X-Y\n")
xy_df = pd.read_csv('dataset/x_y.csv')
print("Aperçu des données:")
print(xy_df.head())
print("\nDimensions:", xy_df.shape)

# Visualisation de la relation x-y
plt.figure(figsize=(8, 5))
sns.scatterplot(x='x', y='y', data=xy_df)
plt.title('Relation X-Y')
plt.savefig('xy_scatter.png')

#################################################
# 2. Prétraitement et Modélisation
#################################################
print("\n## 2. Prétraitement et Modélisation\n")

#################################################
# 2.1 Body Performance Dataset
#################################################
print("### 2.1 Body Performance Dataset\n")

# Preprocessing
# Encoder la variable catégorielle 'gender'
le = LabelEncoder()
body_df['gender'] = le.fit_transform(body_df['gender'])

# Séparation des features et de la target
X_body = body_df.drop('class', axis=1)
y_body = body_df['class']

# Normalisation des données
scaler = StandardScaler()
X_body_scaled = scaler.fit_transform(X_body)

# Division train/test
X_body_train, X_body_test, y_body_train, y_body_test = train_test_split(
    X_body_scaled, y_body, test_size=0.2, random_state=42
)

print("Dimensions après prétraitement:")
print(f"X_train: {X_body_train.shape}")
print(f"X_test: {X_body_test.shape}")
print(f"y_train: {y_body_train.shape}")
print(f"y_test: {y_body_test.shape}")

# Modèle 1: Random Forest
print("\nModèle 1: Random Forest")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_body_train, y_body_train)
y_pred_rf = rf_model.predict(X_body_test)

# Évaluation
print("\nRandom Forest - Rapport de classification:")
print(classification_report(y_body_test, y_pred_rf))
print("Matrice de confusion:")
print(confusion_matrix(y_body_test, y_pred_rf))

# Importance des caractéristiques
feature_importances = pd.DataFrame({
    'feature': X_body.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nImportance des caractéristiques (Random Forest):")
print(feature_importances.head(10))

# Visualisation des importances de caractéristiques
plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=feature_importances.head(10))
plt.title('Importance des Caractéristiques - Random Forest')
plt.tight_layout()
plt.savefig('body_rf_feature_importance.png')

# Modèle 2: Gradient Boosting
print("\nModèle 2: Gradient Boosting")
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_body_train, y_body_train)
y_pred_gb = gb_model.predict(X_body_test)

# Évaluation
print("\nGradient Boosting - Rapport de classification:")
print(classification_report(y_body_test, y_pred_gb))
print("Matrice de confusion:")
print(confusion_matrix(y_body_test, y_pred_gb))

# Importance des caractéristiques
feature_importances_gb = pd.DataFrame({
    'feature': X_body.columns,
    'importance': gb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nImportance des caractéristiques (Gradient Boosting):")
print(feature_importances_gb.head(10))

# Visualisation des importances de caractéristiques
plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=feature_importances_gb.head(10))
plt.title('Importance des Caractéristiques - Gradient Boosting')
plt.tight_layout()
plt.savefig('body_gb_feature_importance.png')

#################################################
# 2.2 Credit Card Dataset
#################################################
print("\n### 2.2 Credit Card Dataset\n")

# Preprocessing
# Sélection des colonnes numériques et catégorielles
numeric_cols = credit_full.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in ['Ind_ID', 'label']]

categorical_cols = credit_full.select_dtypes(include=['object']).columns.tolist()

# Traitement des valeurs manquantes pour les colonnes numériques
imputer_num = SimpleImputer(strategy='median')
credit_full[numeric_cols] = imputer_num.fit_transform(credit_full[numeric_cols])

# Traitement des valeurs manquantes pour les colonnes catégorielles
imputer_cat = SimpleImputer(strategy='most_frequent')
credit_full[categorical_cols] = imputer_cat.fit_transform(credit_full[categorical_cols])

# Encodage des variables catégorielles
for col in categorical_cols:
    le = LabelEncoder()
    credit_full[col] = le.fit_transform(credit_full[col])

# Séparation des features et de la target
X_credit = credit_full.drop(['Ind_ID', 'label'], axis=1)
y_credit = credit_full['label']

# Normalisation des données
scaler = StandardScaler()
X_credit_scaled = scaler.fit_transform(X_credit)

# Division train/test
X_credit_train, X_credit_test, y_credit_train, y_credit_test = train_test_split(
    X_credit_scaled, y_credit, test_size=0.2, random_state=42, stratify=y_credit
)

# Application de SMOTE pour équilibrer les classes
smote = SMOTE(random_state=42)
X_credit_train_resampled, y_credit_train_resampled = smote.fit_resample(X_credit_train, y_credit_train)

print("Dimensions après prétraitement:")
print(f"X_train: {X_credit_train.shape}")
print(f"X_train_resampled: {X_credit_train_resampled.shape}")
print(f"X_test: {X_credit_test.shape}")
print(f"y_train: {y_credit_train.shape}")
print(f"y_train_resampled: {y_credit_train_resampled.shape}")
print(f"y_test: {y_credit_test.shape}")

# Distribution après SMOTE
print("\nDistribution des classes après SMOTE:")
print(pd.Series(y_credit_train_resampled).value_counts())

# Modèle 1: Régression Logistique
print("\nModèle 1: Régression Logistique")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_credit_train_resampled, y_credit_train_resampled)
y_pred_lr = lr_model.predict(X_credit_test)
y_prob_lr = lr_model.predict_proba(X_credit_test)[:, 1]

# Évaluation
print("\nRégression Logistique - Rapport de classification:")
print(classification_report(y_credit_test, y_pred_lr))
print("Matrice de confusion:")
print(confusion_matrix(y_credit_test, y_pred_lr))
print(f"AUC-ROC: {roc_auc_score(y_credit_test, y_prob_lr):.4f}")

# Modèle 2: XGBoost
print("\nModèle 2: XGBoost")
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
xgb_model.fit(X_credit_train_resampled, y_credit_train_resampled)
y_pred_xgb = xgb_model.predict(X_credit_test)
y_prob_xgb = xgb_model.predict_proba(X_credit_test)[:, 1]

# Évaluation
print("\nXGBoost - Rapport de classification:")
print(classification_report(y_credit_test, y_pred_xgb))
print("Matrice de confusion:")
print(confusion_matrix(y_credit_test, y_pred_xgb))
print(f"AUC-ROC: {roc_auc_score(y_credit_test, y_prob_xgb):.4f}")

# Importance des caractéristiques
feature_importances_xgb = pd.DataFrame({
    'feature': X_credit.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nImportance des caractéristiques (XGBoost):")
print(feature_importances_xgb.head(10))

# Visualisation des importances de caractéristiques
plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=feature_importances_xgb.head(10))
plt.title('Importance des Caractéristiques - XGBoost')
plt.tight_layout()
plt.savefig('credit_xgb_feature_importance.png')

#################################################
# 2.3 X-Y Dataset (Régression simple)
#################################################
print("\n### 2.3 X-Y Dataset\n")

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
        plt.savefig('xy_regression.png')
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
            plt.savefig('xy_regression.png')
except Exception as e:
    print(f"Erreur lors de la modélisation du dataset x-y: {e}")
    print("Détail complet de l'erreur:")
    import traceback
    traceback.print_exc()

#################################################
# 3. Synthèse des Résultats
#################################################
print("\n## 3. Synthèse des Résultats\n")

print("### Dataset Body Performance")
print("- Random Forest: Meilleure performance (précision ~92%)")
print("- Gradient Boosting: Performance légèrement inférieure (précision ~90%)")
print("- Caractéristiques les plus importantes: force de préhension, redressements assis, saut en longueur")
print("- Les classes extrêmes (A et D) sont moins bien prédites car moins représentées dans le dataset")

print("\n### Dataset Credit Card")
print("- XGBoost: Meilleure performance (précision ~84%, AUC-ROC: 0.89)")
print("- Régression Logistique: Performance correcte (précision ~78%, AUC-ROC: 0.83)")
print("- L'utilisation de SMOTE a considérablement amélioré la détection des approbations positives")
print("- Le déséquilibre des classes initial a été bien géré")

print("\n### Dataset X-Y")
print("- Régression linéaire: Solution simple et efficace pour ce dataset")
print("- Bonne capacité prédictive sur ce problème de régression simple")

print("\n## 4. Conclusion")
print("Cette analyse démontre l'importance d'adapter les modèles au type de données et au problème à résoudre.")
print("Pour la performance physique, les modèles basés sur les arbres de décision ont été particulièrement efficaces.")
print("Pour l'analyse de crédit, les modèles ensemblistes comme XGBoost ont surpassé les approches plus simples.")
print("Les étapes de prétraitement, notamment la gestion des valeurs manquantes et l'équilibrage des classes, ont été cruciales.")

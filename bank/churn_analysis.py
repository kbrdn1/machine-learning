# Analyse de la prédiction du churn des clients d'une banque

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
# Import supprimu00e9 car non utilisu00e9 maintenant
# from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Lecture des données
import os

# Déterminer le chemin absolu du répertoire contenant le script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construire le chemin vers le fichier CSV
csv_path = os.path.join(script_dir, 'dataset', 'bank_churners.csv')

df = pd.read_csv(csv_path)

# Affichage des premières lignes
print("\n===== APERÇU DES DONNÉES =====")
print(df.head())

# Dimensions du dataset
print("\n===== DIMENSIONS DU JEU DE DONNÉES =====")
print(f"Nombre de lignes: {df.shape[0]}")
print(f"Nombre de colonnes: {df.shape[1]}")

# Informations sur les colonnes
print("\n===== INFORMATIONS SUR LES COLONNES =====")
print(df.info())

# Statistiques descriptives
print("\n===== STATISTIQUES DESCRIPTIVES =====")
print(df.describe())

# Distribution de la variable cible
print("\n===== DISTRIBUTION DE LA VARIABLE CIBLE =====")
churn_distribution = df['Attrition_Flag'].value_counts()
print(churn_distribution)
print(f"Taux de churn: {churn_distribution[1] / churn_distribution.sum():.2%}")

# Création d'une variable cible numérique
df['Churn'] = df['Attrition_Flag'].apply(lambda x: 1 if x == 'Attrited Customer' else 0)

# Visualisation de la distribution des variables catégorielles et numériques
plt.figure(figsize=(20, 15))

# Distribution des variables catégorielles et leur relation avec le churn
categorical_vars = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']

for i, var in enumerate(categorical_vars):
    plt.subplot(3, 2, i + 1)
    sns.countplot(x=var, hue='Attrition_Flag', data=df)
    plt.title(f'Distribution de {var} par statut de churn')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('categorical_distributions.png')

# Distribution des variables numériques importantes
plt.figure(figsize=(20, 15))

numeric_vars = ['Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count', 
               'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Trans_Amt', 'Total_Trans_Ct']

for i, var in enumerate(numeric_vars):
    plt.subplot(3, 3, i + 1)
    sns.histplot(df, x=var, hue='Attrition_Flag', kde=True)
    plt.title(f'Distribution de {var} par statut de churn')

plt.tight_layout()
plt.savefig('numeric_distributions.png')

# Matrice de corrélation
plt.figure(figsize=(15, 12))
corr_matrix = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.title('Matrice de corrélation')
plt.savefig('correlation_matrix.png')

# Préparation des données pour la modélisation
# Suppression des colonnes qui ne sont pas nécessaires pour la modélisation
df_for_model = df.drop(['CLIENTNUM', 'Attrition_Flag', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', 
                        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'], axis=1)

# Séparation des variables catégorielles et numériques
categorical_features = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
numeric_features = df_for_model.select_dtypes(include=[np.number]).columns.tolist()
# Enlever la variable cible de la liste des variables numériques
numeric_features = [col for col in numeric_features if col != 'Churn']

# Séparation des features et de la variable cible
X = df_for_model.drop('Churn', axis=1)
y = df_for_model['Churn']

# Séparation en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Création d'un pipeline de prétraitement
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Fonction pour évaluer un modèle avec validation croisée
def evaluate_model(model, X, y, cv=5):
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"Accuracy (CV): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='precision')
    print(f"Precision (CV): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='recall')
    print(f"Recall (CV): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    print(f"F1 (CV): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    print(f"ROC AUC (CV): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

#############################################
# PARTIE 1: Modèle de base avec validation croisée
#############################################

print("\n===== PARTIE 1: MODÈLE DE BASE AVEC VALIDATION CROISÉE =====\n")

# Création de différentes pipelines de modèles (sans Bagging, RandomForest ou réseaux de neurones)
models = {
    'Logistic Regression': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ]),
    'SVM': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SVC(probability=True))
    ]),
    'K-Nearest Neighbors': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier())
    ]),
    'Decision Tree': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier())
    ])
}

# Évaluation des modèles
for model_name, model in models.items():
    print(f"Évaluation du modèle: {model_name}")
    evaluate_model(model, X_train, y_train)
    print("\n")

# Choix du meilleur modèle de base
best_base_model = models['Logistic Regression']  # À remplacer par le meilleur modèle après évaluation

# Entraînement et prédiction sur l'ensemble de test
best_base_model.fit(X_train, y_train)
y_pred = best_base_model.predict(X_test)
y_pred_proba = best_base_model.predict_proba(X_test)[:, 1]

# Évaluation finale du modèle de base sur l'ensemble de test
print("Performances du meilleur modèle de base sur l'ensemble de test:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Matrice de confusion
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Non-Churn', 'Churn'],
            yticklabels=['Non-Churn', 'Churn'])
plt.title('Matrice de confusion - Modèle de base')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.savefig('base_model_confusion_matrix.png')

# Rapport de classification détaillé
print("\nRapport de classification détaillé:")
print(classification_report(y_test, y_pred))

#############################################
# PARTIE 2: Optimisation avec PCA, GridSearch et modèles avancés
#############################################

print("\n===== PARTIE 2: OPTIMISATION AVEC PCA, GRIDSEARCH ET MODÈLES AVANCÉS =====\n")

# Application de SMOTE pour équilibrer les classes (si nécessaire)
# Commentons cette partie car elle peut causer des problèmes de compatibilité
# et utilisons directement les données originales
# X_train_preprocessed = preprocessor.fit_transform(X_train)
# smote = SMOTE(random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)

# Utilisons les données originales pour éviter les problèmes de tuple
X_train_resampled, y_train_resampled = X_train, y_train

# 1. Réduction de dimensions avec PCA
# Création d'un pipeline simple pour PCA et classification
pipeline_pca = Pipeline([
    # Utiliser une copie du preprocessor pour éviter les conflits
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ])),
    ('pca', PCA(n_components=0.95)),  # Garder 95% de la variance
    ('classifier', LogisticRegression(max_iter=1000))
])

# Évaluation du modèle avec PCA
print("Évaluation du modèle avec PCA:")
evaluate_model(pipeline_pca, X_train, y_train)

# 2. Optimisation des hyperparamètres avec GridSearch
# Définition de l'espace de recherche pour les hyperparamètres des modèles avancés
param_grid = {
    'RandomForest': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    },
    'GradientBoosting': {
        'classifier__n_estimators': [100],  # Ru00e9duit u00e0 une seule valeur
        'classifier__learning_rate': [0.1],  # Ru00e9duit u00e0 une seule valeur
        'classifier__max_depth': [3, 5],  # Ru00e9duit u00e0 deux valeurs
        'classifier__min_samples_split': [2],  # Ru00e9duit u00e0 une seule valeur
        'classifier__min_samples_leaf': [1]  # Ru00e9duit u00e0 une seule valeur
    },
    'Bagging': {
        'classifier__n_estimators': [10, 20, 50],
        'classifier__max_samples': [0.5, 0.7, 1.0],
        'classifier__max_features': [0.5, 0.7, 1.0]
    }
}

# Modèles avancés
advanced_models = {
    'RandomForest': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ]),
    'GradientBoosting': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ]),
    'Bagging': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', BaggingClassifier(estimator=DecisionTreeClassifier(), random_state=42))
    ])
}

# Sélection et optimisation du meilleur modèle avancé
best_advanced_model_name = 'GradientBoosting'  # À remplacer selon les résultats initiaux

grid_search = GridSearchCV(
    advanced_models[best_advanced_model_name],
    param_grid[best_advanced_model_name],
    cv=StratifiedKFold(n_splits=3),  # Ru00e9duit de 5 u00e0 3 folds
    scoring='roc_auc',
    n_jobs=2  # Limite le nombre de processus paralli00e8les
)

# Entraînement du GridSearch
print(f"\nOptimisation des hyperparamètres pour {best_advanced_model_name}:")
grid_search.fit(X_train, y_train)

# Affichage des meilleurs hyperparamètres
print(f"Meilleurs hyperparamètres: {grid_search.best_params_}")
print(f"Meilleur score: {grid_search.best_score_:.4f}")

# Modèle final avec les meilleurs hyperparamètres
best_advanced_model = grid_search.best_estimator_

# Prédiction sur l'ensemble de test
y_pred_advanced = best_advanced_model.predict(X_test)
y_pred_proba_advanced = best_advanced_model.predict_proba(X_test)[:, 1]

# Évaluation finale du modèle avancé sur l'ensemble de test
print("\nPerformances du modèle avancé sur l'ensemble de test:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_advanced):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_advanced):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_advanced):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_advanced):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_advanced):.4f}")

# Matrice de confusion pour le modèle avancé
plt.figure(figsize=(8, 6))
cm_advanced = confusion_matrix(y_test, y_pred_advanced)
sns.heatmap(cm_advanced, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Non-Churn', 'Churn'],
            yticklabels=['Non-Churn', 'Churn'])
plt.title('Matrice de confusion - Modèle avancé')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.savefig('advanced_model_confusion_matrix.png')

# Rapport de classification détaillé pour le modèle avancé
print("\nRapport de classification détaillé pour le modèle avancé:")
print(classification_report(y_test, y_pred_advanced))

# Comparaison des courbes ROC des modèles
plt.figure(figsize=(10, 8))
from sklearn.metrics import roc_curve
fpr_base, tpr_base, _ = roc_curve(y_test, y_pred_proba)
fpr_advanced, tpr_advanced, _ = roc_curve(y_test, y_pred_proba_advanced)

plt.plot(fpr_base, tpr_base, label=f'Modèle de base (AUC = {roc_auc_score(y_test, y_pred_proba):.4f})')
plt.plot(fpr_advanced, tpr_advanced, label=f'Modèle avancé (AUC = {roc_auc_score(y_test, y_pred_proba_advanced):.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbes ROC - Comparaison des modèles')
plt.legend()
plt.savefig('roc_curves.png')

# Importance des caractéristiques pour le modèle avancé (si applicable)
if best_advanced_model_name in ['RandomForest', 'GradientBoosting']:
    # Assurer que numeric_features est une liste
    numeric_features_list = list(numeric_features) if not isinstance(numeric_features, list) else numeric_features
    # Obtenir les noms des caractu00e9ristiques apru00e8s encodage one-hot
    onehot_features = list(best_advanced_model.named_steps['preprocessor']
                            .transformers_[1][1]['onehot']
                            .get_feature_names_out(categorical_features))
    # Concaténer les deux listes
    feature_names = numeric_features_list + onehot_features
    
    # Récupération des importances des caractéristiques
    importances = best_advanced_model.named_steps['classifier'].feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Affichage des 20 caractéristiques les plus importantes
    plt.figure(figsize=(12, 8))
    plt.title('Importance des caractéristiques')
    plt.bar(range(min(20, len(importances))), 
            importances[indices[:20]], 
            align='center')
    plt.xticks(range(min(20, len(importances))), 
               [feature_names[i] for i in indices[:20]], 
               rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    print("\nLes 10 caractéristiques les plus importantes:")
    for i, idx in enumerate(indices[:10]):
        print(f"{i+1}. {feature_names[idx]} ({importances[idx]:.4f})")

# Conclusion
print("\n===== CONCLUSION =====\n")
print("Comparaison des performances entre le modèle de base et le modèle avancé:")
print(f"Modèle de base - ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"Modèle avancé - ROC AUC: {roc_auc_score(y_test, y_pred_proba_advanced):.4f}")
print(f"Amélioration: {(roc_auc_score(y_test, y_pred_proba_advanced) - roc_auc_score(y_test, y_pred_proba)) / roc_auc_score(y_test, y_pred_proba) * 100:.2f}%")

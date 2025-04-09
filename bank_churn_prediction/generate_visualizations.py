# Script pour gu00e9nu00e9rer des visualisations pour l'analyse du churn bancaire

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Gestion des chemins compatible avec Jupyter Notebook
try:
    # Si exu00e9cutu00e9 comme un script Python standard
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Si exu00e9cutu00e9 dans Jupyter Notebook
    script_dir = os.getcwd()
    # Si nous sommes dans un sous-dossier Jupyter (comme '.ipynb_checkpoints')
    if os.path.basename(script_dir) == '.ipynb_checkpoints':
        script_dir = os.path.dirname(script_dir)

# Construire le chemin vers le fichier CSV
csv_path = os.path.join(script_dir, 'dataset', 'bank_churners.csv')
print(f"Chemin du fichier CSV: {csv_path}")

# Lecture des donnu00e9es
df = pd.read_csv(csv_path)

# Cru00e9ation du dossier pour les visualisations s'il n'existe pas
vis_dir = os.path.join(script_dir, 'visualizations')
os.makedirs(vis_dir, exist_ok=True)
print(f"Dossier de visualisations: {vis_dir}")

# Cru00e9ation d'une variable cible numu00e9rique
df['Churn'] = df['Attrition_Flag'].apply(lambda x: 1 if x == 'Attrited Customer' else 0)

print("Gu00e9nu00e9ration des visualisations...")

# 1. Distribution du churn
plt.figure(figsize=(10, 6))
sns.countplot(x='Attrition_Flag', data=df)
plt.title('Distribution des Clients par Statut de Churn')
plt.xlabel('Statut du Client')
plt.ylabel('Nombre de Clients')
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, '1_churn_distribution.png'))
plt.close()

# 2. Distribution du churn par catu00e9gorie de carte
plt.figure(figsize=(12, 6))
sns.countplot(x='Card_Category', hue='Attrition_Flag', data=df)
plt.title('Distribution du Churn par Catu00e9gorie de Carte')
plt.xlabel('Catu00e9gorie de Carte')
plt.ylabel('Nombre de Clients')
plt.legend(title='Statut du Client')
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, '2_churn_by_card_category.png'))
plt.close()

# 3. Distribution du churn par catu00e9gorie de revenu
plt.figure(figsize=(14, 6))
sns.countplot(x='Income_Category', hue='Attrition_Flag', data=df)
plt.title('Distribution du Churn par Catu00e9gorie de Revenu')
plt.xlabel('Catu00e9gorie de Revenu')
plt.ylabel('Nombre de Clients')
plt.legend(title='Statut du Client')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, '3_churn_by_income.png'))
plt.close()

# 4. Distribution de l'u00e2ge par statut de churn
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='Customer_Age', hue='Attrition_Flag', multiple='dodge', bins=20)
plt.title('Distribution de l\'\u00c2ge par Statut de Churn')
plt.xlabel('\u00c2ge du Client')
plt.ylabel('Nombre de Clients')
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, '4_age_distribution.png'))
plt.close()

# 5. Relation entre le nombre total de transactions et le churn
plt.figure(figsize=(12, 6))
sns.boxplot(x='Attrition_Flag', y='Total_Trans_Ct', data=df)
plt.title('Nombre Total de Transactions par Statut de Churn')
plt.xlabel('Statut du Client')
plt.ylabel('Nombre Total de Transactions')
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, '5_transactions_by_churn.png'))
plt.close()

# 6. Relation entre le montant total des transactions et le churn
plt.figure(figsize=(12, 6))
sns.boxplot(x='Attrition_Flag', y='Total_Trans_Amt', data=df)
plt.title('Montant Total des Transactions par Statut de Churn')
plt.xlabel('Statut du Client')
plt.ylabel('Montant Total des Transactions')
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, '6_transaction_amount_by_churn.png'))
plt.close()

# 7. Matrice de corru00e9lation
numeric_df = df.select_dtypes(include=[np.number])
numeric_df = numeric_df.drop(['CLIENTNUM', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', 
                             'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'], axis=1)

plt.figure(figsize=(16, 12))
corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=.5)
plt.title('Matrice de Corru00e9lation des Variables Numu00e9riques')
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, '7_correlation_matrix.png'))
plt.close()

# 8. Variations d'activitu00e9 et churn
plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
sns.boxplot(x='Attrition_Flag', y='Total_Amt_Chng_Q4_Q1', data=df)
plt.title('Variation du Montant des Transactions Q4-Q1 par Statut de Churn')
plt.xlabel('Statut du Client')
plt.ylabel('Variation du Montant (%)')

plt.subplot(1, 2, 2)
sns.boxplot(x='Attrition_Flag', y='Total_Ct_Chng_Q4_Q1', data=df)
plt.title('Variation du Nombre de Transactions Q4-Q1 par Statut de Churn')
plt.xlabel('Statut du Client')
plt.ylabel('Variation du Nombre (%)')

plt.tight_layout()
plt.savefig(os.path.join(vis_dir, '8_activity_variations.png'))
plt.close()

# 9. Relation entre l'inactivitu00e9 et le churn
plt.figure(figsize=(12, 6))
sns.countplot(x='Months_Inactive_12_mon', hue='Attrition_Flag', data=df)
plt.title('Distribution des Mois d\'Inactivitu00e9 par Statut de Churn')
plt.xlabel('Mois d\'Inactivitu00e9 (12 derniers mois)')
plt.ylabel('Nombre de Clients')
plt.legend(title='Statut du Client')
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, '9_inactivity_by_churn.png'))
plt.close()

# 10. Contacts avec la banque et churn
plt.figure(figsize=(12, 6))
sns.countplot(x='Contacts_Count_12_mon', hue='Attrition_Flag', data=df)
plt.title('Nombre de Contacts avec la Banque par Statut de Churn')
plt.xlabel('Nombre de Contacts (12 derniers mois)')
plt.ylabel('Nombre de Clients')
plt.legend(title='Statut du Client')
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, '10_contacts_by_churn.png'))
plt.close()

print("Visualisations gu00e9nu00e9ru00e9es avec succu00e8s dans le dossier 'visualizations'.")

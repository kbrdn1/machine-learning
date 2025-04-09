# Script principal pour exécuter toutes les analyses
# Ce script permet de lancer toutes les analyses du projet multi_dataset_analysis

import os
import subprocess
import sys
import time

def run_script(script_path, title):
    """
    Exécute un script Python et affiche son titre
    """
    print("\n" + "="*80)
    print(f"Exécution de: {title}")
    print("="*80 + "\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            text=True
        )
        print(f"\nScript terminé avec succès: {os.path.basename(script_path)}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nErreur lors de l'exécution du script {os.path.basename(script_path)}: {e}")
        return False

def main():
    """
    Fonction principale qui exécute toutes les analyses
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Liste des analyses à exécuter avec leur titre
    analyses = [
        {
            "path": os.path.join(base_dir, "body_performance_analysis", "body_performance.py"),
            "title": "ANALYSE DE LA PERFORMANCE PHYSIQUE"
        },
        {
            "path": os.path.join(base_dir, "credit_card_analysis", "credit_card.py"),
            "title": "ANALYSE DES DONNÉES DE CARTE DE CRÉDIT"
        },
        {
            "path": os.path.join(base_dir, "xy_regression_analysis", "xy_regression.py"),
            "title": "ANALYSE DE RÉGRESSION X-Y"
        }
    ]
    
    print("\n" + "="*80)
    print("DÉBUT DES ANALYSES DE MACHINE LEARNING")
    print("="*80 + "\n")
    
    start_time = time.time()
    success_count = 0
    
    # Exécution de chaque analyse
    for analysis in analyses:
        if run_script(analysis["path"], analysis["title"]):
            success_count += 1
    
    # Affichage du résumé
    elapsed_time = time.time() - start_time
    print("\n" + "="*80)
    print(f"RÉSUMÉ: {success_count}/{len(analyses)} analyses terminées avec succès")
    print(f"Temps total d'exécution: {elapsed_time:.2f} secondes")
    print("="*80 + "\n")
    
if __name__ == "__main__":
    main()
import subprocess
import sys
import torch
import os

# --- CONFIGURATION INITIALE ---
MODELS_AVAILABLE = {
    "CNN": "CNN.py",
    "RNN": "RNN.py",
    "LSTM": "LSTM.py",
    "GRU": "GRU.py",
    "CfC": "CfC.py",
    "LTC": "ltc_modern_demo.py"
}

DATASETS_AVAILABLE = ["sine", "har", "occupancy", "gesture", "traffic", "physionet"]

def get_input(prompt, default):
    user_input = input(f"{prompt} [{default}]: ")
    if not user_input.strip(): return default
    try:
        # Tente de convertir en entier si possible
        return int(user_input)
    except ValueError:
        return user_input

def run_command(cmd):
    print(f"\n[EXEC] {' '.join(cmd)}")
    try:
        subprocess.run([sys.executable] + cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERREUR] Échec de l'exécution : {e}")
        return False

def main():
    print("="*50)
    print("   MENU DE BENCHMARK GLOBAL")
    print("="*50)
    
    # 1. Configuration des Hyperparamètres
    epochs = get_input("Nombre d'époques (epochs)", 50)
    units = get_input("Nombre d'unités (units)", 32)
    layers = get_input("Nombre de couches (layers)", 1)
    batch_size = get_input("Taille du batch", 128)
    
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = input(f"Device (cuda/cpu) [{default_device}]: ").lower().strip() or default_device
    
    # 2. Sélection des Modèles
    print("\nModèles disponibles :", ", ".join(MODELS_AVAILABLE.keys()))
    m_choice = input("Modèles à tester (séparés par une virgule, ou Entrée pour TOUS) : ").strip()
    if m_choice:
        models_to_run = {k.strip().upper(): MODELS_AVAILABLE[k.strip().upper()] for k in m_choice.split(",") if k.strip().upper() in MODELS_AVAILABLE}
    else:
        models_to_run = MODELS_AVAILABLE

    # 3. Sélection des Datasets
    print("\nDatasets disponibles :", ", ".join(DATASETS_AVAILABLE))
    d_choice = input("Datasets à tester (séparés par une virgule, ou Entrée pour TOUS) : ").strip()
    if d_choice:
        datasets_to_run = [d.strip().lower() for d in d_choice.split(",") if d.strip().lower() in DATASETS_AVAILABLE]
    else:
        datasets_to_run = DATASETS_AVAILABLE

    if not models_to_run or not datasets_to_run:
        print("Erreur : Aucun modèle ou dataset sélectionné.")
        return

    # 4. Confirmation
    print("\n" + "-"*30)
    print(f"RÉSUMÉ DU BENCHMARK :")
    print(f"- Modèles  : {', '.join(models_to_run.keys())}")
    print(f"- Datasets : {', '.join(datasets_to_run)}")
    print(f"- Config   : {epochs} ep, {units} units, {layers} layers, {batch_size} batch, {device}")
    print("-"*30)
    confirm = input("Lancer le benchmark ? (y/n) [y]: ").lower().strip()
    if confirm == 'n':
        print("Annulé.")
        return

    # 5. Boucle d'exécution
    count = 0
    total = len(models_to_run) * len(datasets_to_run)
    
    for ds in datasets_to_run:
        print(f"\n>>> DATASET : {ds.upper()}")
        for name, script in models_to_run.items():
            count += 1
            print(f"\n[{count}/{total}] {name} sur {ds}...")
            
            cmd = [
                script,
                "--dataset", ds,
                "--epochs", str(epochs),
                "--units", str(units),
                "--layers", str(layers),
                "--batch_size", str(batch_size),
                "--device", device
            ]
            run_command(cmd)

    print("\n" + "="*50)
    print("TOUS LES ENTRAÎNEMENTS SONT TERMINÉS")
    print("="*50)
    
    # 6. Comparaison finale
    compare = input("\nGénérer les graphiques de comparaison maintenant ? (y/n) [y]: ").lower().strip()
    if compare != 'n':
        run_command(["compare_results.py"])
    
    print("\n[FIN] Terminé.")

if __name__ == "__main__":
    main()

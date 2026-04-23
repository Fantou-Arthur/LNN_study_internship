import subprocess
import sys
import torch
import os

# --- COULEURS ANSI ---
C_BLUE = "\033[94m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_CYAN = "\033[96m"
C_BOLD = "\033[1m"
C_END = "\033[0m"

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
    user_input = input(f"{C_CYAN}{prompt}{C_END} [{default}]: ")
    if not user_input.strip(): return default
    try: return int(user_input)
    except ValueError: return user_input

def run_command(cmd):
    print(f"\n{C_YELLOW}[EXEC]{C_END} {C_BOLD}{' '.join(cmd)}{C_END}")
    try:
        subprocess.run([sys.executable] + cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"{C_RED}[ERREUR] Échec de l'exécution : {e}{C_END}")
        return False

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"{C_BLUE}{C_BOLD}" + "="*50)
    print("   🚀 ORCHESTRATEUR DE BENCHMARK GLOBAL 🚀")
    print("="*50 + f"{C_END}")
    
    epochs = get_input("Nombre d'époques (epochs)", 50)
    units = get_input("Nombre d'unités (units)", 32)
    layers = get_input("Nombre de couches (layers)", 1)
    batch_size = get_input("Taille du batch", 128)
    
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = input(f"{C_CYAN}Device (cuda/cpu){C_END} [{default_device}]: ").lower().strip() or default_device
    
    print(f"\n{C_YELLOW}Modèles disponibles :{C_END}", ", ".join(MODELS_AVAILABLE.keys()))
    m_choice = input(f"{C_CYAN}Modèles à tester (Entrée pour TOUS) :{C_END} ").strip()
    if m_choice:
        models_to_run = {k.strip().upper(): MODELS_AVAILABLE[k.strip().upper()] for k in m_choice.split(",") if k.strip().upper() in MODELS_AVAILABLE}
    else:
        models_to_run = MODELS_AVAILABLE

    print(f"\n{C_YELLOW}Datasets disponibles :{C_END}", ", ".join(DATASETS_AVAILABLE))
    d_choice = input(f"{C_CYAN}Datasets à tester (Entrée pour TOUS) :{C_END} ").strip()
    if d_choice:
        datasets_to_run = [d.strip().lower() for d in d_choice.split(",") if d.strip().lower() in DATASETS_AVAILABLE]
    else:
        datasets_to_run = DATASETS_AVAILABLE

    if not models_to_run or not datasets_to_run:
        print(f"{C_RED}Erreur : Aucun modèle ou dataset sélectionné.{C_END}")
        return

    print(f"\n{C_BOLD}{C_BLUE}--- RÉSUMÉ DU BENCHMARK ---{C_END}")
    print(f"{C_CYAN}Modèles  :{C_END} {', '.join(models_to_run.keys())}")
    print(f"{C_CYAN}Datasets :{C_END} {', '.join(datasets_to_run)}")
    print(f"{C_CYAN}Config   :{C_END} {epochs} ep, {units} units, {layers} layers, {batch_size} batch, {device}")
    print(f"{C_BLUE}" + "-"*30 + f"{C_END}")
    
    confirm = input(f"{C_BOLD}{C_YELLOW}Lancer le benchmark ? (y/n) [y]: {C_END}").lower().strip()
    if confirm == 'n':
        print(f"{C_RED}Annulé.{C_END}")
        return

    count, total = 0, len(models_to_run) * len(datasets_to_run)
    for ds in datasets_to_run:
        print(f"\n{C_BOLD}{C_GREEN}>>> DATASET : {ds.upper()}{C_END}")
        for name, script in models_to_run.items():
            count += 1
            print(f"\n{C_YELLOW}[{count}/{total}]{C_END} {C_BOLD}{name}{C_END} sur {C_GREEN}{ds}{C_END}...")
            cmd = [script, "--dataset", ds, "--epochs", str(epochs), "--units", str(units), "--layers", str(layers), "--batch_size", str(batch_size), "--device", device]
            run_command(cmd)

    print(f"\n{C_BOLD}{C_BLUE}" + "="*50)
    print("✨ TOUS LES ENTRAÎNEMENTS SONT TERMINÉS ✨")
    print("="*50 + f"{C_END}")
    
    compare = input(f"\n{C_YELLOW}Générer les graphiques de comparaison ? (y/n) [y]: {C_END}").lower().strip()
    if compare != 'n':
        run_command(["compare_results.py"])
    
    print(f"\n{C_GREEN}{C_BOLD}[FIN] Terminé.{C_END}")

if __name__ == "__main__":
    main()

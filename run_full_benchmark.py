import subprocess
import os
import sys
import time

# --- COULEURS ANSI ---
C_BLUE = "\033[94m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_BOLD = "\033[1m"
C_END = "\033[0m"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print(f"\n{C_BLUE}{C_BOLD}" + "="*50)
    print("      🚀 NEURAL NETWORK BENCHMARK SUITE 🚀")
    print("="*50 + f"{C_END}\n")

def run_benchmark():
    datasets = ["sine", "har", "occupancy", "gesture", "traffic", "physionet"]
    models = ["LTC", "CfC", "CNN", "RNN", "LSTM", "GRU"]
    
    print_header()
    
    # 1. Configuration
    print(f"{C_YELLOW}--- CONFIGURATION GLOBALE ---{C_END}")
    epochs = input(f"Nombre d'époques [{C_BOLD}50{C_END}]: ") or "50"
    units = input(f"Nombre d'unités/neurones [{C_BOLD}32{C_END}]: ") or "32"
    layers = input(f"Nombre de couches [{C_BOLD}1{C_END}]: ") or "1"
    batch_size = input(f"Taille du batch [{C_BOLD}128{C_END}]: ") or "128"
    
    device = "cuda" if "cuda" in sys.argv else ("cpu" if "cpu" in sys.argv else "cuda")
    print(f"\n{C_BLUE}Device utilisé : {C_BOLD}{device.upper()}{C_END}")

    # 2. Sélection des datasets
    print(f"\n{C_YELLOW}--- SÉLECTION DES DATASETS ---{C_END}")
    for i, ds in enumerate(datasets):
        print(f"{i+1}. {ds}")
    ds_choice = input(f"\nChoix (ex: 1,2,5 ou 'all') [{C_BOLD}all{C_END}]: ") or "all"
    
    selected_datasets = datasets if ds_choice.lower() == "all" else [datasets[int(i)-1] for i in ds_choice.split(",")]

    # 3. Sélection des modèles
    print(f"\n{C_YELLOW}--- SÉLECTION DES MODÈLES ---{C_END}")
    for i, m in enumerate(models):
        print(f"{i+1}. {m}")
    m_choice = input(f"\nChoix (ex: 1,3 ou 'all') [{C_BOLD}all{C_END}]: ") or "all"
    
    selected_models = models if m_choice.lower() == "all" else [models[int(i)-1] for i in m_choice.split(",")]

    total_runs = len(selected_datasets) * len(selected_models)
    current_run = 0

    print(f"\n{C_BOLD}{C_GREEN}Lancement de {total_runs} tests...{C_END}\n")

    for ds in selected_datasets:
        for model in selected_models:
            current_run += 1
            script_name = "ltc_modern_demo.py" if model == "LTC" else f"{model}.py"
            
            print(f"{C_BLUE}[{current_run}/{total_runs}]{C_END} Entraînement {C_BOLD}{model}{C_END} sur {C_BOLD}{ds}{C_END}...")
            
            cmd = [
                "python", "-u", script_name,
                "--dataset", ds,
                "--epochs", epochs,
                "--units", units,
                "--layers", layers,
                "--batch_size", batch_size,
                "--device", device
            ]
            
            try:
                all_output = []
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                
                final_metrics = {"loss": None, "acc": None}

                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        all_output.append(line)
                        # Extraction des métriques finales pour le résumé
                        if "Test MSE:" in line: final_metrics["loss"] = line.split(":")[-1].strip()
                        if "Test Accuracy:" in line: final_metrics["acc"] = line.split(":")[-1].strip()
                        
                        # Affichage filtré
                        if any(key in line for key in ["Epoch", "Scheduler", "Success", "Test"]):
                            print(f"  {line.strip()}")
                            sys.stdout.flush()
                
                if process.returncode == 0:
                    # Résumé descriptif au lieu de "OK"
                    res_msg = f"✨ {C_BOLD}{model}{C_END} terminé sur {C_BOLD}{ds}{C_END} | "
                    if final_metrics["acc"]: res_msg += f"Précision: {C_GREEN}{final_metrics['acc']}{C_END}"
                    elif final_metrics["loss"]: res_msg += f"Perte: {C_GREEN}{final_metrics['loss']}{C_END}"
                    print(f"  {res_msg}\n")
                else:
                    print(f"{C_RED}  ❌ ÉCHEC : {model} sur {ds} (Code {process.returncode}){C_END}")
                    print(f"{C_RED}  Dernières lignes :{C_END}")
                    for line in all_output[-10:]: print(f"    {line.strip()}")
                    print("")
            except Exception as e:
                print(f"{C_RED}  [ERREUR] Impossible de lancer le script: {e}{C_END}\n")

    print(f"{C_BOLD}{C_GREEN}" + "="*50)
    print("         TOUS LES ENTRAÎNEMENTS SONT TERMINÉS")
    print("="*50 + f"{C_END}\n")

    gen_plots = input(f"Générer les graphiques de comparaison maintenant ? (y/n) [{C_BOLD}y{C_END}]: ") or "y"
    if gen_plots.lower() == "y":
        print(f"\n{C_BLUE}[EXEC]{C_END} compare_results.py")
        subprocess.run(["python", "compare_results.py"])

if __name__ == "__main__":
    try:
        run_benchmark()
    except KeyboardInterrupt:
        print(f"\n\n{C_RED}Benchmark interrompu par l'utilisateur.{C_END}")
        sys.exit(0)

import subprocess
import os
import sys
import time
import re
import numpy as np

# --- COULEURS ANSI ---
C_BLUE = "\033[94m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_BOLD = "\033[1m"
C_END = "\033[0m"

def run_command(cmd):
    """ Exécute une commande et capture la perte d'entraînement finale """
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        bufsize=1, 
        universal_newlines=True
    )
    
    last_train_loss = None
    last_test_metric = None
    
    # Regex pour extraire Loss Train et Test (MSE ou Accuracy)
    # Format: Loss Train: [BOLD]0.1234[END], Test: [YELLOW]0.5678[END]
    loss_pattern = re.compile(r"Loss Train:.*?(\d+\.\d+)")
    test_pattern = re.compile(r"Test:.*?(\d+\.\d+)")

    for line in process.stdout:
        print(line, end="")
        
        # Chercher la dernière perte d'entraînement
        train_match = loss_pattern.search(line)
        if train_match:
            last_train_loss = float(train_match.group(1))
            
        # Chercher la dernière métrique de test
        test_match = test_pattern.search(line)
        if test_match:
            last_test_metric = float(test_match.group(1))

    process.wait()
    return process.returncode, last_train_loss, last_test_metric

def main():
    print(f"\n{C_BOLD}{C_BLUE}==========================================")
    print("   ORCHESTRATEUR BENCHMARK EQUITABLE")
    print(f"=========================================={C_END}\n")

    standard_models = [
        {"name": "RNN", "script": "RNN.py"},
        {"name": "LSTM", "script": "LSTM.py"},
        {"name": "GRU", "script": "GRU.py"},
        {"name": "CNN", "script": "CNN.py"}
    ]
    advanced_models = [
        {"name": "LTC", "script": "ltc_modern_demo.py"},
        {"name": "CfC", "script": "CfC.py"}
    ]
    datasets = ["sine", "har", "occupancy", "gesture", "traffic", "physionet"]

    # --- CONFIGURATION ---
    print(f"{C_BOLD}--- CONFIGURATION ---{C_END}")
    epochs_std = input(f"Époques pour modèles Standards [{C_YELLOW}50{C_END}]: ") or "50"
    epochs_adv_max = input(f"Max époques pour LTC/CfC [{C_YELLOW}500{C_END}]: ") or "500"
    units = input(f"Nombre de neurones [{C_YELLOW}32{C_END}]: ") or "32"
    layers = input(f"Nombre de couches [{C_YELLOW}1{C_END}]: ") or "1"
    
    # Choix du device
    has_cuda = subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    def_device = "cuda" if has_cuda else "cpu"
    
    device_std = input(f"Device pour modèles Standards [{C_YELLOW}{def_device}{C_END}]: ") or def_device
    device_adv = input(f"Device pour LTC/CfC [{C_YELLOW}{def_device}{C_END}]: ") or def_device
    
    print(f"\nCalcul Standards sur : {C_GREEN}{device_std.upper()}{C_END}")
    print(f"Calcul LTC/CfC sur : {C_GREEN}{device_adv.upper()}{C_END}")

    final_results = []

    for dataset in datasets:
        print(f"\n{C_BOLD}{C_BLUE}>>> DATASET: {dataset.upper()} <<<{C_END}")
        
        # PHASE 1 : ÉTABLIR LA LIGNE DE BASE (STANDARDS)
        dataset_train_losses = []
        print(f"\n{C_BOLD}[PHASE 1] Entraînement des modèles standards pour base de référence...{C_END}")
        
        for model in standard_models:
            print(f"\n--- {model['name']} ---")
            cmd = [
                sys.executable, "-u", model["script"],
                "--units", units, "--layers", layers, "--epochs", epochs_std,
                "--dataset", dataset, "--device", device_std
            ]
            code, train_loss, test_metric = run_command(cmd)
            if code == 0 and train_loss is not None:
                dataset_train_losses.append(train_loss)
                print(f"  {C_GREEN}Succès.{C_END} Loss Train finale: {train_loss:.4f}")
            else:
                print(f"  {C_RED}Échec ou perte non capturée.{C_END}")

        if not dataset_train_losses:
            print(f"{C_RED}Aucune base de référence pour {dataset}. On passe au suivant.{C_END}")
            continue

        target_loss = sum(dataset_train_losses) / len(dataset_train_losses)
        print(f"\n{C_BOLD}{C_BLUE}Cible calculée pour {dataset} : Loss Train <= {target_loss:.4f}{C_END}")

        # PHASE 2 : ENTRAÎNER LTC/CFC JUSQU'À LA CIBLE
        print(f"\n{C_BOLD}[PHASE 2] Entraînement LTC/CfC jusqu'à atteindre la cible...{C_END}")
        
        for model in advanced_models:
            print(f"\n--- {model['name']} (Cible: {target_loss:.4f}) ---")
            cmd = [
                sys.executable, "-u", model["script"],
                "--units", units, "--layers", layers, "--epochs", epochs_adv_max,
                "--dataset", dataset, "--device", device_adv,
                "--target_loss", f"{target_loss:.6f}"
            ]
            code, train_loss, test_metric = run_command(cmd)
            
            status = "SUCCESS: Reached" if (train_loss and train_loss <= target_loss * 1.05) else "FAIL: Not reached"
            final_results.append({
                "dataset": dataset,
                "model": model["name"],
                "target": target_loss,
                "final_train_loss": train_loss,
                "test_metric": test_metric,
                "status": status
            })

    # --- RÉSUMÉ FINAL ---
    print(f"\n\n{C_BOLD}{C_BLUE}==========================================")
    print("   RESUME DU BENCHMARK EQUITABLE")
    print(f"=========================================={C_END}")
    
    current_ds = ""
    for res in final_results:
        if res['dataset'] != current_ds:
            current_ds = res['dataset']
            print(f"\n{C_BOLD}Dataset: {current_ds.upper()}{C_END} (Cible Avg Loss: {res['target']:.4f})")
        
        color = C_GREEN if "SUCCESS" in res['status'] else C_RED
        print(f"  - {res['model']}: Loss {res['final_train_loss']:.4f} | Test: {res['test_metric']} | {color}{res['status']}{C_END}")

if __name__ == "__main__":
    main()

import subprocess
import sys
import time
import os
import itertools

# Codes couleur pour le terminal
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
END = "\033[0m"

def print_progress_bar(iteration, total, length=40):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = "█" * filled_length + "-" * (length - filled_length)
    print(f"\r{BOLD}Progression : |{bar}| {percent}% terminé{END}", end="\r")

def run_cnn_benchmarks():
    # --- CONFIGURATION DE LA GRILLE DE RECHERCHE ---
    datasets = ["sine", "occupancy", "physionet"] # Vous pouvez ajouter "har", "gesture", "traffic"
    units_list = [16, 32, 64]
    layers_list = [2, 3]
    activations = ["relu", "leaky_relu"]
    
    epochs = 30
    batch_size = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # -----------------------------------------------

    # Calcul de toutes les combinaisons possibles
    combinations = list(itertools.product(datasets, units_list, layers_list, activations))
    total = len(combinations)
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"{BOLD}{BLUE}======================================================")
    print(f"       BENCHMARK MULTI-COMBINAISONS CNN")
    print(f"======================================================{END}")
    print(f"{BLUE}Hardware : {device.upper()} | Total de combinaisons : {total}{END}\n")

    start_global = time.time()

    for i, (ds, units, layers, act) in enumerate(combinations):
        print_progress_bar(i, total)
        config_desc = f"DS: {ds} | Units: {units} | Layers: {layers} | Act: {act}"
        print(f"\n\n{BOLD}{YELLOW}[{i+1}/{total}] DÉMARRAGE : {config_desc}{END}")
        print(f"{YELLOW}------------------------------------------------------{END}")
        
        cmd = [
            sys.executable, "-u", "CNN.py",
            "--dataset", ds,
            "--units", str(units),
            "--layers", str(layers),
            "--activation", act,
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--device", device
        ]
        
        try:
            # Lancement du processus
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            # Lecture de la sortie en temps réel
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                
            process.wait()
            
            if process.returncode == 0:
                print(f"{GREEN}✔ Combinaison terminée avec succès.{END}")
            else:
                print(f"{RED}✘ ERREUR sur cette combinaison.{END}")
                
        except Exception as e:
            print(f"{RED}⚠ Erreur critique : {e}{END}")

    print_progress_bar(total, total)
    total_time = (time.time() - start_global) / 60
    print(f"\n\n{BOLD}{GREEN}======================================================")
    print(f"   BENCHMARK TERMINÉ EN {total_time:.2f} MINUTES")
    print(f"======================================================{END}")

if __name__ == "__main__":
    import torch # Importé ici pour le choix du device par défaut
    if os.name == 'nt':
        os.system('') # Activer les couleurs ANSI sur Windows
    
    run_cnn_benchmarks()

import subprocess
import sys
import time
import os

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
    print(f"\r{BOLD}Progression globale : |{bar}| {percent}% terminé{END}", end="\r")

def run_all_benchmarks():
    # Configuration des datasets et de leur perte cible respective
    # Dataset: Target_Loss
    datasets_config = {
        "sine": 0.005,
        "occupancy": 0.1,
        "traffic": 0.12,
        "har": 0.5,
        "gesture": 0.3
    }
    total = len(datasets_config)
    
    # Paramètres globaux
    units = 32
    epochs = 100
    batch_size = 64
    device = "cuda"
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"{BOLD}{BLUE}======================================================")
    print(f"       BENCHMARK GLOBAL DES RÉSEAUX LTC")
    print(f"======================================================{END}")
    print(f"{BLUE}Hardware : {device.upper()} | Units : {units} | Max Epochs : {epochs}{END}\n")

    start_global = time.time()

    for i, (ds, t_loss) in enumerate(datasets_config.items()):
        print_progress_bar(i, total)
        print(f"\n\n{BOLD}{YELLOW}[{i+1}/{total}] DÉMARRAGE : {ds.upper()} (Target Loss: {t_loss}){END}")
        print(f"{YELLOW}------------------------------------------------------{END}")
        
        # Construction de la commande avec -u pour désactiver le buffering
        cmd = [
            sys.executable, "-u", "ltc_modern_demo.py",
            "--dataset", ds,
            "--units", str(units),
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--target_loss", str(t_loss),
            "--device", device
        ]
        
        try:
            # On lance sans buffering pour le temps réel
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            last_char_was_r = False
            while True:
                char = process.stdout.read(1)
                if not char and process.poll() is not None:
                    break
                if char:
                    sys.stdout.write(char)
                    sys.stdout.flush()
                
            process.wait()
            print() # Nouvelle ligne après le \r final
            
            if process.returncode == 0:
                print(f"{GREEN}✔ Dataset '{ds}' terminé avec succès.{END}")
            else:
                print(f"{RED}✘ ERREUR sur le dataset '{ds}'.{END}")
                
        except Exception as e:
            print(f"{RED}⚠ Erreur critique sur '{ds}': {e}{END}")

    print_progress_bar(total, total)
    total_time = (time.time() - start_global) / 60
    print(f"\n\n{BOLD}{GREEN}======================================================")
    print(f"   BENCHMARK TERMINÉ EN {total_time:.2f} MINUTES")
    print(f"======================================================{END}")

if __name__ == "__main__":
    # Activer les couleurs sur Windows
    if os.name == 'nt':
        os.system('')
        
    run_all_benchmarks()

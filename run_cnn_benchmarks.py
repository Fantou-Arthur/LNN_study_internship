import subprocess
import sys
import time
import os
import itertools

# Terminal color codes
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
    print(f"\r{BOLD}Progress: |{bar}| {percent}% completed{END}", end="\r")

def run_cnn_benchmarks():
    # --- SEARCH GRID CONFIGURATION ---
    datasets = ["sine", "occupancy", "physionet"] # You can add "har", "gesture", "traffic"
    units_list = [16, 32, 64]
    layers_list = [2, 3]
    activations = ["relu", "leaky_relu"]
    
    epochs = 30
    batch_size = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # -----------------------------------------------

    # Calculation of all possible combinations
    combinations = list(itertools.product(datasets, units_list, layers_list, activations))
    total = len(combinations)
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"{BOLD}{BLUE}======================================================")
    print(f"       MULTI-COMBINATION CNN BENCHMARK")
    print(f"======================================================{END}")
    print(f"{BLUE}Hardware: {device.upper()} | Total combinations: {total}{END}\n")

    start_global = time.time()

    for i, (ds, units, layers, act) in enumerate(combinations):
        print_progress_bar(i, total)
        config_desc = f"DS: {ds} | Units: {units} | Layers: {layers} | Act: {act}"
        print(f"\n\n{BOLD}{YELLOW}[{i+1}/{total}] STARTING: {config_desc}{END}")
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
            # Launch process
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            # Read output in real-time
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                
            process.wait()
            
            if process.returncode == 0:
                print(f"{GREEN}✔ Combination completed successfully.{END}")
            else:
                print(f"{RED}✘ ERROR on this combination.{END}")
                
        except Exception as e:
            print(f"{RED}⚠ Critical error: {e}{END}")

    print_progress_bar(total, total)
    total_time = (time.time() - start_global) / 60
    print(f"\n\n{BOLD}{GREEN}======================================================")
    print(f"   BENCHMARK COMPLETED IN {total_time:.2f} MINUTES")
    print(f"======================================================{END}")

if __name__ == "__main__":
    import torch # Imported here for default device choice
    if os.name == 'nt':
        os.system('') # Enable ANSI colors on Windows
    
    run_cnn_benchmarks()

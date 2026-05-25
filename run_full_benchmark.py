import subprocess
import os
import sys
import time

# --- ANSI COLORS ---
C_BLUE = "\033[94m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_BOLD = "\033[1m"
C_END = "\033[0m"

def run_command(cmd):
    """ Executes a command and displays output in real-time (unbuffered) """
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        bufsize=1, 
        universal_newlines=True
    )
    last_line = ""
    for line in process.stdout:
        print(line, end="")
        if "Test MSE:" in line or "Test Accuracy:" in line:
            last_line = line.strip()
    process.wait()
    return process.returncode, last_line

def main():
    print(f"\n{C_BOLD}{C_BLUE}==========================================")
    print("   🚀 FULL BENCHMARK ORCHESTRATOR")
    print(f"=========================================={C_END}\n")

    # Default configuration
    models = [
        {"name": "LTC", "script": "ltc_modern_demo.py", "epochs_mult": 3}, # LTC/CfC often need more epochs
        {"name": "CfC", "script": "CfC.py", "epochs_mult": 3},
        {"name": "RNN", "script": "RNN.py"},
        {"name": "LSTM", "script": "LSTM.py"},
        {"name": "GRU", "script": "GRU.py"},
        {"name": "CNN", "script": "CNN.py"}
    ]
    datasets = ["sine", "har", "occupancy", "gesture", "traffic", "physionet"]

    # --- STEP 1: CONFIGURATION ---
    print(f"{C_BOLD}--- GLOBAL CONFIGURATION ---{C_END}")
    epochs = input(f"Number of epochs [{C_YELLOW}50{C_END}]: ") or "50"
    units = input(f"Number of units/filters [{C_YELLOW}32{C_END}]: ") or "32"
    layers = input(f"Number of layers [{C_YELLOW}1{C_END}]: ") or "1"
    batch_size = input(f"Batch size [{C_YELLOW}128{C_END}]: ") or "128"
    
    # --- STEP 2: ROBUSTNESS (SPARSITY) ---
    print(f"\n{C_BOLD}--- ROBUSTNESS TEST (Missing Data) ---{C_END}")
    print("Option B will be applied (individual cell removal).")
    sp_train = input(f"Missing percentage TRAIN (0-100) [{C_YELLOW}0{C_END}]: ") or "0"
    sp_test = input(f"Missing percentage TEST (0-100) [{C_YELLOW}0{C_END}]: ") or "0"
    sp_mode = input(f"Mode (random/periodic) [{C_YELLOW}random{C_END}]: ") or "random"
    
    # Convert to ratio 0.0 - 1.0
    sparsity_train = float(sp_train) / 100.0
    sparsity_test = float(sp_test) / 100.0

    device = "cuda" if subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0 else "cpu"
    print(f"\n{C_BOLD}Computing on: {C_GREEN}{device.upper()}{C_END}")
    
    total_tests = len(models) * len(datasets)
    print(f"Launching {total_tests} tests...\n")

    results_summary = []
    start_time = time.time()

    # --- STEP 3: BENCHMARK LOOP ---
    for i, model in enumerate(models):
        for j, dataset in enumerate(datasets):
            idx = i * len(datasets) + j + 1
            print(f"[{idx}/{total_tests}] Training {C_BOLD}{model['name']}{C_END} on {C_YELLOW}{dataset}{C_END}...")
            
            # Apply model-specific overrides (e.g., epochs_mult)
            current_epochs = str(int(int(epochs) * model.get("epochs_mult", 1)))
            current_units = model.get("units_override", units)
            current_layers = model.get("layers_override", layers)

            cmd = [
                sys.executable, "-u", model["script"],
                "--units", current_units,
                "--layers", current_layers,
                "--epochs", current_epochs,
                "--batch_size", batch_size,
                "--dataset", dataset,
                "--device", device,
                "--sparsity_train", str(sparsity_train),
                "--sparsity_test", str(sparsity_test),
                "--sparsity_mode", sp_mode
            ]

            code, last_metrics = run_command(cmd)

            if code == 0:
                results_summary.append(f"✨ {C_GREEN}{model['name']}{C_END} on {dataset} completed. | {last_metrics}")
            else:
                print(f"  {C_RED}[ERROR]{C_END} Script failed with code {code}.")
                results_summary.append(f"❌ {C_RED}{model['name']}{C_END} on {dataset} FAILED.")

    # --- STEP 4: SUMMARY ---
    duration = time.time() - start_time
    print(f"\n{C_BOLD}{C_BLUE}==========================================")
    print(f"   🏁 BENCHMARK COMPLETED in {duration/60:.1f} min")
    print(f"=========================================={C_END}")
    for res in results_summary:
        print(f"  {res}")
    
    print(f"\n{C_BOLD}Weights (.pt) and evaluation data have been saved in the 'results/' folder.{C_END}")

if __name__ == "__main__":
    main()

import subprocess
import os
import sys
import time
import re
import numpy as np

# --- ANSI COLORS ---
C_BLUE = "\033[94m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_BOLD = "\033[1m"
C_END = "\033[0m"

def run_command(cmd):
    """ Executes a command and captures the final training loss """
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
    
    # Regex to extract Train Loss and Test (MSE or Accuracy)
    # Format: Loss Train: [BOLD]0.1234[END], Test: [YELLOW]0.5678[END]
    loss_pattern = re.compile(r"Loss Train:.*?(\d+\.\d+)")
    test_pattern = re.compile(r"Test:.*?(\d+\.\d+)")

    for line in process.stdout:
        print(line, end="")
        
        # Look for the last training loss
        train_match = loss_pattern.search(line)
        if train_match:
            last_train_loss = float(train_match.group(1))
            
        # Look for the last test metric
        test_match = test_pattern.search(line)
        if test_match:
            last_test_metric = float(test_match.group(1))

    process.wait()
    return process.returncode, last_train_loss, last_test_metric

def main():
    print(f"\n{C_BOLD}{C_BLUE}==========================================")
    print("   FAIR BENCHMARK ORCHESTRATOR")
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
    epochs_std = input(f"Epochs for Standard models [{C_YELLOW}50{C_END}]: ") or "50"
    epochs_adv_max = input(f"Max epochs for LTC/CfC [{C_YELLOW}500{C_END}]: ") or "500"
    min_epochs_adv = input(f"Min epochs for LTC/CfC [{C_YELLOW}50{C_END}]: ") or "50"
    units = input(f"Number of units [{C_YELLOW}32{C_END}]: ") or "32"
    layers = input(f"Number of layers [{C_YELLOW}1{C_END}]: ") or "1"
    
    # Device choice
    has_cuda = subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    def_device = "cuda" if has_cuda else "cpu"
    
    device_std = input(f"Device for Standard models [{C_YELLOW}{def_device}{C_END}]: ") or def_device
    device_adv = input(f"Device for LTC/CfC [{C_YELLOW}{def_device}{C_END}]: ") or def_device
    
    print(f"\nComputing Standards on: {C_GREEN}{device_std.upper()}{C_END}")
    print(f"Computing LTC/CfC on: {C_GREEN}{device_adv.upper()}{C_END}")

    final_results = []

    for dataset in datasets:
        print(f"\n{C_BOLD}{C_BLUE}>>> DATASET: {dataset.upper()} <<<{C_END}")
        
        # PHASE 1 : ESTABLISH BASELINE (STANDARDS)
        dataset_train_losses = []
        print(f"\n{C_BOLD}[PHASE 1] Training standard models for baseline...{C_END}")
        
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
                print(f"  {C_GREEN}Success.{C_END} Final Train Loss: {train_loss:.4f}")
            else:
                print(f"  {C_RED}Failure or loss not captured.{C_END}")

        if not dataset_train_losses:
            print(f"{C_RED}No baseline for {dataset}. Skipping.{C_END}")
            continue

        target_loss = float(np.median(dataset_train_losses))
        print(f"\n{C_BOLD}{C_BLUE}Target calculated for {dataset} : Train Loss <= {target_loss:.4f} (Median){C_END}")

        # PHASE 2 : TRAIN LTC/CFC UNTIL TARGET OR EPOCH LIMIT
        print(f"\n{C_BOLD}[PHASE 2] Training LTC/CfC (min {min_epochs_adv} epochs, max {epochs_adv_max})...{C_END}")
        
        for model in advanced_models:
            print(f"\n--- {model['name']} (Target: {target_loss:.4f}) ---")
            cmd = [
                sys.executable, "-u", model["script"],
                "--units", units, "--layers", layers, "--epochs", epochs_adv_max,
                "--dataset", dataset, "--device", device_adv,
                "--target_loss", f"{target_loss:.6f}",
                "--min_epochs", min_epochs_adv
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

    # --- FINAL SUMMARY ---
    print(f"\n\n{C_BOLD}{C_BLUE}==========================================")
    print("   FAIR BENCHMARK SUMMARY")
    print(f"=========================================={C_END}")
    
    current_ds = ""
    for res in final_results:
        if res['dataset'] != current_ds:
            current_ds = res['dataset']
            print(f"\n{C_BOLD}Dataset: {current_ds.upper()}{C_END} (Target Avg Loss: {res['target']:.4f})")
        
        color = C_GREEN if "SUCCESS" in res['status'] else C_RED
        train_loss_str = f"{res['final_train_loss']:.4f}" if res['final_train_loss'] is not None else "N/A"
        test_metric_str = f"{res['test_metric']:.4f}" if res['test_metric'] is not None else "N/A"
        
        print(f"  - {res['model']}: Loss {train_loss_str} | Test: {test_metric_str} | {color}{res['status']}{C_END}")

if __name__ == "__main__":
    main()

import subprocess
import os
import sys
import time
import argparse
import numpy as np
import statistics
from datetime import timedelta

# --- ANSI COLORS ---
C_BLUE = "\033[94m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_BOLD = "\033[1m"
C_CYAN = "\033[96m"
C_PURPLE = "\033[95m"
C_END = "\033[0m"

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def run_command(cmd, description):
    """ Executes a command and shows output in real-time, returns the final test loss if found. """
    print(f"\n{C_BLUE}{C_BOLD}[EXECUTING]{C_END} {description}")
    print(f"{C_YELLOW}> {' '.join(cmd)}{C_END}")
    
    if cmd[0] == sys.executable and "-u" not in cmd:
        cmd.insert(1, "-u")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    final_test_loss = None
    for line in process.stdout:
        print(f"  {line.strip()}", flush=True)
        if "Final Test Loss:" in line:
            try:
                final_test_loss = float(line.split(":")[-1].strip())
            except:
                pass
    
    process.wait()
    if process.returncode != 0:
        print(f"{C_RED}[FAILED]{C_END} Command returned code {process.returncode}", flush=True)
    return process.returncode, final_test_loss

def get_input(prompt, default):
    user_input = input(f"{prompt} [{C_YELLOW}{default}{C_END}]: ")
    if not user_input.strip():
        return default
    return user_input.strip()

def print_progress_bar(iteration, total, start_time, prefix='', suffix='', length=40):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    
    elapsed_time = time.time() - start_time
    if iteration > 0:
        eta_seconds = (elapsed_time / iteration) * (total - iteration)
        eta_str = format_time(eta_seconds)
    else:
        eta_str = "Calculating..."
        
    print(f"\n{C_PURPLE}{C_BOLD}{prefix} |{bar}| {percent}% {suffix} (Remaining: {eta_str}){C_END}\n")

def main():
    parser = argparse.ArgumentParser(description="Automated Study of Training Sparsity Impact")
    parser.add_argument("--datasets", nargs="+", help="Datasets to include")
    parser.add_argument("--epochs_baseline", type=int, help="Epochs for baseline models")
    parser.add_argument("--min_epochs_ltc", type=int, help="Minimum epochs for LTC/CfC")
    parser.add_argument("--max_epochs_ltc", type=int, help="Maximum epochs for LTC/CfC")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--units", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--start_sp", type=int)
    parser.add_argument("--end_sp", type=int)
    parser.add_argument("--step_sp", type=int)
    parser.add_argument("--mode", type=str, choices=["random", "periodic"])
    parser.add_argument("--skip_training", action="store_true")
    
    args = parser.parse_args()

    # --- INTERACTIVE MENU IF NO ARGS ---
    ALL_DATASETS = "sine,har,occupancy,gesture,traffic,physionet"
    if len(sys.argv) <= 1:
        print(f"\n{C_BOLD}{C_BLUE}--- Training Sparsity Study Configuration ---{C_END}")
        ds_input = get_input("Datasets (comma separated)", ALL_DATASETS)
        datasets = [d.strip() for d in ds_input.split(",")]
        
        epochs_baseline = int(get_input("Baseline Epochs (RNN/CNN)", 50))
        min_epochs_ltc = int(get_input("Min Epochs (LTC/CfC)", 20))
        max_epochs_ltc = int(get_input("Max Epochs (LTC/CfC)", 150))
        batch_size = int(get_input("Batch Size", 128))
        
        units = int(get_input("Units/Neurons", 32))
        layers = int(get_input("Layers", 1))
        
        start_sp = int(get_input("Start Training Sparsity %", 0))
        end_sp = int(get_input("End Training Sparsity %", 90))
        step_sp = int(get_input("Sparsity Step %", 10))
        
        mode = get_input("Sparsity Mode (random/periodic)", "random")
        skip_training = False
    else:
        datasets = args.datasets or ALL_DATASETS.split(",")
        epochs_baseline = args.epochs_baseline or 50
        min_epochs_ltc = args.min_epochs_ltc or 20
        max_epochs_ltc = args.max_epochs_ltc or 150
        batch_size = args.batch_size
        units = args.units
        layers = args.layers
        start_sp = args.start_sp if args.start_sp is not None else 0
        end_sp = args.end_sp if args.end_sp is not None else 90
        step_sp = args.step_sp if args.step_sp is not None else 10
        mode = args.mode or "random"
        skip_training = args.skip_training

    cuda_device = "cuda" if subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0 else "cpu"
    
    sparsity_range = list(range(start_sp, end_sp + 1, step_sp))
    total_steps = len(sparsity_range) * len(datasets)
    step_idx = 0
    
    print(f"\n{C_BOLD}{C_BLUE}==========================================")
    print("   🌐 ADVANCED TRAINING SPARSITY STUDY")
    print(f"=========================================={C_END}")
    print(f"Datasets: {datasets}")
    print(f"Sparsity levels to test: {sparsity_range}")
    print(f"Batch Size: {batch_size}")
    print(f"LTC/CfC Device: CPU | Baselines: {cuda_device.upper()}")
    print(f"------------------------------------------\n")

    baselines = [{"name": "RNN", "script": "RNN.py"}, {"name": "LSTM", "script": "LSTM.py"}, 
                 {"name": "GRU", "script": "GRU.py"}, {"name": "CNN", "script": "CNN.py"}]
    specials = [{"name": "LTC", "script": "ltc_modern_demo.py"}, {"name": "CfC", "script": "CfC.py"}]

    start_total_time = time.time()

    for sp_train_pct in sparsity_range:
        sp_train_val = sp_train_pct / 100.0
        print(f"\n{C_BOLD}{C_GREEN}>>> TARGETING TRAINING SPARSITY: {sp_train_pct}% ({mode}) <<<{C_END}")
        
        for ds in datasets:
            step_idx += 1
            print(f"\n{C_BOLD}--- Processing Dataset: {ds.upper()} ({step_idx}/{total_steps}) ---{C_END}")
            
            if not skip_training:
                baseline_losses = []
                # 1. Train Baselines
                for b in baselines:
                    desc = f"Training {b['name']} on {ds} (CUDA)"
                    cmd = [sys.executable, b["script"], "--dataset", ds, "--epochs", str(epochs_baseline),
                           "--units", str(units), "--layers", str(layers), "--batch_size", str(batch_size),
                           "--sparsity_train", str(sp_train_val), "--sparsity_mode", mode, "--device", cuda_device]
                    code, loss = run_command(cmd, desc)
                    if code == 0 and loss is not None: baseline_losses.append(loss)
                
                # 2. Median
                median_loss = statistics.median(baseline_losses) if baseline_losses else 0.0
                print(f"\n{C_CYAN}Median Baseline Loss for {ds}: {median_loss:.6f}{C_END}")

                # 3. Specials
                for s in specials:
                    desc = f"Training {s['name']} on {ds} (CPU) | Target: {median_loss:.6f}"
                    cmd = [sys.executable, s["script"], "--dataset", ds, "--epochs", str(max_epochs_ltc),
                           "--min_epochs", str(min_epochs_ltc), "--target_loss", str(median_loss), "--batch_size", str(batch_size),
                           "--units", str(units), "--layers", str(layers), "--sparsity_train", str(sp_train_val),
                           "--sparsity_mode", mode, "--device", "cpu"]
                    run_command(cmd, desc)
            
            # --- PHASE 2 & 3 (Robustness & Visualization) per Dataset ---
            # Generate robustness curves (only if comparing multiple models for this dataset)
            # Actually we usually run compare_robustness per sparsity level for ALL datasets.
            # But the user wants the progress bar after each dataset. 
            # I'll move compare_robustness outside the dataset loop to keep it efficient, 
            # OR run it per dataset if it supports it. It does (via --datasets).
            
            run_command([sys.executable, "compare_robustness.py", "--datasets", ds, 
                         "--sparsity_train", str(sp_train_val), "--train_mode", mode, "--device", cuda_device],
                        f"Generating Robustness Curves for {ds}")

            run_command([sys.executable, "visualize_predictions.py", "--dataset", ds, 
                         "--sparsity_train", str(sp_train_val), "--sparsity_mode", mode, "--device", cuda_device],
                        f"Generating Prediction Visualizations for {ds}")

            # Update progress after each dataset
            print_progress_bar(step_idx, total_steps, start_total_time, 
                               prefix='Overall Study Progress', suffix='Datasets Complete')

    total_duration = time.time() - start_total_time
    print(f"\n{C_BOLD}{C_GREEN}==========================================")
    print(f"   ✅ STUDY COMPLETED in {format_time(total_duration)}")
    print(f"=========================================={C_END}")

if __name__ == "__main__":
    main()

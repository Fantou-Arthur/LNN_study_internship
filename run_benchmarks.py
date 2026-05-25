import subprocess
import sys
import time
import os

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
    print(f"\r{BOLD}Global Progress: |{bar}| {percent}% completed{END}", end="\r")

def run_all_benchmarks():
    # Dataset configuration and their respective target losses
    # Dataset: Target_Loss
    datasets_config = {
        "sine": 0.005,
        "occupancy": 0.1,
        "traffic": 0.12,
        "har": 0.5,
        "gesture": 0.3
    }
    total = len(datasets_config)
    
    # Global parameters
    units = 32
    epochs = 100
    batch_size = 64
    device = "cuda"
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"{BOLD}{BLUE}======================================================")
    print(f"       GLOBAL LTC NETWORK BENCHMARK")
    print(f"======================================================{END}")
    print(f"{BLUE}Hardware: {device.upper()} | Units: {units} | Max Epochs: {epochs}{END}\n")

    start_global = time.time()

    for i, (ds, t_loss) in enumerate(datasets_config.items()):
        print_progress_bar(i, total)
        print(f"\n\n{BOLD}{YELLOW}[{i+1}/{total}] STARTING: {ds.upper()} (Target Loss: {t_loss}){END}")
        print(f"{YELLOW}------------------------------------------------------{END}")
        
        # Command construction with -u to disable buffering
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
            # Launch without buffering for real-time output
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            while True:
                char = process.stdout.read(1)
                if not char and process.poll() is not None:
                    break
                if char:
                    sys.stdout.write(char)
                    sys.stdout.flush()
                
            process.wait()
            print() # New line after the final \r
            
            if process.returncode == 0:
                print(f"{GREEN}✔ Dataset '{ds}' completed successfully.{END}")
            else:
                print(f"{RED}✘ ERROR on dataset '{ds}'.{END}")
                
        except Exception as e:
            print(f"{RED}⚠ Critical error on '{ds}': {e}{END}")

    print_progress_bar(total, total)
    total_time = (time.time() - start_global) / 60
    print(f"\n\n{BOLD}{GREEN}======================================================")
    print(f"   BENCHMARK COMPLETED IN {total_time:.2f} MINUTES")
    print(f"======================================================{END}")

if __name__ == "__main__":
    # Enable colors on Windows
    if os.name == 'nt':
        os.system('')
        
    run_all_benchmarks()

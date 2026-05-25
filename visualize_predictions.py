import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import glob

# Import models
from ltc_modern_demo import ModernLTCModel
from CfC import ModernCfCModel
from RNN import ModernRNNModel
from LSTM import ModernLSTMModel
from GRU import ModernGRUModel
from CNN import ModernCNNModel

# ANSI Colors
C_BLUE = "\033[94m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_BOLD = "\033[1m"
C_END = "\033[0m"

MODELS = {
    "ltc": ModernLTCModel,
    "cfc": ModernCfCModel,
    "rnn": ModernRNNModel,
    "lstm": ModernLSTMModel,
    "gru": ModernGRUModel,
    "cnn": ModernCNNModel
}

def apply_sparsity(data, sparsity, mode="random"):
    if sparsity <= 0:
        return data.clone(), torch.ones_like(data)
    sparse_data = data.clone()
    batch, seq, feat = sparse_data.shape
    if mode == "random":
        mask = torch.rand(batch, seq, feat, device=data.device) > sparsity
    else:
        period = int(1 / (1 - sparsity)) if sparsity < 1 else 1
        mask = torch.zeros(seq, device=data.device)
        mask[::period] = 1
        mask = mask.view(1, seq, 1).expand(batch, seq, feat)
    return sparse_data * mask.float(), mask

def find_latest_files(model_name, dataset_name, sparsity_train=None, sparsity_mode=None):
    """
    Finds the latest weight and data files. 
    If sparsity_train and sparsity_mode are provided, it looks in the specific subfolder.
    """
    if sparsity_train is not None and sparsity_mode is not None:
        subfolder = f"{int(sparsity_train * 100)}_{sparsity_mode}"
        base_path = os.path.join("results", model_name, dataset_name, subfolder)
    else:
        base_path = os.path.join("results", model_name, dataset_name)
        
    if not os.path.exists(base_path): return None, None
    
    # Recursive search if no specific subfolder, or direct search if subfolder is set
    pattern = "**" if sparsity_train is None else ""
    weight_files = glob.glob(os.path.join(base_path, pattern, "weights_*.pt"), recursive=True)
    data_files = glob.glob(os.path.join(base_path, pattern, "eval_data_*.pt"), recursive=True)
    
    if not weight_files or not data_files: return None, None
    
    latest_weights = max(weight_files, key=os.path.getmtime)
    dir_of_weights = os.path.dirname(latest_weights)
    
    # Extract config string to find matching eval data
    config_str = os.path.basename(latest_weights).replace("weights_", "")
    matching_data = os.path.join(dir_of_weights, "eval_data_" + config_str)
    
    if os.path.exists(matching_data): return latest_weights, matching_data
    return latest_weights, max(data_files, key=os.path.getmtime)

def main():
    parser = argparse.ArgumentParser(description="Multi-model Prediction Visualization")
    parser.add_argument("--dataset", type=str, default="sine", help="Dataset to visualize")
    parser.add_argument("--mode", type=str, default="random", choices=["random", "periodic"], help="Evaluation sparsity mode")
    parser.add_argument("--sample_idx", type=int, default=0, help="Test sample index")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # New arguments for training configuration folder
    parser.add_argument("--sparsity_train", type=float, default=None, help="Training sparsity (0.0-1.0)")
    parser.add_argument("--sparsity_mode", type=str, default=None, help="Training sparsity mode (random/periodic)")
    
    args = parser.parse_args()

    device = torch.device(args.device)
    sparsity_levels = np.arange(0.05, 1.0, 0.05) # 5%, 10%, ..., 95%
    
    # Determine the output directory based on training configuration
    if args.sparsity_train is not None and args.sparsity_mode is not None:
        training_config_folder = f"{int(args.sparsity_train * 100)}_{args.sparsity_mode}"
        output_dir = os.path.join("results", "visualizations", args.dataset, training_config_folder)
    else:
        output_dir = os.path.join("results", "visualizations", args.dataset, "latest")
        print(f"{C_YELLOW}[!] Sparsity train/mode not specified, saving in 'latest' folder.{C_END}")
        
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{C_BOLD}{C_BLUE}--- GENERATING VISUALIZATIONS ({args.dataset.upper()}) ---{C_END}")
    print(f"Targeting models trained with: {C_YELLOW}{args.sparsity_train*100 if args.sparsity_train is not None else 'Any'}% {args.sparsity_mode if args.sparsity_mode else 'Any'}{C_END}")

    # 1. Load models and data
    loaded_models = {}
    x_test, y_test, is_classification, output_size = None, None, None, None

    for m_name in MODELS.keys():
        w_path, d_path = find_latest_files(m_name, args.dataset, args.sparsity_train, args.sparsity_mode)
        if not w_path: 
            # If specified but not found, try any latest if sparsity_train was None
            if args.sparsity_train is None: continue
            else: 
                print(f"  {C_RED}[MISSING]{C_END} No models found for {m_name} with specified config.")
                continue

        print(f"  Loading {m_name} from {C_YELLOW}{os.path.basename(os.path.dirname(w_path))}{C_END}...")
        try:
            data_bundle = torch.load(d_path, map_location="cpu")
            if x_test is None:
                x_test, y_test = data_bundle["x_test"], data_bundle["y_test"]
                is_classification = data_bundle["is_classification"]
                output_size = data_bundle["output_size"]
            
            # Extract config from name
            fname = os.path.basename(w_path)
            parts = fname.split("_")
            units, layers = 32, 1
            for p in parts:
                if p.endswith("u") and p[:-1].isdigit(): units = int(p[:-1])
                if p.endswith("L") and p[:-1].isdigit(): layers = int(p[:-1])

            if m_name == "cnn":
                model = MODELS[m_name](data_bundle["input_size"], units, output_size, layers, "relu").to(device)
            else:
                model = MODELS[m_name](data_bundle["input_size"], units, output_size, layers).to(device)
            
            model.load_state_dict(torch.load(w_path, map_location=device))
            model.eval()
            loaded_models[m_name] = model
        except Exception as e:
            print(f"    {C_RED}Error loading {m_name}: {e}{C_END}")

    if not loaded_models:
        print(f"{C_RED}No models found matching the criteria.{C_END}")
        return

    # 2. Sparsity loop
    sample_x = x_test[args.sample_idx:args.sample_idx+1].to(device)
    sample_y = y_test[args.sample_idx].cpu().numpy()

    for sp in sparsity_levels:
        plt.figure(figsize=(12, 7))
        
        # Time alignment for next-step prediction (like SINE)
        x_offset = 1 if args.dataset.lower() == "sine" else 0
        time_steps = np.arange(len(sample_y)) + x_offset

        # Real signal
        if is_classification:
            plt.step(time_steps, sample_y, label="True", color="black", linewidth=2, where='post', alpha=0.5)
        else:
            plt.plot(time_steps, sample_y, label="True", color="black", linewidth=2, alpha=0.5)

        # Noisy signal (sparsity)
        x_sparse, mask = apply_sparsity(sample_x, sp, args.mode)
        
        # Input data in RED (first feature for display)
        # Input data is at time t, so we plot it at original indices
        m = mask[0, :, 0].cpu().numpy()
        x_in = sample_x[0, :, 0].cpu().numpy()
        plt.scatter(np.where(m > 0)[0], x_in[m > 0], color='red', s=15, label="Input Data", zorder=5)

        # Predictions
        for m_name, model in loaded_models.items():
            with torch.no_grad():
                if m_name == "cnn": out = model(x_sparse)
                else: out, _ = model(x_sparse)
                
                pred = out[0].cpu().numpy()
                if is_classification:
                    pred = np.argmax(pred, axis=-1)
                    plt.step(time_steps, pred, label=f"{m_name.upper()}", linestyle="--", where='post')
                else:
                    plt.plot(time_steps, pred, label=f"{m_name.upper()}", linestyle="--")

        plt.title(f"Predictions on {args.dataset.upper()} | Eval Sparsity: {sp*100:.0f}% ({args.mode})\n(Trained with {args.sparsity_train*100 if args.sparsity_train is not None else 'Unknown'}% {args.sparsity_mode if args.sparsity_mode else ''})")
        fname = f"compare_preds_sp{int(sp*100)}_{args.mode}.png"
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()
        print(f"  {C_GREEN}[OK]{C_END} Saved comparison: {fname}")

        # Generate individual plots for each model
        for m_name, model in loaded_models.items():
            plt.figure(figsize=(12, 7))
            if is_classification:
                plt.step(time_steps, sample_y, label="True", color="black", linewidth=2, where='post', alpha=0.5)
            else:
                plt.plot(time_steps, sample_y, label="True", color="black", linewidth=2, alpha=0.5)
            
            plt.scatter(np.where(m > 0)[0], x_in[m > 0], color='red', s=15, label="Input Data", zorder=5)
            
            with torch.no_grad():
                res = model(x_sparse)
                out = res[0] if isinstance(res, tuple) else res
                pred = out[0].cpu().numpy()
                if is_classification:
                    pred = np.argmax(pred, axis=-1)
                    plt.step(time_steps, pred, label=f"{m_name.upper()}", linestyle="--", color="blue", where='post')
                else:
                    plt.plot(time_steps, pred, label=f"{m_name.upper()}", linestyle="--", color="blue")
            
            plt.title(f"{m_name.upper()} Prediction on {args.dataset.upper()} | Eval Sparsity: {sp*100:.0f}%")
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            
            fname_ind = f"pred_{m_name.lower()}_sp{int(sp*100)}_{args.mode}.png"
            plt.savefig(os.path.join(output_dir, fname_ind))
            plt.close()

    print(f"\n{C_BOLD}Visualizations completed in: {output_dir}{C_END}")

if __name__ == "__main__":
    main()

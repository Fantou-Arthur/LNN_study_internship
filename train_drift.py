import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import time
from drift_loaders import generate_sea, load_electricity, load_nyc_taxi

# Import models
from ltc_modern_demo import ModernLTCModel
from CfC import ModernCfCModel
from CNN import ModernCNNModel

# --- ANSI COLORS ---
C_BLUE = "\033[94m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_BOLD = "\033[1m"
C_END = "\033[0m"

def prequential_eval_custom(model, x, y, criterion, optimizer, device, is_classification, train_block_size, eval_block_size, train_epochs, online_epochs, target_loss=0.0, do_adapt=True):
    """
    Prequential Evaluation: Test on next block, then train on it.
    Supports different initial training vs evaluation block sizes.
    """
    num_seqs = x.shape[0]
    metrics = []
    all_preds = []
    all_targets = []
    
    # 1. Initial training on block 0
    print(f"    {C_BLUE}[Initial Train]{C_END} Training on {train_block_size} samples for {train_epochs} epochs...")
    bx_all, by_all = x[:train_block_size], y[:train_block_size]
    train_ds = TensorDataset(bx_all, by_all)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    
    best_loss = float('inf')
    best_state = None
    final_loss = 0.0
    
    for e in range(train_epochs):
        model.train()
        epoch_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            res = model(bx)
            out = res[0] if isinstance(res, tuple) else res
            
            if not is_classification and out.dim() == 3 and by.dim() == 2:
                out = out[:, -1, :]
                
            if is_classification:
                loss = criterion(out.reshape(-1, out.shape[-1]), by.view(-1))
            else:
                loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        
        # Track BEST model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if e % 20 == 0 or e == train_epochs - 1:
            print(f"      Epoch {e+1}/{train_epochs} | Loss: {avg_loss:.6f}")
        
        final_loss = avg_loss
        if target_loss > 0 and final_loss <= target_loss:
            print(f"      {C_GREEN}[Target Reached]{C_END} Loss {final_loss:.6f} <= {target_loss:.6f} at epoch {e+1}")
            break

    # Restore BEST state for evaluation
    if best_state is not None:
        model.load_state_dict(best_state)
        final_loss = best_loss
        print(f"    {C_GREEN}[Best Model Restored]{C_END} Loss: {final_loss:.6f}")

    # 2. Test then Train on subsequent blocks
    block_idx = 1
    for i in range(train_block_size, num_seqs, eval_block_size):
        end = min(i + eval_block_size, num_seqs)
        if end - i < 1: break
        
        bx, by = x[i:end].to(device), y[i:end].to(device)
        
        # Test
        model.eval()
        with torch.no_grad():
            res = model(bx)
            out = res[0] if isinstance(res, tuple) else res
            
            # Handle dimension mismatch (many-to-one vs many-to-many)
            if not is_classification and out.dim() == 3 and by.dim() == 2:
                out = out[:, -1, :]
            
            # Save for plotting
            all_preds.append(out.cpu().numpy())
            all_targets.append(by.cpu().numpy())

            if is_classification:
                preds = torch.argmax(out, dim=-1)
                metric = 100.0 * (preds == by).sum().item() / by.numel()
            else:
                mse = nn.functional.mse_loss(out, by).item()
                metric = 1.0 - min(mse, 1.0)
            metrics.append(metric)
        
        # Train (if enabled)
        if do_adapt:
            model.train()
            for e in range(online_epochs):
                optimizer.zero_grad()
                res = model(bx)
                out = res[0] if isinstance(res, tuple) else res
                
                if not is_classification and out.dim() == 3 and by.dim() == 2:
                    out = out[:, -1, :]
                
                if is_classification:
                    loss = criterion(out.reshape(-1, out.shape[-1]), by.view(-1))
                else:
                    loss = criterion(out, by)
                loss.backward()
                optimizer.step()
        
        if block_idx % 20 == 0:
            print(f"    [Block {block_idx}] Metric: {metric:.4f} | {'Adapting' if do_adapt else 'Static'}")
        
        block_idx += 1
            
    return metrics, all_preds, all_targets, final_loss

def get_input(prompt, default):
    user_input = input(f"{prompt} [{C_YELLOW}{default}{C_END}]: ")
    if not user_input.strip():
        return default
    return user_input.strip()

def main():
    print(f"\n{C_BOLD}{C_BLUE}=== CONCEPT DRIFT STUDY CONFIGURATION ==={C_END}")
    
    # 1. Global config
    ds_input = get_input("Datasets to test (comma separated)", "OilPrice")
    selected_ds = [d.strip() for d in ds_input.split(",")]
    
    do_adapt_input = get_input("Enable Online Adaptation (y/n)", "n")
    do_adapt = do_adapt_input.lower().startswith('y')
    
    epochs = int(get_input("Online Training Epochs per block", 0))
    
    # 2. Per-model config
    model_classes = {"LTC": ModernLTCModel, "CfC": ModernCfCModel, "CNN": ModernCNNModel}
    model_configs = []
    
    # Custom defaults per model
    m_defaults = {
        "LTC": {"u": 16, "l": 1, "d": "cpu"},
        "CfC": {"u": 16, "l": 1, "d": "cpu"},
        "CNN": {"u": 256, "l": 8, "d": "cuda" if torch.cuda.is_available() else "cpu"}
    }
    
    for m_name in ["LTC", "CfC", "CNN"]:
        d = m_defaults[m_name]
        print(f"\n{C_BOLD}[ Config for {m_name} ]{C_END}")
        units = int(get_input(f"  {m_name} Units/Neurons", d["u"]))
        layers = int(get_input(f"  {m_name} Layers", d["l"]))
        dev_str = get_input(f"  {m_name} Device (cuda/cpu)", d["d"])
        model_configs.append({
            "name": m_name,
            "class": model_classes[m_name],
            "units": units,
            "layers": layers,
            "device": torch.device(dev_str)
        })

    print(f"\n{C_BOLD}{C_BLUE}=== STARTING CONCEPT DRIFT STUDY ==={C_END}")
    
    from drift_loaders import load_oil_price
    all_loaders = {
        "SEA": lambda: (*generate_sea(), True),
        "Electricity": lambda: (*load_electricity(), True),
        "NYCTaxi": lambda: (*load_nyc_taxi(), False),
        "OilPrice": lambda: load_oil_price()
    }
    
    results = {}
    
    for ds_name in selected_ds:
        if ds_name not in all_loaders:
            print(f"{C_RED}[!] Dataset {ds_name} unknown, skipping...{C_END}")
            continue
            
        loader_res = all_loaders[ds_name]()
        if len(loader_res) == 6: # Specialized date split (OilPrice)
            x, y, inp_size, out_size, is_classification, split_idx = loader_res
            # We use split_idx for training, then small blocks for evaluation
            train_b_size = split_idx
            eval_b_size = 30 
            print(f"\n{C_BOLD}--- Dataset: {ds_name} (Long Term Mode) ---{C_END}")
            print(f"  [>] Training until 2018 (Samples: {split_idx})")
            print(f"  [>] Testing 2019-2026 with 30-day evaluation blocks")
            current_do_adapt = False
        else:
            x, y, inp_size, out_size, is_classification = loader_res
            train_b_size = max(5, int(x.shape[0] * 0.05))
            if ds_name == "NYCTaxi": train_b_size = 15
            eval_b_size = train_b_size
            print(f"\n{C_BOLD}--- Dataset: {ds_name} ---{C_END} (Size: {x.shape[0]} sequences)")
            current_do_adapt = do_adapt

        results[ds_name] = {"is_class": is_classification, "models": {}}
        
        for cfg in model_configs:
            print(f"  Training {C_YELLOW}{cfg['name']}{C_END} on {cfg['device']}...")
            model = cfg['class'](inp_size, cfg['units'], out_size, cfg['layers']).to(cfg['device'])
            
            lr = 0.002 if ds_name == "OilPrice" else 0.001
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()
            
            # Use more epochs for initial oil training
            init_epochs = 50 if ds_name == "OilPrice" else epochs
            
            # Run prequential evaluation with split sizing
            m_history, m_preds, m_targets = prequential_eval_custom(model, x, y, criterion, optimizer, cfg['device'], is_classification, train_b_size, eval_b_size, init_epochs, epochs, current_do_adapt)
            
            results[ds_name]["models"][cfg['name']] = {
                "history": m_history,
                "preds": np.concatenate(m_preds) if m_preds else [],
                "targets": np.concatenate(m_targets) if m_targets else []
            }

    # --- VISUALIZATION ---
    print(f"\n{C_BOLD}[>] Generating Drift Performance Plots...{C_END}")
    output_dir = "results/drift_study"
    os.makedirs(output_dir, exist_ok=True)
    
    for ds_name, info in results.items():
        # Plot 1: Metrics (Accuracy/Similarity)
        plt.figure(figsize=(12, 7))
        styles = {"LTC": "-", "CfC": "--", "CNN": "-."}
        
        for m_name, m_data in info["models"].items():
            history = m_data["history"]
            if not history: continue
            
            # Check for NaNs
            if np.isnan(history).any():
                print(f"  {C_YELLOW}[WARNING]{C_END} {m_name} contains NaN values!")
                history = np.nan_to_num(history)

            # Calculate X axis in days
            # For OilPrice, we know eval_b_size=30. 
            # The indices in history are block indices.
            x_days = np.arange(len(history)) * 30 

            # Smoothing
            if len(history) > 15:
                window = min(11, len(history) // 2)
                if window % 2 == 0: window += 1
                smoothed = np.convolve(history, np.ones(window)/window, mode='valid')
                # Adjust x_days for valid convolution
                x_days_smoothed = x_days[window-1:]
            else:
                smoothed = history
                x_days_smoothed = x_days
                
            plt.plot(x_days_smoothed, smoothed, label=m_name, linestyle=styles.get(m_name, "-"), linewidth=2)
        
        ylabel = "Accuracy (%)" if info["is_class"] else "1 - MSE (Similarity)"
        plt.title(f"Performance Evolution: {ds_name.upper()}")
        plt.xlabel("Timeline (Days from test start)")
        plt.ylabel(ylabel)
        plt.legend(loc='lower right', frameon=True, shadow=True)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"drift_{ds_name.lower()}.png"), dpi=150)
        plt.close()

        # Plot 2: Forecast Curves (for regression datasets like OilPrice or NYCTaxi)
        if not info["is_class"]:
            plt.figure(figsize=(12, 7))
            
            # Plot True values once
            first_model = list(info["models"].keys())[0]
            true_vals = info["models"][first_model]["targets"]
            if len(true_vals) > 0:
                plt.plot(true_vals.flatten(), color='black', label='Actual Price', linewidth=1, alpha=0.6)
                
                for m_name, m_data in info["models"].items():
                    preds = m_data["preds"]
                    if len(preds) > 0:
                        plt.plot(preds.flatten(), label=f"{m_name} Prediction", linestyle=styles.get(m_name, "-"), alpha=0.8)
                
                plt.title(f"Actual vs. Predicted Forecast: {ds_name.upper()}")
                plt.xlabel("Days (from 2019)")
                plt.ylabel("Normalized Price")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"forecast_{ds_name.lower()}.png"), dpi=150)
                print(f"  {C_GREEN}[OK]{C_END} Saved forecast plot: forecast_{ds_name.lower()}.png")
                plt.close()

if __name__ == "__main__":
    main()

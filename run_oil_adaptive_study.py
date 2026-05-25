import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import os
import pandas as pd
from drift_loaders import load_oil_price
from CfC import ModernCfCModel
from ltc_modern_demo import ModernLTCModel
from RNN import ModernRNNModel
from LSTM import ModernLSTMModel
from GRU import ModernGRUModel

def calculate_drift_metrics(ref_data, target_data):
    """
    Computes KS, PSI and JS metrics between reference distribution and target window.
    """
    # KS
    sorted_ref = np.sort(ref_data)
    sorted_target = np.sort(target_data)
    all_vals = np.sort(np.unique(np.concatenate([sorted_ref, sorted_target])))
    cdf_ref = np.searchsorted(sorted_ref, all_vals, side='right') / len(sorted_ref)
    cdf_target = np.searchsorted(sorted_target, all_vals, side='right') / len(sorted_target)
    ks = float(np.max(np.abs(cdf_ref - cdf_target)))
    
    # Histogram based (PSI, JS)
    bins = 10
    h_min, h_max = 0.0, 1.0
    ref_hist, _ = np.histogram(ref_data, bins=bins, range=(h_min, h_max))
    target_hist, _ = np.histogram(target_data, bins=bins, range=(h_min, h_max))
    
    # Normalize and add epsilon
    p = (ref_hist / (len(ref_data) + 1e-9)) + 1e-6
    q = (target_hist / (len(target_data) + 1e-9)) + 1e-6
    
    # PSI
    psi = float(np.sum((p - q) * np.log(p / q)))
    
    # JS
    m = 0.5 * (p + q)
    def kl(a, b): return np.sum(a * np.log(a / b))
    js = float(0.5 * kl(p, m) + 0.5 * kl(q, m))
    
    return ks, psi, js

def train_to_target(model, x, y, criterion, optimizer, device, target=0.0005, max_epochs=200):
    model.train()
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    for epoch in range(max_epochs):
        total_loss = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            res, _ = model(bx)
            out = res[0] if isinstance(res, tuple) else res
            if out.dim() == 3: out = out[:, -1, :]
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        if avg_loss <= target:
            break
    return avg_loss

def fine_tune(model, x, y, criterion, optimizer, device, epochs=15):
    model.train()
    # Fine tune on the provided window (usually 3 years)
    # We use a smaller batch or shuffle to adapt quickly
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            res, _ = model(bx)
            out = res[0] if isinstance(res, tuple) else res
            if out.dim() == 3: out = out[:, -1, :]
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

def run_adaptive_study():
    print("=== OIL PRICE ADAPTIVE RETRAINING STUDY (1986-2026) ===")
    
    res_loader = load_oil_price()
    if res_loader is None: return
    x_full, y_full, inp_size, out_size, is_class, _, dates, p_min, p_max = res_loader
    
    # 3 years of data (approx)
    train_size = 3 * 252 # ~756 trading days
    
    # Initial Split
    train_x = x_full[:train_size]
    train_y = y_full[:train_size]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Start Date: {dates[0]}")

    models_cfg = [
        {"name": "CfC", "class": ModernCfCModel, "units": 128},
        {"name": "LTC", "class": ModernLTCModel, "units": 128},
        {"name": "RNN", "class": ModernRNNModel, "units": 128},
        {"name": "LSTM", "class": ModernLSTMModel, "units": 128},
        {"name": "GRU", "class": ModernGRUModel, "units": 128}
    ]
    
    results = {
        "dates": dates[train_size:],
        "actual_price_usd": (y_full[train_size:].flatten() * (p_max - p_min) + p_min).tolist(),
        "drift_metrics": {
            "ks": [], "psi": [], "js": [], "labels": []
        },
        "models": {}
    }

    # Calculate Drift Metrics over time (Windows of 90 days, Step 30)
    print("[>] Calculating Drift Metrics...")
    ref_dist = train_y.flatten().numpy()
    test_dist = y_full[train_size:].flatten().numpy()
    test_dates = dates[train_size:]
    window_size = 90
    step = 30
    for i in range(0, len(test_dist) - window_size, step):
        win = test_dist[i : i+window_size]
        ks, psi, js = calculate_drift_metrics(ref_dist, win)
        results["drift_metrics"]["ks"].append(ks)
        results["drift_metrics"]["psi"].append(psi)
        results["drift_metrics"]["js"].append(js)
        results["drift_metrics"]["labels"].append(test_dates[i + window_size])
    
    for cfg in models_cfg:
        name = cfg['name']
        print(f"\n[!] Initial Training for {name}...")
        model = cfg['class'](inp_size, cfg['units'], out_size, 1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Phase 1: Train to 0.0001
        final_init_loss = train_to_target(model, train_x, train_y, criterion, optimizer, device, target=0.0001)
        print(f"    - Initial Loss: {final_init_loss:.6f}")
        
        # Phase 2: Adaptive Inference
        all_preds = []
        retrain_events = []
        errors_rolling = []
        
        mse_first_month = None
        y_multiplier = 5.0 # Threshold = 5x initial month MSE
        
        # We start with the trained model
        for i in range(train_size, len(x_full)):
            model.eval()
            bx = x_full[i:i+1].to(device)
            by = y_full[i:i+1].to(device)
            
            hx = None
            if name in ["CfC", "LTC"]:
                # Temporal Grounding: x(t) initialization with yesterday's price
                # x_full[i] contains [p_{t-30}, ..., p_{t-1}]. 
                # The most recent value is at index -1.
                p_veille = x_full[i, -1, 0].item()
                hx = [torch.full((1, cfg['units']), p_veille).to(device)]
                
            with torch.no_grad():
                res, _ = model(bx, hx)
                out = res[0] if isinstance(res, tuple) else res
                if out.dim() == 3: out = out[:, -1, :]
                pred_val = out.cpu().item()
                all_preds.append(pred_val)
                
                # Update error window
                err = (pred_val - by.cpu().item())**2
                errors_rolling.append(err)
                if len(errors_rolling) > 30: errors_rolling.pop(0)
            
            # Capture baseline performance (first month)
            if mse_first_month is None and len(errors_rolling) == 30:
                mse_first_month = np.mean(errors_rolling)
                print(f"    - {name} Baseline MSE (First Month): {mse_first_month:.6f}")
            
            # Check for retraining (Relative Drift Detection)
            if mse_first_month is not None and len(errors_rolling) == 30:
                current_mse = np.mean(errors_rolling)
                if current_mse > (y_multiplier * mse_first_month):
                    print(f"    [Retrain] {name} at {dates[i]} (MSE: {current_mse:.5f} > {y_multiplier}x Baseline)")
                    retrain_events.append(dates[i])
                    
                    # Train on the preceding 3 years
                    window_start = max(0, i - train_size)
                    ft_x = x_full[window_start:i]
                    ft_y = y_full[window_start:i]
                    
                    # Switch to train mode and fine-tune
                    fine_tune(model, ft_x, ft_y, criterion, optimizer, device, epochs=10)
                    
                    # Reset baseline and clear rolling error to adapt to the new regime
                    mse_first_month = None
                    errors_rolling = []
            
            if (i - train_size) % 2000 == 0 and i > train_size:
                print(f"    - {name}: Step {i-train_size}/{len(x_full)-train_size}")

        # Temporal Metrics
        abs_errors = np.abs(np.array(all_preds) - y_full[train_size:].flatten().numpy())
        rolling_mae = pd.Series(abs_errors).rolling(window=30, min_periods=1).mean().tolist()
        
        # Growth Rate (%) - Rolling 30d MSE vs previous 30d MSE
        mses = [e for e in errors_rolling] # This is just the last 30, we need the full history
        # Wait, errors_rolling was popped in the loop. Let's recalculate from all_preds.
        full_mses = (np.array(all_preds) - y_full[train_size:].flatten().numpy())**2
        avg_mse = float(np.mean(full_mses))
        rolling_mse = pd.Series(full_mses).rolling(window=30, min_periods=1).mean()
        growth_rate = ((rolling_mse / rolling_mse.shift(30) - 1) * 100).fillna(0).tolist()
        avg_growth = float(np.mean(growth_rate))

        results["models"][name] = {
            "predictions_usd": (np.array(all_preds) * (p_max - p_min) + p_min).tolist(),
            "rolling_mae": rolling_mae,
            "growth_rate": growth_rate,
            "avg_growth": avg_growth,
            "avg_mse": avg_mse,
            "retrain_events": retrain_events,
            "total_retrains": len(retrain_events)
        }

    # Save
    output_path = "results/oil_adaptive_data.js"
    os.makedirs("results", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("var oilAdaptiveData = " + json.dumps(results) + ";")
    
    print(f"\n[OK] Adaptive Study finished. Results saved to {output_path}")

if __name__ == "__main__":
    run_adaptive_study()

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
from train_drift import prequential_eval_custom

def calculate_drift_metrics(ref_data, target_data):
    """
    Computes KS, PSI and JS metrics between reference distribution and target window.
    (Simplified versions for the script to export to JS)
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
    p = (ref_hist / len(ref_data)) + 1e-6
    q = (target_hist / len(target_data)) + 1e-6
    
    # PSI
    psi = float(np.sum((p - q) * np.log(p / q)))
    
    # JS
    m = 0.5 * (p + q)
    def kl(a, b): return np.sum(a * np.log(a / b))
    js = float(0.5 * kl(p, m) + 0.5 * kl(q, m))
    
    return ks, psi, js

def run_short_study():
    print("=== OIL PRICE SHORT-TERM STUDY (2017-2019) ===")
    
    # Load Data
    res_loader = load_oil_price()
    if res_loader is None:
        print("Error loading data")
        return
        
    x_full, y_full, inp_size, out_size, is_class, _, dates, p_min, p_max = res_loader
    
    # Find indices for 2017-2019 split
    try:
        start_idx = next(i for i, d in enumerate(dates) if d.startswith('2017-01'))
        split_idx = next(i for i, d in enumerate(dates) if d.startswith('2020-01'))
    except StopIteration:
        print("Could not find date ranges. Using defaults.")
        start_idx = len(dates) - 2000
        split_idx = len(dates) - 1500

    print(f"[>] Training: {dates[start_idx]} -> {dates[split_idx-1]}")
    print(f"[>] Evaluation: {dates[split_idx]} -> {dates[-1]}")
    
    # Filter Tensors
    train_x = x_full[start_idx:split_idx]
    train_y = y_full[start_idx:split_idx]
    test_x = x_full[split_idx:]
    test_y = y_full[split_idx:]
    test_dates = dates[split_idx:]
    
    sparsity_levels = [0, 20, 50, 80]
    results = {
        "dates": test_dates, 
        "actual_price": test_y.flatten().tolist(),
        "actual_price_usd": (test_y.flatten() * (p_max - p_min) + p_min).tolist(),
        "drift_metrics": {
            "ks": [], "psi": [], "js": [], "labels": []
        },
        "studies": []
    }
    
    # Calculate Drift Metrics over time (Windows of 90 days)
    ref_distribution = train_y.flatten().numpy()
    test_distribution = test_y.flatten().numpy()
    window_size = 90
    step = 30
    for i in range(0, len(test_distribution) - window_size, step):
        win = test_distribution[i : i+window_size]
        ks, psi, js = calculate_drift_metrics(ref_distribution, win)
        results["drift_metrics"]["ks"].append(ks)
        results["drift_metrics"]["psi"].append(psi)
        results["drift_metrics"]["js"].append(js)
        results["drift_metrics"]["labels"].append(test_dates[i + window_size])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    configs = [
        {"name": "CfC", "class": ModernCfCModel, "units": 128, "layers": 1, "max_epochs": 100},
        {"name": "LTC", "class": ModernLTCModel, "units": 128, "layers": 1, "max_epochs": 100},
        {"name": "RNN", "class": ModernRNNModel, "units": 128, "layers": 1, "max_epochs": 100},
        {"name": "LSTM", "class": ModernLSTMModel, "units": 128, "layers": 1, "max_epochs": 100},
        {"name": "GRU", "class": ModernGRUModel, "units": 128, "layers": 1, "max_epochs": 100}
    ]
    
    target_loss = 0.0
    os.makedirs("models", exist_ok=True)

    for cfg in configs:
        name = cfg['name']
        model = cfg['class'](inp_size, cfg['units'], out_size, cfg['layers']).to(device)
        
        print(f"\n[Processing {name}...]")
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        current_target = 0.0 if name in ["CfC", "LTC"] else target_loss
        
        # Train on the 2017-2019 block
        prequential_eval_custom(
            model, train_x, train_y, criterion, optimizer, device, False, 
            len(train_x), 30, cfg['max_epochs'], 0, 
            target_loss=current_target, do_adapt=False
        )
        
        # Sparsity Evaluation with x(0) Initialization for CfC/LTC
        for sp in sparsity_levels:
            print(f"  [Sparsity {sp}%] Benchmarking {name}...")
            study_data = next((s for s in results["studies"] if s["sparsity"] == sp), None)
            if study_data is None:
                study_data = {"sparsity": sp, "models": {}}
                results["studies"].append(study_data)
            
            all_preds = []
            mses = []
            
            # Apply sparsity
            torch.manual_seed(sp)
            cur_test_x = test_x.clone()
            if sp > 0:
                mask = torch.rand(cur_test_x.shape) > (sp / 100.0)
                sparse_x = cur_test_x * mask.to(cur_test_x.dtype)
                for i in range(1, sparse_x.shape[0]):
                    m_row = mask[i]
                    sparse_x[i] = torch.where(m_row > 0, sparse_x[i], sparse_x[i-1])
                cur_test_x = sparse_x

            # Sequential Inference
            total_steps = cur_test_x.shape[0]
            with torch.no_grad():
                model.eval()
                for i in range(total_steps):
                    bx = cur_test_x[i:i+1].to(device)
                    by = test_y[i:i+1].to(device)
                    
                    hx = None
                    if name in ["CfC", "LTC"]:
                        p_veille = x_full[split_idx + i - 1, 0, 0].item()
                        hx = [torch.full((1, cfg['units']), p_veille).to(device)] * cfg['layers']
                    
                    res, _ = model(bx, hx)
                    out = res[0] if isinstance(res, tuple) else res
                    if out.dim() == 3: out = out[:, -1, :]
                    
                    val = out.cpu().item()
                    all_preds.append(val)
                    mses.append(nn.functional.mse_loss(out, by).item())

            # Temporal Metrics Calculation (for sparsity 0% mostly)
            # Rolling MAE (30 days)
            abs_errors = np.abs(np.array(all_preds) - test_y.flatten().numpy())
            rolling_mae = pd.Series(abs_errors).rolling(window=30, min_periods=1).mean().tolist()
            
            # Error Growth Rate (%) - Rolling 30d MSE vs previous 30d MSE
            mse_series = pd.Series(mses)
            rolling_mse = mse_series.rolling(window=30, min_periods=1).mean()
            growth_rate = ((rolling_mse / rolling_mse.shift(30) - 1) * 100).fillna(0).tolist()
            
            avg_growth = float(np.mean(growth_rate)) if growth_rate else 0.0

            study_data["models"][name] = {
                "mse": float(np.mean(mses)),
                "growth_pct": avg_growth,
                "rolling_mae": rolling_mae,
                "growth_rate": growth_rate,
                "predictions": all_preds,
                "predictions_usd": (np.array(all_preds) * (p_max - p_min) + p_min).tolist()
            }
            
    # Save to JS
    output_path = "results/oil_short_data.js"
    os.makedirs("results", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("var oilShortData = " + json.dumps(results) + ";")
    
    print(f"\n[OK] Short Study Results saved to {output_path}")

if __name__ == "__main__":
    run_short_study()

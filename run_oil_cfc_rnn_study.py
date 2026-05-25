import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import os
from drift_loaders import load_oil_price
from CfC import ModernCfCModel
from RNN import ModernRNNModel
from train_drift import prequential_eval_custom

def run_cfc_rnn_comparison():
    print("=== OIL PRICE COMPARISON: CfC vs RNN ===")
    
    # Load Data
    res_loader = load_oil_price()
    x, y, inp_size, out_size, is_class, split_idx, dates = res_loader
    
    # Study 0% to 90% missing data at INFERENCE time
    sparsity_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    results = {
        "dates": dates[split_idx:], # Dates for the test period
        "actual_price": y[split_idx:].flatten().tolist(),
        "studies": []
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. TRAIN OR LOAD MODELS
    configs = [
        {"name": "CfC", "class": ModernCfCModel, "units": 16, "layers": 1, "device": device.type},
        {"name": "RNN", "class": ModernRNNModel, "units": 32, "layers": 1, "device": device.type}
    ]
    trained_models = {}
    os.makedirs("models", exist_ok=True)

    for cfg in configs:
        m_dev = torch.device(cfg['device'])
        model_path = f"models/oil_comp_{cfg['name']}.pth"
        model = cfg['class'](inp_size, cfg['units'], out_size, cfg['layers']).to(m_dev)
        
        if os.path.exists(model_path):
            print(f"\n[Loading {cfg['name']} from {model_path}...]")
            model.load_state_dict(torch.load(model_path, map_location=m_dev))
        else:
            print(f"\n[Training {cfg['name']} on {cfg['device']}...]")
            optimizer = optim.Adam(model.parameters(), lr=0.002)
            criterion = nn.MSELoss()
            # Initial training on the split portion
            prequential_eval_custom(
                model, x, y, criterion, optimizer, m_dev, False, 
                split_idx, 30, 60, 0, do_adapt=False
            )
            torch.save(model.state_dict(), model_path)
            print(f"  Saved to {model_path}")
            
        trained_models[cfg['name']] = (model, m_dev)

    # 2. EVALUATE ON VARYING SPARSITY
    for s in sparsity_levels:
        print(f"\n[Evaluating Sparsity: {s}%]")
        study_data = {"sparsity": s, "models": {}}
        
        for name, (model, m_dev) in trained_models.items():
            model.eval()
            all_preds = []
            mses = []
            
            # Evaluate on test portion
            test_x = x[split_idx:].clone()
            test_y = y[split_idx:].clone()
            
            # Apply sparsity with Sample & Hold logic
            if s > 0:
                mask = (torch.rand(test_x.shape) > (s / 100.0)).float()
                sparse_x = test_x.clone()
                for i in range(1, sparse_x.shape[0]):
                    m_row = mask[i]
                    sparse_x[i] = torch.where(m_row > 0, sparse_x[i], sparse_x[i-1])
                test_x = sparse_x

            # Sequential Inference
            h = None
            total_steps = test_x.shape[0]
            with torch.no_grad():
                for i in range(total_steps):
                    bx = test_x[i:i+1].to(m_dev)
                    by = test_y[i:i+1].to(m_dev)
                    
                    if name == "CfC":
                        res, h = model(bx, h)
                    else: # RNN
                        res, h = model(bx, h)
                    
                    out = res[0] if isinstance(res, tuple) else res
                    if out.dim() == 3: out = out[:, -1, :]
                    
                    val = out.cpu().item()
                    all_preds.append(val)
                    mses.append(nn.functional.mse_loss(out, by).item())

                    if (i + 1) % 500 == 0:
                        print(f"    - {name}: step {i+1}/{total_steps}")
            
            study_data["models"][name] = {
                "mse": float(np.mean(mses)),
                "predictions": all_preds
            }
            
        results["studies"].append(study_data)
        
    # Save to JS
    output_path = "results/oil_cfc_rnn_data.js"
    os.makedirs("results", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("var oilCompData = " + json.dumps(results) + ";")
    
    print(f"\n[OK] Results saved to {output_path}")

if __name__ == "__main__":
    run_cfc_rnn_comparison()

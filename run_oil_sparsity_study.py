import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import os
from drift_loaders import load_oil_price
from ltc_modern_demo import ModernLTCModel
from CfC import ModernCfCModel
from CNN import ModernCNNModel
from train_drift import prequential_eval_custom

def run_sparsity_study():
    print("=== OIL PRICE INFERENCE SPARSITY STUDY ===")
    
    # Load Data
    x, y, inp_size, out_size, is_class, split_idx = load_oil_price()
    
    # Study 0% to 90% missing data at INFERENCE time
    sparsity_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    results = {
        "days": [i for i in range(x.shape[0] - split_idx)],
        "actual_price": y[split_idx:].flatten().tolist(),
        "studies": []
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. TRAIN OR LOAD MODELS
    configs = [
        {"name": "LTC", "class": ModernLTCModel, "units": 16, "layers": 1, "device": "cpu"},
        {"name": "CfC", "class": ModernCfCModel, "units": 16, "layers": 1, "device": "cpu"},
        {"name": "CNN", "class": ModernCNNModel, "units": 256, "layers": 8, "device": "cuda" if torch.cuda.is_available() else "cpu"}
    ]
    trained_models = {}
    os.makedirs("models", exist_ok=True)

    for cfg in configs:
        m_dev = torch.device(cfg['device'])
        model_path = f"models/oil_study_{cfg['name']}.pth"
        model = cfg['class'](inp_size, cfg['units'], out_size, cfg['layers']).to(m_dev)
        
        if os.path.exists(model_path):
            print(f"\n[Loading {cfg['name']} from {model_path}...]")
            model.load_state_dict(torch.load(model_path, map_location=m_dev))
        else:
            print(f"\n[Training {cfg['name']} on {cfg['device']}...]")
            optimizer = optim.Adam(model.parameters(), lr=0.002)
            criterion = nn.MSELoss()
            prequential_eval_custom(
                model, x, y, criterion, optimizer, m_dev, False, 
                split_idx, 30, 60, 0, do_adapt=False
            )
            torch.save(model.state_dict(), model_path)
            print(f"  Saved to {model_path}")
            
        trained_models[cfg['name']] = (model, m_dev)

    # 2. EVALUATE ON VARYING SPARSITY (with Sample & Hold)
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
            
            # 2a. Apply sparsity with TRUE "Sample & Hold" logic (Feature-wise)
            if s > 0:
                mask = (torch.rand(test_x.shape) > (s / 100.0)).float()
                sparse_x = test_x.clone()
                
                # We iterate and if a value is masked, we take the one from the previous step
                # sparse_x shape is typically (samples, features) or (samples, 1, features)
                for i in range(1, sparse_x.shape[0]):
                    # If mask is 0, we take the value from i-1
                    m_row = mask[i]
                    sparse_x[i] = torch.where(m_row > 0, sparse_x[i], sparse_x[i-1])
                
                test_x = sparse_x

            # 2b. Sequential Inference
            h = None
            with torch.no_grad():
                for i in range(test_x.shape[0]):
                    bx = test_x[i:i+1].to(m_dev)
                    by = test_y[i:i+1].to(m_dev)
                    
                    # Handle stateful vs stateless
                    if name in ["LTC", "CfC"]:
                        # Liquid models support (x, h) input/output
                        res, h = model(bx, h)
                    else:
                        res = model(bx)
                    
                    out = res[0] if isinstance(res, tuple) else res
                    if out.dim() == 3: out = out[:, -1, :]
                    
                    val = out.cpu().item()
                    all_preds.append(val)
                    mses.append(nn.functional.mse_loss(out, by).item())
            
            study_data["models"][name] = {
                "mse": float(np.mean(mses)),
                "predictions": all_preds
            }
            
        results["studies"].append(study_data)
        
    # Save to JS (to bypass CORS)
    output_path = "results/oil_sparsity_data.js"
    os.makedirs("results", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("var oilData = " + json.dumps(results) + ";")
    
    print(f"\n[OK] Results saved to {output_path}")

if __name__ == "__main__":
    run_sparsity_study()

import torch
import torch.nn as nn
import torch.optim as optim
import time
import gc
import sys
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from drift_loaders import generate_sea

# Import Models
from CfC import ModernCfCModel
from ltc_modern_demo import ModernLTCModel
from RNN import ModernRNNModel
from LSTM import ModernLSTMModel
try:
    from GRU import ModernGRUModel
except ImportError:
    ModernGRUModel = None

def get_throughput(model_type, units, layers, samples, batch_size, device="cuda"):
    device = torch.device(device)
    
    # 1. Prepare Data
    try:
        train_x, train_y, inp_size, out_size = generate_sea(n_samples=samples)
        loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    except Exception as e:
        return f"DATA_ERROR: {e}"

    # 2. Init Model
    try:
        if model_type == "cfc":
            model = ModernCfCModel(inp_size, units=units, output_size=out_size, num_layers=layers).to(device)
        elif model_type == "ltc":
            model = ModernLTCModel(inp_size, units=units, output_size=out_size, num_layers=layers).to(device)
        elif model_type == "rnn":
            model = ModernRNNModel(inp_size, units=units, output_size=out_size, num_layers=layers).to(device)
        elif model_type == "lstm":
            model = ModernLSTMModel(inp_size, units=units, output_size=out_size, num_layers=layers).to(device)
        elif model_type == "gru" and ModernGRUModel:
            model = ModernGRUModel(inp_size, units=units, output_size=out_size, num_layers=layers).to(device)
        else:
            return f"MODEL_NOT_FOUND: {model_type}"
    except Exception as e:
        return f"INIT_ERROR: {e}"

    optimizer = optim.Adam(model.parameters(), lr=0.002)
    is_classification = (train_y.dtype == torch.long)
    criterion = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()
    
    # 3. Warmup
    try:
        model.train()
        with torch.no_grad():
            warm_x = train_x[:min(batch_size, samples)].to(device)
            model(warm_x)
        if device.type == 'cuda': torch.cuda.synchronize()
    except Exception as e:
        return f"WARMUP_ERROR: {e}"

    # 4. Benchmark
    start_time = time.time()
    try:
        max_batches = 20 # Enough for a good estimate
        batch_count = 0
        
        for i, (bx, by) in enumerate(loader):
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                out, _ = model(bx)
                if is_classification:
                    loss = criterion(out.transpose(1, 2), by)
                else:
                    loss = criterion(out, by)
            
            loss.backward()
            optimizer.step()
            
            batch_count += 1
            if batch_count >= max_batches: break
            
        if device.type == 'cuda': torch.cuda.synchronize()
        duration = time.time() - start_time
        throughput = (batch_count * batch_size) / duration
        
    except Exception as e:
        return f"RUNTIME_ERROR: {e}"
    finally:
        del model, optimizer, loader, train_x, train_y
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()

    return throughput

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ GPU non détecté.")
        sys.exit(1)

    print("=== EXTENDED MULTI-MODEL OPTIMIZATION BENCHMARK (GPU) ===")
    
    grid = []
    
    # LTC: High Precision / ODE
    for u in [16, 64, 256]:
        for b in [64, 128, 256, 512]:
            if u == 256 and b > 128: continue # Safety
            grid.append({"type": "ltc", "units": u, "layers": 1, "samples": 10000, "batch": b})
            
    # CfC: Liquid Optimization
    for u in [128, 512, 1024]:
        for b in [256, 512, 1024, 2048]:
            for l in [1, 4]:
                grid.append({"type": "cfc", "units": u, "layers": l, "samples": 20000, "batch": b})

    # RNN / LSTM: Speed Kings
    for m in ["rnn", "lstm"]:
        for u in [256, 1024]:
            for b in [512, 1024, 2048, 4096]:
                for l in [1, 4]:
                    grid.append({"type": m, "units": u, "layers": l, "samples": 50000, "batch": b})

    results = []
    print(f"Testing {len(grid)} configurations...")
    
    for i, cfg in enumerate(grid):
        print(f"[{i+1}/{len(grid)}] {cfg['type'].upper():<4} | U:{cfg['units']:<4} | L:{cfg['layers']} | B:{cfg['batch']:<5} -> ", end="", flush=True)
        res = get_throughput(cfg['type'], cfg['units'], cfg['layers'], cfg['samples'], cfg['batch'])
        
        if isinstance(res, float):
            print(f"OK ({res:.1f} samples/s)")
            results.append({**cfg, "throughput": res, "status": "SUCCESS"})
        else:
            print(f"FAILED ({res})")
            results.append({**cfg, "throughput": 0, "status": res})

    df = pd.DataFrame(results)
    df.to_csv("optimal_configs_results_extended.csv", index=False)

    print("\n" + "="*80)
    print("🏆 BEST CONFIGURATIONS BY MODEL TYPE (EXTENDED)")
    print("="*80)
    
    for m_type in df['type'].unique():
        subset = df[(df['type'] == m_type) & (df['status'] == "SUCCESS")]
        if not subset.empty:
            best = subset.sort_values(by="throughput", ascending=False).iloc[0]
            print(f"{m_type.upper():<4} : Batch {best['batch']}, Units {best['units']}, Layers {best['layers']} -> {best['throughput']:.2f} samples/sec")

    print("\n✅ Résultats détaillés dans 'optimal_configs_results_extended.csv'")

import torch
import torch.nn as nn
import torch.optim as optim
import time
import argparse
import sys
from torch.utils.data import DataLoader, TensorDataset
from drift_loaders import load_oil_price, generate_sea, load_electricity
from CfC import ModernCfCModel
from ltc_modern_demo import ModernLTCModel

def print_simple_progress(batch_idx, total_batches, epoch, epochs):
    percent = (batch_idx + 1) / total_batches * 100
    bar_length = 20
    filled_length = int(bar_length * (batch_idx + 1) // total_batches)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write(f"\r    Epoch {epoch+1}/{epochs} |{bar}| {percent:.1f}%")
    sys.stdout.flush()

def benchmark_config(model_type, device_name, train_x, train_y, inp_size, out_size, units, layers, epochs=1, batch_size=256):
    device = torch.device(device_name)
    
    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_batches = len(loader)

    if model_type == "cfc":
        model = ModernCfCModel(inp_size, units=units, output_size=out_size, num_layers=layers).to(device)
    else:
        model = ModernLTCModel(inp_size, units=units, output_size=out_size, num_layers=layers).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss() if train_y.dtype == torch.long else nn.MSELoss()
    
    model.train()
    if device.type == 'cuda': torch.cuda.synchronize()
    
    # Warmup
    with torch.no_grad():
        warm_x = train_x[:batch_size].to(device)
        model(warm_x)
    if device.type == 'cuda': torch.cuda.synchronize()
    
    start_time = time.time()
    for epoch in range(epochs):
        for i, (bx, by) in enumerate(loader):
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out, _ = model(bx)
            
            if out.dim() == 3 and by.dim() == 2:
                loss = criterion(out.view(-1, out_size), by.view(-1))
            else:
                loss = criterion(out, by)
                
            loss.backward()
            optimizer.step()
            
            if i % 5 == 0 or i == total_batches - 1:
                print_simple_progress(i, total_batches, epoch, epochs)
        print() 

    if device.type == 'cuda': torch.cuda.synchronize()
    end_time = time.time()
    
    duration = end_time - start_time
    return duration

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CfC/LTC Scalability Benchmark")
    parser.add_argument("--model", type=str, default="cfc", choices=["cfc", "ltc"])
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    print(f"=== {args.model.upper()} SCALABILITY BENCHMARK ===")
    print(f"Dataset: SEA with {args.samples} samples")
    print(f"Settings: {args.epochs} Epochs, Batch Size {args.batch_size}")
    
    print(f"\n[1/3] Preparing data...")
    train_x, train_y, inp_size, out_size = generate_sea(n_samples=args.samples)
    
    if args.model == "ltc":
        configs = [
            {"units": 32, "layers": 1, "label": "LTC Medium (32u, 1L)"},
            {"units": 128, "layers": 1, "label": "LTC Large (128u, 1L)"},
            {"units": 256, "layers": 1, "label": "LTC MASSIVE (256u, 1L)"}
        ]
    else:
        configs = [
            {"units": 64, "layers": 2, "label": "CfC Medium (64u, 2L)"},
            {"units": 256, "layers": 4, "label": "CfC Large (256u, 4L)"},
            {"units": 1024, "layers": 4, "label": "CfC MASSIVE (1024u, 4L)"}
        ]
    
    results = []

    for cfg in configs:
        print(f"\n[2/3] Testing {cfg['label']}...")
        
        # CPU
        print(f"  > Benchmarking CPU...")
        cpu_t = benchmark_config(args.model, "cpu", train_x, train_y, inp_size, out_size, cfg['units'], cfg['layers'], args.epochs, args.batch_size)
        print(f"    Result: {cpu_t:.2f}s")
        
        # GPU
        if torch.cuda.is_available():
            print(f"  > Benchmarking GPU (CUDA)...")
            gpu_t = benchmark_config(args.model, "cuda", train_x, train_y, inp_size, out_size, cfg['units'], cfg['layers'], args.epochs, args.batch_size)
            speedup = cpu_t / gpu_t
            print(f"    Result: {gpu_t:.2f}s | Speedup: x{speedup:.2f}")
            results.append((cfg['label'], cpu_t, gpu_t, speedup))
        else:
            results.append((cfg['label'], cpu_t, None, None))

    print("\n" + "="*65)
    print(f"{'Model Size':<25} | {'CPU (s)':<12} | {'GPU (s)':<12} | {'Speedup':<10}")
    print("-" * 65)
    for label, cpu, gpu, speedup in results:
        gpu_str = f"{gpu:.2f}" if gpu else "N/A"
        sp_str = f"x{speedup:.2f}" if speedup else "N/A"
        print(f"{label:<25} | {cpu:<12.2f} | {gpu_str:<12} | {sp_str:<10}")
    print("="*65)

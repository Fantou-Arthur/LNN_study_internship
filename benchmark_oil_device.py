import torch
import torch.nn as nn
import torch.optim as optim
import time
from drift_loaders import load_oil_price
from CfC import ModernCfCModel

def benchmark_device(device_name, epochs=50):
    device = torch.device(device_name)
    print(f"\n>>> Benchmarking on {device_name.upper()}...")
    
    # Load Data
    x, y, inp_size, out_size, is_class, split_idx = load_oil_price()
    
    # Use only the training portion (up to split_idx)
    train_x = x[:split_idx].to(device)
    train_y = y[:split_idx].to(device)
    
    # Initialize Model
    model = ModernCfCModel(inp_size, units=16, output_size=out_size, num_layers=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.MSELoss()
    
    # Simple training loop
    model.train()
    start_time = time.time()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Full batch for simplicity in benchmarking (or use small batches)
        out, _ = model(train_x)
        if out.dim() == 3: out = out[:, -1, :]
        
        loss = criterion(out, train_y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")
            
    end_time = time.time()
    duration = end_time - start_time
    print(f"✅ Training on {device_name.upper()} completed in {duration:.2f} seconds.")
    return duration

if __name__ == "__main__":
    print("=== CFC OIL PRICE BENCHMARK: CPU VS GPU ===")
    
    # 1. CPU
    cpu_time = benchmark_device("cpu", epochs=50)
    
    # 2. GPU (if available)
    if torch.cuda.is_available():
        gpu_time = benchmark_device("cuda", epochs=50)
        
        speedup = cpu_time / gpu_time
        print(f"\n🚀 Results Summary:")
        print(f"  - CPU: {cpu_time:.2f}s")
        print(f"  - GPU: {gpu_time:.2f}s")
        print(f"  - Speedup: x{speedup:.2f}")
    else:
        print("\n⚠️ CUDA is not available. Skipping GPU benchmark.")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from ncps.torch import CfC
from ncps.wirings import AutoNCP
import matplotlib.pyplot as plt
import json
import os
import time
from datetime import datetime
from ptflops import get_model_complexity_info
import argparse
import sys
from torch.utils.data import DataLoader, TensorDataset

try:
    from sklearn.metrics import f1_score, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

def generate_sine_data(seq_len=64, num_samples=1000):
    times = np.linspace(0, 10 * np.pi, seq_len)
    def create_split(n):
        data = []
        for _ in range(n):
            phase = np.random.uniform(0, 2 * np.pi)
            data.append(np.sin(times + phase))
        data = np.array(data, dtype=np.float32)
        x = data[:, :-1, np.newaxis]
        y = data[:, 1:, np.newaxis]
        return torch.tensor(x), torch.tensor(y)
    x_train, y_train = create_split(num_samples)
    x_test, y_test = create_split(int(num_samples * 0.2))
    return (x_train, y_train), (x_test, y_test), 1, 1

def load_har_data(seq_len=16):
    base_path = "data/har/UCI HAR Dataset"
    if not os.path.exists(base_path): raise FileNotFoundError(f"Dataset HAR non trouvé.")
    def load_split(split):
        x = np.loadtxt(os.path.join(base_path, f"{split}/X_{split}.txt"))
        y = (np.loadtxt(os.path.join(base_path, f"{split}/y_{split}.txt")) - 1).astype(np.int64)
        num_seqs = x.shape[0] // seq_len
        x_seq = x[:num_seqs*seq_len].reshape(num_seqs, seq_len, 561)
        y_seq = y[:num_seqs*seq_len].reshape(num_seqs, seq_len)
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.long)
    return load_split("train"), load_split("test"), 561, 6

def load_occupancy_data(seq_len=16):
    base_path = "data/occupancy"
    def read_file(name):
        df = pd.read_csv(os.path.join(base_path, name))
        x, y = df[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']].values, df['Occupancy'].values.astype(np.int64)
        return x, y
    train_x, train_y = read_file("datatraining.txt")
    test_x, test_y = read_file("datatest.txt")
    def to_seq(x, y):
        num_seqs = x.shape[0] // seq_len
        x_s, y_s = x[:num_seqs*seq_len].reshape(num_seqs, seq_len, 5), y[:num_seqs*seq_len].reshape(num_seqs, seq_len)
        return torch.tensor(x_s, dtype=torch.float32), torch.tensor(y_s, dtype=torch.long)
    return to_seq(train_x, train_y), to_seq(test_x, test_y), 5, 2

def load_gesture_data(seq_len=32):
    base_path = "data/gesture"
    files, convert = ["a1_va3.csv", "a2_va3.csv", "a3_va3.csv", "b1_va3.csv"], {"D":0, "P":1, "S":2, "H":3, "R":4}
    all_x, all_y = [], []
    for f in files:
        df = pd.read_csv(os.path.join(base_path, f))
        all_x.append(df.values[:, :-1].astype(np.float32))
        all_y.append(np.array([convert[v] for v in df["Phase"].values], dtype=np.int64))
    x, y = np.concatenate(all_x), np.concatenate(all_y)
    num_seqs = x.shape[0] // seq_len
    x_s, y_s = x[:num_seqs*seq_len].reshape(num_seqs, seq_len, 32), y[:num_seqs*seq_len].reshape(num_seqs, seq_len)
    split = int(0.8 * num_seqs)
    return (torch.tensor(x_s[:split]), torch.tensor(y_s[:split])), (torch.tensor(x_s[split:]), torch.tensor(y_s[split:])), 32, 5

def load_traffic_data(seq_len=32):
    base_path = "data/traffic"
    df = pd.read_csv(os.path.join(base_path, "Metro_Interstate_Traffic_Volume.csv"))
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['weekday'], df['hour'] = df['date_time'].dt.weekday, np.sin(df['date_time'].dt.hour * np.pi / 24)
    features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weekday', 'hour']
    x, y = df[features].values.astype(np.float32), df['traffic_volume'].values.astype(np.float32)
    x = (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + 1e-5)
    y = (y - np.mean(y)) / (np.std(y) + 1e-5)
    num_seqs = x.shape[0] // seq_len
    x_s, y_s = x[:num_seqs*seq_len].reshape(num_seqs, seq_len, len(features)), y[:num_seqs*seq_len].reshape(num_seqs, seq_len, 1)
    split = int(0.8 * num_seqs)
    return (torch.tensor(x_s[:split]), torch.tensor(y_s[:split])), (torch.tensor(x_s[split:]), torch.tensor(y_s[split:])), len(features), 1

def load_physionet_data(seq_len=32, num_files=500):
    base_path = r"C:\Users\afant\Documents\Denmark Internship\datasets\set-a\set-a"
    files = [f for f in os.listdir(base_path) if f.endswith(".txt")][:num_files]
    params_to_keep, target_param = ['Age', 'Gender', 'GCS', 'Temp'], 'HR'
    all_x, all_y = [], []
    print(f"[>] Chargement de {len(files)} dossiers PhysioNet...")
    for f in files:
        df = pd.read_csv(os.path.join(base_path, f))
        df['minutes'] = df['Time'].apply(lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1]) if isinstance(x, str) else 0)
        patient_data, patient_target, bins = np.zeros((seq_len, len(params_to_keep))), np.zeros((seq_len, 1)), np.linspace(0, 2880, seq_len + 1)
        for i in range(seq_len):
            bin_df = df[(df['minutes'] >= bins[i]) & (df['minutes'] < bins[i+1])]
            for j, p in enumerate(params_to_keep):
                val = bin_df[bin_df['Parameter'] == p]['Value'].mean()
                if not np.isnan(val): patient_data[i, j] = val
            t_val = bin_df[bin_df['Parameter'] == target_param]['Value'].mean()
            if not np.isnan(t_val): patient_target[i, 0] = t_val
        for j in range(len(params_to_keep)):
            last_val = 0
            for i in range(seq_len):
                if patient_data[i, j] == 0: patient_data[i, j] = last_val
                else: last_val = patient_data[i, j]
        last_t = 70.0
        for i in range(seq_len):
            if patient_target[i, 0] == 0: patient_target[i, 0] = last_t
            else: last_t = patient_target[i, 0]
        all_x.append(patient_data); all_y.append(patient_target)
    x = np.array(all_x, dtype=np.float32)
    y = np.array(all_y, dtype=np.float32)
    x = (x - np.mean(x, axis=(0, 1))) / (np.std(x, axis=(0, 1)) + 1e-5)
    y = (y - np.mean(y, axis=(0, 1))) / (np.std(y, axis=(0, 1)) + 1e-5)
    split = int(0.8 * len(files))
    return (torch.tensor(x[:split]), torch.tensor(y[:split])), (torch.tensor(x[split:]), torch.tensor(y[split:])), len(params_to_keep), 1

class ModernCfCModel(nn.Module):
    def __init__(self, input_size, units, output_size, num_layers=1):
        super(ModernCfCModel, self).__init__()
        self.layers = nn.ModuleList()
        current_input = input_size
        for i in range(num_layers):
            if i < num_layers - 1:
                out_neurons = min(units // 2, units - 3)
                if out_neurons < 1: out_neurons = 1
            else:
                out_neurons = output_size
            wiring = AutoNCP(units, out_neurons)
            self.layers.append(CfC(current_input, wiring, batch_first=True))
            current_input = out_neurons
        self.fc = nn.Linear(output_size, output_size)

    def forward(self, x, hx=None):
        for layer in self.layers:
            x, _ = layer(x)
        return self.fc(x), None

def get_input(prompt, default):
    user_input = input(f"{prompt} [{default}]: ")
    return int(user_input) if user_input.strip() else default

def get_bool_input(prompt, default):
    user_input = input(f"{prompt} (y/n) [{'y' if default else 'n'}]: ").lower().strip()
    return user_input == 'y' if user_input else default

parser = argparse.ArgumentParser(description="CfC Modern Demo with Profiler Support")
parser.add_argument("--units", type=int, help="Nombre de neurones")
parser.add_argument("--layers", type=int, default=1, help="Nombre de couches CfC")
parser.add_argument("--epochs", type=int, help="Nombre d'époques")
parser.add_argument("--batch_size", type=int, help="Taille du batch")
parser.add_argument("--device", type=str)
parser.add_argument("--dataset", type=str, choices=["sine", "har", "occupancy", "gesture", "traffic", "physionet"], default="sine")
parser.add_argument("--target_loss", type=float, default=0.0)
parser.add_argument("--profile", action="store_true")
parser.add_argument("--trace", action="store_true")

args = parser.parse_args()

if len(sys.argv) == 1:
    print("\n--- Configuration du modèle CfC (Mode Interactif) ---")
    num_units = get_input("Nombre de neurones (units)", 16)
    num_layers = get_input("Nombre de couches (layers)", 1)
    num_epochs = get_input("Nombre d'époques (epochs)", 50)
    target_loss = float(input("Perte cible [0.0]: ") or 0.0)
    batch_size = get_input("Taille du batch", 128)
    dataset_name = input("Dataset (sine/har/occupancy/gesture/traffic/physionet) [sine]: ").lower().strip() or "sine"
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = input(f"Choix du device (cuda/cpu) [{default_device}]: ").lower().strip() or default_device
    use_profiler = get_bool_input("Activer le profileur ?", False)
    save_trace = get_bool_input("Sauvegarder la trace ?", False) if use_profiler else False
else:
    num_units, num_layers, num_epochs, target_loss, batch_size, dataset_name = args.units or 16, args.layers, args.epochs or 50, args.target_loss, args.batch_size or 128, args.dataset
    device_type = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    use_profiler, save_trace = args.profile, args.trace

device = torch.device(device_type)
print(f"Device: {device} | Units: {num_units} | Layers: {num_layers} | Dataset: {dataset_name}")

if dataset_name == "har": (x_train, y_train), (x_test, y_test), input_size, output_size = load_har_data(); criterion, is_classification = nn.CrossEntropyLoss(), True
elif dataset_name == "occupancy": (x_train, y_train), (x_test, y_test), input_size, output_size = load_occupancy_data(); criterion, is_classification = nn.CrossEntropyLoss(), True
elif dataset_name == "gesture": (x_train, y_train), (x_test, y_test), input_size, output_size = load_gesture_data(); criterion, is_classification = nn.CrossEntropyLoss(), True
elif dataset_name == "traffic": (x_train, y_train), (x_test, y_test), input_size, output_size = load_traffic_data(); criterion, is_classification = nn.MSELoss(), False
elif dataset_name == "physionet": (x_train, y_train), (x_test, y_test), input_size, output_size = load_physionet_data(); criterion, is_classification = nn.MSELoss(), False
else: (x_train, y_train), (x_test, y_test), input_size, output_size = generate_sine_data(); criterion, is_classification = nn.MSELoss(), False

model = ModernCfCModel(input_size, num_units, output_size, num_layers).to(device)
with torch.cuda.device(device) if device.type == 'cuda' else torch.cpu.amp.autocast():
    try: macs, params = get_model_complexity_info(model, (x_train.shape[1], input_size), as_strings=False, print_per_layer_stat=False, verbose=False); flops_per_sample = macs * 2; flops_method = "ptflops"
    except:
        ops_per_step = (input_size * num_units + num_units * num_units) * 10 * num_layers
        flops_per_sample = ops_per_step * x_train.shape[1]
        flops_method = "theoretical_fallback"

print(f"Complexité : {flops_per_sample:.2e} FLOPS/sample ({flops_method})")
optimizer = optim.Adam(model.parameters(), lr=0.01)
start_time_seconds = time.time()
flops_stats = {
    "config": {"units": num_units, "layers": num_layers, "epochs": num_epochs, "batch_size": batch_size, "device": str(device), "flops_method": flops_method, "profiler_enabled": use_profiler},
    "execution": {"start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "end_time": None, "total_duration_seconds": 0},
    "epochs": [], "total_training_flops": 0
}

prof = None
if use_profiler:
    import shutil
    if os.path.exists('./log/cfc_profile'): shutil.rmtree('./log/cfc_profile')
    handler = torch.profiler.tensorboard_trace_handler('./log/cfc_profile') if save_trace else None
    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=handler, record_shapes=True, with_flops=True, profile_memory=True
    )
    prof.start()

dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
for epoch in range(num_epochs):
    model.train(); epoch_loss = 0; epoch_flops = flops_per_sample * 3 * len(x_train)
    for bx, by in dataloader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad(); out, _ = model(bx); loss = criterion(out.view(-1, output_size), by.view(-1)) if is_classification else criterion(out, by)
        loss.backward(); optimizer.step(); epoch_loss += loss.item()
        if use_profiler: prof.step()
    avg_loss = epoch_loss / len(dataloader)
    if (epoch + 1) % 5 == 0 or epoch == 0: print(f"  Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    if target_loss > 0 and avg_loss <= target_loss: break
    flops_stats["epochs"].append({"epoch": epoch + 1, "loss": avg_loss, "flops": epoch_flops})

if use_profiler:
    prof.stop()
    print("\n--- Résumé du Profilage Matériel ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

total_time = time.time() - start_time_seconds
flops_stats["execution"]["total_duration_seconds"] = total_time
flops_stats["execution"]["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

model.eval()
with torch.no_grad():
    print("\n--- Évaluation ---")
    correct, total, total_loss = 0, 0, 0
    all_preds, all_targets = [], []
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size)
    for bx, by in test_loader:
        bx, by = bx.to(device), by.to(device); out, _ = model(bx)
        if is_classification:
            loss = criterion(out.view(-1, output_size), by.view(-1)); preds = torch.argmax(out, dim=-1); correct += (preds == by).sum().item(); total += by.numel()
            all_preds.extend(preds.view(-1).cpu().numpy()); all_targets.extend(by.view(-1).cpu().numpy())
        else:
            loss = criterion(out, by); all_preds.extend(out.view(-1).cpu().numpy()); all_targets.extend(by.view(-1).cpu().numpy())
        total_loss += loss.item()
    avg_loss = total_loss / len(test_loader); accuracy = (100 * correct / total) if is_classification else None
    f1, mae, rmse = None, None, None
    if SKLEARN_AVAILABLE:
        if is_classification: f1 = f1_score(all_targets, all_preds, average='macro')
        else: mae = mean_absolute_error(all_targets, all_preds); rmse = np.sqrt(avg_loss)
    
    if is_classification: print(f"Test Accuracy: {accuracy:.2f}%" + (f" | F1: {f1:.4f}" if f1 else ""))
    else: print(f"Test MSE: {avg_loss:.6f}" + (f" | MAE: {mae:.6f}" if mae else "") + (f" | RMSE: {rmse:.6f}" if rmse else ""))

    prediction, _ = model(x_test[0:1].to(device)); plt.figure(figsize=(12, 8))
    if is_classification:
        p, t = torch.argmax(prediction, dim=-1).cpu().numpy()[0], y_test[0].numpy()
        plt.step(range(len(t)), t, label="True", where='post'); plt.step(range(len(p)), p, label="CfC Pred", linestyle="--", where='post')
    else:
        plt.plot(y_test[0].numpy(), label="True", linewidth=2, color='blue'); plt.plot(prediction[0].cpu().numpy(), label="CfC Pred", linestyle="--", linewidth=2, color='orange')
    
    if is_classification: metric_text = f"Acc: {accuracy:.2f}%" + (f" | F1: {f1:.3f}" if f1 else "")
    else: metric_text = f"MSE: {avg_loss:.4f}" + (f" | MAE: {mae:.3f}" if mae else "")
    full_title = (f"CfC: {num_layers}L | {num_units}u | {num_epochs}e | Dataset: {dataset_name}\n"
                  f"{metric_text} | Total Time: {total_time:.2f}s | Device: {device}")
    plt.title(full_title); plt.legend(); plt.grid(True, alpha=0.3)
    
    flops_stats["evaluation"] = {"test_loss": avg_loss, "test_accuracy_pct": accuracy, "f1_score_macro": f1, "mae": mae, "rmse": rmse}
    output_dir = os.path.join("results", "cfc", dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    tl_suffix = f"_tl{target_loss}" if target_loss > 0 else ""
    base_filename = f"cfc_{num_epochs}e_{num_units}u_{num_layers}L_b{batch_size}_{device.type}{tl_suffix}"
    plt.savefig(os.path.join(output_dir, f"{base_filename}.png"))
    with open(os.path.join(output_dir, f"{base_filename}.json"), "w") as f: json.dump(flops_stats, f, indent=4)
    print(f"\nSuccess! Results saved in '{output_dir}'")

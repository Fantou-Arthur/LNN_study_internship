import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import json
import os
import time
from datetime import datetime
from ptflops import get_model_complexity_info
import argparse
import sys
from torch.utils.data import DataLoader, TensorDataset

# --- COULEURS ANSI ---
C_BLUE = "\033[94m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_BOLD = "\033[1m"
C_END = "\033[0m"

try:
    from sklearn.metrics import f1_score, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

def apply_sparsity(x, percentage, mode='random'):
    """ Applique Option B : Masquage par valeur individuelle (cellule) """
    if percentage <= 0: return x
    if mode == 'random':
        mask = torch.rand(x.shape) > percentage
        return x * mask.to(x.dtype).to(x.device)
    elif mode == 'periodic':
        N = int(1.0 / (percentage + 1e-6))
        if N < 1: N = 1
        mask = torch.ones_like(x)
        for f in range(x.shape[2]):
            offset = f % N
            mask[:, offset::N, f] = 0
        return x * mask.to(x.device)
    return x

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
    print(f"{C_YELLOW}[>] Chargement de {len(files)} dossiers PhysioNet...{C_END}")
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

class ModernLSTMModel(nn.Module):
    def __init__(self, input_size, units, output_size, num_layers=1):
        super(ModernLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, units, num_layers, batch_first=True)
        self.fc = nn.Linear(units, output_size)

    def forward(self, x, hx=None):
        out, hx = self.lstm(x, hx)
        return self.fc(out), hx

def evaluate(model, loader, criterion, device, is_classification, output_size):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device); out, _ = model(bx)
            if is_classification:
                loss = criterion(out.view(-1, output_size), by.view(-1))
                preds = torch.argmax(out, dim=-1); correct += (preds == by).sum().item(); total += by.numel()
                all_preds.extend(preds.view(-1).cpu().numpy()); all_targets.extend(by.view(-1).cpu().numpy())
            else:
                loss = criterion(out, by)
                all_preds.extend(out.view(-1).cpu().numpy()); all_targets.extend(by.view(-1).cpu().numpy())
            total_loss += loss.item()
    return total_loss / len(loader), correct, total, all_preds, all_targets

def get_input(prompt, default):
    user_input = input(f"{prompt} [{default}]: ")
    return int(user_input) if user_input.strip() else default

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM Modern Demo with Sparsity & Persistence")
    parser.add_argument("--units", type=int, help="Nombre de neurones")
    parser.add_argument("--layers", type=int, default=1, help="Nombre de couches")
    parser.add_argument("--epochs", type=int, help="Nombre d'époques")
    parser.add_argument("--batch_size", type=int, help="Taille du batch")
    parser.add_argument("--device", type=str)
    parser.add_argument("--dataset", type=str, default="sine")
    parser.add_argument("--target_loss", type=float, default=0.0)
    parser.add_argument("--sparsity_train", type=float, default=0.0)
    parser.add_argument("--sparsity_test", type=float, default=0.0)
    parser.add_argument("--sparsity_mode", type=str, default="random")

    args = parser.parse_args()

    if len(sys.argv) == 1:
        print(f"\n{C_BOLD}{C_BLUE}--- Configuration du modèle LSTM ---{C_END}")
        num_units = get_input("Nombre d'unités (units)", 32)
        num_layers = get_input("Nombre de couches (layers)", 1)
        num_epochs = get_input("Nombre d'époques (epochs)", 50)
        target_loss = float(input("Perte cible [0.0]: ") or 0.0)
        batch_size = get_input("Taille du batch", 128)
        dataset_name = input("Dataset [sine]: ").lower().strip() or "sine"
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        device_type = input(f"Choix du device [{default_device}]: ").lower().strip() or default_device
        sparsity_train, sparsity_test, sparsity_mode = 0.0, 0.0, "random"
    else:
        num_units, num_layers, num_epochs, target_loss, batch_size, dataset_name = args.units or 32, args.layers, args.epochs or 50, args.target_loss, args.batch_size or 128, args.dataset
        device_type = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
        sparsity_train, sparsity_test, sparsity_mode = args.sparsity_train, args.sparsity_test, args.sparsity_mode

    device = torch.device(device_type)
    print(f"{C_BOLD}Device:{C_END} {C_GREEN}{device}{C_END} | {C_BOLD}Units:{C_END} {num_units} | {C_BOLD}Dataset:{C_END} {C_YELLOW}{dataset_name}{C_END}")

    if dataset_name == "har": (x_train, y_train), (x_test, y_test), input_size, output_size = load_har_data(); criterion, is_classification = nn.CrossEntropyLoss(), True
    elif dataset_name == "occupancy": (x_train, y_train), (x_test, y_test), input_size, output_size = load_occupancy_data(); criterion, is_classification = nn.CrossEntropyLoss(), True
    elif dataset_name == "gesture": (x_train, y_train), (x_test, y_test), input_size, output_size = load_gesture_data(); criterion, is_classification = nn.CrossEntropyLoss(), True
    elif dataset_name == "traffic": (x_train, y_train), (x_test, y_test), input_size, output_size = load_traffic_data(); criterion, is_classification = nn.MSELoss(), False
    elif dataset_name == "physionet": (x_train, y_train), (x_test, y_test), input_size, output_size = load_physionet_data(); criterion, is_classification = nn.MSELoss(), False
    else: (x_train, y_train), (x_test, y_test), input_size, output_size = generate_sine_data(); criterion, is_classification = nn.MSELoss(), False

    # SAUVEGARDE DU TEST SET PROPRE
    eval_data_save = {"x_test": x_test, "y_test": y_test, "input_size": input_size, "output_size": output_size, "is_classification": is_classification}

    # Appliquer la robustesse (Option B)
    x_train = apply_sparsity(x_train, sparsity_train, sparsity_mode)
    x_test_sparsified = apply_sparsity(x_test, sparsity_test, sparsity_mode)

    model = ModernLSTMModel(input_size, num_units, output_size, num_layers).to(device)
    try: macs, params = get_model_complexity_info(model, (x_train.shape[1], input_size), as_strings=False, print_per_layer_stat=False, verbose=False); flops_per_sample = macs * 2
    except: flops_per_sample = 0

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.001)

    start_time_seconds = time.time()
    flops_stats = {
        "config": {"units": num_units, "layers": num_layers, "epochs": num_epochs, "batch_size": batch_size, "device": str(device), 
                   "sparsity": {"train": sparsity_train, "test": sparsity_test, "mode": sparsity_mode}, "model_type": "lstm"},
        "execution": {"start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "end_time": None, "total_duration_seconds": 0},
        "epochs": [], "evaluation": {}
    }

    best_loss, patience_stop, patience_counter = float('inf'), 15, 0
    dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test_sparsified, y_test), batch_size=batch_size)

    for epoch in range(num_epochs):
        model.train(); epoch_loss = 0
        for bx, by in dataloader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(); out, _ = model(bx); loss = criterion(out.view(-1, output_size), by.view(-1)) if is_classification else criterion(out, by)
            loss.backward(); optimizer.step(); epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(dataloader)
        avg_test_loss, _, _, _, _ = evaluate(model, test_loader, criterion, device, is_classification, output_size)
        
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_train_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if (epoch + 1) % 5 == 0 or epoch == 0: 
            print(f"  Epoch [{epoch+1}/{num_epochs}], Loss Train: {C_BOLD}{avg_train_loss:.4f}{C_END}, Test: {C_YELLOW}{avg_test_loss:.4f}{C_END}, LR: {new_lr:.2e}")
        
        if avg_train_loss < best_loss * 0.999: best_loss = avg_train_loss; patience_counter = 0
        else: patience_counter += 1
        
        flops_stats["epochs"].append({"epoch": epoch + 1, "loss": avg_train_loss, "test_loss": avg_test_loss, "flops": flops_per_sample, "lr": new_lr})
        if (target_loss > 0 and avg_train_loss <= target_loss) or patience_counter >= patience_stop: break

    total_time = time.time() - start_time_seconds
    flops_stats["execution"]["total_duration_seconds"] = total_time
    flops_stats["execution"]["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    avg_test_loss, correct, total, all_preds, all_targets = evaluate(model, test_loader, criterion, device, is_classification, output_size)
    accuracy = (100 * correct / total) if is_classification else None
    f1 = f1_score(all_targets, all_preds, average='macro') if (SKLEARN_AVAILABLE and is_classification) else None
    mae = mean_absolute_error(all_targets, all_preds) if (SKLEARN_AVAILABLE and not is_classification) else None
    flops_stats["evaluation"] = {"test_loss": avg_test_loss, "test_accuracy_pct": accuracy, "f1_score_macro": f1, "mae": mae}

    print(f"\n{C_BOLD}--- Évaluation Finale ---{C_END}")
    if is_classification: print(f"{C_BOLD}Test Accuracy:{C_END} {C_GREEN}{accuracy:.2f}%{C_END}")
    else: print(f"{C_BOLD}Test MSE:{C_END} {C_GREEN}{avg_test_loss:.6f}{C_END}")

    # NOM DU FICHIER AVEC SPARSITY
    base_filename = f"lstm_{dataset_name}_spT{int(sparsity_train*100)}_spTe{int(sparsity_test*100)}_{sparsity_mode}_{num_epochs}e_{num_units}u_{num_layers}L_{device.type}"
    output_dir = os.path.join("results", "lstm", dataset_name); os.makedirs(output_dir, exist_ok=True)

    # Sauvegarde des Poids et Données
    torch.save(model.state_dict(), os.path.join(output_dir, f"weights_{base_filename}.pt"))
    torch.save(eval_data_save, os.path.join(output_dir, f"eval_data_{base_filename}.pt"))

    model.eval()
    with torch.no_grad():
        prediction, _ = model(x_test_sparsified[0:1].to(device))
    plt.figure(figsize=(12, 8))
    if is_classification:
        p, t = torch.argmax(prediction, dim=-1).cpu().numpy()[0], y_test[0].numpy()
        plt.step(range(len(t)), t, label="True"); plt.step(range(len(p)), p, label="LSTM Pred", linestyle="--")
    else:
        plt.plot(y_test[0].numpy(), label="True"); plt.plot(prediction[0].detach().cpu().numpy(), label="LSTM Pred", linestyle="--")
    plt.title(f"LSTM: {dataset_name} | Sparsity Test: {sparsity_test*100:.0f}%"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"{base_filename}.png"))

    with open(os.path.join(output_dir, f"{base_filename}.json"), "w") as f: json.dump(flops_stats, f, indent=4)
    print(f"\n{C_GREEN}{C_BOLD}Success! Results, weights and eval_data saved in '{output_dir}'{C_END}")

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import glob

# Import models
from ltc_modern_demo import ModernLTCModel
from CfC import ModernCfCModel
from RNN import ModernRNNModel
from LSTM import ModernLSTMModel
from GRU import ModernGRUModel
from CNN import ModernCNNModel

# Couleurs ANSI
C_BLUE = "\033[94m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_BOLD = "\033[1m"
C_END = "\033[0m"

MODELS = {
    "ltc": ModernLTCModel,
    "cfc": ModernCfCModel,
    "rnn": ModernRNNModel,
    "lstm": ModernLSTMModel,
    "gru": ModernGRUModel,
    "cnn": ModernCNNModel
}

def apply_sparsity(data, sparsity, mode="random"):
    if sparsity <= 0:
        return data.clone()
    sparse_data = data.clone()
    batch, seq, feat = sparse_data.shape
    if mode == "random":
        mask = torch.rand(batch, seq, feat, device=data.device) > sparsity
    else:
        period = int(1 / (1 - sparsity)) if sparsity < 1 else 1
        mask = torch.zeros(seq, device=data.device)
        mask[::period] = 1
        mask = mask.view(1, seq, 1).expand(batch, seq, feat)
    return sparse_data * mask.float()

def find_latest_files(model_name, dataset_name):
    base_path = os.path.join("results", model_name, dataset_name)
    if not os.path.exists(base_path): return None, None
    weight_files = glob.glob(os.path.join(base_path, "**", "weights_*.pt"), recursive=True)
    data_files = glob.glob(os.path.join(base_path, "**", "eval_data_*.pt"), recursive=True)
    if not weight_files or not data_files: return None, None
    latest_weights = max(weight_files, key=os.path.getmtime)
    dir_of_weights = os.path.dirname(latest_weights)
    config_str = os.path.basename(latest_weights).replace("weights_", "")
    matching_data = os.path.join(dir_of_weights, "eval_data_" + config_str)
    if os.path.exists(matching_data): return latest_weights, matching_data
    return latest_weights, max(data_files, key=os.path.getmtime)

def main():
    parser = argparse.ArgumentParser(description="Visualisation des prédictions multi-modèles")
    parser.add_argument("--dataset", type=str, default="sine", help="Dataset à visualiser")
    parser.add_argument("--mode", type=str, default="random", choices=["random", "periodic"])
    parser.add_argument("--sample_idx", type=int, default=0, help="Index de l'échantillon à tester")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    sparsity_levels = np.arange(0.05, 1.0, 0.05) # 5%, 10%, ..., 95%
    
    output_dir = os.path.join("results", "visualizations", args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{C_BOLD}{C_BLUE}--- GÉNÉRATION DES VISUALISATIONS ({args.dataset.upper()}) ---{C_END}")

    # 1. Charger les modèles et les données
    loaded_models = {}
    x_test, y_test, is_classification, output_size = None, None, None, None

    for m_name in MODELS.keys():
        w_path, d_path = find_latest_files(m_name, args.dataset)
        if not w_path: continue

        print(f"  Chargement {m_name}...")
        try:
            data_bundle = torch.load(d_path, map_location="cpu")
            if x_test is None:
                x_test, y_test = data_bundle["x_test"], data_bundle["y_test"]
                is_classification = data_bundle["is_classification"]
                output_size = data_bundle["output_size"]
            
            # Extraire config du nom
            fname = os.path.basename(w_path)
            parts = fname.split("_")
            units, layers = 32, 1
            for p in parts:
                if p.endswith("u") and p[:-1].isdigit(): units = int(p[:-1])
                if p.endswith("L") and p[:-1].isdigit(): layers = int(p[:-1])

            if m_name == "cnn":
                model = MODELS[m_name](data_bundle["input_size"], units, output_size, layers, "relu").to(device)
            else:
                model = MODELS[m_name](data_bundle["input_size"], units, output_size, layers).to(device)
            
            model.load_state_dict(torch.load(w_path, map_location=device))
            model.eval()
            loaded_models[m_name] = model
        except Exception as e:
            print(f"    {C_RED}Erreur: {e}{C_END}")

    if not loaded_models:
        print(f"{C_RED}Aucun modèle trouvé pour ce dataset.{C_END}")
        return

    # 2. Boucle de sparsité
    sample_x = x_test[args.sample_idx:args.sample_idx+1].to(device)
    sample_y = y_test[args.sample_idx].cpu().numpy()

    for sp in sparsity_levels:
        plt.figure(figsize=(12, 7))
        
        # Signal réel
        if is_classification:
            plt.step(range(len(sample_y)), sample_y, label="True", color="black", linewidth=2, where='post', alpha=0.5)
        else:
            plt.plot(sample_y, label="True", color="black", linewidth=2, alpha=0.5)

        # Signal bruité (sparsity)
        x_sparse = apply_sparsity(sample_x, sp, args.mode)
        
        # Prédictions
        for m_name, model in loaded_models.items():
            with torch.no_grad():
                if m_name == "cnn": out = model(x_sparse)
                else: out, _ = model(x_sparse)
                
                pred = out[0].cpu().numpy()
                if is_classification:
                    pred = np.argmax(pred, axis=-1)
                    plt.step(range(len(pred)), pred, label=f"{m_name.upper()}", linestyle="--", where='post')
                else:
                    plt.plot(pred, label=f"{m_name.upper()}", linestyle="--")

        plt.title(f"Prédictions sur {args.dataset.upper()} | Sparsité: {sp*100:.0f}% ({args.mode})")
        plt.legend(loc='upper right', ncol=2)
        plt.grid(True, alpha=0.3)
        
        fname = f"compare_preds_sp{int(sp*100)}_{args.mode}.png"
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()
        print(f"  {C_GREEN}[OK]{C_END} Sauvegardé: {fname}")

    print(f"\n{C_BOLD}Visualisations terminées dans : {output_dir}{C_END}")

if __name__ == "__main__":
    main()

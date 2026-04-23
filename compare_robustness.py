import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import glob
import json
from datetime import datetime

# Import models (ensure they are in the path)
from ltc_modern_demo import ModernLTCModel
from CfC import ModernCfCModel
from RNN import ModernRNNModel
from LSTM import ModernLSTMModel
from GRU import ModernGRUModel
from CNN import ModernCNNModel

# Couleurs pour le terminal
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_BLUE = "\033[94m"
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

DATASETS = ["sine", "har", "occupancy", "gesture", "traffic", "physionet"]

def apply_sparsity(data, sparsity, mode="random"):
    if sparsity <= 0:
        return data.clone()
    
    sparse_data = data.clone()
    batch, seq, feat = sparse_data.shape
    
    if mode == "random":
        mask = torch.rand(batch, seq, feat, device=data.device) > sparsity
        sparse_data = sparse_data * mask.float()
    else:  # periodic
        period = int(1 / (1 - sparsity)) if sparsity < 1 else 1
        mask = torch.zeros(seq, device=data.device)
        mask[::period] = 1
        mask = mask.view(1, seq, 1).expand(batch, seq, feat)
        sparse_data = sparse_data * mask.float()
        
    return sparse_data

def evaluate_robustness(model, x_test, y_test, criterion, is_classification, sparsity, mode, device):
    model.eval()
    x_sparse = apply_sparsity(x_test, sparsity, mode).to(device)
    y_test = y_test.to(device)
    
    with torch.no_grad():
        if isinstance(model, ModernCNNModel):
            output = model(x_sparse)
        else:
            output, _ = model(x_sparse)
            
        if is_classification:
            loss = criterion(output.view(-1, output.shape[-1]), y_test.view(-1))
            preds = torch.argmax(output, dim=-1)
            correct = (preds == y_test).sum().item()
            total = y_test.numel()
            return correct / total # Accuracy
        else:
            loss = criterion(output, y_test)
            return loss.item() # MSE

def find_latest_files(model_name, dataset_name):
    base_path = os.path.join("results", model_name, dataset_name)
    if not os.path.exists(base_path):
        return None, None
    
    weight_files = glob.glob(os.path.join(base_path, "weights_*.pt"))
    data_files = glob.glob(os.path.join(base_path, "eval_data_*.pt"))
    
    if not weight_files or not data_files:
        return None, None
    
    # Trouver le plus récent (basé sur le temps de modif)
    latest_weights = max(weight_files, key=os.path.getmtime)
    
    # Essayer de trouver le eval_data qui correspond au même timestamp/config
    # On cherche celui qui a la même fin de nom après 'weights_'
    config_str = os.path.basename(latest_weights).replace("weights_", "")
    matching_data = os.path.join(base_path, "eval_data_" + config_str)
    
    if os.path.exists(matching_data):
        return latest_weights, matching_data
    else:
        # Fallback sur le plus récent
        return latest_weights, max(data_files, key=os.path.getmtime)

def main():
    parser = argparse.ArgumentParser(description="Comparaison de la Robustesse entre Modèles")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, help="Liste des datasets à tester")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()), help="Liste des modèles à comparer")
    parser.add_argument("--sparsity_mode", type=str, default="random", choices=["random", "periodic"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    sparsity_levels = np.linspace(0, 0.95, 20)
    
    output_dir = os.path.join("results", "comparison")
    os.makedirs(output_dir, exist_ok=True)

    results_all = {}

    for dataset in args.datasets:
        print(f"\n{C_BOLD}{C_BLUE}--- ANALYSE DU DATASET : {dataset.upper()} ---{C_END}")
        results_all[dataset] = {}
        
        for model_name in args.models:
            weights_path, data_path = find_latest_files(model_name, dataset)
            
            if not weights_path:
                print(f"  {C_RED}[SKIP]{C_END} Modèle {model_name} non trouvé pour {dataset}")
                continue
                
            print(f"  {C_YELLOW}[TEST]{C_END} Modèle {model_name}...")
            
            try:
                # Charger les données
                data_bundle = torch.load(data_path, map_location="cpu") # Charger sur CPU d'abord
                x_test = data_bundle["x_test"]
                y_test = data_bundle["y_test"]
                input_size = data_bundle["input_size"]
                output_size = data_bundle["output_size"]
                is_classification = data_bundle["is_classification"]
                
                # Charger le modèle (extraire config du nom de fichier si possible)
                fname = os.path.basename(weights_path)
                parts = fname.split("_")
                units = 32
                layers = 1
                for p in parts:
                    if p.endswith("u") and p[:-1].isdigit(): 
                        units = int(p[:-1])
                    if p.endswith("L") and p[:-1].isdigit(): 
                        layers = int(p[:-1])

                if model_name == "cnn":
                    model = MODELS[model_name](input_size, units, output_size, layers, "relu").to(device)
                else:
                    model = MODELS[model_name](input_size, units, output_size, layers).to(device)
                
                state_dict = torch.load(weights_path, map_location=device)
                model.load_state_dict(state_dict)
                criterion = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()
                
                # Déplacer les données sur le bon device
                x_test = x_test.to(device)
                y_test = y_test.to(device)
                
                model_results = []
                for sp in sparsity_levels:
                    metric = evaluate_robustness(model, x_test, y_test, criterion, is_classification, sp, args.sparsity_mode, device)
                    model_results.append(metric)
                
                results_all[dataset][model_name] = {
                    "sparsity": sparsity_levels.tolist(),
                    "metrics": model_results,
                    "is_classification": is_classification
                }
                
            except Exception as e:
                print(f"    {C_RED}Erreur lors du test de {model_name} : {e}{C_END}")

        # Tracer le graphique pour ce dataset
        if results_all[dataset]:
            plt.figure(figsize=(10, 6))
            is_classif = any(r["is_classification"] for r in results_all[dataset].values())
            metric_name = "Accuracy" if is_classif else "MSE"
            
            for m_name, m_res in results_all[dataset].items():
                plt.plot(np.array(m_res["sparsity"]) * 100, m_res["metrics"], label=m_name.upper(), marker='o', markersize=4)
            
            plt.title(f"Robustesse sur {dataset.upper()} (Mode: {args.sparsity_mode})")
            plt.xlabel("Sparsité (%)")
            plt.ylabel(metric_name)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            save_path = os.path.join(output_dir, f"robustness_compare_{dataset}_{args.sparsity_mode}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"  {C_GREEN}[OK]{C_END} Graphique comparatif sauvegardé : {save_path}")

    # Sauvegarder les résultats numériques
    with open(os.path.join(output_dir, f"results_robustness_{args.sparsity_mode}.json"), "w") as f:
        json.dump(results_all, f, indent=4)

if __name__ == "__main__":
    main()

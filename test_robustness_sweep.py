import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys
from torch.utils.data import DataLoader, TensorDataset

# Importer les classes de modèles
try:
    from ltc_modern_demo import ModernLTCModel
    from CfC import ModernCfCModel
    from RNN import ModernRNNModel
    from LSTM import ModernLSTMModel
    from GRU import ModernGRUModel
    from CNN import ModernCNNModel
except ImportError:
    print("Erreur : Impossible d'importer les classes de modèles. Assurez-vous d'être dans le dossier racine.")
    sys.exit(1)

# --- COULEURS ANSI ---
C_BLUE = "\033[94m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_BOLD = "\033[1m"
C_END = "\033[0m"

def apply_sparsity(x, percentage, mode='random'):
    if percentage <= 0: return x
    if mode == 'random':
        mask = torch.rand(x.shape) > percentage
        return x * mask.to(x.dtype)
    elif mode == 'periodic':
        N = int(1.0 / (percentage + 1e-6))
        if N < 1: N = 1
        mask = torch.ones_like(x)
        for f in range(x.shape[2]):
            offset = f % N
            mask[:, offset::N, f] = 0
        return x * mask
    return x

def evaluate(model, loader, criterion, device, is_classification, output_size):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            # Gestion CfC/LTC qui renvoient (out, hx) vs CNN qui renvoie out
            out_raw = model(bx)
            out = out_raw[0] if isinstance(out_raw, tuple) else out_raw
            
            if is_classification:
                loss = criterion(out.view(-1, output_size), by.view(-1))
                preds = torch.argmax(out, dim=-1)
                correct += (preds == by).sum().item()
                total += by.numel()
            else:
                loss = criterion(out, by)
            total_loss += loss.item()
    
    score = (100.0 * correct / total) if is_classification else (total_loss / len(loader))
    return score

def main():
    print(f"\n{C_BOLD}{C_BLUE}>>> INITIALISATION DU SCRIPT DE SWEEP v1.1 <<<{C_END}")
    parser = argparse.ArgumentParser(description="Sweep de robustesse (5% à 95% de sparsity)")
    parser.add_argument("--weights", type=str, required=True, help="Chemin vers le fichier .pt des poids")
    parser.add_argument("--data", type=str, required=True, help="Chemin vers le fichier .pt des données d'éval")
    parser.add_argument("--mode", type=str, default="random", help="Mode de sparsity (random/periodic)")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{C_BOLD}{C_BLUE}--- ANALYSE DE ROBUSTESSE ---{C_END}")
    
    # 1. Charger les données
    print(f"Chargement des données : {C_YELLOW}{args.data}{C_END}")
    data_bundle = torch.load(args.data)
    x_test_clean = data_bundle["x_test"]
    y_test = data_bundle["y_test"]
    input_size = data_bundle["input_size"]
    output_size = data_bundle["output_size"]
    is_classification = data_bundle["is_classification"]

    # 2. Déterminer le modèle et charger les poids
    # On essaye de deviner le type de modèle via le nom du fichier
    filename = os.path.basename(args.weights).lower()
    
    # Parsing des hyperparamètres depuis le nom du fichier (convention : weights_model_dataset_spT_spTe_mode_epochs_units_layersL_device.pt)
    parts = filename.replace(".pt", "").split("_")
    # Format attendu : [weights, model, dataset, spT, spTe, mode, epochs, units, layersL, device]
    # Mais si l'utilisateur change le nom, on a besoin de robustesse.
    
    model_type = "ltc" # default
    units = 32
    layers = 1
    
    for p in parts:
        if "u" in p and p.replace("u", "").isdigit(): units = int(p.replace("u", ""))
        if "l" in p and p.replace("l", "").replace("L", "").isdigit(): layers = int(p.replace("l", "").replace("L", ""))
        for t in ["ltc", "cfc", "rnn", "lstm", "gru", "cnn"]:
            if t in p: model_type = t

    print(f"Modèle détecté : {C_GREEN}{model_type.upper()}{C_END} | Units: {units} | Layers: {layers}")

    model_classes = {
        "ltc": ModernLTCModel, "cfc": ModernCfCModel, "rnn": ModernRNNModel,
        "lstm": ModernLSTMModel, "gru": ModernGRUModel, "cnn": ModernCNNModel
    }
    
    if model_type == "cnn":
        model = model_classes[model_type](input_size, units, output_size, layers)
    else:
        model = model_classes[model_type](input_size, units, output_size, layers)
        
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()
    
    # 3. Sweep
    sparsity_levels = np.arange(0.0, 0.96, 0.01)
    results = []

    print(f"\n{C_BOLD}Lancement du sweep (Mode: {args.mode})...{C_END}")
    for sp in sparsity_levels:
        # Appliquer sparsity sur une copie des données
        x_sp = apply_sparsity(x_test_clean.clone(), sp, args.mode)
        loader = DataLoader(TensorDataset(x_sp, y_test), batch_size=args.batch_size)
        score = evaluate(model, loader, criterion, device, is_classification, output_size)
        results.append(score)
        if int(sp * 100) % 5 == 0:
            print(f"  Sparsity {sp*100:4.1f}% -> {'Précision' if is_classification else 'MSE'}: {C_GREEN}{score:.4f}{C_END}")

    # 4. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(sparsity_levels * 100, results, marker='o', linestyle='-', linewidth=2, color='#2ecc71')
    plt.title(f"Test de Robustesse : {model_type.upper()} ({args.mode})")
    plt.xlabel("Données Manquantes (%)")
    plt.ylabel("Précision (%)" if is_classification else "MSE (Perte)")
    plt.grid(True, alpha=0.3)
    
    # Inverser l'axe Y pour le MSE (plus bas c'est mieux, donc on veut voir la montée de l'erreur)
    # Pas besoin de l'inverser, la courbe montera, ce qui est logique pour une erreur.
    
    output_path = args.weights.replace(".pt", "_robustness_sweep.png").replace("weights_", "")
    plt.savefig(output_path)
    print(f"\nGraphique de robustesse sauvegardé : {C_YELLOW}{output_path}{C_END}")

if __name__ == "__main__":
    main()

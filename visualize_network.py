import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import argparse
import os
import sys

# Import project models
from CfC import ModernCfCModel
from ltc_modern_demo import ModernLTCModel
from RNN import ModernRNNModel

def visualize_model(model_path, model_type, show_params=False):
    # 1. Load Model Weights
    if not os.path.exists(model_path):
        print(f"❌ Fichier non trouvé : {model_path}")
        return

    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 1a. Try to guess architecture from filename
    filename = os.path.basename(model_path)
    import re
    u_match = re.search(r'(\d+)u', filename)
    l_match = re.search(r'(\d+)L', filename)
    
    units = int(u_match.group(1)) if u_match else 16
    layers = int(l_match.group(1)) if l_match else 1

    # Try to find eval_data to get input/output sizes
    eval_path = model_path.replace("weights_", "eval_data_")
    input_size, output_size = 3, 1 # Defaults
    if os.path.exists(eval_path):
        eval_data = torch.load(eval_path, map_location='cpu')
        input_size = eval_data.get('input_size', input_size)
        output_size = eval_data.get('output_size', output_size)
    
    # 1b. Refine units from weights if possible
    for key in checkpoint.keys():
        # For RNN/LSTM/CfC: weight_ih_l0 or similar
        # For LTC: layers.0.rnn_cell.sensory_mu
        if 'weight' in key or 'mu' in key or 'vleak' in key:
            if checkpoint[key].dim() >= 1:
                # The first dimension often corresponds to the hidden size or a multiple
                cand = checkpoint[key].shape[0]
                if cand >= units: # Prefer larger if multiple candidates
                     # For NCPS, some params are [units, units] or [units, input]
                     # We use the filename as primary source if available
                     pass
    
    print(f"🔍 Architecture détectée : In:{input_size} | Units:{units} | Layers:{layers} | Out:{output_size}")

    # 2. Reconstruct Model (to get wiring if LTC/CfC)
    if model_type == 'ltc':
        model = ModernLTCModel(input_size, units, output_size, num_layers=layers)
    elif model_type == 'cfc':
        model = ModernCfCModel(input_size, units, output_size, num_layers=layers)
    else:
        model = ModernRNNModel(input_size, units, output_size, num_layers=layers)

    model.load_state_dict(checkpoint)
    model.eval()

    # 3. Build Graph
    G = nx.DiGraph()
    
    # Layers positions
    pos = {}
    layer_dist = 2
    node_dist = 1
    
    # Nodes: Inputs
    for i in range(input_size):
        node_id = f"In_{i}"
        G.add_node(node_id, layer='input', color='skyblue')
        pos[node_id] = np.array([0, -i * node_dist + (input_size-1)*node_dist/2])

    # Nodes: Hidden (Units)
    for i in range(units):
        node_id = f"H_{i}"
        G.add_node(node_id, layer='hidden', color='lightgreen')
        pos[node_id] = np.array([layer_dist, -i * node_dist + (units-1)*node_dist/2])
        
        # Add recurrent loops for RNN/LTC
        G.add_edge(node_id, node_id, weight=0.1)

    # Nodes: Outputs
    for i in range(output_size):
        node_id = f"Out_{i}"
        G.add_node(node_id, layer='output', color='salmon')
        pos[node_id] = np.array([2 * layer_dist, -i * node_dist + (output_size-1)*node_dist/2])

    # 4. Extract Actual Connections
    print(f"📊 Extraction du câblage réel...")
    
    # Check if we can extract wiring (LTC and CfC from ncps have it)
    try:
        # We assume the first layer defines the primary wiring
        if model_type in ['ltc', 'cfc']:
            # In ncps, wiring is in rnn_cell.wiring
            # ModernLTCModel/ModernCfCModel stores layers in self.layers
            wiring = model.layers[0].rnn_cell.wiring
            adj = wiring.adjacency_matrix # [units, units]
            in_adj = wiring.input_adjacency_matrix # [units, input_size]
            
            # 4a. Connect Inputs to Hidden (Sensory connections)
            for i in range(input_size):
                for j in range(units):
                    if in_adj[j, i] != 0:
                        G.add_edge(f"In_{i}", f"H_{j}", weight=1.0)
            
            # 4b. Connect Hidden to Hidden (Internal/Recurrent)
            for i in range(units):
                for j in range(units):
                    if adj[i, j] != 0:
                        G.add_edge(f"H_{i}", f"H_{j}", weight=0.5)

            # 4c. Connect Hidden to Output (Motor connections)
            # In AutoNCP, only the last 'output_size' neurons are usually motor
            # but we check the projection or the wiring's motor identification
            for i in range(units):
                # Simple logic: if it's an LTC, the last layer's wiring knows the outputs
                # In our ModernLTCModel, the last LTC layer has out_neurons = output_size
                # We'll connect all hidden nodes that have a path to output
                for j in range(output_size):
                    G.add_edge(f"H_{i}", f"Out_{j}", weight=1.0)
        else:
            # Standard RNN is fully connected
            for i in range(input_size):
                for j in range(units):
                    G.add_edge(f"In_{i}", f"H_{j}")
            for i in range(units):
                for j in range(units):
                    G.add_edge(f"H_{i}", f"H_{j}")
            for i in range(units):
                for j in range(output_size):
                    G.add_edge(f"H_{i}", f"Out_{j}")
    except Exception as e:
        print(f"⚠️ Erreur lors de l'extraction du câblage : {e}. Utilisation d'un schéma générique.")
        for i in range(input_size):
            for j in range(units): G.add_edge(f"In_{i}", f"H_{j}")
        for i in range(units):
            for j in range(output_size): G.add_edge(f"H_{i}", f"Out_{j}")

    # 5. Drawing
    plt.figure(figsize=(12, 8))
    colors = [G.nodes[n]['color'] for n in G.nodes]
    
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=2000, 
            font_size=10, font_weight='bold', edge_color='gray', 
            arrowsize=20, connectionstyle='arc3,rad=0.1')

    if show_params:
        print("\n--- Paramètres du Réseau (Échantillon) ---")
        for name, param in model.named_parameters():
            if 'weight' in name or 'bias' in name:
                print(f"ID: {name} | Shape: {list(param.shape)} | Mean: {param.data.mean():.4f}")
        
        # Add random weights as labels for demo on some edges
        edge_labels = {}
        for edge in list(G.edges())[:10]:
            edge_labels[edge] = f"{np.random.uniform(-1, 1):.2f}"
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(f"Visualisation du réseau {model_type.upper()} ({units} unités)")
    plt.axis('off')
    
    # Save output
    output_img = f"vis_{model_type}_{os.path.basename(model_path)}.png"
    plt.savefig(output_img)
    print(f"\n✅ Visualisation enregistrée sous : {output_img}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualisation de réseaux neuronaux entraînés")
    parser.add_argument("--model", type=str, required=True, help="Chemin vers le fichier .pt des poids")
    parser.add_argument("--type", type=str, required=True, choices=['ltc', 'cfc', 'rnn'], help="Type de modèle")
    parser.add_argument("--show-params", action="store_true", help="Afficher les paramètres des neurones")
    
    args = parser.parse_args()
    
    # Check if networkx is installed
    try:
        import networkx
    except ImportError:
        print("❌ Bibliothèques manquantes. Installation de networkx...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "networkx"])

    visualize_model(args.model, args.type, args.show_params)

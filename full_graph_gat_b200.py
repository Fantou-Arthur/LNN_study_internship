import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
# On utilise le DataLoader standard de PyTorch pour éviter les dépendances C++
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
import time
import json
import os
import argparse
from tqdm import tqdm

# --- ARGUMENTS LIGNE DE COMMANDE ---
parser = argparse.ArgumentParser(description="Full-Graph Benchmark B200 Optimized")
parser.add_argument('--epochs', '-e', type=int, default=30, help="Époques initiales")
parser.add_argument('--retrain-epochs', '-re', type=int, default=10, help="Époques de réentraînement")
args = parser.parse_known_args()[0]

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: Vérifiez que ce chemin est accessible sur votre cluster
CSV_PATH = r"C:\Users\afant\Downloads\comtradeExports_updatedH5[240924]-wb.csv\comtradeExports_updatedH5[240924]-wb.csv"
PARQUET_PATH = os.path.join(SCRIPT_DIR, "trade_data.parquet")

BATCH_SIZE = 32768 # Taille de batch large pour le B200
HIDDEN_DIM = 64
LEARNING_RATE = 0.001
THRESHOLD_RETRAIN = 1.15

# --- ARCHITECTURES ---

class FullGraphGAT(nn.Module):
    def __init__(self, in_channels, edge_dim):
        super().__init__()
        self.conv1 = GATConv(in_channels, HIDDEN_DIM, heads=4, edge_dim=edge_dim, concat=True)
        self.conv2 = GATConv(HIDDEN_DIM * 4, HIDDEN_DIM, heads=1, edge_dim=edge_dim, concat=False)
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2 + edge_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x, edge_index, edge_attr, target_edge_index, target_edge_attr):
        # x: [N, in_channels], edge_index: [2, E_context], target_edge_index: [2, BATCH]
        h = torch.relu(self.conv1(x, edge_index, edge_attr))
        h = self.conv2(h, edge_index, edge_attr)
        
        # On extrait les embeddings des nœuds source et destination pour le batch actuel
        src, dst = target_edge_index
        combined = torch.cat([h[src], h[dst], target_edge_attr], dim=-1)
        return self.fc(combined)

class SeqModel(nn.Module):
    def __init__(self, model_type, in_dim):
        super().__init__()
        if model_type == 'CfC': self.core = CfC(in_dim, HIDDEN_DIM, batch_first=True)
        elif model_type == 'LTC': self.core = LTC(in_dim, AutoNCP(HIDDEN_DIM, 1), batch_first=True)
        elif model_type == 'LSTM': self.core = nn.LSTM(in_dim, HIDDEN_DIM, batch_first=True)
        else: self.core = nn.RNN(in_dim, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x):
        # x shape: [Batch, SeqLen, Feats] -> [Batch, 1, 4]
        out, _ = self.core(x)
        if isinstance(out, tuple): out = out[0]
        return self.fc(out[:, -1, :])

# --- UTILS ---

def calculate_drift(ref, curr):
    ks, _ = ks_2samp(ref, curr)
    hist_p, _ = np.histogram(ref, bins=20, density=True)
    hist_q, _ = np.histogram(curr, bins=20, density=True)
    js = jensenshannon(hist_p + 1e-6, hist_q + 1e-6)
    return float(ks), 0.0, float(js) # PSI omis pour rapidité

def get_pyg_data(df, country_map):
    df_c = df.copy()
    df_c['src'] = df_c['reporterCode'].map(country_map)
    df_c['dst'] = df_c['partnerCode'].map(country_map)
    df_c = df_c.dropna(subset=['src', 'dst'])
    
    ei = torch.tensor([df_c['src'].values, df_c['dst'].values], dtype=torch.long)
    ea = torch.tensor(np.log1p(df_c[['dist', 'cmdCode', 'gdp_o', 'gdp_d']].values), dtype=torch.float)
    y = torch.tensor(np.log1p(df_c['primaryValue'].values), dtype=torch.float).view(-1, 1)
    
    # 250 pays x 16 features aléatoires
    x = torch.randn(len(country_map), 16)
    return Data(x=x, edge_index=ei, edge_attr=ea, y=y)

# --- ENGINE ---

def run_adaptive_benchmark():
    print(f"\n[🚀] BENCHMARK ADAPTATIF B200 (Version Compatible Cluster)")
    
    # Chargement des données
    cols = ['reporterCode', 'partnerCode', 'period', 'primaryValue', 'gdp_o', 'gdp_d', 'dist', 'cmdCode']
    if os.path.exists(PARQUET_PATH):
        df = pd.read_parquet(PARQUET_PATH, columns=cols)
    else:
        # Fallback si parquet absent
        df = pd.read_csv(CSV_PATH, usecols=cols, nrows=20000000).fillna(0)

    all_countries = np.unique(np.concatenate([df['reporterCode'], df['partnerCode']]))
    country_map = {int(c): i for i, c in enumerate(all_countries)}
    
    model_names = ['GAT', 'CfC', 'LTC', 'LSTM']
    metrics = {
        'dates': [], 'actual_price_usd': [],
        'models': {m: {'predictions_usd': [], 'rolling_mae': [], 'retrain_events': []} for m in model_names},
        'drift_metrics': {'labels': [], 'ks': [], 'js': []}
    }

    # Initialisation des modèles
    models = {n: (FullGraphGAT(16, 4) if n == 'GAT' else SeqModel(n, 4)).to(DEVICE) for n in model_names}
    opts = {n: optim.Adam(m.parameters(), lr=LEARNING_RATE) for n, m in models.items()}
    scaler, crit = torch.amp.GradScaler('cuda'), nn.MSELoss()

    # 1. Entraînement Initial (2017-2019)
    train_df = df[df['period'] <= 2019]
    train_data = get_pyg_data(train_df, country_map)
    
    # On crée un Dataset d'arêtes simple
    edge_dataset = TensorDataset(train_data.edge_index.t(), train_data.edge_attr, train_data.y)
    loader = DataLoader(edge_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"\n[🔥] Phase 1 : Entraînement Initial (2017-2019)")
    # On déplace le graphe de contexte sur le GPU une seule fois
    ctx_x = train_data.x.to(DEVICE)
    ctx_ei = train_data.edge_index.to(DEVICE)
    ctx_ea = train_data.edge_attr.to(DEVICE)

    t_start = time.time()
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for b_ei, b_ea, b_y in pbar:
            b_ei, b_ea, b_y = b_ei.to(DEVICE).t(), b_ea.to(DEVICE), b_y.to(DEVICE)
            
            with torch.amp.autocast('cuda'):
                for name, m in models.items():
                    opts[name].zero_grad()
                    if name == 'GAT':
                        p = m(ctx_x, ctx_ei, ctx_ea, b_ei, b_ea)
                    else:
                        p = m(b_ea.unsqueeze(1))
                    
                    l = crit(p, b_y)
                    scaler.scale(l).backward()
                    scaler.step(opts[name])
            scaler.update()
    
    print(f"  > Temps entraînement initial : {time.time() - t_start:.2f}s")

    # 2. Évaluation Adaptative (2020-2023)
    print(f"\n[📈] Phase 2 : Évaluation Adaptative")
    ref_vals = np.log1p(train_df['primaryValue'].values[:100000])
    prev_mae = {n: 100.0 for n in model_names}

    for yr in [2020, 2021, 2022, 2023]:
        yr_df = df[df['period'] == yr]
        if len(yr_df) == 0: continue
        yr_data = get_pyg_data(yr_df, country_map)
        yr_loader = DataLoader(TensorDataset(yr_data.edge_index.t(), yr_data.edge_attr, yr_data.y), 
                               batch_size=BATCH_SIZE, shuffle=False)
        
        # Calcul Drift
        curr_vals = np.log1p(yr_df['primaryValue'].values[:100000])
        ks, _, js = calculate_drift(ref_vals, curr_vals)
        metrics['drift_metrics']['labels'].append(str(yr))
        metrics['drift_metrics']['ks'].append(ks); metrics['drift_metrics']['js'].append(js)
        metrics['dates'].append(str(yr))
        metrics['actual_price_usd'].append(float(np.expm1(curr_vals).mean()))

        for n in model_names:
            errs, preds = [], []
            models[n].eval()
            with torch.no_grad():
                for b_ei, b_ea, b_y in yr_loader:
                    b_ei, b_ea, b_y = b_ei.to(DEVICE).t(), b_ea.to(DEVICE), b_y.to(DEVICE)
                    if n == 'GAT': p = models[n](ctx_x, ctx_ei, ctx_ea, b_ei, b_ea)
                    else: p = models[n](b_ea.unsqueeze(1))
                    errs.append(torch.abs(p - b_y).mean().item())
                    preds.append(torch.expm1(p).mean().item())
            
            mae = np.mean(errs)
            metrics['models'][n]['rolling_mae'].append(float(mae))
            metrics['models'][n]['predictions_usd'].append(float(np.mean(preds)))
            
            # Retrain si dérive
            if mae > THRESHOLD_RETRAIN * prev_mae[n]:
                print(f"    [⚠] Drift détecté pour {n} en {yr} ! Réentraînement...")
                metrics['models'][n]['retrain_events'].append(str(yr))
                models[n].train()
                for _ in range(args.retrain_epochs):
                    for b_ei, b_ea, b_y in yr_loader:
                        b_ei, b_ea, b_y = b_ei.to(DEVICE).t(), b_ea.to(DEVICE), b_y.to(DEVICE)
                        opts[n].zero_grad()
                        with torch.amp.autocast('cuda'):
                            p = models[n](ctx_x, ctx_ei, ctx_ea, b_ei, b_ea) if n == 'GAT' else models[n](b_ea.unsqueeze(1))
                            l = crit(p, b_y)
                            scaler.scale(l).backward()
                            scaler.step(opts[n])
                        scaler.update()
            prev_mae[n] = mae
        print(f"  > Année {yr} terminée. GAT MAE: {prev_mae['GAT']:.4f}")

    # Sauvegarde des résultats
    with open("full_adaptive_metrics.js", "w") as f:
        f.write(f"const b200AdaptiveMetricsData = {json.dumps(metrics, indent=2)};")
    print(f"\n[🏁] Terminé ! Fichier 'full_adaptive_metrics.js' généré.")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    run_adaptive_benchmark()

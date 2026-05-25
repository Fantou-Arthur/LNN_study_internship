import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
from ncps.torch import CfC
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

# --- CONFIGURATION EXPERTE ---
FILE_PATH = r"C:\Users\afant\Downloads\comtrade_sampled_10pct.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
BATCH_SIZE = 512
LR = 0.002

# On utilise maintenant 10 colonnes en entrée
INPUT_COLS = ['primaryValue', 'qty', 'netWgt', 'gdp_o', 'pop_o', 'gdpcap_o', 'gdp_d', 'pop_d', 'gdpcap_d', 'dist']
# Et on prédit un vecteur de 3 valeurs pour 2023
TARGET_COLS = ['primaryValue', 'qty', 'netWgt']

def load_and_preprocess():
    print(f"Chargement des données ({len(INPUT_COLS)} features)...")
    df = pd.read_csv(FILE_PATH)
    
    # Log transformation pour toutes les features numériques
    for col in INPUT_COLS:
        df[col] = np.log1p(df[col].fillna(0))
    
    # Agrégation par paire et année
    df_grouped = df.groupby(['reporterCode', 'partnerCode', 'period'])[INPUT_COLS].mean().reset_index()
    return df_grouped

class ExpertGAT(nn.Module):
    def __init__(self, node_in, edge_in, out_size):
        super().__init__()
        self.conv1 = GATConv(node_in, 64, heads=4, dropout=0.1)
        self.conv2 = GATConv(64 * 4, 32, heads=1, dropout=0.1)
        # Sortie multivariée (taille 3)
        self.fc = nn.Linear(32 * 2 + edge_in, out_size)

    def forward(self, x, edge_index, edge_attr):
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        src, dst = edge_index
        # Concaténation des features des pays et des caractéristiques de l'échange
        combined = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        return self.fc(combined)

class ExpertCfC(nn.Module):
    def __init__(self, input_size, units, out_size):
        super().__init__()
        self.cfc = CfC(input_size, units, batch_first=True)
        # Sortie multivariée (taille 3)
        self.fc = nn.Linear(units, out_size)

    def forward(self, x):
        out, _ = self.cfc(x)
        return self.fc(out[:, -1, :])

def train():
    df = load_and_preprocess()
    
    # 1. Pivot pour la cible 2023 (Vecteur de 3 valeurs)
    target_df = df[df['period'] == 2023].set_index(['reporterCode', 'partnerCode'])[TARGET_COLS]
    y_target = torch.tensor(target_df.values, dtype=torch.float).to(DEVICE)
    
    # 2. Préparation GAT
    all_countries = pd.unique(df[['reporterCode', 'partnerCode']].values.ravel())
    c_map = {code: i for i, code in enumerate(all_countries)}
    num_nodes = len(all_countries)
    
    # Edges basés sur les paires de 2023
    edge_index = torch.tensor([
        [c_map[r] for r, p in target_df.index],
        [c_map[p] for r, p in target_df.index]
    ], dtype=torch.long).to(DEVICE)
    
    # Node features (GDP, Pop, GDPcap de l'exportateur en 2022)
    node_feats = np.zeros((num_nodes, 3))
    df_2022_nodes = df[df['period'] == 2022].groupby('reporterCode')[['gdp_o', 'pop_o', 'gdpcap_o']].mean()
    for code, i in c_map.items():
        if code in df_2022_nodes.index:
            node_feats[i] = df_2022_nodes.loc[code].values
    x_nodes = torch.tensor(node_feats, dtype=torch.float).to(DEVICE)
    
    # Edge features (Distance + Valeurs 2022)
    df_2022_edges = df[df['period'] == 2022].set_index(['reporterCode', 'partnerCode'])
    edge_attr_cols = ['dist', 'primaryValue', 'qty', 'netWgt']
    edge_attr = df_2022_edges[edge_attr_cols].reindex(target_df.index, fill_value=0)
    edge_tensor = torch.tensor(edge_attr.values, dtype=torch.float).to(DEVICE)
    
    # 3. Préparation CfC (Cube : Paires x 6 ans x 10 features)
    years = [2017, 2018, 2019, 2020, 2021, 2022]
    cfc_input = []
    for yr in years:
        yr_data = df[df['period'] == yr].set_index(['reporterCode', 'partnerCode'])[INPUT_COLS]
        cfc_input.append(yr_data.reindex(target_df.index, fill_value=0).values)
    
    # Shape : (Batch, 6, 10)
    x_cfc = torch.tensor(np.array(cfc_input), dtype=torch.float).transpose(0, 1).to(DEVICE)
    
    # --- MODÈLES ---
    print(f"Modèles : GAT (Nodes: {num_nodes}) | CfC (Units: 128)")
    gat = ExpertGAT(x_nodes.shape[1], edge_tensor.shape[1], len(TARGET_COLS)).to(DEVICE)
    cfc = ExpertCfC(len(INPUT_COLS), 128, len(TARGET_COLS)).to(DEVICE)
    
    opt_gat = optim.Adam(gat.parameters(), lr=LR)
    opt_cfc = optim.Adam(cfc.parameters(), lr=LR)
    crit = nn.MSELoss()
    
    g_hist, c_hist = [], []
    
    print("\nLancement de l'entraînement Expert...")
    for epoch in range(EPOCHS):
        gat.train(); cfc.train()
        
        # GAT
        opt_gat.zero_grad()
        p_gat = gat(x_nodes, edge_index, edge_tensor)
        loss_g = crit(p_gat, y_target)
        loss_g.backward(); opt_gat.step()
        
        # CfC
        opt_cfc.zero_grad()
        p_cfc = cfc(x_cfc)
        loss_c = crit(p_cfc, y_target)
        loss_c.backward(); opt_cfc.step()
        
        g_hist.append(loss_g.item())
        c_hist.append(loss_c.item())
        
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Époque {epoch+1:03d}/{EPOCHS} | MSE GAT: {loss_g.item():.4f} | MSE CfC: {loss_c.item():.4f}")

    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(g_hist, label="GAT Expert (Spatial + Node/Edge Feats)")
    plt.plot(c_hist, label="CfC Expert (Full Time Series 10 features)")
    plt.yscale('log'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.title("MSE Prédiction 2023 : Multi-output (Value, Qty, NetWgt)")
    plt.savefig('comparison_mse_expert.png')
    print("\nTerminé ! Graphique sauvegardé : 'comparison_mse_expert.png'")

if __name__ == "__main__":
    train()

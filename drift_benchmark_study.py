import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv
from ncps.torch import CfC
import os

# --- CONFIGURATION PROTOCOLE PAPIER ---
FILE_PATH = r"C:\Users\afant\Downloads\comtrade_sampled_10pct.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_YEARS = [2017, 2018, 2019]
TEST_YEARS = [2020, 2021, 2022, 2023]
INPUT_COLS = ['primaryValue', 'qty', 'netWgt', 'gdp_o', 'pop_o', 'gdpcap_o', 'gdp_d', 'pop_d', 'gdpcap_d', 'dist']
TARGET_COLS = ['primaryValue']

# --- UTILS DRIFT ---
def calculate_drift(ref, curr):
    ks_stat, _ = ks_2samp(ref, curr)
    
    def get_psi(expected, actual, buckets=10):
        try:
            quantiles = np.linspace(0, 1, buckets + 1)
            bins = np.unique(np.quantile(expected, quantiles))
            expected_percents = np.histogram(expected, bins=bins)[0] / len(expected)
            actual_percents = np.histogram(actual, bins=bins)[0] / len(actual)
            expected_percents = np.clip(expected_percents, 1e-6, None)
            actual_percents = np.clip(actual_percents, 1e-6, None)
            return np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        except: return 0.0

    psi_stat = get_psi(ref, curr)
    hist_ref, _ = np.histogram(ref, bins=20, density=True)
    hist_curr, _ = np.histogram(curr, bins=20, density=True)
    js_stat = jensenshannon(hist_ref + 1e-6, hist_curr + 1e-6)
    return ks_stat, psi_stat, js_stat

# --- MODELS ---
class GAT(nn.Module):
    def __init__(self, node_in, edge_in, out_size=1):
        super().__init__()
        self.conv1 = GATConv(node_in, 32, heads=4)
        self.conv2 = GATConv(32*4, 16)
        self.fc = nn.Linear(16*2 + edge_in, out_size)
    def forward(self, x, edge_index, edge_attr):
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        src, dst = edge_index
        combined = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        return self.fc(combined)

class LiquidCfC(nn.Module):
    def __init__(self, in_size, units=64, out_size=1):
        super().__init__()
        self.cfc = CfC(in_size, units, batch_first=True)
        self.fc = nn.Linear(units, out_size)
    def forward(self, x):
        out, _ = self.cfc(x)
        return self.fc(out[:, -1, :])

def prepare_year_data(df, target_year, input_years):
    target_df = df[df['period'] == target_year].set_index(['reporterCode', 'partnerCode'])[TARGET_COLS]
    y = torch.tensor(target_df.values, dtype=torch.float).to(DEVICE)
    
    prev_year = target_year - 1
    df_prev = df[df['period'] == prev_year]
    all_c = pd.unique(df[['reporterCode', 'partnerCode']].values.ravel())
    c_map = {code: i for i, code in enumerate(all_c)}
    
    edge_idx = torch.tensor([[c_map[r] for r, p in target_df.index],
                             [c_map[p] for r, p in target_df.index]], dtype=torch.long).to(DEVICE)
    
    node_feats = np.zeros((len(all_c), 3))
    df_nodes = df_prev.groupby('reporterCode')[['gdp_o', 'pop_o', 'gdpcap_o']].mean()
    for code, i in c_map.items():
        if code in df_nodes.index: node_feats[i] = df_nodes.loc[code].values
    x_nodes = torch.tensor(node_feats, dtype=torch.float).to(DEVICE)
    
    df_edges = df_prev.set_index(['reporterCode', 'partnerCode'])
    edge_attr = df_edges[['dist', 'primaryValue']].reindex(target_df.index, fill_value=0)
    edge_tensor = torch.tensor(edge_attr.values, dtype=torch.float).to(DEVICE)
    
    cfc_input = []
    for yr in input_years:
        yr_data = df[df['period'] == yr].set_index(['reporterCode', 'partnerCode'])[INPUT_COLS]
        cfc_input.append(yr_data.reindex(target_df.index, fill_value=0).values)
    x_cfc = torch.tensor(np.array(cfc_input), dtype=torch.float).transpose(0, 1).to(DEVICE)
    
    return x_nodes, edge_idx, edge_tensor, x_cfc, y

def main():
    print("Chargement des données...")
    df_raw = pd.read_csv(FILE_PATH)
    
    print("Agrégation par paires de pays (Fusion des codes produits HS-6)...")
    df = df_raw.groupby(['reporterCode', 'partnerCode', 'period']).agg({
        'primaryValue': 'sum', 'qty': 'sum', 'netWgt': 'sum',
        'gdp_o': 'mean', 'pop_o': 'mean', 'gdpcap_o': 'mean',
        'gdp_d': 'mean', 'pop_d': 'mean', 'gdpcap_d': 'mean', 'dist': 'first'
    }).reset_index()
    
    for col in INPUT_COLS: df[col] = np.log1p(df[col].fillna(0))
    
    print("\n[🎬] Phase 1 : Entraînement Initial (2019)")
    x_n, e_idx, e_a, x_c, y = prepare_year_data(df, 2019, [2017, 2018])
    
    gat = GAT(x_n.shape[1], e_a.shape[1]).to(DEVICE)
    cfc = LiquidCfC(x_c.shape[2], 128).to(DEVICE)
    
    opt_g = optim.Adam(gat.parameters(), lr=0.005)
    opt_c = optim.Adam(cfc.parameters(), lr=0.005)
    crit = nn.MSELoss()
    
    for epoch in range(50):
        gat.train(); opt_g.zero_grad(); crit(gat(x_n, e_idx, e_a), y).backward(); opt_g.step()
        cfc.train(); opt_c.zero_grad(); crit(cfc(x_c), y).backward(); opt_c.step()
        if (epoch+1) % 10 == 0: print(f"  Warmup Epoch {epoch+1}/50 Done")
    
    ref_distribution = df[df['period'].isin(BASE_YEARS)]['primaryValue'].values
    bench_results = []
    
    for yr in TEST_YEARS:
        print(f"\n>>> Évaluation de l'année {yr}")
        history_years = list(range(2017, yr))
        x_n, e_idx, e_a, x_c, y = prepare_year_data(df, yr, history_years)
        
        curr_dist = df[df['period'] == yr]['primaryValue'].values
        ks, psi, js = calculate_drift(ref_distribution, curr_dist)
        alerts = sum([ks > 0.1, psi > 0.25, js > 0.1])
        retrain = alerts >= 2
        
        gat.eval(); cfc.eval()
        with torch.no_grad():
            mse_g = crit(gat(x_n, e_idx, e_a), y).item()
            mse_c = crit(cfc(x_c), y).item()
        
        print(f"Drift: KS={ks:.3f} | PSI={psi:.3f} | JS={js:.3f} | Alertes={alerts}")
        print(f"Performance: MSE GAT={mse_g:.4f} | MSE CfC={mse_c:.4f}")
        
        bench_results.append({
            "Year": yr, "KS": ks, "PSI": psi, "JS": js, 
            "MSE_GAT": mse_g, "MSE_CfC": mse_c, "Retrain": retrain
        })
        
        if retrain:
            print("--- ALERTE DRIFT : Ré-entraînement adaptatif ---")
            for _ in range(20):
                gat.train(); opt_g.zero_grad(); crit(gat(x_n, e_idx, e_a), y).backward(); opt_g.step()
                cfc.train(); opt_c.zero_grad(); crit(cfc(x_c), y).backward(); opt_c.step()

    res_df = pd.DataFrame(bench_results)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(res_df['Year'], res_df['KS'], label='KS Stat', marker='o', color='orange')
    plt.axhline(y=0.1, color='r', linestyle='--', label='Warning')
    plt.title("Intensité du Drift (Papier 2026)")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(res_df['Year'], res_df['MSE_GAT'], label='GAT MSE', marker='s')
    plt.plot(res_df['Year'], res_df['MSE_CfC'], label='CfC MSE', marker='d')
    plt.title("Robustesse GAT vs CfC")
    plt.legend()
    plt.tight_layout()
    plt.savefig('drift_results_corrected.png')
    print("\n[🏁] Étude terminée. Graphique : 'drift_results_corrected.png'")

if __name__ == "__main__":
    main()

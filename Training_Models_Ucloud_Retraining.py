import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
parser = argparse.ArgumentParser(description="Benchmark Résilience B200")
parser.add_argument('--epochs', '-e', type=int, default=80, help="Nombre d'époques pour l'entraînement initial (réduit de 100 à 80)")
parser.add_argument('--retrain-epochs', '-re', type=int, default=20, help="Nombre d'époques pour le réentraînement (fine-tuning)")
args = parser.parse_known_args()[0]

# --- CONFIGURATION UCLOUD / B200 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "trade_data.parquet")

if not os.path.exists(DATA_PATH):
    DATA_PATH = r"C:\Users\afant\Downloads\comtradeExports_updatedH5[240924]-wb.csv\comtradeExports_updatedH5[240924]-wb.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = args.epochs 
RETRAIN_EPOCHS = args.retrain_epochs
BATCH_SIZE = 10000 
TRAIN_YEARS = [2017, 2018, 2019] # Critique 1 : Années jusqu'à 2019
TEST_YEARS = [2020, 2021, 2022, 2023] # Test sur le choc COVID et suivant

os.makedirs(os.path.join(SCRIPT_DIR, "results"), exist_ok=True)
os.makedirs(os.path.join(SCRIPT_DIR, "models"), exist_ok=True)

class B200GAT(nn.Module):
    def __init__(self, node_dim=3, edge_dim=2, hidden=32):
        super().__init__()
        self.conv1 = GATConv(node_dim, hidden, heads=4, edge_dim=edge_dim, concat=True)
        self.conv2 = GATConv(hidden * 4, hidden, heads=4, edge_dim=edge_dim, concat=True)
        self.fc = nn.Linear(hidden * 4 * 2 + edge_dim, 1) 

    def compute_node_emb(self, x, macro_edge_index, macro_edge_attr):
        # Étape 1 : Le GAT apprend sur le squelette global des pays (Macro-Graphe)
        h1 = torch.relu(self.conv1(x, macro_edge_index, macro_edge_attr))
        return torch.relu(self.conv2(h1, macro_edge_index, macro_edge_attr))

    def predict_edge(self, src_emb, dst_emb, edge_attr):
        # Étape 2 : Prédiction sur les flux détaillés (Micro-Graphe)
        combined = torch.cat([src_emb, dst_emb, edge_attr], dim=-1)
        return self.fc(combined)

class B200CfC(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.cfc = CfC(in_dim, 128, batch_first=True)
        self.fc = nn.Linear(128, 1)
    def forward(self, x, hx=None):
        out, hx_new = self.cfc(x, hx)
        return self.fc(out[:, -1, :]), hx_new

class B200LTC(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        wiring = AutoNCP(64, 1)
        self.ltc = LTC(in_dim, wiring, batch_first=True)
    def forward(self, x, hx=None):
        out, hx_new = self.ltc(x, hx)
        return out[:, -1, :], hx_new

class B200RNN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.rnn = nn.RNN(in_dim, 128, batch_first=True)
        self.fc = nn.Linear(128, 1)
    def forward(self, x, hx=None):
        out, hx_new = self.rnn(x, hx)
        return self.fc(out[:, -1, :]), hx_new

class B200LSTM(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, 128, batch_first=True)
        self.fc = nn.Linear(128, 1)
    def forward(self, x, hx=None):
        out, hx_new = self.lstm(x, hx)
        return self.fc(out[:, -1, :]), hx_new

class B200GRU(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.gru = nn.GRU(in_dim, 128, batch_first=True)
        self.fc = nn.Linear(128, 1)
    def forward(self, x, hx=None):
        out, hx_new = self.gru(x, hx)
        return self.fc(out[:, -1, :]), hx_new

def detach_state(state):
    if state is None: return None
    if isinstance(state, tuple):
        return tuple([h.detach() if isinstance(h, torch.Tensor) else h for h in state])
    return state.detach() if isinstance(state, torch.Tensor) else state

def run_scientific_benchmark():
    print(f"[🚀] Benchmark Résilience B200 (Batching actif | Batch: {BATCH_SIZE})")
    
    # --- CHARGEMENT OPTIMISÉ ---
    print(f"    📂 Chargement des données (HS-6)...", flush=True)
    cols = ['reporterCode', 'partnerCode', 'period', 'primaryValue', 'gdp_o', 'pop_o', 'gdp_d', 'pop_d', 'dist', 'qty', 'cmdCode']
    
    if DATA_PATH.endswith('.csv'):
        df = pd.read_csv(DATA_PATH, nrows=200000, usecols=cols)
    else:
        df = pd.read_parquet(DATA_PATH, columns=cols)
        
    df['period'] = pd.to_numeric(df['period']).astype(np.int32)
    
    # --- FILTRAGE DES ANNÉES ---
    years_needed = set(TRAIN_YEARS) | set(TEST_YEARS) | {y+1 for y in TRAIN_YEARS} | {y+1 for y in TEST_YEARS}
    df = df[df['period'].isin(years_needed)]
    print(f"    ✅ Dataset chargé : {len(df)} lignes.", flush=True)
    
    df = df.fillna(0)
    
    seq_features = ['gdp_o', 'pop_o', 'gdp_d', 'pop_d', 'dist', 'qty'] 
    edge_features = ['dist', 'qty'] 
    
    # Optimisation mémoire immédiate
    float_cols = seq_features + ['primaryValue']
    df[float_cols] = df[float_cols].astype(np.float32)
    df['reporterCode'] = pd.to_numeric(df['reporterCode'], errors='coerce').fillna(0).astype(np.int32)
    df['partnerCode'] = pd.to_numeric(df['partnerCode'], errors='coerce').fillna(0).astype(np.int32)
    df['cmdCode'] = pd.to_numeric(df['cmdCode'], errors='coerce').fillna(0).astype(np.int32)
    
    model_names = ['GAT', 'CfC', 'LTC', 'RNN', 'LSTM', 'GRU']
    metrics = {
        'dates': [], 'actual_price_usd': [],
        'models': {m: {'predictions_usd': [], 'rolling_mae': [], 'growth_rate': [], 'retrain_events': [], 'time_cost_seconds': 0.0, 'retrain_time_seconds': 0.0} for m in model_names},
        'drift_metrics': {'labels': [], 'ks': [], 'psi': [], 'js': []}
    }

    def get_lagged_data(df, yr):
        # Lagged merge: on joint sur Pays O, Pays D ET Code Produit
        df_f = df[df['period'] == yr].copy()
        df_t = df[df['period'] == yr + 1][['reporterCode', 'partnerCode', 'cmdCode', 'primaryValue']]
        merged = pd.merge(df_f, df_t, on=['reporterCode', 'partnerCode', 'cmdCode'], how='left', suffixes=('', '_target'))
        merged['primaryValue_target'] = merged['primaryValue_target'].fillna(0)
        return merged

    models = {name: None for name in model_names}
    models['GAT'] = B200GAT(node_dim=3, edge_dim=len(edge_features)).to(DEVICE)
    models['CfC'] = B200CfC(len(seq_features)).to(DEVICE)
    models['LTC'] = B200LTC(len(seq_features)).to(DEVICE)
    models['RNN'] = B200RNN(len(seq_features)).to(DEVICE)
    models['LSTM'] = B200LSTM(len(seq_features)).to(DEVICE)
    models['GRU'] = B200GRU(len(seq_features)).to(DEVICE)
    
    opts = {name: optim.Adam(m.parameters(), lr=0.001) for name, m in models.items() if m is not None}
    opts['GAT'] = optim.Adam(models['GAT'].parameters(), lr=0.01)
    opts['LTC'] = optim.Adam(models['LTC'].parameters(), lr=0.005) 
    crit = nn.MSELoss()

    # --- SCHEDULERS ET EARLY STOPPING ---
    schedulers = {name: ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, verbose=True) for name, opt in opts.items()}
    best_losses = {name: float('inf') for name in model_names}
    patience_counters = {name: 0 for name in model_names}
    early_stop_patience = 15
    active_models = set(model_names)

    # --- PHASE 1 : ENTRAINEMENT ---
    print(f"\n[1/2] Entraînement historique ({EPOCHS} époques)...")
    final_states = {name: None for name in ['CfC', 'LTC', 'RNN', 'LSTM', 'GRU']}
    
    try:
        # Préparation des données une seule fois pour toutes les époques
        train_data_blocks = []
        print(f"    🔍 Préparation des données (Lagged Merge)...", flush=True)
        for yr in tqdm(TRAIN_YEARS, desc="Préparation Années"):
            d = get_lagged_data(df, yr)
            if len(d) > 0:
                # On pré-calcule les tenseurs pour gagner du temps
                x_seq = torch.tensor(np.log1p(d[seq_features].values), dtype=torch.float)
                y = torch.tensor(np.log1p(d['primaryValue_target'].values), dtype=torch.float).view(-1, 1)
                reporters = torch.tensor(d['reporterCode'].values % 1000, dtype=torch.long)
                partners = torch.tensor(d['partnerCode'].values % 1000, dtype=torch.long)
                edge_index = torch.stack([reporters, partners])
                edge_attr = torch.tensor(np.log1p(d[edge_features].values), dtype=torch.float)
                
                # Node Features GAT (GDP, POP, Valeur Commerciale Globale)
                reporter_trade = d.groupby('reporterCode')['primaryValue'].sum().to_dict()
                partner_trade = d.groupby('partnerCode')['primaryValue'].sum().to_dict()
                rep_trade_vals = d['reporterCode'].map(reporter_trade).fillna(0).values
                part_trade_vals = d['partnerCode'].map(partner_trade).fillna(0).values
                
                node_features = torch.zeros(1001, 3)
                node_features[reporters, 0] = torch.tensor(np.log1p(d['gdp_o'].values), dtype=torch.float)
                node_features[reporters, 1] = torch.tensor(np.log1p(d['pop_o'].values), dtype=torch.float)
                node_features[reporters, 2] = torch.tensor(np.log1p(rep_trade_vals), dtype=torch.float)
                
                node_features[partners, 0] = torch.tensor(np.log1p(d['gdp_d'].values), dtype=torch.float)
                node_features[partners, 1] = torch.tensor(np.log1p(d['pop_d'].values), dtype=torch.float)
                node_features[partners, 2] = torch.tensor(np.log1p(part_trade_vals), dtype=torch.float)
                
                # --- Création du Macro-Graphe (Agrégé) pour le GAT ---
                df_macro = d.groupby(['reporterCode', 'partnerCode']).agg({'dist': 'mean', 'qty': 'sum'}).reset_index()
                macro_reporters = torch.tensor(df_macro['reporterCode'].values % 1000, dtype=torch.long)
                macro_partners = torch.tensor(df_macro['partnerCode'].values % 1000, dtype=torch.long)
                macro_edge_index = torch.stack([macro_reporters, macro_partners])
                macro_edge_attr = torch.tensor(np.log1p(df_macro[['dist', 'qty']].values), dtype=torch.float)
                
                train_data_blocks.append({
                    'yr': yr, 'x_seq': x_seq, 'y': y, 
                    'edge_index': edge_index, 'edge_attr': edge_attr, # Micro-Graphe (Produits)
                    'macro_edge_index': macro_edge_index, 'macro_edge_attr': macro_edge_attr, # Macro-Graphe (Pays)
                    'x_gat': node_features, 'len': len(d),
                    'reporters': reporters, 'partners': partners
                })

        # --- LIBÉRATION MÉMOIRE CRITIQUE ---
        print(f"    ♻️ Libération de la mémoire (suppression du dataset source)...", flush=True)
        del df # On supprime les 13 Go de RAM inutiles
        import gc
        gc.collect()

        print(f"\n🚀 Démarrage de l'entraînement ({EPOCHS} époques)...", flush=True)
        training_loss_history = {m: [] for m in model_names}
        for epoch in range(EPOCHS):
            states = {name: None for name in ['CfC', 'LTC', 'RNN', 'LSTM', 'GRU']}
            epoch_loss_sum = {m: 0.0 for m in model_names}
            epoch_batches = 0
            
            # Barre de progression pour les batchs de l'époque
            total_rows = sum([b['len'] for b in train_data_blocks])
            pbar = tqdm(total=total_rows, desc=f"Époque {epoch+1}/{EPOCHS}", unit="rows")
            
            for block in train_data_blocks:
                x_seq_gpu = block['x_seq'].to(DEVICE)
                y_gpu = block['y'].to(DEVICE)
                reporters_all = block['reporters'].to(DEVICE)
                partners_all = block['partners'].to(DEVICE)
                edge_attr_all = block['edge_attr'].to(DEVICE)
                
                macro_ei_gpu = block['macro_edge_index'].to(DEVICE)
                macro_ea_gpu = block['macro_edge_attr'].to(DEVICE)
                x_gat_gpu = block['x_gat'].to(DEVICE)

                for i in range(0, block['len'], BATCH_SIZE):
                    end = i + BATCH_SIZE
                    
                    # --- Entraînement du GAT ---
                    if 'GAT' in active_models:
                        t0_gat = time.time()
                        models['GAT'].train()
                        opts['GAT'].zero_grad()
                        
                        # 1. Calcul des embeddings pays
                        node_emb = models['GAT'].compute_node_emb(x_gat_gpu, macro_ei_gpu, macro_ea_gpu)
                        
                        # 2. Prédiction sur le batch actuel
                        batch_rep = reporters_all[i:end]
                        batch_part = partners_all[i:end]
                        batch_ea = edge_attr_all[i:end]
                        
                        pred_g = models['GAT'].predict_edge(node_emb[batch_rep], node_emb[batch_part], batch_ea)
                        loss_g = crit(pred_g, y_gpu[i:end])
                        
                        # 3. Rétropropagation
                        loss_g.backward()
                        opts['GAT'].step()
                        metrics['models']['GAT']['time_cost_seconds'] += (time.time() - t0_gat)
                        epoch_loss_sum['GAT'] += loss_g.item()


                    # --- Entraînement des Séquentiels ---
                    for m_name in ['CfC', 'LTC', 'RNN', 'LSTM', 'GRU']:
                        if m_name not in active_models: continue
                        t0_seq = time.time()
                        models[m_name].train()
                        curr_x = x_seq_gpu[i:end].unsqueeze(1)
                        bs_now = curr_x.size(0)
                        
                        if states[m_name] is not None:
                            try:
                                if isinstance(states[m_name], tuple):
                                    if states[m_name][0].size(1) != bs_now: states[m_name] = None
                                elif states[m_name].shape[-2] != bs_now and states[m_name].shape[0] != bs_now:
                                    states[m_name] = None
                            except: states[m_name] = None

                        pred, states[m_name] = models[m_name](curr_x, states[m_name])
                        states[m_name] = detach_state(states[m_name])
                        loss = crit(pred, y_gpu[i:end])
                        loss.backward(); opts[m_name].step(); opts[m_name].zero_grad()
                        metrics['models'][m_name]['time_cost_seconds'] += (time.time() - t0_seq)
                        epoch_loss_sum[m_name] += loss.item()

                    epoch_batches += 1
                    pbar.update(BATCH_SIZE)
            
            pbar.close()
            final_states = states

            for m in model_names:
                avg_loss = epoch_loss_sum[m] / epoch_batches if epoch_batches > 0 else (training_loss_history[m][-1] if epoch > 0 else 0)
                training_loss_history[m].append(avg_loss)
                
                if m in active_models:
                    # Mise à jour du scheduler
                    schedulers[m].step(avg_loss)
                    
                    # Logique Early Stopping
                    if avg_loss < best_losses[m] * 0.999: # Amélioration d'au moins 0.1%
                        best_losses[m] = avg_loss
                        patience_counters[m] = 0
                    else:
                        patience_counters[m] += 1
                        
                    if patience_counters[m] >= early_stop_patience:
                        print(f"    [🛑] {m} a convergé (Early Stopping à l'époque {epoch+1})")
                        active_models.remove(m)

            if not active_models:
                print(f"    [🏁] Tous les modèles ont convergé. Fin de l'entraînement.")
                break

        with open(os.path.join(SCRIPT_DIR, "results", "training_loss_history.json"), "w") as f:
            json.dump(training_loss_history, f, indent=2)

        for m_name, m in models.items():
            torch.save(m.state_dict(), os.path.join(SCRIPT_DIR, f"models/{m_name.lower()}_trained_2019.pt"))

        # --- PHASE 2 : EVALUATION FIXE ---
        print(f"\n[2/2] Évaluation de la résilience...")
        for m in models.values(): m.eval()
        states = final_states
        
        # --- RECHARGEMENT DES DONNÉES (TEST) ---
        # Le dataset a été supprimé de la RAM en Phase 1 pour éviter l'OOM. On le recharge.
        print(f"    📂 Rechargement des données pour l'évaluation...", flush=True)
        if DATA_PATH.endswith('.csv'):
            df = pd.read_csv(DATA_PATH, nrows=200000, usecols=cols)
        else:
            df = pd.read_parquet(DATA_PATH, columns=cols)
            
        df['period'] = pd.to_numeric(df['period']).astype(np.int32)
        years_needed_test = set(TEST_YEARS) | {y+1 for y in TEST_YEARS} | {2019}
        df = df[df['period'].isin(years_needed_test)]
        df = df.fillna(0)
        
        float_cols = seq_features + ['primaryValue']
        df[float_cols] = df[float_cols].astype(np.float32)
        df['reporterCode'] = pd.to_numeric(df['reporterCode'], errors='coerce').fillna(0).astype(np.int32)
        df['partnerCode'] = pd.to_numeric(df['partnerCode'], errors='coerce').fillna(0).astype(np.int32)
        df['cmdCode'] = pd.to_numeric(df['cmdCode'], errors='coerce').fillna(0).astype(np.int32)
        
        ref_data = df[df['period'] == 2019]
        ref_dist = np.log1p(ref_data['primaryValue'].sample(n=min(200000, len(ref_data)), random_state=42).values) if len(ref_data) > 0 else np.random.randn(100)
        
        EVAL_YEARS = [2019, 2020, 2021, 2022] # Prédictions pour 2020 à 2023
        prev_mae = {m: None for m in model_names}
        for yr in EVAL_YEARS:
            data_yr = get_lagged_data(df, yr)
            if len(data_yr) == 0: continue
            
            x_seq = torch.tensor(np.log1p(data_yr[seq_features].values), dtype=torch.float).to(DEVICE)
            y_val = np.log1p(data_yr['primaryValue_target'].values)
            y_tensor = torch.tensor(y_val, dtype=torch.float).view(-1, 1).to(DEVICE)
            reporters = torch.tensor(data_yr['reporterCode'].values % 1000, dtype=torch.long).to(DEVICE)
            partners = torch.tensor(data_yr['partnerCode'].values % 1000, dtype=torch.long).to(DEVICE)
            edge_index = torch.stack([reporters, partners]).to(DEVICE)
            edge_attr = torch.tensor(np.log1p(data_yr[edge_features].values), dtype=torch.float).to(DEVICE)
            
            # Construction des Node Features pour le GAT (GDP, POP, Valeur Commerciale Globale)
            reporter_trade = data_yr.groupby('reporterCode')['primaryValue'].sum().to_dict()
            partner_trade = data_yr.groupby('partnerCode')['primaryValue'].sum().to_dict()
            rep_trade_vals = data_yr['reporterCode'].map(reporter_trade).fillna(0).values
            part_trade_vals = data_yr['partnerCode'].map(partner_trade).fillna(0).values
            
            node_features = torch.zeros(1001, 3).to(DEVICE)
            node_features[reporters, 0] = torch.tensor(np.log1p(data_yr['gdp_o'].values), dtype=torch.float).to(DEVICE)
            node_features[reporters, 1] = torch.tensor(np.log1p(data_yr['pop_o'].values), dtype=torch.float).to(DEVICE)
            node_features[reporters, 2] = torch.tensor(np.log1p(rep_trade_vals), dtype=torch.float).to(DEVICE)
            
            node_features[partners, 0] = torch.tensor(np.log1p(data_yr['gdp_d'].values), dtype=torch.float).to(DEVICE)
            node_features[partners, 1] = torch.tensor(np.log1p(data_yr['pop_d'].values), dtype=torch.float).to(DEVICE)
            node_features[partners, 2] = torch.tensor(np.log1p(part_trade_vals), dtype=torch.float).to(DEVICE)
            x_gat = node_features
            
            # --- Création du Macro-Graphe pour l'évaluation GAT ---
            df_macro_test = data_yr.groupby(['reporterCode', 'partnerCode']).agg({'dist': 'mean', 'qty': 'sum'}).reset_index()
            macro_rep_test = torch.tensor(df_macro_test['reporterCode'].values % 1000, dtype=torch.long)
            macro_part_test = torch.tensor(df_macro_test['partnerCode'].values % 1000, dtype=torch.long)
            macro_ei_test = torch.stack([macro_rep_test, macro_part_test]).to(DEVICE)
            macro_ea_test = torch.tensor(np.log1p(df_macro_test[['dist', 'qty']].values), dtype=torch.float).to(DEVICE)

            # --- Calcul du Drift (Observabilité) ---
            current_dist = np.log1p(data_yr['primaryValue'].sample(n=min(200000, len(data_yr)), random_state=42).values)
            ks_stat, _ = ks_2samp(ref_dist, current_dist)
            
            # Calcul PSI et JS (Histogrammes à 10 bacs basés sur les quantiles de référence)
            try:
                breakpoints = np.percentile(ref_dist, np.arange(0, 11) * 10)
                # S'assurer que les bins sont strictement croissants (problème des quantiles avec beaucoup de 0)
                breakpoints = np.unique(breakpoints)
                if len(breakpoints) < 2:
                    breakpoints = 10 # Fallback simple
                p, _ = np.histogram(ref_dist, bins=breakpoints)
                q, _ = np.histogram(current_dist, bins=breakpoints)
                p = np.where(p == 0, 0.0001, p) / len(ref_dist)
                q = np.where(q == 0, 0.0001, q) / len(current_dist)
                psi_val = np.sum((q - p) * np.log(q / p))
                js_val = jensenshannon(p, q)
            except:
                psi_val, js_val = 0.0, 0.0

            metrics['drift_metrics']['labels'].append(str(yr))
            metrics['drift_metrics']['ks'].append(float(ks_stat))
            metrics['drift_metrics']['psi'].append(float(psi_val))
            metrics['drift_metrics']['js'].append(float(js_val))
            print(f"  [Drift] Année {yr} - KS: {ks_stat:.4f} | PSI: {psi_val:.4f} | JS: {js_val:.4f}")

            # Inférence par Batch
            all_preds = {m: [] for m in model_names}
            with torch.no_grad():
                # 1. Calcul des embeddings pays (Macro-Graphe de l'année de test)
                node_emb = models['GAT'].compute_node_emb(x_gat, macro_ei_test, macro_ea_test)
                
                for i in range(0, len(data_yr), BATCH_SIZE):
                    end = i + BATCH_SIZE
                    
                    # 2. Prédiction sur le Micro-Graphe
                    batch_rep = reporters[i:end]
                    batch_part = partners[i:end]
                    batch_ea = edge_attr[i:end]
                    all_preds['GAT'].append(models['GAT'].predict_edge(node_emb[batch_rep], node_emb[batch_part], batch_ea))

                    for m_name in ['CfC', 'LTC', 'RNN', 'LSTM', 'GRU']:
                        # Vérification taille batch pour l'évaluation
                        curr_x_eval = x_seq[i:end].unsqueeze(1)
                        bs_eval = curr_x_eval.size(0)
                        if states[m_name] is not None:
                            try:
                                if isinstance(states[m_name], tuple):
                                    if states[m_name][0].size(1) != bs_eval: states[m_name] = None
                                elif states[m_name].shape[-2] != bs_eval and states[m_name].shape[0] != bs_eval:
                                    states[m_name] = None
                            except:
                                states[m_name] = None
                                
                        p, states[m_name] = models[m_name](curr_x_eval, states[m_name])
                        all_preds[m_name].append(p)

            metrics['dates'].append(str(yr + 1)) # On a prédit yr+1
            metrics['actual_price_usd'].append(float(np.mean(data_yr['primaryValue_target'])))
            
            for m_name in model_names:
                y_pred = torch.cat(all_preds[m_name])
                mae = torch.abs(y_pred - y_tensor).mean().item()
                metrics['models'][m_name]['predictions_usd'].append(float(torch.expm1(y_pred).mean().item()))
                metrics['models'][m_name]['rolling_mae'].append(float(mae))
                
                if prev_mae[m_name] is not None and mae > 1.10 * prev_mae[m_name]:
                    print(f"    [!] Drift détecté pour {m_name} en {yr} (MAE: {prev_mae[m_name]:.4f} -> {mae:.4f}). Réentraînement...")
                    metrics['models'][m_name]['retrain_events'].append(str(yr))
                    
                    retrain_years = [y for y in [yr-2, yr-1, yr] if y >= TRAIN_YEARS[0]]
                    retrain_blocks = []
                    for ry in retrain_years:
                        d_ry = get_lagged_data(df, ry)
                        if len(d_ry) > 0:
                            x_seq_ry = torch.tensor(np.log1p(d_ry[seq_features].values), dtype=torch.float)
                            y_ry = torch.tensor(np.log1p(d_ry['primaryValue_target'].values), dtype=torch.float).view(-1, 1)
                            reporters_ry = torch.tensor(d_ry['reporterCode'].values % 1000, dtype=torch.long)
                            partners_ry = torch.tensor(d_ry['partnerCode'].values % 1000, dtype=torch.long)
                            edge_attr_ry = torch.tensor(np.log1p(d_ry[edge_features].values), dtype=torch.float)
                            
                            rep_trade = d_ry.groupby('reporterCode')['primaryValue'].sum().to_dict()
                            part_trade = d_ry.groupby('partnerCode')['primaryValue'].sum().to_dict()
                            rep_vals = d_ry['reporterCode'].map(rep_trade).fillna(0).values
                            part_vals = d_ry['partnerCode'].map(part_trade).fillna(0).values
                            
                            nf = torch.zeros(1001, 3)
                            nf[reporters_ry, 0] = torch.tensor(np.log1p(d_ry['gdp_o'].values), dtype=torch.float)
                            nf[reporters_ry, 1] = torch.tensor(np.log1p(d_ry['pop_o'].values), dtype=torch.float)
                            nf[reporters_ry, 2] = torch.tensor(np.log1p(rep_vals), dtype=torch.float)
                            nf[partners_ry, 0] = torch.tensor(np.log1p(d_ry['gdp_d'].values), dtype=torch.float)
                            nf[partners_ry, 1] = torch.tensor(np.log1p(d_ry['pop_d'].values), dtype=torch.float)
                            nf[partners_ry, 2] = torch.tensor(np.log1p(part_vals), dtype=torch.float)
                            
                            df_macro_ry = d_ry.groupby(['reporterCode', 'partnerCode']).agg({'dist': 'mean', 'qty': 'sum'}).reset_index()
                            mac_rep = torch.tensor(df_macro_ry['reporterCode'].values % 1000, dtype=torch.long)
                            mac_part = torch.tensor(df_macro_ry['partnerCode'].values % 1000, dtype=torch.long)
                            mac_ei = torch.stack([mac_rep, mac_part])
                            mac_ea = torch.tensor(np.log1p(df_macro_ry[['dist', 'qty']].values), dtype=torch.float)
                            
                            retrain_blocks.append({
                                'yr': ry, 'x_seq': x_seq_ry.to(DEVICE), 'y': y_ry.to(DEVICE), 
                                'reporters': reporters_ry.to(DEVICE), 'partners': partners_ry.to(DEVICE),
                                'edge_attr': edge_attr_ry.to(DEVICE),
                                'macro_ei': mac_ei.to(DEVICE), 'macro_ea': mac_ea.to(DEVICE), 'nf': nf.to(DEVICE),
                                'len': len(d_ry)
                            })
                    
                    m = models[m_name]
                    opt = opts[m_name]
                    t0_rt = time.time()
                    m.train()
                    
                    pbar_rt = tqdm(total=RETRAIN_EPOCHS * sum(b['len'] for b in retrain_blocks), desc=f"Retrain {m_name} ({RETRAIN_EPOCHS} ep)", unit="rows", leave=False)
                    for epoch_rt in range(RETRAIN_EPOCHS):
                        rt_state = None
                        for b in retrain_blocks:
                            for i in range(0, b['len'], BATCH_SIZE):
                                end = i + BATCH_SIZE
                                opt.zero_grad()
                                
                                if m_name == 'GAT':
                                    n_emb = m.compute_node_emb(b['nf'], b['macro_ei'], b['macro_ea'])
                                    pred_rt = m.predict_edge(n_emb[b['reporters'][i:end]], n_emb[b['partners'][i:end]], b['edge_attr'][i:end])
                                    loss_rt = crit(pred_rt, b['y'][i:end])
                                else:
                                    curr_x_rt = b['x_seq'][i:end].unsqueeze(1)
                                    bs_rt = curr_x_rt.size(0)
                                    if rt_state is not None:
                                        try:
                                            if isinstance(rt_state, tuple):
                                                if rt_state[0].size(1) != bs_rt: rt_state = None
                                            elif rt_state.shape[-2] != bs_rt and rt_state.shape[0] != bs_rt:
                                                rt_state = None
                                        except: rt_state = None
                                    pred_rt, rt_state = m(curr_x_rt, rt_state)
                                    rt_state = detach_state(rt_state)
                                    loss_rt = crit(pred_rt, b['y'][i:end])
                                
                                loss_rt.backward()
                                opt.step()
                                pbar_rt.update(end - i)
                                
                    pbar_rt.close()
                    metrics['models'][m_name]['retrain_time_seconds'] += (time.time() - t0_rt)
                    m.eval()
                    if m_name != 'GAT':
                        with torch.no_grad():
                            eval_state = None
                            for b in retrain_blocks:
                                for i in range(0, b['len'], BATCH_SIZE):
                                    end = i + BATCH_SIZE
                                    curr_x_ev = b['x_seq'][i:end].unsqueeze(1)
                                    bs_ev = curr_x_ev.size(0)
                                    if eval_state is not None:
                                        try:
                                            if isinstance(eval_state, tuple):
                                                if eval_state[0].size(1) != bs_ev: eval_state = None
                                            elif eval_state.shape[-2] != bs_ev and eval_state.shape[0] != bs_ev:
                                                eval_state = None
                                        except: eval_state = None
                                    _, eval_state = m(curr_x_ev, eval_state)
                            states[m_name] = eval_state

                prev_mae[m_name] = mae

            print(f"  Year {yr} | MAE GAT: {metrics['models']['GAT']['rolling_mae'][-1]:.4f}")

    except Exception as e:
        print(f"\n[❌] CRASH détecté : {str(e)}")
        import traceback
        traceback.print_exc()

    # Finalisation JSON
    for m in model_names:
        metrics['models'][m]['avg_mse'] = float(np.mean(metrics['models'][m]['rolling_mae'])**2)
        growth = np.diff(metrics['models'][m]['rolling_mae'])
        metrics['models'][m]['avg_growth'] = float(np.mean(growth) * 100) if len(growth) > 0 else 0.0

    with open(os.path.join(SCRIPT_DIR, "results/b200_adaptive_metrics.js"), "w") as f:
        f.write(f"const b200AdaptiveMetricsData = {json.dumps(metrics, indent=2)};")

if __name__ == "__main__":
    run_scientific_benchmark()

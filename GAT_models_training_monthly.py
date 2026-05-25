import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP
import time
import json
import os
import argparse
import sys
import traceback
from tqdm import tqdm

# ==========================================
# CONFIGURATION ET PARAMÈTRES
# ==========================================
parser = argparse.ArgumentParser(description="STGNN Monthly Benchmark (GAT + Temporal Models)")
parser.add_argument('--epochs', '-e', type=int, default=20, help="Nombre d'époques")
parser.add_argument('--use-rgat', action='store_true', help="Activer le RGAT (Relational GAT)")
args = parser.parse_known_args()[0]

# IMPORTANT: Flag pour activer/désactiver le GAT Relationnel
# Par défaut, False (utilise le GAT standard plus rapide).
USE_RGAT = args.use_rgat

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "trade_data_monthly.parquet")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CRITIQUE 2 : Optimisation B200 (Saturation GPU) ---
# Le Batch Size a été réduit à 25000. 
# Explication : Le réseau LTC utilise un solveur d'équations différentielles (ODE) qui alloue 
# énormément de mémoire pour chaque arête. Un batch de 250k est physiquement impossible pour un LTC.
BATCH_SIZE = 25000

def generate_periods(start_year, end_year):
    periods = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            periods.append(y * 100 + m)
    return periods

TRAIN_PERIODS = generate_periods(2016, 2019)
TEST_PERIODS = generate_periods(2020, 2023)

os.makedirs(os.path.join(SCRIPT_DIR, "results"), exist_ok=True)
os.makedirs(os.path.join(SCRIPT_DIR, "models"), exist_ok=True)

def next_month(p):
    year = p // 100
    month = p % 100
    if month == 12: return (year + 1) * 100 + 1
    else: return p + 1

# ==========================================
# ARCHITECTURES: GAT & STGNN
# ==========================================

# 1. Le GAT de base (Encodeur Spatial)
class SpatialGAT(nn.Module):
    def __init__(self, node_dim=2, edge_dim=2, hidden=32, use_rgat=False, num_products=5204, prod_emb_dim=16):
        super().__init__()
        self.use_rgat = use_rgat
        self.hidden = hidden
        
        if self.use_rgat:
            # RGAT : On inclut l'embedding du produit dans l'attention spatiale (Edge Attributes)
            self.prod_emb = nn.Embedding(num_products, prod_emb_dim)
            eff_edge_dim = edge_dim + prod_emb_dim
        else:
            eff_edge_dim = edge_dim
            
        self.conv1 = GATConv(node_dim, hidden, heads=4, edge_dim=eff_edge_dim, concat=True)
        self.conv2 = GATConv(hidden * 4, hidden, heads=1, edge_dim=eff_edge_dim, concat=False)

    def forward(self, x, edge_index, edge_attr, product_ids=None):
        if self.use_rgat and product_ids is not None:
            p_emb = self.prod_emb(product_ids)
            edge_attr = torch.cat([edge_attr, p_emb], dim=-1)
            
        h = torch.relu(self.conv1(x, edge_index, edge_attr))
        return self.conv2(h, edge_index, edge_attr)

# 2. Modèle GAT Classique (Baseline)
class ClassicGAT(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        # Le GAT classique prédit juste avec src, dst, et edge_attr
        # Pas d'utilisation de features séquentielles complexes
        self.fc = nn.Linear(encoder.hidden * 2 + 2, 1)

    def forward(self, node_emb, target_ei, target_ea):
        src, dst = target_ei
        combined = torch.cat([node_emb[src], node_emb[dst], target_ea], dim=-1)
        return self.fc(combined)

# 3. Le STGNN Hybrid (GAT + LTC/CfC/RNN)
class STGNN(nn.Module):
    def __init__(self, encoder, temporal_type, in_dim_seq, num_products, prod_emb_dim=16):
        super().__init__()
        self.encoder = encoder # GAT Spatial
        self.temporal_type = temporal_type
        self.prod_emb = nn.Embedding(num_products, prod_emb_dim)
        
        # Le "Super-Vecteur" d'entrée Z = [GAT_src, GAT_dst, Prod_emb, Seq_features]
        z_dim = (encoder.hidden * 2) + prod_emb_dim + in_dim_seq
        
        if temporal_type == 'CfC': 
            self.temporal = CfC(z_dim, 64, batch_first=True)
            self.fc = nn.Linear(64, 1)
        elif temporal_type == 'LTC': 
            self.temporal = LTC(z_dim, AutoNCP(64, 1), batch_first=True)
            self.fc = nn.Identity() 
        elif temporal_type == 'LSTM': 
            self.temporal = nn.LSTM(z_dim, 64, batch_first=True)
            self.fc = nn.Linear(64, 1)
        elif temporal_type == 'GRU':
            self.temporal = nn.GRU(z_dim, 64, batch_first=True)
            self.fc = nn.Linear(64, 1)
        else: # RNN standard
            self.temporal = nn.RNN(z_dim, 64, batch_first=True)
            self.fc = nn.Linear(64, 1)

    def forward(self, node_emb, target_ei, seq_features, product_ids, hx=None):
        src, dst = target_ei
        p_emb = self.prod_emb(product_ids)
        
        # Création du Super-Vecteur Z
        z = torch.cat([node_emb[src], node_emb[dst], p_emb, seq_features], dim=-1).unsqueeze(1)
        
        # Ajustement des dimensions hx pour RNN/GRU/LSTM (qui attendent [num_layers, batch, hidden])
        if self.temporal_type in ['RNN', 'GRU'] and hx is not None:
            hx = hx.unsqueeze(0)
        elif self.temporal_type == 'LSTM' and hx is not None:
            hx = (hx[0].unsqueeze(0), hx[1].unsqueeze(0))
            
        # La librairie ncps gère le delta de temps implicitement à 1.0 si 'times' n'est pas fourni.
        # Sur certaines versions de la librairie, passer 'times' dans le wrapper RNN global lève une TypeError.
        out, hx_new = self.temporal(z, hx)
        
        # On reformate hx_new en [batch, hidden] pour simplifier le stockage global
        if self.temporal_type in ['RNN', 'GRU']:
            hx_new = hx_new.squeeze(0)
        elif self.temporal_type == 'LSTM':
            hx_new = (hx_new[0].squeeze(0), hx_new[1].squeeze(0))
            
        if isinstance(out, tuple): out = out[0]
        return self.fc(out[:, -1, :]), hx_new

# ==========================================
# LOSS FONCTION (Pondérée)
# ==========================================
class WeightedMSELoss(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, pred, target):
        weights = torch.abs(target) + 1.0
        squared_errors = (pred - target) ** 2
        return torch.mean(weights * squared_errors)

# ==========================================
# MOTEUR D'ENTRAÎNEMENT
# ==========================================
def run_stgnn_benchmark():
    rgat_str = "RELATIONAL GAT (RGAT)" if USE_RGAT else "STANDARD GAT"
    print(f"[🚀] Démarrage du Benchmark STGNN avec {rgat_str}")
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Monthly Parquet file not found.")
        return

    df = pd.read_parquet(DATA_PATH)
    print(f"✅ Data Loaded: {len(df)} rows.")
    
    seq_features = ['primaryValue', 'month', 'gdp_o', 'pop_o', 'gdp_d', 'pop_d', 'dist', 'qty']
    df['month'] = df['period'] % 100
    for col in ['gdp_o', 'pop_o', 'gdp_d', 'pop_d', 'dist', 'qty']:
        if col not in df.columns: df[col] = 0.0
        
    print(f"⏳ Normalisation de {len(df)} lignes en cours...")
    # --- CRITIQUE 3 : Normalisation (Z-Score) ---
    features_to_normalize = ['gdp_o', 'pop_o', 'gdp_d', 'pop_d', 'dist', 'qty']
    for col in features_to_normalize:
        mean_val = df[col].mean()
        std_val = df[col].std()
        if std_val > 1e-6: df[col] = (df[col] - mean_val) / std_val
        else: df[col] = 0.0
    
    df[seq_features] = df[seq_features].astype(np.float32)
    df['primaryValue'] = np.log1p(df['primaryValue'])
    
    all_countries = np.unique(np.concatenate([df['reporterCode'], df['partnerCode']]))
    country_map = {int(c): i for i, c in enumerate(all_countries)}
    num_nodes = len(all_countries)

    all_products = np.unique(df['cmdCode'])
    num_products = len(all_products)
    product_map = {int(p): i for i, p in enumerate(all_products)}
    
    print("⏳ Création des identifiants de flux uniques (Edge IDs) pour 295M de lignes...")
    # Création d'un ID unique par flux (Edge ID) pour le TBPTT de manière ultra-optimisée (C++)
    # Au lieu d'assembler des chaînes de caractères (très lent), on utilise ngroup()
    df['edge_id'] = df.groupby(['reporterCode', 'partnerCode', 'cmdCode'], sort=False).ngroup()
    num_unique_edges = df['edge_id'].nunique()
    
    print(f"📊 Nodes: {num_nodes} | Products: {num_products} | Unique Flows: {num_unique_edges}")
    print(f"🔄 Entraînement configuré pour {args.epochs} Epochs au total.")

    # Modèles à évaluer
    model_names = ['GAT_Classic', 'STGNN_LTC', 'STGNN_CfC', 'STGNN_LSTM', 'STGNN_GRU', 'STGNN_RNN']
    
    # Création des encodeurs spatiaux
    # On partage le même type d'encodeur (RGAT ou GAT) pour tous les modèles
    encoders = {n: SpatialGAT(use_rgat=USE_RGAT, num_products=num_products).to(DEVICE) for n in model_names}
    
    models = {
        'GAT_Classic': ClassicGAT(encoders['GAT_Classic']).to(DEVICE),
        'STGNN_LTC': STGNN(encoders['STGNN_LTC'], 'LTC', len(seq_features), num_products).to(DEVICE),
        'STGNN_CfC': STGNN(encoders['STGNN_CfC'], 'CfC', len(seq_features), num_products).to(DEVICE),
        'STGNN_LSTM': STGNN(encoders['STGNN_LSTM'], 'LSTM', len(seq_features), num_products).to(DEVICE),
        'STGNN_GRU': STGNN(encoders['STGNN_GRU'], 'GRU', len(seq_features), num_products).to(DEVICE),
        'STGNN_RNN': STGNN(encoders['STGNN_RNN'], 'RNN', len(seq_features), num_products).to(DEVICE)
    }
    opts = {n: optim.Adam(m.parameters(), lr=0.001) for n, m in models.items()}
    crit = WeightedMSELoss()

    metrics = {
        'dates': [], 'actual_price_usd': [],
        'models': {m: {'predictions_usd': [], 'mse': [], 'mae': [], 'smape': [], 'r2': [], 'time_cost_seconds': 0.0} for m in model_names}
    }

    print(f"\n[1/2] Construction des fenêtres temporelles (Graphes)...")
    train_blocks = []
    # Préparation des graphes (mois par mois)
    for p in tqdm(TRAIN_PERIODS, desc="Prep"):
        p_next = next_month(p)
        df_f = df[df['period'] == p].copy()
        df_t = df[df['period'] == p_next][['reporterCode', 'partnerCode', 'cmdCode', 'primaryValue', 'edge_id']]
        
        # --- CRITIQUE 4 : Biais de Survie (Left Join) ---
        merged = pd.merge(df_f, df_t, on=['reporterCode', 'partnerCode', 'cmdCode'], how='left', suffixes=('', '_target'))
        merged['primaryValue_target'] = merged['primaryValue_target'].fillna(0.0)
        merged['edge_id_target'] = merged['edge_id_target'].fillna(merged['edge_id'])
        
        if len(merged) > 1000:
            df_macro = merged.groupby(['reporterCode', 'partnerCode']).agg({'primaryValue': 'mean', 'qty': 'sum'}).reset_index()
            
            m_rep = torch.tensor(df_macro['reporterCode'].map(country_map).values, dtype=torch.long)
            m_part = torch.tensor(df_macro['partnerCode'].map(country_map).values, dtype=torch.long)
            m_ei = torch.stack([m_rep, m_part])
            m_ea = torch.tensor(df_macro[['primaryValue', 'qty']].values, dtype=torch.float)
            
            train_blocks.append({
                'x_seq': torch.tensor(merged[seq_features].values, dtype=torch.float),
                'y': torch.tensor(merged['primaryValue_target'].values, dtype=torch.float).view(-1, 1),
                'reporters': torch.tensor(merged['reporterCode'].map(country_map).values, dtype=torch.long),
                'partners': torch.tensor(merged['partnerCode'].map(country_map).values, dtype=torch.long),
                'products': torch.tensor(merged['cmdCode'].map(product_map).fillna(0).values, dtype=torch.long),
                'edge_attr': torch.tensor(merged[['primaryValue', 'qty']].values, dtype=torch.float),
                'edge_ids': torch.tensor(merged['edge_id_target'].values, dtype=torch.long),
                'macro_ei': m_ei, 'macro_ea': m_ea,
                'macro_prod': torch.zeros(len(df_macro), dtype=torch.long) 
            })

    print(f"\n[2/2] Entraînement End-to-End (STGNN & TBPTT simulé)...")
    
    # Allocation du registre de mémoire globale pour le TBPTT (une case par flux)
    hx_states = {}
    for n in model_names:
        if n == 'GAT_Classic': continue
        if n == 'STGNN_LSTM':
            hx_states[n] = (torch.zeros(num_unique_edges, 64, device=DEVICE), torch.zeros(num_unique_edges, 64, device=DEVICE))
        else:
            hx_states[n] = torch.zeros(num_unique_edges, 64, device=DEVICE)

    # --- CRITIQUE 2 : Optimisation B200 (Automatic Mixed Precision) ---
    scalers = {n: torch.amp.GradScaler('cuda') for n in model_names}

    for epoch in range(args.epochs):
        epoch_losses = {n: 0.0 for n in model_names}
        epoch_counts = {n: 0 for n in model_names}
        
        # Inversion des boucles : On entraîne un modèle complètement sur ses fenêtres temporelles, 
        # puis on passe au suivant. Cela divise l'utilisation de la VRAM par 6 !
        for n in model_names:
            if n in hx_states:
                if n == 'STGNN_LSTM': hx_states[n][0].zero_(); hx_states[n][1].zero_()
                else: hx_states[n].zero_()
            
            pbar = tqdm(train_blocks, desc=f"Epoch {epoch+1} - {n}", leave=False)
            for month_idx, b in enumerate(pbar):
                x_gpu, y_gpu = b['x_seq'].to(DEVICE), b['y'].to(DEVICE)
                rep_gpu, part_gpu, prod_gpu = b['reporters'].to(DEVICE), b['partners'].to(DEVICE), b['products'].to(DEVICE)
                ea_gpu = b['edge_attr'].to(DEVICE)
                edge_ids_gpu = b['edge_ids'].to(DEVICE)
                m_ei_gpu, m_ea_gpu, m_prod_gpu = b['macro_ei'].to(DEVICE), b['macro_ea'].to(DEVICE), b['macro_prod'].to(DEVICE)
                node_x = torch.randn(num_nodes, 2).to(DEVICE)
                target_ei = torch.stack([rep_gpu, part_gpu])

                t_start = time.time()
                m = models[n]
                opt = opts[n]
                opt.zero_grad()
                
                month_loss = 0
                steps = 0
                for i in range(0, len(x_gpu), BATCH_SIZE):
                    end = i + BATCH_SIZE
                    
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        # OPTION A: Le GAT est calculé DANS le mini-batch pour isoler le graphe de calcul
                        node_emb = m.encoder(node_x, m_ei_gpu, m_ea_gpu, m_prod_gpu if USE_RGAT else None)
                        
                        if n == 'GAT_Classic':
                            p_out = m(node_emb, target_ei[:, i:end], ea_gpu[i:end])
                        else:
                            edge_ids_batch = edge_ids_gpu[i:end]
                            if n == 'STGNN_LSTM': hx_batch = (hx_states[n][0][edge_ids_batch], hx_states[n][1][edge_ids_batch])
                            else: hx_batch = hx_states[n][edge_ids_batch]
                                
                            p_out, hx_new = m(node_emb, target_ei[:, i:end], x_gpu[i:end], prod_gpu[i:end], hx=hx_batch)
                            
                            # OPTION A: On coupe la mémoire instantanément pour éviter le BPTT et sauver la VRAM
                            if n == 'STGNN_LSTM':
                                hx_states[n][0][edge_ids_batch] = hx_new[0].detach().float()
                                hx_states[n][1][edge_ids_batch] = hx_new[1].detach().float()
                            else:
                                hx_states[n][edge_ids_batch] = hx_new.detach().float()
                        
                        loss = crit(p_out, y_gpu[i:end])
                        
                    # OPTION A: Backward immédiat pour détruire le graphe et libérer la mémoire (Mini-Batch SGD)
                    scalers[n].scale(loss).backward()
                    scalers[n].step(opt)
                    scalers[n].update()
                    opt.zero_grad()
                    
                    month_loss += loss.item()
                    steps += 1
                        
                if steps > 0:
                    epoch_losses[n] += (month_loss / steps)
                    epoch_counts[n] += 1
                    
                metrics['models'][n]['time_cost_seconds'] += (time.time() - t_start)
                
        print(f"Epoch {epoch+1}/{args.epochs} | STGNN_LTC Loss: {epoch_losses['STGNN_LTC']/max(1,epoch_counts['STGNN_LTC']):.4f} | GAT_Classic Loss: {epoch_losses['GAT_Classic']/max(1,epoch_counts['GAT_Classic']):.4f}")

    # --- ÉVALUATION ---
    print(f"\n[3/3] Évaluation et Calcul des Métriques...")
    for n in model_names: models[n].eval()
    
    with torch.no_grad():
        for p in TEST_PERIODS:
            p_next = next_month(p)
            df_f = df[df['period'] == p].copy()
            df_t = df[df['period'] == p_next][['reporterCode', 'partnerCode', 'cmdCode', 'primaryValue', 'edge_id']]
            merged = pd.merge(df_f, df_t, on=['reporterCode', 'partnerCode', 'cmdCode'], how='left', suffixes=('', '_target'))
            merged['primaryValue_target'] = merged['primaryValue_target'].fillna(0.0)
            merged['edge_id_target'] = merged['edge_id_target'].fillna(merged['edge_id'])
            if len(merged) < 100: continue
            
            x_all = torch.tensor(merged[seq_features].values, dtype=torch.float).to(DEVICE)
            y_all = torch.tensor(merged['primaryValue_target'].values, dtype=torch.float).view(-1, 1).to(DEVICE)
            rep_all = torch.tensor(merged['reporterCode'].map(country_map).fillna(0).values, dtype=torch.long).to(DEVICE)
            part_all = torch.tensor(merged['partnerCode'].map(country_map).fillna(0).values, dtype=torch.long).to(DEVICE)
            prod_all = torch.tensor(merged['cmdCode'].map(product_map).fillna(0).values, dtype=torch.long).to(DEVICE)
            ea_all = torch.tensor(merged[['primaryValue', 'qty']].values, dtype=torch.float).to(DEVICE)
            edge_ids_all = torch.tensor(merged['edge_id_target'].values, dtype=torch.long).to(DEVICE)
            target_ei = torch.stack([rep_all, part_all])

            df_macro_ev = merged.groupby(['reporterCode', 'partnerCode']).agg({'primaryValue': 'mean', 'qty': 'sum'}).reset_index()
            m_ei_eval = torch.stack([
                torch.tensor(df_macro_ev['reporterCode'].map(country_map).values, dtype=torch.long),
                torch.tensor(df_macro_ev['partnerCode'].map(country_map).values, dtype=torch.long)
            ]).to(DEVICE)
            m_ea_eval = torch.tensor(df_macro_ev[['primaryValue', 'qty']].values, dtype=torch.float).to(DEVICE)
            m_prod_eval = torch.zeros(len(df_macro_ev), dtype=torch.long).to(DEVICE)
            node_x_eval = torch.randn(num_nodes, 2).to(DEVICE)

            metrics['dates'].append(str(p))
            metrics['actual_price_usd'].append(float(np.expm1(merged['primaryValue_target'].values).mean()))

            for n in model_names:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    node_emb = models[n].encoder(node_x_eval, m_ei_eval, m_ea_eval, m_prod_eval if USE_RGAT else None)
                    all_preds = []
                    for i in range(0, len(x_all), BATCH_SIZE):
                        end = i + BATCH_SIZE
                        if n == 'GAT_Classic': 
                            out = models[n](node_emb, target_ei[:, i:end], ea_all[i:end])
                        else: 
                            edge_ids_batch = edge_ids_all[i:end]
                            if n == 'STGNN_LSTM': hx_batch = (hx_states[n][0][edge_ids_batch], hx_states[n][1][edge_ids_batch])
                            else: hx_batch = hx_states[n][edge_ids_batch]
                            
                            out, hx_new = models[n](node_emb, target_ei[:, i:end], x_all[i:end], prod_all[i:end], hx=hx_batch)
                            
                            # Continuer le tracking de la mémoire pendant l'évaluation
                            if n == 'STGNN_LSTM':
                                hx_states[n][0][edge_ids_batch] = hx_new[0].detach().float()
                                hx_states[n][1][edge_ids_batch] = hx_new[1].detach().float()
                            else:
                                hx_states[n][edge_ids_batch] = hx_new.detach().float()
                                
                        all_preds.append(out)
                
                # Conversion en float32 pour l'évaluation mathématique
                y_p = torch.cat(all_preds).float()
                
                # Metrics (Log Space pour MSE/MAE, USD space pour Moyenne et sMAPE)
                mse = torch.nn.functional.mse_loss(y_p, y_all).item()
                mae = torch.abs(y_p - y_all).mean().item()
                
                # Transform to USD for sMAPE
                pred_usd = torch.expm1(y_p)
                true_usd = torch.expm1(y_all)
                smape = torch.mean(2 * torch.abs(pred_usd - true_usd) / (torch.abs(pred_usd) + torch.abs(true_usd) + 1e-8)).item() * 100
                
                # R2 Score (approximated on Log space)
                ss_res = torch.sum((y_all - y_p) ** 2)
                ss_tot = torch.sum((y_all - torch.mean(y_all)) ** 2)
                r2 = (1 - ss_res / (ss_tot + 1e-8)).item()
                
                metrics['models'][n]['mse'].append(mse)
                metrics['models'][n]['mae'].append(mae)
                metrics['models'][n]['smape'].append(smape)
                metrics['models'][n]['r2'].append(r2)
                metrics['models'][n]['predictions_usd'].append(float(pred_usd.mean().item()))

    with open(os.path.join(SCRIPT_DIR, "results/stgnn_metrics.js"), "w") as f:
        f.write(f"const stgnnMetrics = {json.dumps(metrics, indent=2)};")

    print(f"\n✅ Benchmark STGNN terminé. Résultats exportés dans results/stgnn_metrics.js")

if __name__ == "__main__":
    try:
        run_stgnn_benchmark()
    except BaseException as e:
        print(traceback.format_exc())

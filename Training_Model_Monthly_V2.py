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
import sys
import traceback
from tqdm import tqdm

# --- COMMAND LINE ARGUMENTS ---
# Keeps the exact options of the previous training script
parser = argparse.ArgumentParser(description="Monthly Resilience Benchmark B200 V2")
parser.add_argument('--epochs', '-e', type=int, default=50, help="Number of initial epochs")
parser.add_argument('--retrain-epochs', '-re', type=int, default=10, help="Number of epochs for retraining")
args = parser.parse_known_args()[0]

# --- MONTHLY CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "trade_data_monthly.parquet")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 15000 

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

# --- ARCHITECTURES V2 ---
# Implements Option C with shared Product Embeddings
class MonthlyGAT_V2(nn.Module):
    def __init__(self, node_dim=2, edge_dim=2, num_products=5204, prod_emb_dim=16, hidden=32):
        super().__init__()
        # Graph convolution over country macro-graph
        self.conv1 = GATConv(node_dim, hidden, heads=4, edge_dim=edge_dim, concat=True)
        self.conv2 = GATConv(hidden * 4, hidden, heads=1, edge_dim=edge_dim, concat=False)
        # Shared product embedding layer
        self.prod_emb = nn.Embedding(num_products, prod_emb_dim)
        # Combines: source node embedding + edge attributes + product embedding
        self.fc = nn.Linear(hidden + edge_dim + prod_emb_dim, 1)

    def compute_node_emb(self, x, edge_index, edge_attr):
        h = torch.relu(self.conv1(x, edge_index, edge_attr))
        return self.conv2(h, edge_index, edge_attr)

    def predict_edges(self, node_emb, target_ei, target_ea, prod):
        src, dst = target_ei
        e_prod = self.prod_emb(prod)
        combined = torch.cat([node_emb[src], target_ea, e_prod], dim=-1)
        return self.fc(combined)

class MonthlyRNN_V2(nn.Module):
    def __init__(self, model_type, in_dim, num_countries, num_products, emb_dim=8, prod_emb_dim=16):
        super().__init__()
        self.model_type = model_type
        # Learn country embeddings
        self.emb = nn.Embedding(num_countries, emb_dim)
        # Learn product embeddings
        self.prod_emb = nn.Embedding(num_products, prod_emb_dim)
        # Sequence features + country origin/destination embeddings + product embedding
        combined_dim = in_dim + (emb_dim * 2) + prod_emb_dim
        
        if model_type == 'CfC': 
            self.rnn = CfC(combined_dim, 64, batch_first=True)
            self.fc = nn.Linear(64, 1)
        elif model_type == 'LTC': 
            self.rnn = LTC(combined_dim, AutoNCP(64, 1), batch_first=True)
            self.fc = nn.Identity() 
        elif model_type == 'LSTM': 
            self.rnn = nn.LSTM(combined_dim, 64, batch_first=True)
            self.fc = nn.Linear(64, 1)
        else: 
            self.rnn = nn.GRU(combined_dim, 64, batch_first=True)
            self.fc = nn.Linear(64, 1)

    def forward(self, x, rep, part, prod, hx=None):
        e_r = self.emb(rep).unsqueeze(1)
        e_p = self.emb(part).unsqueeze(1)
        e_prod = self.prod_emb(prod).unsqueeze(1)
        x_combined = torch.cat([x, e_r, e_p, e_prod], dim=-1)
        
        out, hx_new = self.rnn(x_combined, hx)
        if isinstance(out, tuple): out = out[0]
        return self.fc(out[:, -1, :]), hx_new

# --- LOSS FUNCTION V2 ---
# Custom Weighted MSE to resolve the flat low-variance prediction paradox on USD-scale.
# Weights each trade flow's squared error by its log-target magnitude.
class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # target is already np.log1p(primaryValue), so it is >= 0.
        # Adding 1.0 ensures that small/zero flows still get a small gradient weight of 1.0,
        # but large-value flows (e.g. log-value 14.0) get a high weight of 15.0.
        weights = torch.abs(target) + 1.0
        squared_errors = (pred - target) ** 2
        return torch.mean(weights * squared_errors)

# --- ENGINE ---
def run_monthly_benchmark():
    print(f"[🚀] Adaptive Monthly Benchmark B200 V2 - Product Aware Embeddings")
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Monthly Parquet file not found.")
        return

    # --- LOGGING TO FILE ---
    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.start_time = time.strftime("%Y%m%d_%H%M%S")
            log_name = f"training_log_v2_{self.start_time}.txt"
            self.log = open(os.path.join(SCRIPT_DIR, log_name), "a", encoding="utf-8")
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    sys.stdout = Logger()

    df = pd.read_parquet(DATA_PATH)
    print(f"✅ Data Loaded: {len(df)} rows.")
    
    seq_features = ['primaryValue', 'month', 'gdp_o', 'pop_o', 'gdp_d', 'pop_d', 'dist', 'qty']
    df['month'] = df['period'] % 100
    
    # Fill missing columns with 0 if they don't exist in the monthly Parquet
    for col in ['gdp_o', 'pop_o', 'gdp_d', 'pop_d', 'dist', 'qty']:
        if col not in df.columns: df[col] = 0.0
    
    df[seq_features] = df[seq_features].astype(np.float32)
    df['primaryValue'] = np.log1p(df['primaryValue'])
    
    # Country mapping
    all_countries = np.unique(np.concatenate([df['reporterCode'], df['partnerCode']]))
    country_map = {int(c): i for i, c in enumerate(all_countries)}
    num_nodes = len(all_countries)

    # Product mapping
    all_products = np.unique(df['cmdCode'])
    product_map = {int(p): i for i, p in enumerate(all_products)}
    num_products = len(all_products)
    print(f"📊 Unique Countries: {num_nodes} | Unique Products: {num_products}")

    model_names = ['GAT', 'CfC', 'LTC', 'LSTM', 'GRU']
    metrics = {
        'dates': [], 'actual_price_usd': [],
        'models': {m: {'predictions_usd': [], 'rolling_mae': [], 'avg_mse': 0, 'avg_growth': 0.0, 'retrain_events': [], 'time_cost_seconds': 0.0, 'retrain_time_seconds': 0.0} for m in model_names},
        'drift_metrics': {'labels': [], 'ks': [], 'psi': [], 'js': []}
    }

    models = {
        'GAT': MonthlyGAT_V2(node_dim=2, edge_dim=2, num_products=num_products).to(DEVICE),
        'CfC': MonthlyRNN_V2('CfC', in_dim=len(seq_features), num_countries=num_nodes, num_products=num_products).to(DEVICE),
        'LTC': MonthlyRNN_V2('LTC', in_dim=len(seq_features), num_countries=num_nodes, num_products=num_products).to(DEVICE),
        'LSTM': MonthlyRNN_V2('LSTM', in_dim=len(seq_features), num_countries=num_nodes, num_products=num_products).to(DEVICE),
        'GRU': MonthlyRNN_V2('GRU', in_dim=len(seq_features), num_countries=num_nodes, num_products=num_products).to(DEVICE)
    }
    opts = {n: optim.Adam(m.parameters(), lr=0.001) for n, m in models.items()}
    crit = WeightedMSELoss()

    # --- SCHEDULERS AND EARLY STOPPING ---
    schedulers = {name: ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5) for name, opt in opts.items()}
    best_losses = {name: float('inf') for name in model_names}
    patience_counters = {name: 0 for name in model_names}
    early_stop_patience = 15
    active_models = set(model_names)

    # --- PHASE 1: INITIAL TRAINING ---
    print(f"\n[1/2] Initial Training ({args.epochs} epochs)...")
    train_blocks = []
    for p in tqdm(TRAIN_PERIODS, desc="Preparation"):
        p_next = next_month(p)
        df_f = df[df['period'] == p].copy()
        df_t = df[df['period'] == p_next][['reporterCode', 'partnerCode', 'cmdCode', 'primaryValue']]
        merged = pd.merge(df_f, df_t, on=['reporterCode', 'partnerCode', 'cmdCode'], how='inner', suffixes=('', '_target'))
        
        if len(merged) > 1000:
            # --- Macro-Graph (Aggregated by Country) ---
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
                'macro_ei': m_ei, 'macro_ea': m_ea
            })

    for epoch in range(args.epochs):
        epoch_losses = {n: 0.0 for n in model_names}
        epoch_counts = {n: 0 for n in model_names}
        pbar = tqdm(train_blocks, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        
        for b in pbar:
            x_gpu, y_gpu = b['x_seq'].to(DEVICE), b['y'].to(DEVICE)
            rep_gpu, part_gpu = b['reporters'].to(DEVICE), b['partners'].to(DEVICE)
            prod_gpu = b['products'].to(DEVICE)
            ea_gpu = b['edge_attr'].to(DEVICE)
            m_ei_gpu, m_ea_gpu = b['macro_ei'].to(DEVICE), b['macro_ea'].to(DEVICE)
            node_x = torch.randn(num_nodes, 2).to(DEVICE)

            for n in model_names:
                if n not in active_models: continue
                t_start = time.time()
                m = models[n]
                opt = opts[n]
                
                for i in range(0, len(x_gpu), BATCH_SIZE):
                    end = i + BATCH_SIZE
                    opt.zero_grad()
                    
                    if n == 'GAT':
                        node_emb = m.compute_node_emb(node_x, m_ei_gpu, m_ea_gpu)
                        p_out = m.predict_edges(node_emb, torch.stack([rep_gpu[i:end], part_gpu[i:end]]), ea_gpu[i:end], prod_gpu[i:end])
                    else:
                        p_out, _ = m(x_gpu[i:end].unsqueeze(1), rep_gpu[i:end], part_gpu[i:end], prod_gpu[i:end])
                    
                    loss = crit(p_out, y_gpu[i:end])
                    loss.backward()
                    opt.step()
                    
                    epoch_losses[n] += loss.item()
                    epoch_counts[n] += 1
                
                metrics['models'][n]['time_cost_seconds'] += (time.time() - t_start)
            
            if epoch_counts['LTC'] > 0:
                pbar.set_postfix({'LTC_loss': f"{epoch_losses['LTC']/epoch_counts['LTC']:.4f}"})

        log_msg = f" [Epoch {epoch+1}/{args.epochs}]"
        for n in model_names:
            if epoch_counts[n] > 0:
                avg = epoch_losses[n] / epoch_counts[n]
                log_msg += f" | {n}: {avg:.4f}"
        print(log_msg)

        for n in list(active_models):
            avg_loss = epoch_losses[n] / epoch_counts[n] if epoch_counts[n] > 0 else 1e9
            schedulers[n].step(avg_loss)
            
            if avg_loss < best_losses[n] * 0.999:
                best_losses[n] = avg_loss
                patience_counters[n] = 0
            else:
                patience_counters[n] += 1
            
            if patience_counters[n] >= early_stop_patience:
                print(f"  [🛑] {n} converged at epoch {epoch+1}")
                active_models.remove(n)

        if not active_models:
            print(f"  [🏁] All models converged.")
            break

    # --- PHASE 2: ADAPTIVE EVALUATION ---
    print(f"\n[2/2] Monthly Evaluation with Adaptive Retraining...")
    ref_vals = train_blocks[-1]['y'].numpy().flatten()
    prev_mae = {m: None for m in model_names}
    eval_memory = train_blocks[-12:].copy()

    for p in TEST_PERIODS:
        p_next = next_month(p)
        df_f = df[df['period'] == p].copy()
        df_t = df[df['period'] == p_next][['reporterCode', 'partnerCode', 'cmdCode', 'primaryValue']]
        merged = pd.merge(df_f, df_t, on=['reporterCode', 'partnerCode', 'cmdCode'], how='inner', suffixes=('', '_target'))
        if len(merged) < 100: continue
        
        x_all = torch.tensor(merged[seq_features].values, dtype=torch.float).to(DEVICE)
        y_all = torch.tensor(merged['primaryValue_target'].values, dtype=torch.float).view(-1, 1).to(DEVICE)
        rep_all = torch.tensor(merged['reporterCode'].map(country_map).fillna(0).values, dtype=torch.long).to(DEVICE)
        part_all = torch.tensor(merged['partnerCode'].map(country_map).fillna(0).values, dtype=torch.long).to(DEVICE)
        prod_all = torch.tensor(merged['cmdCode'].map(product_map).fillna(0).values, dtype=torch.long).to(DEVICE)
        ea_all = torch.tensor(merged[['primaryValue', 'qty']].values, dtype=torch.float).to(DEVICE)

        # Macro-Graph for Inference
        df_macro_ev = merged.groupby(['reporterCode', 'partnerCode']).agg({'primaryValue': 'mean', 'qty': 'sum'}).reset_index()
        m_ei_eval = torch.stack([
            torch.tensor(df_macro_ev['reporterCode'].map(country_map).values, dtype=torch.long),
            torch.tensor(df_macro_ev['partnerCode'].map(country_map).values, dtype=torch.long)
        ]).to(DEVICE)
        m_ea_eval = torch.tensor(df_macro_ev[['primaryValue', 'qty']].values, dtype=torch.float).to(DEVICE)

        ks, _ = ks_2samp(ref_vals, merged['primaryValue_target'].values)
        try:
            curr_v = merged['primaryValue_target'].values
            p_hist, _ = np.histogram(ref_vals, bins=10, range=(ref_vals.min(), ref_vals.max()))
            q_hist, _ = np.histogram(curr_v, bins=10, range=(ref_vals.min(), ref_vals.max()))
            p_hist = np.where(p_hist == 0, 0.0001, p_hist) / len(ref_vals)
            q_hist = np.where(q_hist == 0, 0.0001, q_hist) / len(curr_v)
            psi_val = np.sum((q_hist - p_hist) * np.log(q_hist / p_hist))
            js_val = jensenshannon(p_hist, q_hist)
        except: psi_val, js_val = 0.0, 0.0

        node_x_eval = torch.randn(num_nodes, 2).to(DEVICE)
        all_preds = {n: [] for n in model_names}
        for n, m in models.items():
            m.eval()
            with torch.no_grad():
                if n == 'GAT':
                    node_emb = m.compute_node_emb(node_x_eval, m_ei_eval, m_ea_eval)
                for i in range(0, len(x_all), BATCH_SIZE):
                    end = i + BATCH_SIZE
                    if n == 'GAT': p_out = m.predict_edges(node_emb, torch.stack([rep_all[i:end], part_all[i:end]]), ea_all[i:end], prod_all[i:end])
                    else: p_out, _ = m(x_all[i:end].unsqueeze(1), rep_all[i:end], part_all[i:end], prod_all[i:end])
                    all_preds[n].append(p_out)

        metrics['dates'].append(str(p))
        metrics['drift_metrics']['labels'].append(str(p))
        metrics['drift_metrics']['ks'].append(float(ks)); metrics['drift_metrics']['psi'].append(float(psi_val)); metrics['drift_metrics']['js'].append(float(js_val))
        metrics['actual_price_usd'].append(float(np.expm1(merged['primaryValue_target'].values).mean()))

        for n in model_names:
            y_p = torch.cat(all_preds[n])
            mae = torch.abs(y_p - y_all).mean().item()
            metrics['models'][n]['rolling_mae'].append(mae)
            metrics['models'][n]['predictions_usd'].append(float(torch.expm1(y_p).mean().item()))

            if prev_mae[n] is not None and mae > 1.10 * prev_mae[n]:
                print(f"  [!] Drift {n} ({p}) | MAE: {mae:.4f}. Retraining...")
                metrics['models'][n]['retrain_events'].append(str(p))
                t0_rt = time.time()
                models[n].train()
                for _ in range(args.retrain_epochs):
                    for b_rt in eval_memory[-6:]:
                        xr_all, yr_all = b_rt['x_seq'].to(DEVICE), b_rt['y'].to(DEVICE)
                        rr_all, pr_all, er_all = b_rt['reporters'].to(DEVICE), b_rt['partners'].to(DEVICE), b_rt['edge_attr'].to(DEVICE)
                        prodr_all = b_rt['products'].to(DEVICE)
                        m_ei_rt, m_ea_rt = b_rt['macro_ei'].to(DEVICE), b_rt['macro_ea'].to(DEVICE)
                        node_xr = torch.randn(num_nodes, 2).to(DEVICE)
                        
                        for i_rt in range(0, len(xr_all), BATCH_SIZE):
                            end_rt = i_rt + BATCH_SIZE
                            opts[n].zero_grad()
                            if n == 'GAT':
                                node_emb_r = models[n].compute_node_emb(node_xr, m_ei_rt, m_ea_rt)
                                out_r = models[n].predict_edges(node_emb_r, torch.stack([rr_all[i_rt:end_rt], pr_all[i_rt:end_rt]]), er_all[i_rt:end_rt], prodr_all[i_rt:end_rt])
                            else:
                                out_r, _ = models[n](xr_all[i_rt:end_rt].unsqueeze(1), rr_all[i_rt:end_rt], pr_all[i_rt:end_rt], prodr_all[i_rt:end_rt])
                            
                            loss_r = crit(out_r, yr_all[i_rt:end_rt])
                            loss_r.backward(); opts[n].step()
                metrics['models'][n]['retrain_time_seconds'] += (time.time() - t0_rt)
            prev_mae[n] = mae

        # Save this month's data block for future retraining
        df_macro_rt = merged.groupby(['reporterCode', 'partnerCode']).agg({'primaryValue': 'mean', 'qty': 'sum'}).reset_index()
        m_rep_rt = torch.tensor(df_macro_rt['reporterCode'].map(country_map).values, dtype=torch.long)
        m_part_rt = torch.tensor(df_macro_rt['partnerCode'].map(country_map).values, dtype=torch.long)
        m_ei_rt_block = torch.stack([m_rep_rt, m_part_rt])
        m_ea_rt_block = torch.tensor(df_macro_rt[['primaryValue', 'qty']].values, dtype=torch.float)

        eval_memory.append({
            'x_seq': x_all.cpu(), 'y': y_all.cpu(), 
            'reporters': rep_all.cpu(), 'partners': part_all.cpu(), 
            'products': prod_all.cpu(),
            'edge_attr': torch.tensor(merged[['primaryValue_target', 'qty']].values, dtype=torch.float).cpu(),
            'macro_ei': m_ei_rt_block, 'macro_ea': m_ea_rt_block
        })
        if len(eval_memory) > 12: eval_memory.pop(0)
        
        best_n = min(model_names, key=lambda n: metrics['models'][n]['rolling_mae'][-1])
        print(f"  Period {p} | GAT: {metrics['models']['GAT']['rolling_mae'][-1]:.4f} | LTC: {metrics['models']['LTC']['rolling_mae'][-1]:.4f} | Best: {best_n}")

        # Incremental save for live dashboard (V2 metrics)
        for m in model_names:
            maes = metrics['models'][m]['rolling_mae']
            metrics['models'][m]['avg_mse'] = float(np.mean(maes)**2) if maes else 0.0
            growth = np.diff(maes)
            metrics['models'][m]['avg_growth'] = float(np.mean(growth) * 100) if len(growth) > 0 else 0.0

        with open(os.path.join(SCRIPT_DIR, "results/monthly_v2_metrics.js"), "w") as f:
            f.write(f"const b200AdaptiveMetricsDataV2 = {json.dumps(metrics, indent=2)};")

    print(f"\n✅ Monthly adaptive benchmark V2 completed.")

if __name__ == "__main__":
    try:
        run_monthly_benchmark()
    except BaseException as e:
        print("\n" + "!"*30)
        print(f"⚠️ SCRIPT STOPPED OR FATAL ERROR:")
        print(traceback.format_exc())
        print("!"*30)

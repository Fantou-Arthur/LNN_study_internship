import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import dgl
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
import warnings
import json

# Disable redundant warnings for clean terminal logging
warnings.filterwarnings('ignore')

# Import custom modular components
from gat_model import GATRegressionModel
from smape_util import calculate_smape
from evaluator import evaluate_dataset

def main():
    parser = argparse.ArgumentParser(description="Accelerated GAT Model Training on B200 GPU")
    parser.add_argument('--data-path', type=str, 
                        default="../comtradeExports_updatedH5[240924]-wb.csv.gz", 
                        help="Path to the UN Comtrade dataset (CSV or CSV.GZ)")
    parser.add_argument('--batch-size', type=int, default=10000, 
                        help="Training and evaluation batch size")
    parser.add_argument('--epochs', type=int, default=100, 
                        help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.01, 
                        help="Learning rate for Adam optimizer")
    args = parser.parse_args()

    # ==============================================================
    # HARDWARE SPEEDUP INITIALIZATION (B200 optimization)
    # ==============================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print(f"B200 ACCELERATED GAT TRAINING SYSTEM")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
        # Enable TF32 for matrix multiplications on B200 if supported
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print("=" * 60)

    # ==============================================================
    # DATA LOADING & PREPROCESSING
    # ==============================================================
    print("\nLOADING & PREPROCESSING DATA...")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset not found at: {args.data_path}. Please check your relative path.")
        
    df = pd.read_csv(args.data_path)
    
    # Scale variables back to research base dimensions
    df['gdp_o'] = df['gdp_o'] / 1_000_000
    df['gdp_d'] = df['gdp_d'] / 1_000_000
    df['gdpcap_o'] = df['gdpcap_o'] / 1_000_000
    df['gdpcap_d'] = df['gdpcap_d'] / 1_000_000
    df['dist'] = df['dist'] / 1_000

    # Clean commodity codes
    df = df[df['cmdCode'] != 'TOTAL']
    df = df[df['cmdCode'].astype(str).str.isdigit()]
    df['cmdCode'] = df['cmdCode'].astype(int)
    df = df[df['cmdCode'] > 99999]  # 6-digit classification filter

    # Drop duplicate trade flows
    data_all = df[['refYear', 'reporterCode', 'partnerCode', 'cmdCode', 'gdpcap_o', 'pop_o', 'gdpcap_d', 'pop_d', 'dist', 'primaryValue']]
    data_all = data_all.drop_duplicates(['refYear', 'reporterCode', 'partnerCode', 'cmdCode'], keep='first').reset_index(drop=True)

    # ==============================================================
    # YEAR SPLIT: Train (2017-2018), Val (2019-2022), Test (2023)
    # ==============================================================
    data_train_full = data_all[data_all['refYear'].isin([2017, 2018])].copy()
    data_val_full = data_all[data_all['refYear'].isin([2019, 2020, 2021, 2022])].copy()
    data_2023 = data_all[data_all['refYear'] == 2023].copy()

    print(f"Train data size (2017-2018): {data_train_full.shape[0]:,}")
    print(f"Validation data size (2019-2022): {data_val_full.shape[0]:,}")
    print(f"Test 2023 size: {data_2023.shape[0]:,}")
    print(f"Years included in training: {sorted(data_train_full['refYear'].unique())}")

    # Handle missing physical values using reporter/partner mean grouping with global mean fallbacks
    for col in ['gdpcap_o', 'pop_o']:
        data_train_full[col] = data_train_full.groupby(['refYear', 'reporterCode'])[col].transform(lambda x: x.fillna(x.mean() if pd.notnull(x.mean()) else 0))
    for col in ['gdpcap_d', 'pop_d']:
        data_train_full[col] = data_train_full.groupby(['refYear', 'partnerCode'])[col].transform(lambda x: x.fillna(x.mean() if pd.notnull(x.mean()) else 0))
    data_train_full['dist'] = data_train_full.groupby(['refYear', 'reporterCode', 'partnerCode'])['dist'].transform(lambda x: x.fillna(x.mean() if pd.notnull(x.mean()) else 0))

    # Global fallback for any remaining NaNs
    for col in ['gdpcap_o', 'pop_o', 'gdpcap_d', 'pop_d', 'dist']:
        global_mean = data_train_full[col].mean()
        fallback_val = global_mean if pd.notnull(global_mean) else 0
        data_train_full[col] = data_train_full[col].fillna(fallback_val)

    # ==============================================================
    # TRAIN / VALIDATION SET ASSIGNMENT (Years 2017-2018 vs 2019-2022)
    # ==============================================================
    print("\nAssigning train/val datasets...")
    train_data = data_train_full.copy()
    val_data = data_val_full.copy()

    print(f"Train set size: {train_data.shape[0]:,}")
    print(f"Validation set size: {val_data.shape[0]:,}")

    # ==============================================================
    # FEATURE NORMALIZATION
    # ==============================================================
    agg_features = train_data.groupby(['refYear', 'reporterCode', 'cmdCode']).agg({
        'primaryValue': 'mean',
        'dist': 'first',
        'gdpcap_d': 'sum',
        'gdpcap_o': 'sum',
        'pop_d': 'sum',
        'pop_o': 'sum'
    }).reset_index()

    country_code_map = {code: i for i, code in enumerate(agg_features['reporterCode'])}
    train_data = train_data.copy()
    train_data['nodeID'] = train_data['reporterCode'].map(country_code_map)
    train_data['partnerNodeID'] = train_data['partnerCode'].map(country_code_map)

    # Filter out rows with unmapped countries (NaNs) to prevent invalid node IDs in DGL
    train_data_filtered = train_data.dropna(subset=['nodeID', 'partnerNodeID']).copy()
    train_data_filtered['nodeID'] = train_data_filtered['nodeID'].astype(np.int64)
    train_data_filtered['partnerNodeID'] = train_data_filtered['partnerNodeID'].astype(np.int64)

    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()

    features_to_normalize = agg_features[['primaryValue', 'refYear', 'cmdCode', 'dist', 'gdpcap_d', 'gdpcap_o', 'pop_o', 'pop_d']]
    normalized_features = scaler_features.fit_transform(features_to_normalize)
    normalized_target = scaler_target.fit_transform(agg_features[['primaryValue']])

    # ==============================================================
    # BUILDING THE DGL DENSE GRAPH ON CUDA
    # ==============================================================
    print("\nBuilding graph and hosting on B200 GPU...")
    # Map edges and build graph structure
    src_nodes = train_data_filtered['nodeID'].to_numpy()
    dst_nodes = train_data_filtered['partnerNodeID'].to_numpy()
    g = dgl.graph((src_nodes, dst_nodes))
    
    # Pack attributes into node data dictionary
    feat_tensor = torch.tensor(normalized_features, dtype=torch.float32)
    g.ndata['feat'] = feat_tensor
    g = dgl.add_self_loop(g)

    # Pin graph directly on GPU VRAM to speed up message passing convolutions
    g = g.to(device)
    print(f"Graph hosted on {g.device} successfully!")
    print(f"Graph dimensions: {g.num_nodes():,} nodes | {g.num_edges():,} edges")

    # ==============================================================
    # GAT MODEL & OPTIMIZER INITIALIZATION
    # ==============================================================
    in_feats = 8
    hidden_feats = 32
    num_heads = 4

    model = GATRegressionModel(in_feats, hidden_feats, num_heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # ==============================================================
    # ACCELERATED GPU TRAINING
    # ==============================================================
    losses = []
    num_nodes = g.num_nodes()
    num_batches = num_nodes // args.batch_size + (num_nodes % args.batch_size > 0)

    print("\n" + "=" * 60)
    print("TRAINING PROCESS (2017-2018)")
    print("=" * 60)

    # Convert targets to tensor and host on active device
    target_all_tensor = torch.tensor(normalized_target[:, 0], dtype=torch.float32, device=device).view(-1, 1)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        for i in range(num_batches):
            # Batch slicing
            batch_start = i * args.batch_size
            batch_end = min((i + 1) * args.batch_size, num_nodes)
            batch_nodes = list(range(batch_start, batch_end))
            
            # Extract CUDA-pinned subgraph (extremely fast on B200)
            batch_graph = g.subgraph(batch_nodes)
            
            logits = model(batch_graph, batch_graph.ndata['feat'])
            target = target_all_tensor[batch_nodes]
            
            loss = criterion(logits.view(-1, 1), target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        losses.append(epoch_loss)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:3d}/{args.epochs} | Loss: {epoch_loss:.6f}')

    print("\n✓ Training complete!")

    # Save loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(losses, linewidth=2, color='#0ea5e9')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('GAT Training Loss (2017-2018) - Accelerated on B200')
    plt.grid(True, alpha=0.3)
    plot_name = 'training_loss_2017_2018.png'
    plt.savefig(plot_name, dpi=300)
    print(f"✓ Loss curve figure saved as: {plot_name}")

    # ==============================================================
    # MULTI-DATASET DIAGNOSTICS & EVALUATIONS
    # ==============================================================
    # 1. Validation evaluation
    results_val, y_true_val, y_pred_val = evaluate_dataset(
        model, val_data, agg_features, country_code_map,
        scaler_features, scaler_target, args.batch_size, "Validation (2019-2022)", device
    )

    # 2. Test 2023 evaluation
    results_2023, y_true_2023, y_pred_2023 = evaluate_dataset(
        model, data_2023, agg_features, country_code_map,
        scaler_features, scaler_target, args.batch_size, "Test 2023", device
    )

    # ==============================================================
    # DIAGNOSTIC BY VALUE RANGE (Validation 2019-2022)
    # ==============================================================
    print("\n" + "=" * 120)
    print("DIAGNOSTIC BY VALUE RANGE (Validation 2019-2022)")
    print("=" * 120)

    if y_true_val is not None:
        y_true_flat = y_true_val.flatten()
        y_pred_flat = y_pred_val.flatten()
        
        tranches = [
            ('< 100K USD', y_true_flat < 100_000),
            ('100K - 1M USD', (y_true_flat >= 100_000) & (y_true_flat < 1_000_000)),
            ('1M - 10M USD', (y_true_flat >= 1_000_000) & (y_true_flat < 10_000_000)),
            ('10M - 100M USD', (y_true_flat >= 10_000_000) & (y_true_flat < 100_000_000)),
            ('> 100M USD', y_true_flat >= 100_000_000)
        ]
        
        for nom, mask in tranches:
            if mask.sum() > 1:
                r2_t = r2_score(y_true_flat[mask], y_pred_flat[mask])
                smape_t = calculate_smape(y_true_flat[mask], y_pred_flat[mask])
                print(f"{nom:25s} | n={mask.sum():>10,} | R²={r2_t:>8.4f} | SMAPE={smape_t:>6.2f}%")

    # ==============================================================
    # SUMMARY TABLE (>= 100M USD)
    # ==============================================================
    print("\n" + "=" * 100)
    print("SUMMARY TABLE (High-value flows >= 100M USD)")
    print("=" * 100)

    all_results = [
        ("Validation (2019-2022)", results_val),
        ("Test 2023", results_2023)
    ]

    print(f"\n{'Dataset':<25} | {'n':>8} | {'MSE':>15} | {'MAE':>12} | {'R²':>8} | {'SMAPE':>10}")
    print("-" * 90)

    for name, results in all_results:
        if results and results.get('gte_100M'):
            r = results['gte_100M']
            print(f"{name:<25} | {r['n']:>8,} | {r['mse']:>15,.2f} | {r['mae']:>12,.2f} | {r['r2']:>8.4f} | {r['smape']:>9.2f}%")

    # ==============================================================
    # LATEX GENERATOR
    # ==============================================================
    print("\n" + "=" * 100)
    print("LATEX TABLE GENERATION (>= 100M USD)")
    print("=" * 100)

    print(r"""
\begin{table}[h]
\centering
\caption{GAT Model Performance for Trade Flow Prediction (trained on 2017-2018, flows $\geq$ 100M USD)}
\label{tab:results_2017_2018}
\begin{tabular}{l|r|r|r|c|c}
Dataset & n & MSE & MAE & R² & SMAPE \\
\hline""")

    for name, results in all_results:
        if results and results.get('gte_100M'):
            r = results['gte_100M']
            print(f"{name} & {r['n']:,} & {r['mse']:,.2f} & {r['mae']:,.2f} & {r['r2']:.2f} & {r['smape']:.2f}\\% \\\\")

    print(r"""\end{tabular}
\end{table}
""")

    # ==============================================================
    # EXPORT RESULTS TO JSON FOR DASHBOARD
    # ==============================================================
    print("\n" + "=" * 100)
    print("EXPORTING RESULTS FOR DASHBOARD")
    print("=" * 100)
    
    dashboard_data = {}
    for name, results in all_results:
        if results:
            key_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").lower()
            clean_results = {}
            for threshold_name, metrics in results.items():
                clean_metrics = {}
                for k, v in metrics.items():
                    # Convert numpy types to native Python types for JSON serialization
                    if hasattr(v, 'item'):
                        clean_metrics[k] = v.item()
                    elif isinstance(v, (np.floating, float)):
                        # Handle NaNs and Infs safely for JSON
                        clean_metrics[k] = None if np.isnan(v) or np.isinf(v) else float(v)
                    elif isinstance(v, (np.integer, int)):
                        clean_metrics[k] = int(v)
                    else:
                        clean_metrics[k] = v
                clean_results[threshold_name] = clean_metrics
            dashboard_data[key_name] = clean_results
            
    json_path = 'dashboard_stats_gat_2017_2018.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dashboard_data, f, indent=4)
    print(f"✓ Evaluation results saved for dashboard to: {json_path}")

    # ==============================================================
    # MODEL CHECKPOINT SAVING
    # ==============================================================
    checkpoint_name = 'model_gat_2017_2018.pth'
    torch.save(model.state_dict(), checkpoint_name)
    print(f"\n✓ Accelerated GAT model saved as: {checkpoint_name}")

if __name__ == "__main__":
    main()

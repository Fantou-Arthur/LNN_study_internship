import pandas as pd
import numpy as np
import torch
import dgl
from metrics_util import evaluate_metrics

def evaluate_dataset(model, df_eval, train_agg_features, country_code_map, scaler_features, scaler_target, batch_size, dataset_name, device):
    """
    Evaluate GAT model on validation/test datasets.
    Optimized for execution on a B200 GPU if available.
    
    Args:
        model (torch.nn.Module): Trained model.
        df_eval (pd.DataFrame): Raw evaluation dataframe.
        train_agg_features (pd.DataFrame): Preprocessed aggregator dataframe.
        country_code_map (dict): Mapping of reporterCode to nodeID.
        scaler_features (MinMaxScaler): Scaler used on the features.
        scaler_target (MinMaxScaler): Scaler used on the target variable.
        batch_size (int): Evaluation batch size.
        dataset_name (str): Evaluation dataset name tag.
        device (torch.device): Device on which computations are run (CUDA/CPU).
        
    Returns:
        tuple: (metrics dictionary, true values, predicted values)
    """
    print(f"\n{'='*120}")
    print(f"EVALUATION: {dataset_name}")
    print(f"{'='*120}")
    
    df_eval = df_eval.copy()
    
    # Fill missing values for physical features with global fallbacks (safe from Copy-on-Write issues)
    for col in ['gdpcap_o', 'pop_o', 'gdpcap_d', 'pop_d', 'dist']:
        global_mean = df_eval[col].mean()
        fallback_val = global_mean if pd.notnull(global_mean) else 0
        df_eval[col] = df_eval[col].fillna(fallback_val)
    
    df_eval['nodeID'] = df_eval['reporterCode'].map(country_code_map)
    df_eval['partnerNodeID'] = df_eval['partnerCode'].map(country_code_map)
    
    df_eval_filtered = df_eval.dropna(subset=['nodeID', 'partnerNodeID']).copy()
    df_eval_filtered['nodeID'] = df_eval_filtered['nodeID'].astype(int)
    df_eval_filtered['partnerNodeID'] = df_eval_filtered['partnerNodeID'].astype(int)
    
    print(f"Original size: {len(df_eval):,}, After filtering unknown countries: {len(df_eval_filtered):,}")
    
    if len(df_eval_filtered) == 0:
        print(f"⚠️ No data points available for {dataset_name}")
        return None, None, None
    
    # Aggregate flows by reporter and commodity
    agg_features_eval = df_eval_filtered.groupby(['refYear', 'reporterCode', 'cmdCode']).agg({
        'primaryValue': 'mean',
        'dist': 'first',
        'gdpcap_d': 'sum',
        'gdpcap_o': 'sum',
        'pop_d': 'sum',
        'pop_o': 'sum'
    }).reset_index()
    
    eval_country_map = {code: i for i, code in enumerate(agg_features_eval['reporterCode'])}
    
    features_eval = agg_features_eval[['primaryValue', 'refYear', 'cmdCode', 'dist', 'gdpcap_d', 'gdpcap_o', 'pop_o', 'pop_d']]
    normalized_features_eval = scaler_features.transform(features_eval)
    normalized_target_eval = scaler_target.transform(agg_features_eval[['primaryValue']])
    
    df_eval_filtered['evalNodeID'] = df_eval_filtered['reporterCode'].map(eval_country_map)
    df_eval_filtered['evalPartnerID'] = df_eval_filtered['partnerCode'].map(eval_country_map)
    df_eval_filtered = df_eval_filtered.dropna(subset=['evalNodeID', 'evalPartnerID'])
    df_eval_filtered['evalNodeID'] = df_eval_filtered['evalNodeID'].astype(int)
    df_eval_filtered['evalPartnerID'] = df_eval_filtered['evalPartnerID'].astype(int)
    
    # Build evaluation DGL graph
    eval_g = dgl.graph((df_eval_filtered['evalNodeID'].to_numpy(), 
                        df_eval_filtered['evalPartnerID'].to_numpy()))
    
    eval_feat_tensor = torch.tensor(normalized_features_eval, dtype=torch.float32)
    eval_g.ndata['feat'] = eval_feat_tensor
    eval_g = dgl.add_self_loop(eval_g)
    
    # Move graph to the accelerated B200 GPU device
    eval_g = eval_g.to(device)
    
    model.eval()
    predictions = []
    num_nodes_eval = eval_g.num_nodes()
    
    # Subgraph mini-batch evaluation
    for i in range(0, num_nodes_eval, batch_size):
        batch_nodes = list(range(i, min(i + batch_size, num_nodes_eval)))
        batch_graph = eval_g.subgraph(batch_nodes)
        
        with torch.no_grad():
            logits = model(batch_graph, batch_graph.ndata['feat'])
            predictions.append(logits.unsqueeze(1).cpu()) # Move back to CPU memory
            
    predictions = torch.cat(predictions, dim=0)
    predictions_np = predictions.view(predictions.size(0), -1).numpy()
    
    y_pred = scaler_target.inverse_transform(predictions_np)
    y_true = scaler_target.inverse_transform(normalized_target_eval)
    
    results = {}
    results['all'] = evaluate_metrics(y_true, y_pred, threshold=None, dataset_name=dataset_name)
    results['gte_1M'] = evaluate_metrics(y_true, y_pred, threshold=1_000_000, dataset_name=dataset_name)
    results['gte_10M'] = evaluate_metrics(y_true, y_pred, threshold=10_000_000, dataset_name=dataset_name)
    results['gte_100M'] = evaluate_metrics(y_true, y_pred, threshold=100_000_000, dataset_name=dataset_name)
    
    return results, y_true, y_pred

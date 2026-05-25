import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from smape_util import calculate_smape

def evaluate_metrics(y_true, y_pred, threshold=None, dataset_name="Dataset"):
    """
    Evaluate regression metrics (MSE, MAE, R², SMAPE) with an optional threshold filter.
    
    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted values.
        threshold (float, optional): If specified, filters results where true value is >= threshold.
        dataset_name (str): Label for the evaluated dataset.
        
    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    if threshold is not None:
        mask = y_true_flat >= threshold
        if mask.sum() < 10:
            print(f"⚠️ Not enough data points >= {threshold:,}")
            return None
        y_true_flat = y_true_flat[mask]
        y_pred_flat = y_pred_flat[mask]
        n_total = len(y_true.flatten())
        pct = len(y_true_flat) / n_total * 100
        scope = f"(>= {threshold:,} USD, {pct:.1f}%)"
    else:
        scope = "(all flows)"
    
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)
    smape = calculate_smape(y_true_flat, y_pred_flat)
    
    print(f"{dataset_name:20s} {scope:35s} | n={len(y_true_flat):>10,} | R²={r2:.4f} | SMAPE={smape:.2f}%")
    
    return {'mse': mse, 'mae': mae, 'r2': r2, 'smape': smape, 'n': len(y_true_flat)}

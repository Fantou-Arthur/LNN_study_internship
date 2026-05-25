import numpy as np

def calculate_smape(y_true, y_pred):
    """
    Calculate SMAPE (0-200%, lower is better).
    
    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted values.
        
    Returns:
        float: The mean SMAPE value.
    """
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator > 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(numerator[mask] / denominator[mask]) * 100

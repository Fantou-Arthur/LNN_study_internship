import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def generate_sea(n_samples=100000, seq_len=16):
    """
    Generates a synthetic dataset with concept drifts (SEA Concept).
    Rules change every 1/4th of the dataset by default.
    """
    print(f"[>] Generating LONG SEA Concept dataset ({n_samples} samples)...")
    
    # We maintain the structure compatible with the requested logic
    # Reshape for sequence models
    num_seqs = n_samples // seq_len
    
    x = np.random.uniform(0, 10, (num_seqs * seq_len, 3)).astype(np.float32)
    y = np.zeros(num_seqs * seq_len, dtype=np.int64)
    
    # Define drift points (5 rules)
    drift_points = [0, num_seqs // 5, 2 * num_seqs // 5, 3 * num_seqs // 5, 4 * num_seqs // 5, num_seqs]
    thresholds = [8.0, 9.5, 7.0, 9.0, 7.5]
    
    for i in range(len(thresholds)):
        start, end = drift_points[i] * seq_len, drift_points[i+1] * seq_len
        thresh = thresholds[i]
        y[start:end] = (x[start:end, 0] + x[start:end, 1] > thresh).astype(np.int64)
        
    x = x / 10.0
    # For simplicity in this study, let's use seq_len=16
    seq_len = 16
    num_seqs = len(x) // seq_len
    x_s = x[:num_seqs*seq_len].reshape(num_seqs, seq_len, 3)
    y_s = y[:num_seqs*seq_len].reshape(num_seqs, seq_len) # We want labels for each step
    
    return torch.tensor(x_s), torch.tensor(y_s), 3, 2

def load_electricity():
    """
    Fetches the Electricity (NSW) dataset from OpenML.
    """
    print("[>] Fetching Electricity (NSW) from OpenML...")
    data = fetch_openml(data_id=151, as_frame=True, parser='auto')
    df = data.frame
    
    # Preprocess
    le = LabelEncoder()
    df['class'] = le.fit_transform(df['class']) # DOWN=0, UP=1
    
    features = ['date', 'day', 'period', 'nswprice', 'nswdemand', 'vicprice', 'vicdemand', 'transfer']
    x = df[features].values.astype(np.float32)
    y = df['class'].values.astype(np.int64)
    
    # Scale features
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    
    seq_len = 24 # 24 periods (half a day)
    num_seqs = len(x) // seq_len
    x_s = x[:num_seqs*seq_len].reshape(num_seqs, seq_len, len(features))
    y_s = y[:num_seqs*seq_len].reshape(num_seqs, seq_len)
    
    return torch.tensor(x_s), torch.tensor(y_s), len(features), 2

def load_oil_price():
    """
    Loads WTI Crude Oil price data from FRED.
    Splits at the end of 2018 for long-term forecasting test.
    """
    print("[>] Fetching WTI Crude Oil Price from FRED...")
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILWTICO"
    try:
        df = pd.read_csv(url)
        df.columns = ["date", "price"]
        # Handle missing values (FRED uses '.')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['price'] = df['price'].ffill()
        df = df.dropna().reset_index(drop=True)
        
        prices = df['price'].values.astype(np.float32)
        
        # Normalize
        p_min, p_max = prices.min(), prices.max()
        prices_norm = (prices - p_min) / (p_max - p_min)
        
        # Create sequences (lookback of 30 days)
        seq_len = 30
        x_list, y_list = [], []
        for i in range(len(prices_norm) - seq_len):
            x_list.append(prices_norm[i:i+seq_len])
            y_list.append(prices_norm[i+seq_len])
            
        x = torch.from_numpy(np.array(x_list)).unsqueeze(-1) # (N, 30, 1)
        y = torch.from_numpy(np.array(y_list)).unsqueeze(-1) # (N, 1)
        
        # Find index for 2019 split
        split_idx = df[df['date'].str.startswith('2019')].index[0] - seq_len
        
        return x, y, 1, 1, False, split_idx, df['date'].tolist()[seq_len:], p_min, p_max
    except Exception as e:
        print(f"Error loading oil data: {e}")
        return None

def load_nyc_taxi():
    """
    Fetches the NYC Taxi dataset from Numenta Anomaly Benchmark (NAB).
    Features: hour, day_of_week, value (t)
    Target: value (t+1)
    """
    print("[>] Fetching NYC Taxi dataset...")
    url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv"
    df = pd.read_csv(url)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Normalize
    scaler = MinMaxScaler()
    df[['value', 'hour', 'day_of_week']] = scaler.fit_transform(df[['value', 'hour', 'day_of_week']])
    
    x_vals = df[['value', 'hour', 'day_of_week']].values.astype(np.float32)
    y_vals = df['value'].values.astype(np.float32)
    
    # Task: Predict next value
    seq_len = 48 # 24 hours (data is every 30 mins)
    num_seqs = (len(x_vals) - 1) // seq_len
    x_s = np.zeros((num_seqs, seq_len, 3), dtype=np.float32)
    y_s = np.zeros((num_seqs, seq_len, 1), dtype=np.float32)
    
    for i in range(num_seqs):
        start = i * seq_len
        x_s[i] = x_vals[start:start+seq_len]
        y_s[i] = y_vals[start+1:start+seq_len+1].reshape(-1, 1)
        
    return torch.tensor(x_s), torch.tensor(y_s), 3, 1

if __name__ == "__main__":
    # Test
    x, y, inp, out = generate_sea()
    print(f"SEA: x={x.shape}, y={y.shape}")
    x, y, inp, out = load_electricity()
    print(f"Electricity: x={x.shape}, y={y.shape}")
    x, y, inp, out = load_nyc_taxi()
    print(f"NYC Taxi: x={x.shape}, y={y.shape}")

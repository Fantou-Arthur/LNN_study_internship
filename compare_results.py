import os
import json
import pandas as pd
import numpy as np

def load_all_results(results_dir="results"):
    """
    Scans the results directory and aggregates training stats (time, flops) 
    and robustness data (accuracy/mse over eval sparsity).
    """
    data = {}
    
    # 1. Collect training stats from model-specific folders
    for root, dirs, files in os.walk(results_dir):
        if "comparison" in root: continue # Skip comparison folder for now
        for file in files:
            if file.endswith(".json"):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r') as f:
                        res = json.load(f)
                    
                    config = res.get("config", {})
                    model_type = config.get("model_type", "unknown").upper()
                    dataset = root.split(os.sep)[-2] if os.sep in root else "unknown"
                    
                    # Identify training config
                    sp_train = int(config.get("sparsity", {}).get("train", 0) * 100)
                    sp_mode = config.get("sparsity", {}).get("mode", "random")
                    config_key = f"{dataset}_{sp_train}_{sp_mode}"
                    
                    if config_key not in data:
                        data[config_key] = {"dataset": dataset, "train_sp": sp_train, "mode": sp_mode, "models": {}}
                    
                    if model_type not in data[config_key]["models"]:
                        data[config_key]["models"][model_type] = {}
                        
                    m_entry = data[config_key]["models"][model_type]
                    m_entry["train_time"] = res.get("execution", {}).get("total_duration_seconds", 0)
                    m_entry["flops"] = res.get("epochs", [{}])[0].get("flops", 0) if res.get("epochs") else 0
                    
                except Exception as e:
                    print(f"Error reading {file}: {e}")

    # 2. Collect robustness data from comparison folder
    comp_dir = os.path.join(results_dir, "comparison")
    if os.path.exists(comp_dir):
        for config_folder in os.listdir(comp_dir):
            folder_path = os.path.join(comp_dir, config_folder)
            if not os.path.isdir(folder_path): continue
            
            rob_path = os.path.join(folder_path, "results_robustness_random.json")
            if os.path.exists(rob_path):
                try:
                    with open(rob_path, 'r') as f:
                        rob_data = json.load(f)
                    
                    # rob_data is { dataset: { model: { sparsity: [], metrics: [] } } }
                    for dataset, models in rob_data.items():
                        # training config is in folder name (e.g. "0_random")
                        config_key = f"{dataset}_{config_folder}"
                        
                        if config_key not in data: continue
                        
                        for m_name, m_rob in models.items():
                            m_upper = m_name.upper()
                            if m_upper in data[config_key]["models"]:
                                data[config_key]["models"][m_upper]["robustness"] = {
                                    "sparsity": [int(s*100) for s in m_rob["sparsity"]],
                                    "metrics": m_rob["metrics"],
                                    "is_classification": m_rob.get("is_classification", False)
                                }
                except Exception as e:
                    print(f"Error reading robustness in {config_folder}: {e}")

    return data

def main():
    print("[>] Aggregating all results for dashboard...")
    data = load_all_results()
    
    output_path = "dashboard_stats.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
        
    print(f"[OK] Consolidated data saved to {output_path}")

if __name__ == "__main__":
    main()

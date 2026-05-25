import numpy as np
import json
import os
from drift_loaders import load_oil_price

def run_export_for_interactive():
    print("=== EXPORTING RAW DATA FOR INTERACTIVE DASHBOARD ===")
    
    # 1. Load Data
    res_loader = load_oil_price()
    _, y, _, _, _, _, dates, p_min, p_max = res_loader
    
    # We also want model predictions to compare them
    results_path = "results/oil_modern_data.js"
    models_data = {}
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            content = f.read().replace("var oilModernData = ", "").rstrip(";")
            modern_json = json.loads(content)
            # Use Sparsity 0% results
            study_0 = next(s for s in modern_json["studies"] if s["sparsity"] == 0)
            for name, m in study_0["models"].items():
                models_data[name] = m["predictions"]

    # Export everything needed for JS-side calculation
    export_data = {
        "dates": dates.tolist() if isinstance(dates, np.ndarray) else list(dates),
        "actual": y.flatten().tolist(),
        "models": models_data,
        "p_min": float(p_min),
        "p_max": float(p_max)
    }

    output_path = "results/paper_drift_metrics.js"
    os.makedirs("results", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("var paperRawData = " + json.dumps(export_data) + ";")
    
    print(f"\n[OK] Raw data exported to {output_path}")
    print("The dashboard will now handle calculations locally.")

if __name__ == "__main__":
    run_export_for_interactive()

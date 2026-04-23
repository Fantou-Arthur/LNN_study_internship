import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# --- COULEURS ANSI ---
C_BLUE = "\033[94m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_BOLD = "\033[1m"
C_END = "\033[0m"

def load_results(results_dir="results"):
    data = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith(".json"):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r') as f:
                        res = json.load(f)
                    
                    model_type = root.split(os.sep)[-2] if os.sep in root else "unknown"
                    dataset = root.split(os.sep)[-1]
                    eval_data = res.get("evaluation", {})
                    
                    entry = {
                        "model": model_type.upper(),
                        "dataset": dataset,
                        "accuracy": eval_data.get("test_accuracy_pct") or eval_data.get("accuracy"),
                        "test_loss": eval_data.get("test_loss") or res.get("test_loss"),
                        "time": res.get("execution", {}).get("total_duration_seconds", 0),
                        "flops": res.get("epochs", [{}])[0].get("flops", 0) if res.get("epochs") else 0
                    }
                    data.append(entry)
                except Exception as e:
                    print(f"{C_RED}Erreur lecture {file}: {e}{C_END}")
    return pd.DataFrame(data)

def generate_plots(df):
    if df.empty: return
    os.makedirs("comparison_plots", exist_ok=True)
    
    for dataset in df['dataset'].unique():
        ds_df = df[df['dataset'] == dataset]
        models = ds_df['model'].tolist()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Comparaison Benchmarking - Dataset: {dataset}", fontsize=16)

        # 1. Accuracy / Loss
        is_classif = ds_df['accuracy'].notna().any()
        if is_classif:
            values = ds_df['accuracy'].fillna(0)
            axes[0].bar(models, values, color='#3498db')
            axes[0].set_title("Précision (Test %)")
        else:
            values = ds_df['test_loss'].fillna(0)
            axes[0].bar(models, values, color='#e74c3c')
            axes[0].set_title("Perte (MSE Test)")

        # 2. Execution Time
        axes[1].bar(models, ds_df['time'], color='#2ecc71')
        axes[1].set_title("Temps d'exécution (s)")

        # 3. Complexity (FLOPS)
        axes[2].bar(models, ds_df['flops'], color='#f1c40f')
        axes[2].set_title("Complexité (FLOPS/sample)")
        axes[2].set_yscale('log')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"comparison_plots/comparison_{dataset}.png")
        print(f"{C_GREEN}[OK]{C_END} Dashboard généré pour {C_BOLD}{dataset}{C_END}")

if __name__ == "__main__":
    print(f"\n{C_BOLD}{C_BLUE}=== ANALYSEUR DE RÉSULTATS ==={C_END}\n")
    df_results = load_results()
    if not df_results.empty:
        generate_plots(df_results)
        print(f"\n{C_YELLOW}{C_BOLD}--- Tableau Récapitulatif Global ---{C_END}")
        summary = df_results.groupby(['dataset', 'model'])[['accuracy', 'test_loss', 'time', 'flops']].mean()
        print(summary)
        print(f"\n{C_BLUE}{C_BOLD}=============================={C_END}")
    else:
        print(f"{C_RED}Aucune donnée trouvée dans 'results/'.{C_END}")

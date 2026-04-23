import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- COULEURS ANSI ---
C_BLUE = "\033[94m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_BOLD = "\033[1m"
C_END = "\033[0m"

def load_results(results_dir="results"):
    all_data = []
    histories = {} 
    
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith(".json"):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r') as f:
                        res = json.load(f)
                    
                    model_type = root.split(os.sep)[-2].upper() if os.sep in root else "UNKNOWN"
                    dataset = root.split(os.sep)[-1]
                    eval_data = res.get("evaluation", {})
                    
                    entry = {
                        "model": model_type,
                        "dataset": dataset,
                        "accuracy": eval_data.get("test_accuracy_pct"),
                        "test_loss": eval_data.get("test_loss"),
                        "time": res.get("execution", {}).get("total_duration_seconds", 0),
                        "flops_per_sample": res.get("epochs", [{}])[0].get("flops", 0) if res.get("epochs") else 0
                    }
                    all_data.append(entry)
                    
                    if dataset not in histories: histories[dataset] = {}
                    
                    epochs_list = res.get("epochs", [])
                    if epochs_list:
                        train_losses = [e["loss"] for e in epochs_list]
                        test_losses = [e.get("test_loss") for e in epochs_list]
                        histories[dataset][model_type] = {
                            "train_losses": train_losses,
                            "test_losses": test_losses,
                            "flops_per_epoch": entry["flops_per_sample"] 
                        }
                        
                except Exception as e:
                    print(f"{C_RED}Erreur lecture {file}: {e}{C_END}")
    
    return pd.DataFrame(all_data), histories

def generate_plots(df, histories):
    if df.empty: return
    os.makedirs("comparison_plots", exist_ok=True)
    
    # Palette de couleurs fixe pour la cohérence entre train/test
    color_palette = plt.cm.get_cmap('tab10', 10)
    
    for dataset in df['dataset'].unique():
        ds_df = df[df['dataset'] == dataset]
        
        # --- Figure 1: Bar Charts (Stats Finales) ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Stats Finales - Dataset: {dataset}", fontsize=16, fontweight='bold')
        
        models = ds_df['model'].tolist()
        if ds_df['accuracy'].notna().any():
            axes[0].bar(models, ds_df['accuracy'].fillna(0), color='#3498db')
            axes[0].set_title("Précision Finale (%)")
        else:
            axes[0].bar(models, ds_df['test_loss'].fillna(0), color='#e74c3c')
            axes[0].set_title("Perte Finale (MSE)")
            
        axes[1].bar(models, ds_df['time'], color='#2ecc71')
        axes[1].set_title("Temps Total (s)")
        
        axes[2].bar(models, ds_df['flops_per_sample'], color='#f1c40f')
        axes[2].set_title("Complexité (FLOPS/sample)")
        axes[2].set_yscale('log')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"comparison_plots/stats_{dataset}.png")
        
        # --- Figure 2: Courbes de Convergence (Train vs Test) ---
        if dataset in histories:
            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
            fig2.suptitle(f"Courbes de Convergence (Train vs Test) - Dataset: {dataset}", fontsize=16, fontweight='bold')
            
            for i, (model_name, h_data) in enumerate(histories[dataset].items()):
                color = color_palette(i)
                train_losses = h_data["train_losses"]
                test_losses = h_data["test_losses"]
                epochs = np.arange(1, len(train_losses) + 1)
                
                # Plot 1: Loss vs Epoch
                ax1.plot(epochs, train_losses, color=color, label=f"{model_name} (Train)", linewidth=2)
                if any(t is not None for t in test_losses):
                    ax1.plot(epochs, [t for t in test_losses if t is not None], color=color, label=f"{model_name} (Test)", linestyle='--', alpha=0.7)
                
                # Plot 2: Loss vs Cumulative Complexity
                cum_flops = epochs * h_data["flops_per_epoch"]
                ax2.plot(cum_flops, train_losses, color=color, label=f"{model_name} (Train)", linewidth=2)
                if any(t is not None for t in test_losses):
                    ax2.plot(cum_flops, [t for t in test_losses if t is not None], color=color, label=f"{model_name} (Test)", linestyle='--', alpha=0.7)

            ax1.set_title("Convergence par Époque")
            ax1.set_xlabel("Époque")
            ax1.set_ylabel("Perte")
            ax1.set_yscale('log')
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize='small', ncol=2)

            ax2.set_title("Efficacité Énergétique (Loss vs Travail)")
            ax2.set_xlabel("Complexité Cumulée (FLOPS relatifs)")
            ax2.set_ylabel("Perte")
            ax2.set_yscale('log')
            ax2.set_xscale('log')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize='small', ncol=2)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f"comparison_plots/convergence_{dataset}.png")
            
        print(f"{C_GREEN}[OK]{C_END} Dashboards générés pour {C_BOLD}{dataset}{C_END}")

if __name__ == "__main__":
    print(f"\n{C_BOLD}{C_BLUE}=== ANALYSEUR DE RÉSULTATS (V3 - TRAIN vs TEST) ==={C_END}\n")
    df_results, histories = load_results()
    if not df_results.empty:
        generate_plots(df_results, histories)
        print(f"\n{C_YELLOW}{C_BOLD}--- Résumé des Performances ---{C_END}")
        summary = df_results.groupby(['dataset', 'model'])[['accuracy', 'test_loss', 'time', 'flops_per_sample']].mean()
        print(summary)
    else:
        print(f"{C_RED}Aucune donnée trouvée.{C_END}")

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

def load_all_results(base_path="results"):
    results_list = []
    if not os.path.exists(base_path):
        print(f"Erreur: Le dossier {base_path} n'existe pas.")
        return None

    # Parcourir results/<model>/<dataset>/<file>.json
    for model_name in os.listdir(base_path):
        model_path = os.path.join(base_path, model_name)
        if not os.path.isdir(model_path): continue
        
        for dataset_name in os.listdir(model_path):
            dataset_path = os.path.join(model_path, dataset_name)
            if not os.path.isdir(dataset_path): continue
            
            for file in os.listdir(dataset_path):
                if file.endswith(".json"):
                    with open(os.path.join(dataset_path, file), "r") as f:
                        try:
                            data = json.load(f)
                            # Extraire les infos utiles
                            res = {
                                "model": model_name.upper(),
                                "dataset": dataset_name,
                                "filename": file,
                                "units": data["config"].get("units"),
                                "layers": data["config"].get("layers"),
                                "device": data["config"].get("device"),
                                "time": data["execution"].get("total_duration_seconds", 0),
                                "test_loss": data["evaluation"].get("test_loss"),
                                "accuracy": data["evaluation"].get("test_accuracy_pct"),
                                "f1": data["evaluation"].get("f1_score_macro"),
                                "mae": data["evaluation"].get("mae"),
                                "flops": data["epochs"][0].get("flops", 0) if data.get("epochs") else 0
                            }
                            results_list.append(res)
                        except Exception as e:
                            print(f"Erreur lecture {file}: {e}")
    return pd.DataFrame(results_list)

def plot_dataset_comparison(df, dataset_name, output_dir="comparison_plots"):
    df_ds = df[df["dataset"] == dataset_name].copy()
    if df_ds.empty: return

    # Si plusieurs tests pour un modèle, on prend le dernier ou le meilleur ?
    # Ici, on va grouper par modèle et prendre la moyenne ou le max pour simplifier
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Comparaison des Modèles - Dataset: {dataset_name.upper()}", fontsize=20, fontweight='bold')

    # 1. Performance (Accuracy ou MSE)
    is_classification = df_ds["accuracy"].notnull().any()
    ax1 = axes[0, 0]
    if is_classification:
        perf_data = df_ds.groupby("model")["accuracy"].max()
        perf_data.plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
        ax1.set_ylabel("Précision (%)")
        ax1.set_title("Meilleure Précision par Modèle")
    else:
        perf_data = df_ds.groupby("model")["test_loss"].min()
        perf_data.plot(kind='bar', ax=ax1, color='salmon', edgecolor='black')
        ax1.set_ylabel("MSE (Perte)")
        ax1.set_title("Plus basse perte (MSE) par Modèle")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # 2. Temps d'entraînement
    ax2 = axes[0, 1]
    time_data = df_ds.groupby("model")["time"].mean()
    time_data.plot(kind='bar', ax=ax2, color='lightgreen', edgecolor='black')
    ax2.set_ylabel("Temps (secondes)")
    ax2.set_title("Temps d'entraînement moyen")
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # 3. Complexité (FLOPS)
    ax3 = axes[1, 0]
    flops_data = df_ds.groupby("model")["flops"].mean()
    flops_data.plot(kind='bar', ax=ax3, color='orange', edgecolor='black')
    ax3.set_ylabel("FLOPS / sample")
    ax3.set_yscale('log') # Log scale car les différences peuvent être énormes
    ax3.set_title("Complexité Matérielle (Échelle Log)")
    ax3.grid(axis='y', linestyle='--', alpha=0.7)

    # 4. Compromis : Précision vs FLOPS
    ax4 = axes[1, 1]
    for model in df_ds["model"].unique():
        sub = df_ds[df_ds["model"] == model]
        metric = sub["accuracy"] if is_classification else sub["test_loss"]
        ax4.scatter(sub["flops"], metric, label=model, s=100, alpha=0.7)
    
    ax4.set_xscale('log')
    ax4.set_xlabel("FLOPS / sample")
    ax4.set_ylabel("Précision (%)" if is_classification else "MSE")
    ax4.set_title("Compromis : Efficacité vs Performance")
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, f"comparison_{dataset_name}.png")
    plt.savefig(save_path)
    print(f"[OK] Dashboard généré pour {dataset_name} -> {save_path}")
    plt.close()

if __name__ == "__main__":
    print("\n--- Analyseur de Résultats de Benchmarking ---")
    df = load_all_results()
    
    if df is not None and not df.empty:
        datasets = df["dataset"].unique()
        print(f"Datasets détectés : {', '.join(datasets)}")
        
        for ds in datasets:
            plot_dataset_comparison(df, ds)
            
        # Affichage d'un tableau résumé dans la console
        print("\n--- Tableau Récapitulatif Global ---")
        summary = df.groupby(["dataset", "model"]).agg({
            "accuracy": "max",
            "test_loss": "min",
            "time": "mean",
            "flops": "mean"
        }).round(4)
        print(summary)
    else:
        print("Aucun résultat JSON trouvé dans le dossier 'results/'.")

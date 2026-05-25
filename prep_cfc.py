import pandas as pd
import numpy as np
import torch

file_path = r"C:\Users\afant\Downloads\comtrade_sampled_10pct.csv"

def build_sequences():
    print("Chargement des données pour le CfC...")
    df = pd.read_csv(file_path)
    
    # On se concentre sur les flux globaux par paire de pays
    # (On agrège tous les produits pour simplifier la séquence temporelle)
    df_grouped = df.groupby(['reporterCode', 'partnerCode', 'period'])['primaryValue'].sum().reset_index()
    
    # Pivot pour avoir les années en colonnes
    # Index: (Reporter, Partner) | Colonnes: 2017, 2018, ..., 2023
    df_pivot = df_grouped.pivot(index=['reporterCode', 'partnerCode'], columns='period', values='primaryValue')
    
    # Remplissage des valeurs manquantes par 0 (ou interpolation)
    df_pivot = df_pivot.fillna(0)
    
    print(f"Nombre de paires (séquences) identifiées : {len(df_pivot)}")
    
    # Transformation en tenseur (Samples, TimeSteps, Features)
    # Ici Features = 1 (la valeur de l'export)
    data_array = df_pivot.values
    samples = data_array.shape[0]
    timesteps = data_array.shape[1]
    
    # Reshape pour PyTorch (Batch, Seq_Len, Input_Size)
    sequences = torch.tensor(data_array, dtype=torch.float).view(samples, timesteps, 1)
    
    print(f"\n--- Structure des Séquences CfC ---")
    print(f"Shape finale : {sequences.shape} (Paires, Années, Valeur)")
    
    # Exemple de Target : Prédire l'année 2023 à partir de 2017-2022
    X = sequences[:, :-1, :] # 2017 -> 2022
    y = sequences[:, -1, :]  # 2023
    
    print(f"X (Input) shape : {X.shape}")
    print(f"y (Target) shape : {y.shape}")
    
    return X, y

if __name__ == "__main__":
    X, y = build_sequences()
    # torch.save({'X': X, 'y': y}, 'comtrade_sequences.pt')

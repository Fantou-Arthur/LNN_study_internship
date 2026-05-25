import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data

file_path = r"C:\Users\afant\Downloads\comtrade_sampled_10pct.csv"

def build_graph():
    print("Chargement des données pour le GAT...")
    df = pd.read_csv(file_path)
    
    # Pour un premier test, on se concentre sur une seule année (ex: 2022)
    # Sinon le graphe aura des arêtes redondantes pour chaque année
    df_year = df[df['period'] == 2022].copy()
    
    # 1. Mapping des codes pays en indices continus [0, N-1]
    all_countries = pd.concat([df_year['reporterCode'], df_year['partnerCode']]).unique()
    country_to_idx = {code: i for i, code in enumerate(all_countries)}
    num_nodes = len(all_countries)
    
    print(f"Nombre de pays (nœuds) : {num_nodes}")
    
    # 2. Création des arêtes (Edges)
    # Source: reporter, Cible: partner
    edge_index = torch.tensor([
        df_year['reporterCode'].map(country_to_idx).values,
        df_year['partnerCode'].map(country_to_idx).values
    ], dtype=torch.long)
    
    # 3. Caractéristiques des arêtes (Edge Attributes)
    # On peut mettre la valeur de l'exportation et la distance
    edge_attr = torch.tensor(df_year[['primaryValue', 'dist']].values, dtype=torch.float)
    
    # 4. Caractéristiques des nœuds (Node Features)
    # On récupère le PIB et la Population par pays (on prend la moyenne sur l'année)
    node_features = np.zeros((num_nodes, 4)) # [gdp_o, pop_o, gdp_d, pop_d] par exemple
    
    # On remplit avec les données disponibles dans le CSV
    # (Note: c'est une simplification, idéalement on croise avec une table pays)
    for code, idx in country_to_idx.items():
        # On cherche les infos de ce pays quand il est 'reporter' (o)
        info = df_year[df_year['reporterCode'] == code].iloc[:1]
        if not info.empty:
            node_features[idx, 0] = info['gdp_o'].values[0]
            node_features[idx, 1] = info['pop_o'].values[0]
            
    x = torch.tensor(node_features, dtype=torch.float)
    
    # 5. Création de l'objet Data de PyTorch Geometric
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    print("\n--- Structure du Graphe ---")
    print(data)
    return data

if __name__ == "__main__":
    try:
        import torch_geometric
        graph_data = build_graph()
        # torch.save(graph_data, 'comtrade_graph_2022.pt')
    except ImportError:
        print("Erreur : PyTorch Geometric n'est pas installé. Fais : pip install torch-geometric")

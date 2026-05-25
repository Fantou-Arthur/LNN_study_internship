import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import os

csv_path = r"C:\Users\afant\Downloads\comtradeExports_updatedH5[240924]-wb.csv\comtradeExports_updatedH5[240924]-wb.csv"
parquet_path = "trade_data.parquet"

if not os.path.exists(csv_path):
    print(f"❌ Fichier CSV non trouvé : {csv_path}")
else:
    if os.path.exists(parquet_path):
        os.remove(parquet_path) # On repart sur du propre
        
    print("🚀 RE-CONVERSION ROBUSTE (PyArrow Writer)...")
    
    # Lecture par blocs de 2M
    reader = pd.read_csv(csv_path, chunksize=2000000, low_memory=False)
    
    writer = None
    for chunk in tqdm(reader, total=28):
        # On force les types pour éviter les conflits lors de l'écriture
        chunk['period'] = chunk['period'].astype(str)
        
        # Conversion en Table PyArrow
        table = pa.Table.from_pandas(chunk)
        
        if writer is None:
            # On initialise l'écrivain avec le schéma du premier bloc
            writer = pq.ParquetWriter(parquet_path, table.schema, compression='snappy')
        
        writer.write_table(table)

    if writer:
        writer.close()

    print(f"\n✅ RE-CONVERSION terminée !")
    print(f"Vérification : {os.path.getsize(parquet_path) / 1e9:.2f} Go sur le disque.")

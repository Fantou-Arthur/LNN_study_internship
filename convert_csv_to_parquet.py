import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
from tqdm import tqdm

# Chemins
CSV_PATH = r"C:\Users\afant\Documents\Denmark Internship\liquid_time_constant_networks-master\Monthly_dataset\Comtrade2016-23.csv\Comtrade2016-23.csv"
OUTPUT_PATH = r"C:\Users\afant\Documents\Denmark Internship\liquid_time_constant_networks-master\liquid_time_constant_networks-master\trade_data_monthly.parquet"

def convert_monthly_to_parquet():
    if not os.path.exists(CSV_PATH):
        print(f"❌ Erreur : Fichier CSV non trouvé à {CSV_PATH}")
        return

    print(f"🚀 Conversion optimisée (Stream PyArrow)...")
    print(f"📂 Source : {CSV_PATH}")
    
    dtypes = {
        'refYear': 'int32',
        'refMonth': 'int32',
        'reporterCode': 'int32',
        'partnerCode': 'int32',
        'cmdCode': 'int32',
        'primaryValue': 'float32'
    }

    chunksize = 2_000_000 
    reader = pd.read_csv(CSV_PATH, dtype=dtypes, chunksize=chunksize)
    
    writer = None
    
    try:
        for chunk in tqdm(reader, desc="Progression"):
            # Création de la colonne period format YYYYMM
            chunk['period'] = chunk['refYear'] * 100 + chunk['refMonth']
            
            # Sélection des colonnes essentielles
            chunk = chunk[['period', 'reporterCode', 'partnerCode', 'cmdCode', 'primaryValue']]
            
            # Conversion en table PyArrow
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            
            # Initialisation du Writer au premier passage
            if writer is None:
                writer = pq.ParquetWriter(OUTPUT_PATH, table.schema, compression='snappy')
            
            # Écriture du chunk
            writer.write_table(table)
            
    except Exception as e:
        print(f"❌ Erreur pendant la conversion : {e}")
    finally:
        if writer:
            writer.close()

    if os.path.exists(OUTPUT_PATH):
        print(f"✅ Conversion terminée ! Fichier : {OUTPUT_PATH}")
        print(f"📊 Taille finale : {os.path.getsize(OUTPUT_PATH) / (1024**3):.2f} GB")

if __name__ == "__main__":
    convert_monthly_to_parquet()

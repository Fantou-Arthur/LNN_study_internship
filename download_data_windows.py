import os
import urllib.request
import zipfile
import gzip
import shutil

def download_and_extract():
    # Création de l'arborescence
    base_data_dir = "data"
    os.makedirs(base_data_dir, exist_ok=True)

    datasets = [
        {
            "name": "har",
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
            "type": "zip"
        },
        {
            "name": "gesture",
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00302/gesture_phase_dataset.zip",
            "type": "zip"
        },
        {
            "name": "occupancy",
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip",
            "type": "zip"
        },
        {
            "name": "traffic",
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz",
            "type": "gz",
            "filename": "Metro_Interstate_Traffic_Volume.csv"
        }
    ]

    for ds in datasets:
        ds_dir = os.path.join(base_data_dir, ds["name"])
        if os.path.exists(ds_dir):
            print(f"[*] Dataset {ds['name']} déjà présent, on passe...")
            continue
        
        os.makedirs(ds_dir, exist_ok=True)
        temp_file = os.path.join(base_data_dir, f"temp_{ds['name']}")
        
        print(f"[>] Téléchargement de {ds['name']}...")
        try:
            urllib.request.urlretrieve(ds["url"], temp_file)
            
            if ds["type"] == "zip":
                print(f"[!] Extraction ZIP pour {ds['name']}...")
                with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                    zip_ref.extractall(ds_dir)
            
            elif ds["type"] == "gz":
                print(f"[!] Extraction GZ pour {ds['name']}...")
                out_path = os.path.join(ds_dir, ds["filename"])
                with gzip.open(temp_file, 'rb') as f_in:
                    with open(out_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            
            os.remove(temp_file)
            print(f"[+] {ds['name']} terminé !\n")
            
        except Exception as e:
            print(f"[X] Erreur pour {ds['name']}: {e}")

if __name__ == "__main__":
    print("--- Téléchargement des datasets pour Liquid Time-Constant Networks ---")
    download_and_extract()
    print("--- Terminé ! ---")

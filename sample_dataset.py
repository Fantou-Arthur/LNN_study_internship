import os
import random
import time

input_path = r"C:\Users\afant\Downloads\comtradeExports_updatedH5[240924]-wb.csv\comtradeExports_updatedH5[240924]-wb.csv"
output_path = r"C:\Users\afant\Downloads\comtrade_sampled_10pct.csv"

sample_rate = 0.10  # 10%

def sample_csv():
    if not os.path.exists(input_path):
        print(f"Erreur : Le fichier source n'a pas été trouvé à l'adresse : {input_path}")
        return

    file_size = os.path.getsize(input_path)
    print(f"Fichier source trouvé : {file_size / (1024**3):.2f} Go")
    print(f"Échantillonnage à {sample_rate*100}% vers {output_path}...")

    start_time = time.time()
    
    try:
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f_in:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                # Header
                header = f_in.readline()
                if not header:
                    print("Erreur : Le fichier est vide.")
                    return
                f_out.write(header)
                
                count = 0
                saved = 0
                
                for line in f_in:
                    count += 1
                    if random.random() < sample_rate:
                        f_out.write(line)
                        saved += 1
                    
                    if count % 1000000 == 0:
                        elapsed = time.time() - start_time
                        print(f"Lignes traitées : {count:,} | Sauvegardées : {saved:,} | Temps écoulé : {elapsed:.1f}s")
        
        total_time = time.time() - start_time
        print(f"\nTerminé avec succès !")
        print(f"Total lignes traitées : {count:,}")
        print(f"Total lignes sauvegardées : {saved:,}")
        print(f"Temps total : {total_time:.1f}s")
        print(f"Taille finale approximative : {os.path.getsize(output_path) / (1024**2):.2f} Mo")

    except Exception as e:
        print(f"Une erreur est survenue : {e}")

if __name__ == "__main__":
    sample_csv()

import json

import pandas as pd
import redis
from tqdm import tqdm

from config.project_config import App_config, CacheDb_settings

redis_client = redis.Redis(host=CacheDb_settings.host, 
                           port=CacheDb_settings.port, 
                           db=CacheDb_settings.db, 
                           decode_responses=True,
                           password=CacheDb_settings.password.get_secret_value())

ID = "SK_ID_CURR"
BATCH_SIZE = 10000

def load_csv_to_redis(csv_path : str):
    '''
    Chargement des données historique agrégées dans redis
    '''
    print("Lecture du fichier csv...")
    df_agg = pd.read_csv(csv_path)

    print("Conversion des np.nan en None pour compatibilité JSON (redis)...")
    df_agg = df_agg.astype(object).where(pd.notnull(df_agg), None)

    print("Conversion en dictionnaires pour insertion rapide...")
    records = df_agg.to_dict(orient='records')

    print("Insertion dans Redis...")
    pipe = redis_client.pipeline(transaction=False)
   
    total_rows = len(records)

    for idx, data in tqdm(enumerate(records), total=total_rows, desc="Injection Redis"):
        client_id = str(int(data.pop(ID)))
        
        pipe.set(f"client:{client_id}", json.dumps(data))

        if (idx + 1) % BATCH_SIZE == 0:
            pipe.execute()
            tqdm.write(f"   -> Batch exécuté : {idx + 1} / {total_rows} clients")

    pipe.execute()
    print("Injection dans Redis terminée avec succès!")

if __name__ == "__main__":
    load_csv_to_redis(App_config.paths.output_aggregation)
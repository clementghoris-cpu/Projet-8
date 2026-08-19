import json

import numpy as np
import pandas as pd
import redis

from config.project_config import App_config, CacheDb_settings

redis_client = redis.Redis(host=CacheDb_settings.host, 
                port=CacheDb_settings.port,
                db=CacheDb_settings.db, 
                decode_responses=True, 
                password=CacheDb_settings.password.get_secret_value())

with open(App_config.paths.model_columns, "r", encoding="utf-8") as f:
    EXPECTED_COLUMNS = json.load(f)

def get_redis_features_batch(sk_ids: list[int]) -> list[dict]:
    """Récupère les agrégations historiques du client depuis Redis."""
    try:
        pipe = redis_client.pipeline()

        for sk_id in sk_ids:
            pipe.get(f"client:{sk_id}")

        results = pipe.execute()
    except Exception:   # noqa: BLE001
        return [{} for _ in sk_ids]

    features = []
    for res in results:
        if res:
            features.append(json.loads(res))
        else:
            features.append({})

    return features


def preprocess_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage de base et création des ratios primaires."""
    df_clean = df.copy()

    # Correction anomalie DAYS_EMPLOYED (365243 = retraités/sans emploi)
    if 'DAYS_EMPLOYED' in df_clean.columns :
        df_clean['DAYS_EMPLOYED_ANOM'] = (df_clean['DAYS_EMPLOYED'] == 365243).astype(int)
        df_clean['DAYS_EMPLOYED'] = df_clean['DAYS_EMPLOYED'].replace(365243, np.nan)

    # Conversion en années (plus lisible)
    if 'DAYS_BIRTH' in df_clean.columns:
        df_clean['AGE_YEARS'] = (-df_clean['DAYS_BIRTH'] / 365).astype(int)
    if 'DAYS_EMPLOYED' in df_clean.columns:
        df_clean['YEARS_EMPLOYED'] = (-df_clean['DAYS_EMPLOYED'] / 365)

    # Feature engineering
    if all(col in df_clean.columns for col in ['AMT_CREDIT', 'AMT_INCOME_TOTAL']):
        df_clean['CREDIT_INCOME_RATIO'] = (df_clean['AMT_CREDIT'] / df_clean['AMT_INCOME_TOTAL'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    if all(col in df_clean.columns for col in ['AMT_ANNUITY', 'AMT_INCOME_TOTAL']):
        df_clean['ANNUITY_INCOME_RATIO'] = (df_clean['AMT_ANNUITY'] / df_clean['AMT_INCOME_TOTAL'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    if all(col in df_clean.columns for col in ['AMT_CREDIT', 'AMT_GOODS_PRICE']):
        df_clean['CREDIT_GOODS_RATIO'] = (df_clean['AMT_CREDIT'] / df_clean['AMT_GOODS_PRICE'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    return df_clean

def add_polynomial_and_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Génère des features polynomiales (puissances) et d'interaction (multiplications/divisions)
    """
    df_fe = df.copy()
    
    # Puissances polynomiales sur les colonnes EXT_SOURCE
    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    for col in ext_cols:
        if col in df_fe.columns:
            df_fe[f"{col}_SQ"] = df_fe[col] ** 2
            df_fe[f"{col}_CUBIC"] = df_fe[col] ** 3

    # Interactions par Multiplication
    interaction_multiplications = {
        'EXT1_EXT2': ('EXT_SOURCE_1', 'EXT_SOURCE_2'),
        'EXT1_EXT3': ('EXT_SOURCE_1', 'EXT_SOURCE_3'),
        'EXT2_EXT3': ('EXT_SOURCE_2', 'EXT_SOURCE_3'),
        'EXT2_INCOME_RATIO': ('EXT_SOURCE_2', 'CREDIT_INCOME_RATIO'),
        'EXT3_INCOME_RATIO': ('EXT_SOURCE_3', 'CREDIT_INCOME_RATIO'),
        'EXT2_ANNUITY_RATIO': ('EXT_SOURCE_2', 'ANNUITY_INCOME_RATIO'),
        'AGE_EXT2': ('AGE_YEARS', 'EXT_SOURCE_2'),
        'AGE_EXT3': ('AGE_YEARS', 'EXT_SOURCE_3'),
        'EXT2_INST_LATE': ('EXT_SOURCE_2', 'INST_LATE_RATE'),
        'EXT3_INST_LATE': ('EXT_SOURCE_3', 'INST_LATE_RATE'),
    }

    for feat_name, (col_a, col_b) in interaction_multiplications.items():
        if col_a in df_fe.columns and col_b in df_fe.columns:
            df_fe[feat_name] = df_fe[col_a] * df_fe[col_b]
            df_fe[feat_name] = df_fe[feat_name].replace([np.inf, -np.inf], np.nan)
            

    # Interactions par Division
    interaction_divisions = {
        'EXT2_DIV_EXT3': ('EXT_SOURCE_2', 'EXT_SOURCE_3'),
        'EXT3_DIV_EXT2': ('EXT_SOURCE_3', 'EXT_SOURCE_2'),
        'EXT1_DIV_EXT2': ('EXT_SOURCE_1', 'EXT_SOURCE_2'),
        'CREDIT_DIV_AGE': ('AMT_CREDIT', 'AGE_YEARS'),
    }

    for feat_name, (col_a, col_b) in interaction_divisions.items():
        if col_a in df_fe.columns and col_b in df_fe.columns:
            df_fe[feat_name] = df_fe[col_a] / df_fe[col_b].replace(0, np.nan)
            df_fe[feat_name] = df_fe[feat_name].replace([np.inf, -np.inf], np.nan)
            
    return df_fe

def encode_categorical_features(df):
    """
    Encode les variables catégorielles  :
    - LabelEncoding sur les colonnes binaires (<= 2 catégories).
    - One-Hot Encoding sur les colonnes multi-classes (> 2 catégories).
    """
    df_enc = df.copy()
    
    # Isoler la clé SK_ID_CURR pour ne pas la transformer lors de get_dummies
    sk_id = df_enc.pop('SK_ID_CURR') if 'SK_ID_CURR' in df_enc.columns else None
        
    # 4. One-Hot Encoding pour le reste des variables catégorielles (> 2 catégories)
    df_enc = pd.get_dummies(df_enc, dtype=int)
    
    # 5. Réinjection de SK_ID_CURR dans le DataFrame
    if sk_id is not None:
        df_enc['SK_ID_CURR'] = sk_id
       
    return df_enc

def build_full_features(pydantic_df: pd.DataFrame, expected_columns: list) -> pd.DataFrame:
    """
    Orchestre le pipeline complet.
    Fonctionne pour 1 ligne (API) ou pour N lignes (Batch)
    """
    sk_ids = pydantic_df['SK_ID_CURR'].astype(int).tolist()
    
    # Récupération Redis
    redis_features_list = get_redis_features_batch(sk_ids)
    redis_df = pd.DataFrame(redis_features_list) 

    pydantic_df = pd.concat([
            pydantic_df.reset_index(drop=True), 
            redis_df.reset_index(drop=True)
        ], axis=1)

    # Nettoyage de base (Ratios primaires)
    df_clean = preprocess_basic_features(pydantic_df)
    
    # Feature Engineering avancé (Votre fonction)
    df_fe = add_polynomial_and_interaction_features(df_clean)
    
    # Encodage One-Hot
    df_encoded = encode_categorical_features(df_fe)
    
    # Alignement strict avec le modèle d'entraînement
    df_final = df_encoded.reindex(columns=expected_columns, fill_value=0)

    return df_final

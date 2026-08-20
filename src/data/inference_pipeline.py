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

# ============================================================
# Preprocessing pour une seule ligne
# ============================================================
def preprocess_basic_features_dict(data: dict) -> None:
    """Nettoyage de base et ratios primaires directement sur le dictionnaire (In-Place)."""
    days_emp = data.get("DAYS_EMPLOYED")
    if days_emp is not None:
        data["DAYS_EMPLOYED_ANOM"] = 1 if days_emp == 365243 else 0
        data["DAYS_EMPLOYED"] = np.nan if days_emp == 365243 else days_emp

        if not np.isnan(data["DAYS_EMPLOYED"]):
            data["YEARS_EMPLOYED"] = -data["DAYS_EMPLOYED"] / 365.0

    days_birth = data.get("DAYS_BIRTH")
    if days_birth is not None and not np.isnan(days_birth):
        data["AGE_YEARS"] = int(-days_birth / 365.0)

    # Ratios financiers
    income = data.get("AMT_INCOME_TOTAL")
    income_valid = income if (income and income != 0) else np.nan

    credit, annuity, goods = data.get("AMT_CREDIT"), data.get("AMT_ANNUITY"), data.get("AMT_GOODS_PRICE")

    if credit is not None and not np.isnan(income_valid):
        data["CREDIT_INCOME_RATIO"] = credit / income_valid
    if annuity is not None and not np.isnan(income_valid):
        data["ANNUITY_INCOME_RATIO"] = annuity / income_valid
    if credit is not None and goods is not None and goods != 0:
        data["CREDIT_GOODS_RATIO"] = credit / goods

def add_polynomial_and_interaction_dict(data: dict) -> None:
    """Calcul des polynômes et interactions directement sur le dictionnaire (In-Place)."""
    # Puissances EXT_SOURCE
    for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        val = data.get(col)
        if val is not None and not np.isnan(val):
            data[f"{col}_SQ"] = val ** 2
            data[f"{col}_CUBIC"] = val ** 3

    # Multiplications
    interaction_mult = {
        "EXT1_EXT2": ("EXT_SOURCE_1", "EXT_SOURCE_2"),
        "EXT1_EXT3": ("EXT_SOURCE_1", "EXT_SOURCE_3"),
        "EXT2_EXT3": ("EXT_SOURCE_2", "EXT_SOURCE_3"),
        "EXT2_INCOME_RATIO": ("EXT_SOURCE_2", "CREDIT_INCOME_RATIO"),
        "EXT3_INCOME_RATIO": ("EXT_SOURCE_3", "CREDIT_INCOME_RATIO"),
        "EXT2_ANNUITY_RATIO": ("EXT_SOURCE_2", "ANNUITY_INCOME_RATIO"),
        "AGE_EXT2": ("AGE_YEARS", "EXT_SOURCE_2"),
        "AGE_EXT3": ("AGE_YEARS", "EXT_SOURCE_3"),
        "EXT2_INST_LATE": ("EXT_SOURCE_2", "INST_LATE_RATE"),
        "EXT3_INST_LATE": ("EXT_SOURCE_3", "INST_LATE_RATE"),
    }

    for feat_name, (col_a, col_b) in interaction_mult.items():
        val_a, val_b = data.get(col_a), data.get(col_b)
        if val_a is not None and val_b is not None and not (np.isnan(val_a) or np.isnan(val_b)):
            res = val_a * val_b
            data[feat_name] = np.nan if np.isinf(res) else res

    # Divisions
    interaction_div = {
        "EXT2_DIV_EXT3": ("EXT_SOURCE_2", "EXT_SOURCE_3"),
        "EXT3_DIV_EXT2": ("EXT_SOURCE_3", "EXT_SOURCE_2"),
        "EXT1_DIV_EXT2": ("EXT_SOURCE_1", "EXT_SOURCE_2"),
        "CREDIT_DIV_AGE": ("AMT_CREDIT", "AGE_YEARS"),
    }

    for feat_name, (col_a, col_b) in interaction_div.items():
        val_a, val_b = data.get(col_a), data.get(col_b)
        if val_a is not None and val_b is not None and val_b != 0 and not (np.isnan(val_a) or np.isnan(val_b)):
            res = val_a / val_b
            data[feat_name] = np.nan if np.isinf(res) else res

def encode_categorical_dict(data: dict) -> None:
    """Encode les variables catégorielles texte en valeurs numériques."""
    # Encodage binaire direct pour FLAG_OWN_CAR (Y -> 1, N -> 0)
    if "FLAG_OWN_CAR" in data and isinstance(data["FLAG_OWN_CAR"], str):
        data["FLAG_OWN_CAR"] = 1 if data["FLAG_OWN_CAR"].upper() == "Y" else 0

    # Encodage binaire direct pour FLAG_OWN_REALTY (Y -> 1, N -> 0)
    if "FLAG_OWN_REALTY" in data and isinstance(data["FLAG_OWN_REALTY"], str):
        data["FLAG_OWN_REALTY"] = 1 if data["FLAG_OWN_REALTY"].upper() == "Y" else 0

    # Encodage binaire pour CODE_GENDER (M -> 1, F -> 0) si utilisé ainsi
    if "CODE_GENDER" in data and isinstance(data["CODE_GENDER"], str):
        data["CODE_GENDER"] = 1 if data["CODE_GENDER"].upper() in ["M", "1"] else 0

    # Pour les variables One-Hot (ex: NAME_CONTRACT_TYPE, EMERGENCYSTATE_MODE...)
    # Si la valeur est sous forme de texte, on génère la clé One-Hot correspondante
    # Exemple : NAME_CONTRACT_TYPE = "Cash loans" -> NAME_CONTRACT_TYPE_Cash loans = 1
    cat_cols = ["NAME_CONTRACT_TYPE", "NAME_TYPE_SUITE", "NAME_INCOME_TYPE", 
                "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", 
                "OCCUPATION_TYPE", "WEEKDAY_APPR_PROCESS_START", "ORGANIZATION_TYPE", "EMERGENCYSTATE_MODE"]

    for col in cat_cols:
        val = data.get(col)
        if val is not None and isinstance(val, str):
            ohe_key = f"{col}_{val}"
            data[ohe_key] = 1

def align_dict_to_dataframe(data: dict, expected_columns: list[str]) -> pd.DataFrame:
    """Mappe le dictionnaire enrichi sur expected_columns pour sortir un DataFrame 1 ligne."""
    row = []
    for col in expected_columns:
        val = data.get(col)
        
        # Si la colonne est absente ou vaut None/NaN
        if val is None or (isinstance(val, float) and np.isnan(val)):
            row.append(np.nan)
        # Si la valeur est un booléen
        elif isinstance(val, bool):
            row.append(int(val))
        # Si la valeur est numérique
        elif isinstance(val, (int, float, np.number)):
            row.append(val)
        # Si la valeur est encore une string non convertie (sécurité)
        elif isinstance(val, str):
            try:
                row.append(float(val))
            except ValueError:
                row.append(0.0)
        else:
            row.append(0.0)

    # Conversion explicite du DataFrame en float32/float64 pour LightGBM
    df_out = pd.DataFrame([row], columns=expected_columns)
    return df_out.astype(np.float64)

def build_full_features_single(input_dict: dict, redis_dict: dict, expected_columns: list[str]) -> pd.DataFrame:
    """Orchestrateur rapide pour 1 seule ligne."""
    data = {**input_dict, **redis_dict}
    
    preprocess_basic_features_dict(data)
    add_polynomial_and_interaction_dict(data)
    encode_categorical_dict(data)
    
    return align_dict_to_dataframe(data, expected_columns)

# ============================================================
# Preprocessing pour les batchs
# ============================================================

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

def build_full_features_batch(pydantic_df: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
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

# ===============================================================

def build_full_features(pydantic_df: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    """Point d'entrée principal avec A/B Routing (Single vs Batch)."""
    if len(pydantic_df) == 1:
        sk_id = pydantic_df['SK_ID_CURR'].values[0]
        input_dict = pydantic_df.iloc[0].to_dict()
        redis_dict = get_redis_features_batch([int(sk_id)])[0]
        
        return build_full_features_single(input_dict, redis_dict, expected_columns)
    
    return build_full_features_batch(pydantic_df, expected_columns)
import numpy as np

def preprocess_app_data(df):
    """
    Effectue un nettoyage et un feature engineering sur les données initiales (reçu en paramètres de la requête)
    """
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
        df_clean['CREDIT_INCOME_RATIO'] = df_clean['AMT_CREDIT'] / df_clean['AMT_INCOME_TOTAL']
    if all(col in df_clean.columns for col in ['AMT_ANNUITY', 'AMT_INCOME_TOTAL']):
        df_clean['ANNUITY_INCOME_RATIO'] = df_clean['AMT_ANNUITY'] / df_clean['AMT_INCOME_TOTAL']
    if all(col in df_clean.columns for col in ['AMT_CREDIT', 'AMT_GOODS_PRICE']):
        df_clean['CREDIT_GOODS_RATIO'] = df_clean['AMT_CREDIT'] / df_clean['AMT_GOODS_PRICE']

    return df_clean
import numpy as np
import pandas as pd


def process_previous_app(df_prev_app : pd.DataFrame) -> pd.DataFrame:
    df = df_prev_app.copy()

    # Nettoyage des valeurs aberrantes spécifiques à previous_application (365243 = NaN)
    days_cols = ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']
    for col in days_cols:
        if col in df.columns:
            df[col] = df[col].replace(365243, np.nan)

    # Feature Engineering au niveau de la ligne (Ratios Métiers)
    # Ratio Montant Demandé / Montant Accordé
    df['APP_CREDIT_PERC'] = df['AMT_APPLICATION'] / df['AMT_CREDIT'].replace(0, np.nan)
    # Différence entre le montant demandé et accordé
    df['AMT_DECLINED'] = df['AMT_APPLICATION'] - df['AMT_CREDIT']
    # Ratio Annuité / Montant du crédit
    df['ANNUITY_CREDIT_PERC'] = df['AMT_ANNUITY'] / df['AMT_CREDIT'].replace(0, np.nan)
    # Ratio Acompte / Prix du bien
    df['DOWN_PAYMENT_PERC'] = df['AMT_DOWN_PAYMENT'] / df['AMT_GOODS_PRICE'].replace(0, np.nan)

    # Traitement des variables catégorielles (One-Hot Encoding)
    cat_cols = [c for c in df.columns if df[c].dtype == 'object']
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=True)
    encoded_cols = [c for c in df.columns if any(c.startswith(f"{orig}_") for orig in cat_cols)]

    cols_to_exclude = ['SK_ID_CURR', 'SK_ID_PREV']
    num_cols = [c for c in df.select_dtypes(include=['number']).columns 
                if c not in cols_to_exclude and c not in encoded_cols]

    agg_dict = {}
    
    # Pour les colonnes numériques (min, max, mean, sum, var)
    for col in num_cols:
        agg_dict[col] = ['min', 'max', 'mean', 'sum', 'var']

    # Pour les colonnes encodées (mean = proportion/taux, sum = volume)
    for col in encoded_cols:
        agg_dict[col] = ['mean', 'sum']

    agg_dict['SK_ID_PREV'] = ['count']

    # Agrégation principale par client (SK_ID_CURR)
    prev_agg = df.groupby('SK_ID_CURR').agg(agg_dict)
    prev_agg.columns = [f"PREV_{col}_{stat}".upper() for col, stat in prev_agg.columns]

    if 'PREV_SK_ID_PREV_COUNT' in prev_agg.columns:
        prev_agg.rename(columns={'PREV_SK_ID_PREV_COUNT': 'PREV_APP_COUNT'}, inplace=True)

    # Sous-agrégations ciblées : Demandes Approuvées vs Demandes Refusées
    # Permet de capturer le profil financier selon l'issue de la décision
    approved_df = df[df['NAME_CONTRACT_STATUS_Approved'] == 1] if 'NAME_CONTRACT_STATUS_Approved' in df.columns else df[df['NAME_CONTRACT_STATUS'] == 'Approved']
    refused_df  = df[df['NAME_CONTRACT_STATUS_Refused'] == 1] if 'NAME_CONTRACT_STATUS_Refused' in df.columns else df[df['NAME_CONTRACT_STATUS'] == 'Refused']

    sub_cols = ['AMT_APPLICATION', 'AMT_CREDIT', 'AMT_ANNUITY', 'APP_CREDIT_PERC', 'AMT_DECLINED']
    sub_cols = [c for c in sub_cols if c in df.columns]

    # Agrégation des demandes Approuvées
    approved_agg = approved_df.groupby('SK_ID_CURR').agg({col: ['mean', 'max', 'sum'] for col in sub_cols})
    approved_agg.columns = [f"PREV_APPROVED_{col}_{stat}".upper() for col, stat in approved_agg.columns]

    # Agrégation des demandes Refusées
    refused_agg = refused_df.groupby('SK_ID_CURR').agg({col: ['mean', 'max', 'sum'] for col in sub_cols})
    refused_agg.columns = [f"PREV_REFUSED_{col}_{stat}".upper() for col, stat in refused_agg.columns]

    # 8. Fusion finale
    final_prev = (prev_agg
                  .merge(approved_agg, on='SK_ID_CURR', how='left')
                  .merge(refused_agg, on='SK_ID_CURR', how='left')
                  .reset_index())

    print(f"    Agrégation previous application : {final_prev.shape}")
    return final_prev
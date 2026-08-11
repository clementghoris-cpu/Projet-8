import pandas as pd
import numpy as np

def process_install_payments(df_install_pay : pd.DataFrame) -> pd.DataFrame:
    '''
    Agrégation des données du fichier 'installments_payments'
    '''
    df = df_install_pay.copy()

    # Feature Engineering au niveau ligne (Ratios Métier)
    # Différence de paiement (Positif = Sous-payé, Négatif = Sur-payé)
    df['PAYMENT_DIFF'] = df['AMT_INSTALMENT'] - df['AMT_PAYMENT']
    
    # Ratio de paiement (Proportion payée)
    df['PAYMENT_PERC'] = df['AMT_PAYMENT'] / df['AMT_INSTALMENT'].replace(0, np.nan)
    
    # Retard en jours (Jours effectifs - Jours prévus)
    df['DAYS_LATE'] = df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']
    df['DAYS_LATE_POS'] = df['DAYS_LATE'].clip(lower=0) # Uniquement les jours de retard stricts
    df['IS_LATE'] = (df['DAYS_LATE_POS'] > 0).astype(int)
    
    # Tendance de retard
    df['LATE_TREND'] = df['DAYS_LATE_POS'] / (df['DAYS_INSTALMENT'].abs() + 1)

    cols_to_exclude = ['SK_ID_CURR', 'SK_ID_PREV']
    num_cols = [c for c in df.select_dtypes(include=['number']).columns if c not in cols_to_exclude]

    agg_dict = {col: ['min', 'max', 'mean', 'sum', 'var'] for col in num_cols}
    agg_dict['SK_ID_PREV'] = ['count'] # Nombre total d'échéances payées

    inst_agg = df.groupby('SK_ID_CURR').agg(agg_dict)  
    inst_agg.columns = [f"INST_{col}_{stat}".upper() for col, stat in inst_agg.columns]

    if 'INST_SK_ID_PREV_COUNT' in inst_agg.columns:
        inst_agg.rename(columns={'INST_SK_ID_PREV_COUNT': 'INST_COUNT'}, inplace=True)

    # Agrégation ciblée sur les 180 derniers jours (~6 mois)
    recent_df = df[df['DAYS_INSTALMENT'] >= -180]
    
    recent_cols = ['DAYS_LATE_POS', 'PAYMENT_DIFF', 'PAYMENT_PERC', 'IS_LATE']
    recent_cols = [c for c in recent_cols if c in recent_df.columns]
    
    recent_agg_dict = {col: ['mean', 'max', 'sum'] for col in recent_cols}
    
    inst_recent_agg = recent_df.groupby('SK_ID_CURR').agg(recent_agg_dict)
    inst_recent_agg.columns = [f"INST_RECENT_180D_{col}_{stat}".upper() for col, stat in inst_recent_agg.columns]

    final_inst = inst_agg.merge(inst_recent_agg, on='SK_ID_CURR', how='left').reset_index()

    return final_inst
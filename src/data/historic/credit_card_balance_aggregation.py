import pandas as pd
import numpy as np

def process_credit_card(df_credit_card : pd.DataFrame) -> pd.DataFrame:
    '''
    Agrégation des données du fichier 'credit_card_balance.csv'
    '''

    df = df_credit_card.copy()

    # Traitement des variables catégorielles
    cat_cols = [c for c in df.columns if df[c].dtype == 'object']
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=True)
    encoded_cols = [c for c in df.columns if any(c.startswith(f"{orig}_") for orig in cat_cols)]

    # Ratios métiers clés
    if 'AMT_BALANCE' in df.columns and 'AMT_CREDIT_LIMIT_ACTUAL' in df.columns:
        df['UTILIZATION'] = (
            df['AMT_BALANCE'] / df['AMT_CREDIT_LIMIT_ACTUAL'].replace(0, np.nan)
        ).clip(0, 1)

    if 'AMT_PAYMENT_TOTAL_CURRENT' in df.columns and 'AMT_BALANCE' in df.columns:
        df['PAYMENT_RATIO'] = (
            df['AMT_PAYMENT_TOTAL_CURRENT'] / df['AMT_BALANCE'].replace(0, np.nan)
        ).clip(0, 1)

    # Liste des variables numériques à agréger
    num_cols = [
        'MONTHS_BALANCE', 'AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL',
        'AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT',
        'AMT_DRAWINGS_POS_CURRENT', 'AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_CURRENT',
        'AMT_PAYMENT_TOTAL_CURRENT', 'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE',
        'AMT_TOTAL_RECEIVABLE', 'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_CURRENT',
        'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_POS_CURRENT', 'CNT_INSTALMENT_MATURE_CUM',
        'SK_DPD', 'SK_DPD_DEF', 'UTILIZATION', 'PAYMENT_RATIO'
    ]

    # Filtrer uniquement les colonnes réellement présentes
    num_cols = [c for c in num_cols if c in df.columns]

    # Construction du dictionnaire d'agrégation global
    agg_dict = {}
    for col in num_cols:
        agg_dict[col] = ['min', 'max', 'mean', 'sum', 'var']

    for col in encoded_cols:
        agg_dict[col] = ['mean', 'sum']

    agg_dict['SK_ID_PREV'] = ['count']

    # Agrégation globale par client (SK_ID_CURR)
    cc_agg = df.groupby('SK_ID_CURR').agg(agg_dict)
    cc_agg.columns = [f"CC_{col}_{stat}".upper() for col, stat in cc_agg.columns]

    if 'CC_SK_ID_PREV_COUNT' in cc_agg.columns:
        cc_agg.rename(columns={'CC_SK_ID_PREV_COUNT': 'CC_COUNT'}, inplace=True)

    # Agrégation ciblée sur la période récente (6 derniers mois)
    # Permet de détecter une détérioration récente du comportement financier
    recent_df = df[df['MONTHS_BALANCE'] >= -6]
    
    recent_num_cols = ['UTILIZATION', 'PAYMENT_RATIO', 'SK_DPD', 'AMT_BALANCE']
    recent_num_cols = [c for c in recent_num_cols if c in recent_df.columns]

    recent_agg_dict = {col: ['mean', 'max'] for col in recent_num_cols}

    # Si des colonnes récentes existent, on aggrège et on merge
    if recent_agg_dict:
        cc_recent_agg = recent_df.groupby('SK_ID_CURR').agg(recent_agg_dict)
        cc_recent_agg.columns = [f"CC_RECENT_6M_{col}_{stat}".upper() for col, stat in cc_recent_agg.columns]
        final_cc = cc_agg.merge(cc_recent_agg, on='SK_ID_CURR', how='left').reset_index()
    else:
        final_cc = cc_agg.reset_index()

    return final_cc
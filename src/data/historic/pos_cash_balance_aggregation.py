import pandas as pd


def process_pos_cash_balance(df_pos_cash_bal : pd.DataFrame) -> pd.DataFrame:
    '''
    Agrégation des données du fichier 'pos_cash_balance.csv'
    '''
    df = df_pos_cash_bal.copy()

    # Traitement des variables catégorielles
    cat_cols = [c for c in df.columns if df[c].dtype == 'object']
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=True)
    encoded_cols = [c for c in df.columns if any(c.startswith(f"{orig}_") for orig in cat_cols)]

    # Création d'indicateurs métier binaire au niveau ligne
    df['IS_DPD'] = (df['SK_DPD'] > 0).astype(int)
    df['IS_DPD_DEF'] = (df['SK_DPD_DEF'] > 0).astype(int)

    cols_to_exclude = ['SK_ID_CURR', 'SK_ID_PREV']
    num_cols = [c for c in df.select_dtypes(include=['number']).columns 
                if c not in cols_to_exclude and c not in encoded_cols]

    agg_dict = {}
    
    for col in num_cols:
        agg_dict[col] = ['min', 'max', 'mean', 'sum', 'var']

    for col in encoded_cols:
        agg_dict[col] = ['mean', 'sum']

    # Nombre total d'enregistrements mensuels
    agg_dict['SK_ID_PREV'] = ['count']

    # Agrégation principale par SK_ID_CURR
    pos_agg = df.groupby('SK_ID_CURR').agg(agg_dict)    
    pos_agg.columns = [f"POS_{col}_{stat}".upper() for col, stat in pos_agg.columns]

    if 'POS_SK_ID_PREV_COUNT' in pos_agg.columns:
        pos_agg.rename(columns={'POS_SK_ID_PREV_COUNT': 'POS_COUNT'}, inplace=True)

    # Agrégation ciblée sur les 6 derniers mois (MONTHS_BALANCE >= -6)
    recent_df = df[df['MONTHS_BALANCE'] >= -6]
    
    recent_cols = ['SK_DPD', 'SK_DPD_DEF', 'CNT_INSTALMENT_FUTURE', 'IS_DPD']
    recent_cols = [c for c in recent_cols if c in recent_df.columns]
    
    recent_agg_dict = {col: ['mean', 'max', 'sum'] for col in recent_cols}
    
    pos_recent_agg = recent_df.groupby('SK_ID_CURR').agg(recent_agg_dict)
    pos_recent_agg.columns = [f"POS_RECENT_6M_{col}_{stat}".upper() for col, stat in pos_recent_agg.columns]

    # Fusion de l'agrégation globale et récente
    final_pos = pos_agg.merge(pos_recent_agg, on='SK_ID_CURR', how='left').reset_index()

    print(f"    Agrégation POS_CASH balance : {final_pos.shape}")
    return final_pos
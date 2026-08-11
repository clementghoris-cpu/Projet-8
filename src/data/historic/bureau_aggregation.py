import pandas as pd

def process_bureau(df_bureau : pd.DataFrame) -> pd.DataFrame:
    '''
    Agrégation des données du fichier 'bureau.csv'
    '''
    df = df_bureau.copy()

    # Traitement des variables catégorielles par One-Hot Encoding
    cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=True)
    
    # Isoler les nouvelles colonnes créées par le One-Hot Encoding
    encoded_cols = [c for c in df.columns if any(c.startswith(f"{orig}_") for orig in cat_cols)]

    # Définition des variables numériques à agréger
    num_cols = [
        'DAYS_CREDIT', 
        'DAYS_CREDIT_ENDDATE', 
        'DAYS_ENDDATE_FACT',
        'AMT_CREDIT_MAX_OVERDUE', 
        'CNT_CREDIT_PROLONG', 
        'AMT_CREDIT_SUM',
        'AMT_CREDIT_SUM_DEBT', 
        'AMT_CREDIT_SUM_OVERDUE', 
        'AMT_CREDIT_SUM_LIMIT',
        'DAYS_CREDIT_UPDATE'
    ]

    agg_dict = {}
    
    # Agrégation des variables numériques
    for col in num_cols:
        if col in df.columns:
            agg_dict[col] = ['min', 'max', 'mean', 'sum']
            
    # Agrégation sur les colonnes encodées (somme et moyenne = fréquence)
    for col in encoded_cols:
        agg_dict[col] = ['mean', 'sum']
            
    agg_dict['SK_ID_BUREAU'] = ['count']
    bureau_agg = df.groupby('SK_ID_CURR').agg(agg_dict)

    bureau_agg.columns = [f"BUREAU_{col}_{stat}".upper() for col, stat in bureau_agg.columns]

    # Renommer le count global pour plus de clarté
    if 'BUREAU_SK_ID_BUREAU_COUNT' in bureau_agg.columns:
        bureau_agg.rename(columns={'BUREAU_SK_ID_BUREAU_COUNT': 'BUREAU_LOAN_COUNT'}, inplace=True)

    # Calcul optimisé des ratios métier globaux
    if 'BUREAU_AMT_CREDIT_SUM_DEBT_SUM' in bureau_agg.columns and 'BUREAU_AMT_CREDIT_SUM_SUM' in bureau_agg.columns:
        bureau_agg['BUREAU_DEBT_RATIO'] = (
            bureau_agg['BUREAU_AMT_CREDIT_SUM_DEBT_SUM'] / 
            (bureau_agg['BUREAU_AMT_CREDIT_SUM_DEBT_SUM'] + 1)
        )

    bureau_agg.reset_index(inplace=True)
    return bureau_agg

def process_bureau_balance(df_bureau_bal : pd.DataFrame, df_bureau : pd.DataFrame) -> pd.DataFrame:
    '''
    Agrégation des données du fichier 'bureau_balance.csv'
    '''
    df = df_bureau_bal.copy()

    # Traitement numérique du statut (retards)
    # 0/C/X = 0, 1 = 1, ..., 5 = 5 (permet des calculs de retards max/moyens)
    status_map = {'C': 0, 'X': 0, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
    df['STATUS_NUM'] = df['STATUS'].map(status_map)

    # One-Hot Encoding du statut brut
    df = pd.get_dummies(df, columns=['STATUS'], prefix='STATUS', dummy_na=False)
    status_cols = [c for c in df.columns if c.startswith('STATUS_')]

    # Première agrégation par SK_ID_BUREAU (par compte bureau)
    agg_bureau = {
        'MONTHS_BALANCE': ['min', 'max', 'size'],
        'STATUS_NUM': ['mean', 'max']
    }
    # Ajouter les sommes et moyennes des variables encodées
    for col in status_cols:
        agg_bureau[col] = ['mean', 'sum']

    bb_bureau_agg = df.groupby('SK_ID_BUREAU').agg(agg_bureau)    
    bb_bureau_agg.columns = [f"{col}_{stat}".upper() for col, stat in bb_bureau_agg.columns]
    bb_bureau_agg.reset_index(inplace=True)

    # Merging avec bureau pour récupérer les SK_ID_CURR
    bureau_map = df_bureau[['SK_ID_BUREAU', 'SK_ID_CURR']].drop_duplicates()
    bb_merged = bureau_map.merge(bb_bureau_agg, on='SK_ID_BUREAU', how='inner')

    # Deuxième agrégation par SK_ID_CURR (par client)
    # Exclure les clés d'identification pour la boucle
    features_to_agg = [c for c in bb_merged.columns if c not in ['SK_ID_BUREAU', 'SK_ID_CURR']]
    
    agg_curr = {}
    for col in features_to_agg:
        agg_curr[col] = ['min', 'max', 'mean', 'sum']

    bb_curr_agg = bb_merged.groupby('SK_ID_CURR').agg(agg_curr)
    bb_curr_agg.columns = [f"BB_{col}_{stat}".upper() for col, stat in bb_curr_agg.columns]
    bb_curr_agg.reset_index(inplace=True)

    print(f"bureau_bal_curr_agg : {bb_curr_agg.shape}")
    return bb_curr_agg

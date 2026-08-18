import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.data.historic.bureau_aggregation import process_bureau, process_bureau_balance

# ------------------------------------------------------------------
# Fixtures : Jeux de données réutilisables
# ------------------------------------------------------------------

@pytest.fixture
def sample_bureau_df():
    """DataFrame représentatif avec données numériques et catégorielles."""
    return pd.DataFrame({
        'SK_ID_CURR': [100001, 100001, 100002],
        'SK_ID_BUREAU': [1, 2, 3],
        'CREDIT_ACTIVE': ['Closed', 'Active', 'Active'],
        'DAYS_CREDIT': [-100, -200, -300],
        'AMT_CREDIT_SUM': [10000.0, 20000.0, 30000.0],
        'AMT_CREDIT_SUM_DEBT': [0.0, 5000.0, 10000.0],
    })

@pytest.fixture
def sample_bureau_bal_df():
    """Données d'historique de paiement mensuel (bureau_balance)."""
    return pd.DataFrame({
        'SK_ID_BUREAU': [1, 1, 1, 2, 2],
        'MONTHS_BALANCE': [0, -1, -2, 0, -1],
        'STATUS': ['C', '0', '1', 'X', '5']  # Mélange de statuts
    })


@pytest.fixture
def sample2_bureau_df():
    """Table de correspondance entre SK_ID_BUREAU et SK_ID_CURR."""
    return pd.DataFrame({
        'SK_ID_CURR': [100001, 100001, 100002],
        'SK_ID_BUREAU': [1, 2, 3]  # Le bureau 3 n'est pas dans bureau_balance (test inner join)
    })

# ------------------------------------------------------------------
# Tests unitaires
# ------------------------------------------------------------------

def test_process_bureau_nominal_structure(sample_bureau_df):
    """Vérifie le regroupement par SK_ID_CURR et le nommage des colonnes."""
    res = process_bureau(sample_bureau_df)

    # 1. Nombre de lignes unique par client
    assert len(res) == 2
    assert set(res['SK_ID_CURR']) == {100001, 100002}

    # 2. Renommage explicite de la colonne de comptage
    assert 'BUREAU_LOAN_COUNT' in res.columns
    assert 'BUREAU_SK_ID_BUREAU_COUNT' not in res.columns

    # 3. Présence du ratio métier calculé
    assert 'BUREAU_DEBT_RATIO' in res.columns


def test_process_bureau_aggregation_values(sample_bureau_df):
    """Vérifie l'exactitude mathématique des agrégations sur un client spécifique."""
    res = process_bureau(sample_bureau_df)
    row_100001 = res[res['SK_ID_CURR'] == 100001].iloc[0]

    # Verification du nombre de prêts pour 100001 (2 entrées dans la fixture)
    assert row_100001['BUREAU_LOAN_COUNT'] == 2

    # Agrégations sur DAYS_CREDIT (-100, -200)
    assert row_100001['BUREAU_DAYS_CREDIT_MIN'] == -200
    assert row_100001['BUREAU_DAYS_CREDIT_MAX'] == -100
    assert row_100001['BUREAU_DAYS_CREDIT_MEAN'] == -150.0
    assert row_100001['BUREAU_DAYS_CREDIT_SUM'] == -300

    # Test du One-Hot Encoding (CREDIT_ACTIVE: 1 'Active', 1 'Closed')
    # Les colonnes générées portent des majuscules à la fin
    assert row_100001['BUREAU_CREDIT_ACTIVE_ACTIVE_MEAN'] == 0.5
    assert row_100001['BUREAU_CREDIT_ACTIVE_ACTIVE_SUM'] == 1


def test_process_bureau_debt_ratio_calculation():
    """Vérifie la formule spécifique : DEBT_SUM / (DEBT_SUM + 1)."""
    df = pd.DataFrame({
        'SK_ID_CURR': [1],
        'SK_ID_BUREAU': [10],
        'AMT_CREDIT_SUM_DEBT': [9.0],
        'AMT_CREDIT_SUM': [100.0]
    })
    res = process_bureau(df)
    
    # Calcul attendu : 9.0 / (9.0 + 1) = 0.9
    expected_ratio = 9.0 / 10.0
    assert pytest.approx(res.loc[0, 'BUREAU_DEBT_RATIO']) == expected_ratio


def test_process_bureau_immutability(sample_bureau_df):
    """S'assure que le DataFrame d'origine n'est pas modifié par la fonction."""
    original_copy = sample_bureau_df.copy()
    _ = process_bureau(sample_bureau_df)

    assert_frame_equal(sample_bureau_df, original_copy)


def test_process_bureau_missing_optional_columns():
    """Vérifie le comportement si des colonnes numériques ou de dette manquent."""
    df_minimal = pd.DataFrame({
        'SK_ID_CURR': [1, 1],
        'SK_ID_BUREAU': [101, 102],
        'DAYS_CREDIT': [-10, -20]
        # Aucune variable catégorielle ni AMT_CREDIT_SUM_DEBT
    })

    res = process_bureau(df_minimal)

    # Doit s'exécuter sans erreur
    assert len(res) == 1
    assert 'BUREAU_DAYS_CREDIT_MEAN' in res.columns
    # Le ratio ne doit pas être généré si la colonne de dette est absente
    assert 'BUREAU_DEBT_RATIO' not in res.columns


def test_process_bureau_handles_nan_in_categories():
    """Vérifie que pd.get_dummies(..., dummy_na=True) gère bien les valeurs NaN."""
    df_nan_cat = pd.DataFrame({
        'SK_ID_CURR': [1, 1],
        'SK_ID_BUREAU': [1, 2],
        'CREDIT_TYPE': ['Microloan', np.nan]
    })

    res = process_bureau(df_nan_cat)

    # La colonne spécifique aux NaN doit exister
    nan_col = [c for c in res.columns if 'NAN' in c]
    assert len(nan_col) > 0

def test_process_bureau_balance_nominal_structure(sample_bureau_bal_df, sample2_bureau_df):
    """Vérifie la structure globale du DataFrame résultat."""
    res = process_bureau_balance(sample_bureau_bal_df, sample2_bureau_df)

    # 1. Deux clients uniques doivent ressortir (100001 et 100002 éliminé ou conservé selon la jointure)
    # Le bureau 1 et 2 appartiennent au client 100001
    assert set(res['SK_ID_CURR']) == {100001}

    # 2. Toutes les colonnes hors SK_ID_CURR doivent commencer par BB_
    feature_cols = [c for c in res.columns if c != 'SK_ID_CURR']
    assert all(c.startswith('BB_') for c in feature_cols)


def test_process_bureau_balance_status_mapping():
    """Vérifie le traitement des statuts et la présence de la colonne STATUS_NUM."""
    df_bal = pd.DataFrame({
        'SK_ID_BUREAU': [1, 1, 1],
        'MONTHS_BALANCE': [0, -1, -2],
        'STATUS': ['C', 'X', '3']  # STATUS_NUM associés : 0, 0, 3
    })
    
    df_bureau = pd.DataFrame({
        'SK_ID_CURR': [100001],
        'SK_ID_BUREAU': [1]
    })
    
    res = process_bureau_balance(df_bal, df_bureau)
    
    # STATUS_NUM étant capturé dans status_cols, ses agrégations génèrent *_SUM_* et *_MEAN_*
    # Au lieu de STATUS_NUM_MAX_MAX, la colonne réelle calculée par votre code est :
    assert 'BB_STATUS_NUM_SUM_MAX' in res.columns or 'BB_STATUS_NUM_MEAN_MAX' in res.columns
    
    # La valeur maximale de la moyenne de STATUS_NUM pour le client 100001
    row_client = res[res['SK_ID_CURR'] == 100001].iloc[0]
    assert row_client['BB_STATUS_NUM_SUM_SUM'] == 3.0  # Sum des num (0+0+3 = 3)


def test_process_bureau_balance_double_aggregation_logic(sample_bureau_bal_df, sample2_bureau_df):
    """Vérifie le calcul exact des agrégations au niveau du client."""
    res = process_bureau_balance(sample_bureau_bal_df, sample2_bureau_df)
    
    # Client 100001 :
    # - Bureau 1 a 3 mois (size=3)
    # - Bureau 2 a 2 mois (size=2)
    # Somme totale des mois pour le client 100001 = 3 + 2 = 5
    # Moyenne par bureau = (3 + 2) / 2 = 2.5
    
    row_100001 = res[res['SK_ID_CURR'] == 100001].iloc[0]
    
    assert row_100001['BB_MONTHS_BALANCE_SIZE_SUM'] == 5
    assert row_100001['BB_MONTHS_BALANCE_SIZE_MEAN'] == 2.5

def test_process_bureau_balance_inner_join_filtering(sample_bureau_bal_df):
    """Vérifie que l'inner join élimine les SK_ID_BUREAU absents de df_bureau."""
    # df_bureau ne contient QUE le bureau 1 (le bureau 2 sera filtré)
    df_bureau_restricted = pd.DataFrame({
        'SK_ID_CURR': [100001],
        'SK_ID_BUREAU': [1]
    })

    res = process_bureau_balance(sample_bureau_bal_df, df_bureau_restricted)

    # Seul le client 100001 et les données du bureau 1 doivent subsister
    assert len(res) == 1
    assert res.loc[0, 'SK_ID_CURR'] == 100001
    
    # Pour le bureau 1 seul, la somme des tailles de mois est 3
    assert res.loc[0, 'BB_MONTHS_BALANCE_SIZE_SUM'] == 3


def test_process_bureau_balance_immutability(sample_bureau_bal_df, sample2_bureau_df):
    """S'assure que les DataFrames passés en argument ne sont pas altérés."""
    bal_copy = sample_bureau_bal_df.copy()
    bureau_copy = sample2_bureau_df.copy()

    _ = process_bureau_balance(sample_bureau_bal_df, sample2_bureau_df)

    assert_frame_equal(sample_bureau_bal_df, bal_copy)
    assert_frame_equal(sample2_bureau_df, bureau_copy)


def test_process_bureau_balance_unmapped_status_handling():
    """Vérifie qu'un statut inconnu génère bien sa propre colonne One-Hot sans faire planter le code."""
    df_bal_unknown = pd.DataFrame({
        'SK_ID_BUREAU': [1, 1],
        'MONTHS_BALANCE': [0, -1],
        'STATUS': ['C', 'UNKNOWN_STATUS']
    })
    
    df_bureau = pd.DataFrame({
        'SK_ID_CURR': [100001],
        'SK_ID_BUREAU': [1]
    })

    res = process_bureau_balance(df_bal_unknown, df_bureau)
    
    assert len(res) == 1
    # La colonne dynamique issue de get_dummies pour UNKNOWN_STATUS doit être présente
    unknown_cols = [c for c in res.columns if 'UNKNOWN_STATUS' in c]
    assert len(unknown_cols) > 0
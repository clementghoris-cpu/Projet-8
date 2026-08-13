import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from src.data.historic.credit_card_balance_aggregation import process_credit_card

# ------------------------------------------------------------------
# Fixtures : Jeux de données de test
# ------------------------------------------------------------------

@pytest.fixture
def sample_credit_card_df():
    """DataFrame représentatif avec colonnes numériques, catégorielles et temporelles."""
    return pd.DataFrame({
        'SK_ID_CURR': [100001, 100001, 100002, 100002],
        'SK_ID_PREV': [1, 2, 3, 4],
        'MONTHS_BALANCE': [-1, -12, -2, -8],  # Mélange récent (>= -6) et ancien (< -6)
        'AMT_BALANCE': [500.0, 1000.0, 0.0, 2000.0],
        'AMT_CREDIT_LIMIT_ACTUAL': [1000.0, 1000.0, 0.0, 1000.0],  # Test division par zéro
        'AMT_PAYMENT_TOTAL_CURRENT': [600.0, 100.0, 0.0, 500.0],
        'NAME_CONTRACT_STATUS': ['Active', 'Active', 'Completed', 'Active']
    })


# ------------------------------------------------------------------
# Tests unitaires
# ------------------------------------------------------------------

def test_process_credit_card_structure(sample_credit_card_df):
    """Vérifie le regroupement par SK_ID_CURR, le renommage de COUNT et les préfixes."""
    res = process_credit_card(sample_credit_card_df)

    # 1. Nombre de clients uniques retenus
    assert len(res) == 2
    assert set(res['SK_ID_CURR']) == {100001, 100002}

    # 2. Renommage de la colonne de comptage
    assert 'CC_COUNT' in res.columns
    assert 'CC_SK_ID_PREV_COUNT' not in res.columns

    # 3. Présence des colonnes issues des 6 derniers mois
    recent_cols = [c for c in res.columns if c.startswith('CC_RECENT_6M_')]
    assert len(recent_cols) > 0


def test_process_credit_card_ratios_calculation():
    """Vérifie l'exactitude des calculs de UTILIZATION et PAYMENT_RATIO (div par zero + clip)."""
    df = pd.DataFrame({
        'SK_ID_CURR': [1, 2, 3],
        'SK_ID_PREV': [10, 20, 30],
        'MONTHS_BALANCE': [-1, -1, -1],
        'AMT_BALANCE': [1500.0, 500.0, 100.0],  # Cas 1: > limit, Cas 2: normal, Cas 3: div par zéro (limit=0)
        'AMT_CREDIT_LIMIT_ACTUAL': [1000.0, 1000.0, 0.0],
        'AMT_PAYMENT_TOTAL_CURRENT': [2000.0, 250.0, 50.0]  # Cas 1: payment > balance
    })

    res = process_credit_card(df)

    # Client 1:
    # UTILIZATION = 1500 / 1000 = 1.5 -> clippé à 1.0
    # PAYMENT_RATIO = 2000 / 1500 = 1.33 -> clippé à 1.0
    c1 = res[res['SK_ID_CURR'] == 1].iloc[0]
    assert c1['CC_UTILIZATION_MAX'] == 1.0
    assert c1['CC_PAYMENT_RATIO_MAX'] == 1.0

    # Client 2:
    # UTILIZATION = 500 / 1000 = 0.5
    # PAYMENT_RATIO = 250 / 500 = 0.5
    c2 = res[res['SK_ID_CURR'] == 2].iloc[0]
    assert c2['CC_UTILIZATION_MEAN'] == 0.5
    assert c2['CC_PAYMENT_RATIO_MEAN'] == 0.5

    # Client 3:
    # AMT_CREDIT_LIMIT_ACTUAL = 0 -> UTILIZATION devient NaN (ignoré par l'agrégation mean/max)
    c3 = res[res['SK_ID_CURR'] == 3].iloc[0]
    assert np.isnan(c3['CC_UTILIZATION_MEAN']) or c3['CC_UTILIZATION_MEAN'] == 0.0


def test_process_credit_card_recent_filter(sample_credit_card_df):
    """Vérifie que l'agrégation récente ne prend en compte QUE les lignes avec MONTHS_BALANCE >= -6."""
    res = process_credit_card(sample_credit_card_df)
    
    row_100001 = res[res['SK_ID_CURR'] == 100001].iloc[0]

    # Pour 100001:
    # Global (2 lignes) : AMT_BALANCE = [500.0, 1000.0] -> MEAN = 750.0
    # Récent (1 ligne avec MONTHS_BALANCE=-1) : AMT_BALANCE = [500.0] -> MEAN = 500.0
    assert row_100001['CC_AMT_BALANCE_MEAN'] == 750.0
    assert row_100001['CC_RECENT_6M_AMT_BALANCE_MEAN'] == 500.0


def test_process_credit_card_client_without_recent_history():
    """Vérifie le comportement (Left Merge) si un client n'a pas de données dans les 6 derniers mois."""
    df_old_only = pd.DataFrame({
        'SK_ID_CURR': [1, 1],
        'SK_ID_PREV': [10, 11],
        'MONTHS_BALANCE': [-12, -24],  # Rien >= -6
        'AMT_BALANCE': [500.0, 1000.0],
        'AMT_CREDIT_LIMIT_ACTUAL': [1000.0, 1000.0],
        'AMT_PAYMENT_TOTAL_CURRENT': [100.0, 100.0]
    })

    res = process_credit_card(df_old_only)

    assert len(res) == 1
    # Les colonnes de l'agrégation globale sont présentes
    assert res.loc[0, 'CC_COUNT'] == 2
    
    # Les colonnes récents doivent exister mais contenir des NaN suite au left join
    assert 'CC_RECENT_6M_AMT_BALANCE_MEAN' in res.columns
    assert np.isnan(res.loc[0, 'CC_RECENT_6M_AMT_BALANCE_MEAN'])


def test_process_credit_card_missing_optional_num_cols():
    """Vérifie que la fonction s'exécute même si plusieurs colonnes de num_cols sont absentes."""
    df_minimal = pd.DataFrame({
        'SK_ID_CURR': [1, 1],
        'SK_ID_PREV': [10, 20],
        'MONTHS_BALANCE': [-1, -2],
        'AMT_BALANCE': [100.0, 200.0],
        'AMT_CREDIT_LIMIT_ACTUAL': [500.0, 500.0]
        # Beaucoup de colonnes comme AMT_DRAWINGS_* sont absentes
    })

    res = process_credit_card(df_minimal)

    assert len(res) == 1
    assert 'CC_AMT_BALANCE_MEAN' in res.columns
    # Les colonnes non fournies ne doivent pas figurer dans le résultat
    assert 'CC_SK_DPD_MEAN' not in res.columns


def test_process_credit_card_immutability(sample_credit_card_df):
    """Vérifie que la fonction ne modifie pas le DataFrame original passé en argument."""
    original_copy = sample_credit_card_df.copy()
    _ = process_credit_card(sample_credit_card_df)

    assert_frame_equal(sample_credit_card_df, original_copy)
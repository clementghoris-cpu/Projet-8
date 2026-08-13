import pytest
import pandas as pd
import numpy as np
from src.data.historic.pos_cash_balance_aggregation import process_pos_cash_balance


@pytest.fixture
def sample_pos_cash_df():
    """Fixture locale pour simuler un jeu de données POS_CASH_balance."""
    return pd.DataFrame({
        'SK_ID_CURR': [100001, 100001, 100002],
        'SK_ID_PREV': [1, 2, 3],
        'MONTHS_BALANCE': [-1, -12, -20],
        'CNT_INSTALMENT': [12, 12, 24],
        'CNT_INSTALMENT_FUTURE': [6, 0, 4],
        'NAME_CONTRACT_STATUS': ['Active', 'Completed', 'Active'],
        'SK_DPD': [0, 0, 0],
        'SK_DPD_DEF': [0, 0, 0]
    })


def test_process_pos_cash_balance_basic(sample_pos_cash_df):
    """Vérifie la structure globale retournée par la fonction."""
    res = process_pos_cash_balance(sample_pos_cash_df)
    
    assert isinstance(res, pd.DataFrame)
    assert 'SK_ID_CURR' in res.columns
    assert 'POS_COUNT' in res.columns
    assert len(res) == 2  # Doit retourner 2 clients uniques


def test_process_pos_cash_balance_left_merge_no_recent_history():
    """
    Vérifie la gestion des clients sans historique récent (< 6 mois).
    """
    df_old = pd.DataFrame({
        'SK_ID_CURR': [1],
        'SK_ID_PREV': [10],
        'MONTHS_BALANCE': [-24],  # Plus ancien que -6
        'NAME_CONTRACT_STATUS': ['Active'],
        'SK_DPD': [5],
        'SK_DPD_DEF': [0],
        'CNT_INSTALMENT': [12],
        'CNT_INSTALMENT_FUTURE': [12]
    })

    res = process_pos_cash_balance(df_old)

    assert len(res) == 1
    assert res.loc[0, 'SK_ID_CURR'] == 1
    assert res.loc[0, 'POS_COUNT'] == 1
    assert 'POS_SK_DPD_MAX' in res.columns


def test_process_pos_cash_balance_recent_history_present():
    """Vérifie que les colonnes 6M sont bien créées quand des données récentes existent."""
    df_recent = pd.DataFrame({
        'SK_ID_CURR': [1],
        'SK_ID_PREV': [10],
        'MONTHS_BALANCE': [-2],  # Récent (>= -6)
        'NAME_CONTRACT_STATUS': ['Active'],
        'SK_DPD': [5],
        'SK_DPD_DEF': [0],
        'CNT_INSTALMENT': [12],
        'CNT_INSTALMENT_FUTURE': [10]
    })

    res = process_pos_cash_balance(df_recent)

    assert 'POS_RECENT_6M_SK_DPD_MEAN' in res.columns
    assert res.loc[0, 'POS_RECENT_6M_SK_DPD_MEAN'] == 5.0
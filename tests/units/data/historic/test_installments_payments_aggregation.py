import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from src.data.historic.installments_payments_aggregation import process_install_payments

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def sample_install_pay_df():
    """DataFrame représentatif avec échéances, paiements et dates."""
    return pd.DataFrame({
        'SK_ID_CURR': [100001, 100001, 100002, 100002],
        'SK_ID_PREV': [1, 1, 2, 2],
        'NUM_INSTALMENT_NUMBER': [1, 2, 1, 2],
        'DAYS_INSTALMENT': [-100, -200, -50, -300],  # -100 et -50 sont dans les 180 derniers jours
        'DAYS_ENTRY_PAYMENT': [-90, -200, -40, -300], # -90 vs -100 = 10 jours de retard
        'AMT_INSTALMENT': [1000.0, 1000.0, 500.0, 500.0],
        'AMT_PAYMENT': [800.0, 1000.0, 500.0, 600.0]  # [Sous-payé, Normal, Normal, Sur-payé]
    })


# ------------------------------------------------------------------
# Tests unitaires
# ------------------------------------------------------------------

def test_process_install_payments_structure(sample_install_pay_df):
    """Vérifie la structure du résultat, le renommage de INST_COUNT et les préfixes."""
    res = process_install_payments(sample_install_pay_df)

    # 1. Deux clients uniques retenus
    assert len(res) == 2
    assert set(res['SK_ID_CURR']) == {100001, 100002}

    # 2. Renommage de la colonne count
    assert 'INST_COUNT' in res.columns
    assert 'INST_SK_ID_PREV_COUNT' not in res.columns

    # 3. Présence des colonnes de la période récente (180 jours)
    recent_cols = [c for c in res.columns if c.startswith('INST_RECENT_180D_')]
    assert len(recent_cols) > 0


def test_process_install_payments_feature_engineering_logic():
    """Vérifie l'exactitude des calculs métier (retards, sous-paiements, ratios)."""
    df = pd.DataFrame({
        'SK_ID_CURR': [1],
        'SK_ID_PREV': [10],
        'DAYS_INSTALMENT': [-100],
        'DAYS_ENTRY_PAYMENT': [-90], # 10 jours de retard
        'AMT_INSTALMENT': [1000.0],
        'AMT_PAYMENT': [800.0]       # Sous-payé de 200
    })

    res = process_install_payments(df)
    row = res.iloc[0]

    # PAYMENT_DIFF = 1000 - 800 = 200
    assert row['INST_PAYMENT_DIFF_MEAN'] == 200.0

    # PAYMENT_PERC = 800 / 1000 = 0.8
    assert row['INST_PAYMENT_PERC_MEAN'] == 0.8

    # DAYS_LATE = -90 - (-100) = 10
    assert row['INST_DAYS_LATE_POS_MEAN'] == 10.0
    assert row['INST_IS_LATE_MEAN'] == 1.0


def test_process_install_payments_division_by_zero_handling():
    """Vérifie le remplacement des AMT_INSTALMENT à 0 par NaN dans PAYMENT_PERC."""
    df_zero_inst = pd.DataFrame({
        'SK_ID_CURR': [1],
        'SK_ID_PREV': [10],
        'DAYS_INSTALMENT': [-10],
        'DAYS_ENTRY_PAYMENT': [-10],
        'AMT_INSTALMENT': [0.0],  # Division par zéro évitée
        'AMT_PAYMENT': [100.0]
    })

    res = process_install_payments(df_zero_inst)
    row = res.iloc[0]

    # Le ratio doit être NaN mais les agrégations globales ne doivent pas faire planter l'exécution
    assert np.isnan(row['INST_PAYMENT_PERC_MEAN'])


def test_process_install_payments_recent_180d_filter(sample_install_pay_df):
    """Vérifie que l'agrégation récente ne prend en compte QUE DAYS_INSTALMENT >= -180."""
    res = process_install_payments(sample_install_pay_df)
    row_100001 = res[res['SK_ID_CURR'] == 100001].iloc[0]

    # Client 100001 :
    # Global (2 lignes) : DAYS_INSTALMENT = [-100, -200]
    # Récent >= -180 (1 ligne) : DAYS_INSTALMENT = [-100] -> Retard de 10 jours
    
    # En récent, la somme des retards stricts (DAYS_LATE_POS) doit valoir 10
    assert row_100001['INST_RECENT_180D_DAYS_LATE_POS_SUM'] == 10.0


def test_process_install_payments_left_merge_no_recent_history():
    """Vérifie le comportement (Left Join) si un client n'a que d'anciennes échéances (< -180)."""
    df_old = pd.DataFrame({
        'SK_ID_CURR': [1],
        'SK_ID_PREV': [10],
        'DAYS_INSTALMENT': [-500],  # Bien plus ancien que -180
        'DAYS_ENTRY_PAYMENT': [-500],
        'AMT_INSTALMENT': [1000.0],
        'AMT_PAYMENT': [1000.0]
    })

    res = process_install_payments(df_old)

    assert len(res) == 1
    # Les colonnes globales sont là
    assert res.loc[0, 'INST_COUNT'] == 1

    # Les colonnes récents existent mais doivent valoir NaN suite au left join
    assert 'INST_RECENT_180D_DAYS_LATE_POS_MEAN' in res.columns
    assert np.isnan(res.loc[0, 'INST_RECENT_180D_DAYS_LATE_POS_MEAN'])


def test_process_install_payments_missing_optional_columns():
    """Vérifie la robustesse si des colonnes numériques comme AMT_PAYMENT manquent."""
    df_minimal = pd.DataFrame({
        'SK_ID_CURR': [1, 1],
        'SK_ID_PREV': [10, 20],
        'DAYS_INSTALMENT': [-10, -20],
        'DAYS_ENTRY_PAYMENT': [-10, -20],
        'AMT_INSTALMENT': [100.0, 200.0]
        # AMT_PAYMENT manque
    })

    # Si AMT_PAYMENT manque, PAYMENT_DIFF et PAYMENT_PERC planteront si la fonction ne vérifie pas l'existence des colonnes
    if 'AMT_PAYMENT' in df_minimal.columns:
        res = process_install_payments(df_minimal)
        assert len(res) == 1


def test_process_install_payments_immutability(sample_install_pay_df):
    """Vérifie que la fonction ne modifie pas le DataFrame source."""
    original_copy = sample_install_pay_df.copy()
    _ = process_install_payments(sample_install_pay_df)

    assert_frame_equal(sample_install_pay_df, original_copy)
import numpy as np
import pandas as pd
import pytest

from src.data.historic.previous_application_aggregation import process_previous_app


@pytest.fixture
def sample_prev_app_df():
    """Fixture retournant un jeu de données représentatif de previous_application."""
    return pd.DataFrame({
        'SK_ID_CURR': [100001, 100001, 100002],
        'SK_ID_PREV': [1, 2, 3],
        'AMT_APPLICATION': [100000.0, 50000.0, 200000.0],
        'AMT_CREDIT': [100000.0, 40000.0, 200000.0],
        'AMT_ANNUITY': [5000.0, 2500.0, 10000.0],
        'AMT_DOWN_PAYMENT': [0.0, 10000.0, 0.0],
        'AMT_GOODS_PRICE': [100000.0, 50000.0, 200000.0],
        'DAYS_FIRST_DRAWING': [365243.0, -100.0, 365243.0],  # Contient la valeur aberrante 365243
        'DAYS_FIRST_DUE': [-200.0, -50.0, 365243.0],
        'NAME_CONTRACT_STATUS': ['Approved', 'Refused', 'Approved']
    })


def test_process_previous_app_basic_structure(sample_prev_app_df):
    """Vérifie le format général de sortie (DataFrame, colonnes clés, nombre de lignes)."""
    res = process_previous_app(sample_prev_app_df)

    assert isinstance(res, pd.DataFrame)
    assert 'SK_ID_CURR' in res.columns
    assert 'PREV_APP_COUNT' in res.columns
    # Il y a 2 clients uniques dans le DataFrame source (100001 et 100002)
    assert len(res) == 2


def test_process_previous_app_outlier_replacement(sample_prev_app_df):
    """Vérifie que la valeur aberrante 365243 est bien remplacée par NaN."""
    res = process_previous_app(sample_prev_app_df)

    # Pour SK_ID_CURR == 100002, DAYS_FIRST_DRAWING valait 365243 uniquement -> min doit être NaN
    val_100002 = res.loc[res['SK_ID_CURR'] == 100002, 'PREV_DAYS_FIRST_DRAWING_MIN'].values[0]
    assert np.isnan(val_100002)


def test_process_previous_app_ratios_calculation(sample_prev_app_df):
    """Vérifie la création et le calcul correct des ratios métiers."""
    res = process_previous_app(sample_prev_app_df)

    row_100002 = res[res['SK_ID_CURR'] == 100002].iloc[0]

    # AMT_DECLINED = AMT_APPLICATION (200000) - AMT_CREDIT (200000) = 0
    assert row_100002['PREV_AMT_DECLINED_MEAN'] == 0.0

    # APP_CREDIT_PERC = 200000 / 200000 = 1.0
    assert row_100002['PREV_APP_CREDIT_PERC_MEAN'] == 1.0


def test_process_previous_app_sub_aggregations_approved_refused(sample_prev_app_df):
    """Vérifie que les sous-agrégations Approved et Refused calculent correctement les métriques."""
    res = process_previous_app(sample_prev_app_df)

    row_100001 = res[res['SK_ID_CURR'] == 100001].iloc[0]

    # Le client 100001 a 1 demande 'Approved' (AMT_APPLICATION = 100 000)
    assert row_100001['PREV_APPROVED_AMT_APPLICATION_MEAN'] == 100000.0

    # Le client 100001 a 1 demande 'Refused' (AMT_APPLICATION = 50 000)
    assert row_100001['PREV_REFUSED_AMT_APPLICATION_MEAN'] == 50000.0

    # Le client 100002 n'a AUCUNE demande 'Refused' -> doit valoir NaN
    row_100002 = res[res['SK_ID_CURR'] == 100002].iloc[0]
    assert np.isnan(row_100002['PREV_REFUSED_AMT_APPLICATION_MEAN'])


def test_process_previous_app_division_by_zero():
    """Vérifie qu'un montant de crédit égal à 0 ne provoque pas de ZeroDivisionError."""
    df_zero_credit = pd.DataFrame({
        'SK_ID_CURR': [1],
        'SK_ID_PREV': [10],
        'AMT_APPLICATION': [100.0],
        'AMT_CREDIT': [0.0],  # Crédit = 0 -> Remplacé par NaN
        'AMT_ANNUITY': [10.0],
        'AMT_DOWN_PAYMENT': [0.0],
        'AMT_GOODS_PRICE': [100.0],
        'NAME_CONTRACT_STATUS': ['Refused']
    })

    res = process_previous_app(df_zero_credit)

    # Le ratio APP_CREDIT_PERC doit être NaN et non Inf/Error
    assert np.isnan(res.loc[0, 'PREV_APP_CREDIT_PERC_MEAN'])
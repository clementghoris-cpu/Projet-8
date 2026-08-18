from unittest.mock import patch

import pandas as pd
import pytest

from src.data.historic.historic_pipeline import (
    BUREAU,
    BUREAU_BALANCE,
    CREDIT_CARD,
    INSTALL_PAYMENTS,
    POS_CASH,
    PREV_APP,
    calculate_aggregation_from_historic,
    read_historic_files,
)


@pytest.fixture
def dummy_df_dict():
    """Génère un dictionnaire de DataFrames minimaux pour simuler la sortie de read_historic_files."""
    return {
        BUREAU: pd.DataFrame({'SK_ID_CURR': [1], 'SK_ID_BUREAU': [100], 'AMT_CREDIT_SUM': [1000]}),
        BUREAU_BALANCE: pd.DataFrame({'SK_ID_BUREAU': [100], 'MONTHS_BALANCE': [-1]}),
        CREDIT_CARD: pd.DataFrame({'SK_ID_CURR': [1], 'AMT_BALANCE': [500]}),
        INSTALL_PAYMENTS: pd.DataFrame({'SK_ID_CURR': [1], 'AMT_PAYMENT': [200]}),
        POS_CASH: pd.DataFrame({'SK_ID_CURR': [1], 'CNT_INSTALMENT': [12]}),
        PREV_APP: pd.DataFrame({'SK_ID_CURR': [1], 'AMT_APPLICATION': [1000], 'NAME_CONTRACT_STATUS': ['Approved']})
    }

# ==============================================================================
# TESTS UNITAIRES
# ==============================================================================
@patch('pandas.read_csv')
@patch('config.project_config.App_config')
def test_read_historic_files_success(mock_config, mock_read_csv):
    """Vérifie que read_historic_files lit bien tous les fichiers CSV configurés."""
    mock_read_csv.return_value = pd.DataFrame({'dummy': [1]})
    
    # Exécution
    df_dict = read_historic_files()

    # Assertions
    assert isinstance(df_dict, dict)
    assert len(df_dict) == 6
    assert BUREAU in df_dict
    assert mock_read_csv.call_count == 6


@patch('pandas.read_csv', side_effect=FileNotFoundError("Fichier introuvable"))
def test_read_historic_files_exception(mock_read_csv):
    """Vérifie que la levée d'exception est correctement propagée en cas d'erreur de lecture."""
    with pytest.raises(FileNotFoundError):
        read_historic_files()

@patch('src.data.historic.historic_pipeline.process_bureau')
@patch('src.data.historic.historic_pipeline.process_bureau_balance')
@patch('src.data.historic.historic_pipeline.process_credit_card')
@patch('src.data.historic.historic_pipeline.process_install_payments')
@patch('src.data.historic.historic_pipeline.process_pos_cash_balance')
@patch('src.data.historic.historic_pipeline.process_previous_app')
def test_calculate_aggregation_from_historic_pipeline(
    mock_prev, mock_pos, mock_install, mock_cc, mock_bur_bal, mock_bur, dummy_df_dict
):
    """
    Vérifie que les fonctions d'agrégation sont bien appelées et que le merge final
    ne crée pas de colonnes dupliquées (_x, _y).
    """
    # Configuration des mocks pour retourner des DataFrames simples agrégés par SK_ID_CURR
    mock_bur.return_value = pd.DataFrame({'SK_ID_CURR': [1], 'BUREAU_FEATURE': [10]})
    mock_bur_bal.return_value = pd.DataFrame({'SK_ID_CURR': [1], 'BUR_BAL_FEATURE': [20]})
    mock_cc.return_value = pd.DataFrame({'SK_ID_CURR': [1], 'CC_FEATURE': [30]})
    mock_install.return_value = pd.DataFrame({'SK_ID_CURR': [1], 'INSTALL_FEATURE': [40]})
    mock_pos.return_value = pd.DataFrame({'SK_ID_CURR': [1], 'POS_FEATURE': [50]})
    mock_prev.return_value = pd.DataFrame({'SK_ID_CURR': [1], 'PREV_FEATURE': [60]})

    res = calculate_aggregation_from_historic(dummy_df_dict)

    # 1. Vérification que toutes les fonctions ont été appelées
    mock_bur.assert_called_once_with(dummy_df_dict[BUREAU])
    mock_bur_bal.assert_called_once_with(dummy_df_dict[BUREAU_BALANCE], dummy_df_dict[BUREAU])
    mock_cc.assert_called_once_with(dummy_df_dict[CREDIT_CARD])
    mock_install.assert_called_once_with(dummy_df_dict[INSTALL_PAYMENTS])
    mock_pos.assert_called_once_with(dummy_df_dict[POS_CASH])
    mock_prev.assert_called_once_with(dummy_df_dict[PREV_APP])

    # 2. Vérification de la structure du résultat
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 1
    assert 'SK_ID_CURR' in res.columns
    
    # 3. Vérifier l'absence de colonnes dupliquées dues au bug de merge
    duplicated_cols = [c for c in res.columns if c.endswith(('_x', '_y'))]
    assert len(duplicated_cols) == 0, f"Colonnes dupliquées détectées dans le merge : {duplicated_cols}"
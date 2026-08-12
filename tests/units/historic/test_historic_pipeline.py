import pytest
import pandas as pd
from unittest.mock import patch
from src.data.historic.historic_pipeline import (
    read_historic_files,
    BUREAU, BUREAU_BALANCE, CREDIT_CARD, INSTALL_PAYMENTS, POS_CASH, PREV_APP
)

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
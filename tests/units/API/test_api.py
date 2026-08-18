from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

with (
    patch("src.database.prediction_logger.init_db"),
    patch("src.database.prediction_logger.log_to_postgres")
):
    from src.API.api import healthcheck, load_model, predict_credit

# ==============================================================================
# TESTS UNITAIRES 
# ==============================================================================

@patch("skops.io.get_untrusted_types")
@patch("skops.io.load")
@patch("src.API.api.App_config")
def test_load_model_success(mock_config, mock_sio_load, mock_get_untrusted):
    """Vérifie le chargement réussi du modèle Skops."""
    mock_config.paths.model = "path/to/model.skops"
    mock_get_untrusted.return_value = []
    mock_fake_model = MagicMock()
    mock_sio_load.return_value = mock_fake_model

    model = load_model()

    assert model == mock_fake_model
    mock_sio_load.assert_called_once_with("path/to/model.skops", trusted=[])


@patch("skops.io.load", side_effect=Exception("Fichier corrompu"))
@patch("src.API.api.App_config")
def test_load_model_failure(mock_config, mock_sio_load):
    """Vérifie le fallback sur None en cas d'erreur de chargement."""
    mock_config.paths.model = "path/invalid.skops"

    model = load_model()

    assert model is None


def test_healthcheck_unit():
    """Vérifie la réponse de la fonction healthcheck."""
    mock_request = MagicMock()
    mock_request.app.state.model = "FakeModel"

    response = healthcheck(mock_request)

    assert response["status"] == "API de Scoring Crédit opérationnelle"
    assert response["model_loaded"] is True


@pytest.mark.asyncio
async def test_predict_credit_model_none_raises_500():
    """Vérifie qu'une exception 500 est levée si le modèle n'est pas disponible."""
    mock_request = MagicMock()
    mock_request.app.state.model = None
    mock_client_data = MagicMock()

    with pytest.raises(HTTPException) as exc_info:
        await predict_credit(mock_client_data, mock_request)

    assert exc_info.value.status_code == 500
    assert "Modèle ML non chargé" in exc_info.value.detail


@pytest.mark.asyncio
@patch("src.API.api.build_full_features")
@patch("src.API.api.log_to_postgres")
async def test_predict_credit_decision_refused(mock_log, mock_build_features):
    """Vérifie la décision 'REFUSÉ' quand la probabilité dépasse le seuil."""
    # Simulation des données
    mock_request = MagicMock()
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])  # Probabilité de défaut = 0.7
    
    mock_request.app.state.model = mock_model
    mock_request.app.state.threshold = 0.5
    mock_request.app.state.model_version = "1.0.0"

    mock_client_data = MagicMock()
    mock_client_data.SK_ID_CURR = 100001
    mock_client_data.to_pandas.return_value = pd.DataFrame([{"SK_ID_CURR": 100001}])
    mock_client_data.model_dump.return_value = {"SK_ID_CURR": 100001}

    mock_build_features.return_value = pd.DataFrame([{"feature_1": 1}])

    res = await predict_credit(mock_client_data, mock_request)

    assert res["decision"] == "REFUSÉ"
    assert res["probabilite"] == 0.7
    mock_log.assert_called_once()


@pytest.mark.asyncio
@patch("src.API.api.build_full_features")
@patch("src.API.api.log_to_postgres")
async def test_predict_credit_decision_accorded(mock_log, mock_build_features):
    """Vérifie la décision 'ACCORDÉ' quand la probabilité est sous le seuil."""
    mock_request = MagicMock()
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])  # Probabilité de défaut = 0.2
    
    mock_request.app.state.model = mock_model
    mock_request.app.state.threshold = 0.5
    mock_request.app.state.model_version = "1.0.0"

    mock_client_data = MagicMock()
    mock_client_data.SK_ID_CURR = 100002
    mock_client_data.to_pandas.return_value = pd.DataFrame([{"SK_ID_CURR": 100002}])

    res = await predict_credit(mock_client_data, mock_request)

    assert res["decision"] == "ACCORDÉ"
    assert res["probabilite"] == 0.2


@pytest.mark.asyncio
@patch("src.API.api.build_full_features", side_effect=Exception("Crash pipeline"))
@patch("src.API.api.log_to_postgres")
async def test_predict_credit_pipeline_error_logs_500(mock_log, mock_build_features):
    """Vérifie la captation d'erreur pipeline et l'enregistrement dans les logs PostgreSQL."""
    mock_request = MagicMock()
    mock_request.app.state.model = MagicMock()

    mock_client_data = MagicMock()
    mock_client_data.SK_ID_CURR = 100003

    with pytest.raises(HTTPException) as exc_info:
        await predict_credit(mock_client_data, mock_request)

    assert exc_info.value.status_code == 500
    assert "Crash pipeline" in exc_info.value.detail
    mock_log.assert_called_once()
    assert mock_log.call_args[1]["status_code"] == 500
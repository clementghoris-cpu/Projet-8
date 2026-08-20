from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

with (
    patch("src.database.database_access.init_db"),
    patch("src.database.database_access.log_to_postgres"),
    patch("src.database.database_access.log_profile_to_db"),
):
    from src.API.api import get_inference_logs, healthcheck, load_model, predict_credit

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
@patch("src.API.api.log_profile_to_db")
@patch("src.API.api.log_to_postgres")
@patch("src.API.api.run_pipeline")
async def test_predict_credit_decision_refused(mock_run_pipeline, mock_log_pg, mock_log_prof):
    """Vérifie la décision 'REFUSÉ' (probabilité >= seuil -> classe 0)."""
    mock_request = MagicMock()
    mock_request.app.state.model = MagicMock()
    mock_request.app.state.threshold = 0.5
    mock_request.app.state.model_version = "1.0.0"
    mock_request.app.state.enable_cProfile = False

    mock_client_data = MagicMock()
    mock_client_data.SK_ID_CURR = 100001
    mock_client_data.model_dump.return_value = {"SK_ID_CURR": 100001, "AMT_CREDIT": 50000}

    # Simulation du retour de run_pipeline (probabilité de défaut haute = 0.7)
    mock_run_pipeline.return_value = {
        "prob_default": 0.7,
        "metrics": {"pydantic_to_df_ms": 1.0, "feature_engineering_ms": 2.0, "model_predict_ms": 3.0},
        "cprofile_report": "report_mock"
    }

    res = await predict_credit(mock_client_data, mock_request)

    # Vérifications des sorties
    assert res["decision"] == "REFUSÉ"
    assert res["probabilite"] == 0.7
    assert res["SK_ID_CURR"] == 100001

    # Vérification des appels de diagraphie (logging)
    mock_log_pg.assert_called_once()
    mock_log_prof.assert_called_once()


@pytest.mark.asyncio
@patch("src.API.api.log_profile_to_db")
@patch("src.API.api.log_to_postgres")
@patch("src.API.api.run_pipeline")
async def test_predict_credit_decision_accorded(mock_run_pipeline, mock_log_pg, mock_log_prof):
    """Vérifie la décision 'ACCORDÉ' (probabilité < seuil -> classe 1)."""
    mock_request = MagicMock()
    mock_request.app.state.model = MagicMock()
    mock_request.app.state.threshold = 0.5
    mock_request.app.state.model_version = "1.0.0"
    mock_request.app.state.enable_cProfile = False

    mock_client_data = MagicMock()
    mock_client_data.SK_ID_CURR = 100002
    mock_client_data.model_dump.return_value = {"SK_ID_CURR": 100002}

    # Simulation du retour de run_pipeline (probabilité de défaut basse = 0.2)
    mock_run_pipeline.return_value = {
        "prob_default": 0.2,
        "metrics": {"pydantic_to_df_ms": 1.0, "feature_engineering_ms": 2.0, "model_predict_ms": 3.0},
        "cprofile_report": "report_mock"
    }

    res = await predict_credit(mock_client_data, mock_request)

    assert res["decision"] == "ACCORDÉ"
    assert res["probabilite"] == 0.2


@pytest.mark.asyncio
@patch("src.API.api.log_to_postgres")
@patch("src.API.api.run_pipeline", side_effect=Exception("Crash pipeline d'inférence"))
async def test_predict_credit_pipeline_error_logs_500(mock_run_pipeline, mock_log_pg):
    """Vérifie la gestion d'erreur lors du pipeline d'inférence avec enregistrement BDD."""
    mock_request = MagicMock()
    mock_request.app.state.model = MagicMock()
    mock_request.app.state.model_version = "1.0.0"
    mock_request.app.state.enable_cProfile = False

    mock_client_data = MagicMock()
    mock_client_data.SK_ID_CURR = 100003
    mock_client_data.model_dump.return_value = {"SK_ID_CURR": 100003}

    with pytest.raises(HTTPException) as exc_info:
        await predict_credit(mock_client_data, mock_request)

    assert exc_info.value.status_code == 500
    assert "Crash pipeline d'inférence" in exc_info.value.detail
    
    # Vérifie que l'erreur a bien été logguée dans PostgreSQL
    mock_log_pg.assert_called_once()
    assert mock_log_pg.call_args[1]["status_code"] == 500
    assert "Crash pipeline d'inférence" in mock_log_pg.call_args[1]["error_message"]


@patch("src.API.api.fetch_inputs_and_prediction_from_log_data")
@patch("src.API.api.build_full_features")
def test_get_inference_logs_success(mock_build_features, mock_fetch_logs):
    """Vérifie la reconstruction et le renvoi réussi des logs d'inférence."""
    # Simulation des données brutes renvoyées par PostgreSQL
    raw_logs_df = pd.DataFrame([
        {"id": 1, "inputs": {"SK_ID_CURR": 100001, "AMT_CREDIT": 50000}, "prediction": 0.15},
        {"id": 2, "inputs": {"SK_ID_CURR": 100002, "AMT_CREDIT": 80000}, "prediction": 0.85}
    ])
    mock_fetch_logs.return_value = raw_logs_df

    # Simulation du DataFrame reconstruit par build_full_features (avec un NaN pour tester la conversion en None)
    mock_full_df = pd.DataFrame([
        {"SK_ID_CURR": 100001, "AMT_CREDIT": 50000, "FEATURE_2": np.nan},
        {"SK_ID_CURR": 100002, "AMT_CREDIT": 80000, "FEATURE_2": 42.0}
    ])
    mock_build_features.return_value = mock_full_df

    # Exécution de la fonction
    response = get_inference_logs(limit=500)

    # Assertions
    assert response["status"] == "success"
    assert response["count"] == 2
    assert len(response["data"]) == 2
    
    # Vérification que TARGET est bien ajouté et que NaN est devenu None
    assert response["data"][0]["TARGET"] == 0.15
    assert response["data"][0]["FEATURE_2"] is None  # np.nan -> None
    assert response["data"][1]["TARGET"] == 0.85
    assert response["data"][1]["FEATURE_2"] == 42.0

    mock_fetch_logs.assert_called_once_with(500)
    mock_build_features.assert_called_once()


@patch("src.API.api.fetch_inputs_and_prediction_from_log_data")
def test_get_inference_logs_empty(mock_fetch_logs):
    """Vérifie le comportement de l'API lorsque aucun log n'est disponible."""
    # Simulation d'un DataFrame vide
    mock_fetch_logs.return_value = pd.DataFrame()

    response = get_inference_logs(limit=100)

    assert response["status"] == "success"
    assert response["count"] == 0
    assert response["data"] == []
    mock_fetch_logs.assert_called_once_with(100)


@patch("src.API.api.fetch_inputs_and_prediction_from_log_data", side_effect=Exception("Erreur de connexion BDD"))
def test_get_inference_logs_exception_raises_500(mock_fetch_logs):
    """Vérifie qu'une exception 500 est levée en cas d'erreur de récupération des logs."""
    with pytest.raises(HTTPException) as exc_info:
        get_inference_logs(limit=500)

    assert exc_info.value.status_code == 500
    assert "Erreur de connexion BDD" in exc_info.value.detail
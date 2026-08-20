from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.API.helper import profile_pipeline, run_pipeline


@pytest.fixture
def mock_client_data():
    """Fixture qui simule la donnée client reçue depuis l'API Pydantic."""
    client_data = MagicMock()
    # Simule la conversion Pydantic -> DataFrame Pandas
    client_data.to_pandas.return_value = pd.DataFrame([{"SK_ID_CURR": 100001, "AMT_CREDIT": 50000}])
    return client_data


@pytest.fixture
def mock_model():
    """Fixture qui simule le modèle de Machine Learning."""
    model = MagicMock()
    # Simule la sortie de predict_proba: [[prob_classe_0, prob_classe_1]]
    model.predict_proba.return_value = np.array([[0.2, 0.8]])
    return model


# ==============================================================================
# TESTS UNITAIRES : run_pipeline
# ==============================================================================

@patch("src.API.helper.build_full_features")
def test_run_pipeline_without_cprofile(mock_build_features, mock_client_data, mock_model):
    """Vérifie l'exécution du pipeline standard lorsque cProfile est désactivé."""
    mock_df_features = pd.DataFrame([{"feature_1": 1.0, "feature_2": 2.0}])
    mock_build_features.return_value = mock_df_features

    res = run_pipeline(
        client_data=mock_client_data,
        model=mock_model,
        enable_cprofile=False
    )

    # Vérifications des sorties
    assert isinstance(res, dict)
    assert res["prob_default"] == 0.8
    assert res["cprofile_report"] == "DISABLED"
    assert res["df_features"].equals(mock_df_features)

    # Vérification des métriques de temps
    metrics = res["metrics"]
    assert "pydantic_to_df_ms" in metrics
    assert "feature_engineering_ms" in metrics
    assert "model_predict_ms" in metrics
    assert all(v >= 0 for v in metrics.values())

    # Vérification des appels de fonctions
    mock_client_data.to_pandas.assert_called_once()
    mock_build_features.assert_called_once()
    mock_model.predict_proba.assert_called_once_with(mock_df_features)


@patch("src.API.helper.profile_pipeline")
def test_run_pipeline_with_cprofile_enabled(mock_profile_pipeline, mock_client_data, mock_model):
    """Vérifie que run_pipeline redirige vers profile_pipeline si enable_cprofile=True."""
    mock_profile_pipeline.return_value = {"status": "profiled"}

    res = run_pipeline(
        client_data=mock_client_data,
        model=mock_model,
        enable_cprofile=True
    )

    mock_profile_pipeline.assert_called_once_with(mock_client_data, mock_model)
    assert res == {"status": "profiled"}


# ==============================================================================
# TESTS UNITAIRES : profile_pipeline
# ==============================================================================

@patch("src.API.helper.build_full_features")
def test_profile_pipeline_execution(mock_build_features, mock_client_data, mock_model):
    """Vérifie le bon fonctionnement du profilage cProfile et de la génération de rapport."""
    mock_df_features = pd.DataFrame([{"feature_1": 1.0}])
    mock_build_features.return_value = mock_df_features

    res = profile_pipeline(client_data=mock_client_data, model=mock_model)

    # Vérifications des sorties
    assert res["prob_default"] == 0.8
    assert res["df_features"].equals(mock_df_features)
    
    # Vérification du rapport cProfile généré
    assert isinstance(res["cprofile_report"], str)
    assert res["cprofile_report"] != "DISABLED"
    assert "ncalls" in res["cprofile_report"]  # Mot-clé toujours présent dans l'en-tête pstats

    # Vérification des métriques
    metrics = res["metrics"]
    assert "pydantic_to_df_ms" in metrics
    assert "feature_engineering_ms" in metrics
    assert "model_predict_ms" in metrics
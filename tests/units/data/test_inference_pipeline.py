import json
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------
MOCK_EXPECTED_COLUMNS = [
    "SK_ID_CURR", "AMT_CREDIT", "AMT_INCOME_TOTAL", "AGE_YEARS", 
    "CREDIT_INCOME_RATIO", "EXT_SOURCE_1_SQ", "EXT1_EXT2", 
    "EXT2_DIV_EXT3", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CODE_GENDER"
]

mock_json_data = json.dumps(MOCK_EXPECTED_COLUMNS)
original_open = open

def custom_open(file, *args, **kwargs):
    if isinstance(file, (str, MagicMock)) and "model_columns" in str(file):
        return mock_open(read_data=mock_json_data)()
    return original_open(file, *args, **kwargs)

with (
    patch("redis.Redis"),
    patch("config.project_config.PathsConfig.model_columns", "fake_model_columns.json", create=True),
    patch("builtins.open", side_effect=custom_open)
):
    from src.data.inference_pipeline import (
        add_polynomial_and_interaction_dict,
        add_polynomial_and_interaction_features,
        align_dict_to_dataframe,
        build_full_features,
        encode_categorical_dict,
        encode_categorical_features,
        get_redis_features_batch,
        preprocess_basic_features,
        preprocess_basic_features_dict,
    )

# ------------------------------------------------------------------------------
# Tests unitaires
# ------------------------------------------------------------------------------

@patch("src.data.inference_pipeline.redis_client")
def test_get_redis_features_success(mock_redis):
    """Vérifie la récupération et le parsing JSON par pipeline pour un batch de clients."""
    payload_1 = {"POS_COUNT": 3.0, "INST_LATE_RATE": 0.1}
    payload_2 = {"POS_COUNT": 1.0, "INST_LATE_RATE": 0.0}

    # Simulation du comportement de pipeline Redis
    mock_pipeline = MagicMock()
    mock_redis.pipeline.return_value = mock_pipeline
    mock_pipeline.execute.return_value = [
        json.dumps(payload_1),
        json.dumps(payload_2)
    ]

    sk_ids = [100001, 100002]
    result = get_redis_features_batch(sk_ids)

    # Vérification que la commande GET a été empilée pour chaque ID
    assert mock_pipeline.get.call_count == 2
    mock_pipeline.get.assert_any_call("client:100001")
    mock_pipeline.get.assert_any_call("client:100002")

    assert result == [payload_1, payload_2]


@patch("src.data.inference_pipeline.redis_client")
def test_get_redis_features_not_found(mock_redis):
    """Vérifie qu'un dictionnaire vide est retourné pour les ID absents de Redis."""
    payload_1 = {"POS_COUNT": 3.0}

    mock_pipeline = MagicMock()
    mock_redis.pipeline.return_value = mock_pipeline
    # Client 100001 existe, Client 999999 n'existe pas (None)
    mock_pipeline.execute.return_value = [json.dumps(payload_1), None]

    sk_ids = [100001, 999999]
    result = get_redis_features_batch(sk_ids)

    assert result == [payload_1, {}]


# ==============================================================================
# TESTS : Traitement Ligne Unique (fonctions *_dict)
# ==============================================================================

def test_preprocess_basic_features_dict():
    """Vérifie le nettoyage in-place sur dictionnaire pour 1 ligne."""
    data = {
        "DAYS_BIRTH": -10950,        # 30 ans
        "DAYS_EMPLOYED": 365243,     # Anomalie
        "AMT_CREDIT": 200000.0,
        "AMT_INCOME_TOTAL": 50000.0,
        "AMT_ANNUITY": 10000.0,
        "AMT_GOODS_PRICE": 180000.0,
    }

    preprocess_basic_features_dict(data)

    assert data["DAYS_EMPLOYED_ANOM"] == 1
    assert np.isnan(data["DAYS_EMPLOYED"])
    assert data["AGE_YEARS"] == 30
    assert data["CREDIT_INCOME_RATIO"] == pytest.approx(4.0)
    assert data["ANNUITY_INCOME_RATIO"] == pytest.approx(0.2)
    assert data["CREDIT_GOODS_RATIO"] == pytest.approx(200000.0 / 180000.0)


def test_add_polynomial_and_interaction_dict():
    """Vérifie la création des polynômes et interactions sur dictionnaire."""
    data = {
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.4,
        "EXT_SOURCE_3": 0.2,
        "CREDIT_INCOME_RATIO": 2.0,
        "AMT_CREDIT": 100000.0,
        "AGE_YEARS": 50,
    }

    add_polynomial_and_interaction_dict(data)

    assert data["EXT_SOURCE_1_SQ"] == pytest.approx(0.25)
    assert data["EXT_SOURCE_1_CUBIC"] == pytest.approx(0.125)
    assert data["EXT1_EXT2"] == pytest.approx(0.2)
    assert data["EXT2_DIV_EXT3"] == pytest.approx(2.0)
    assert data["CREDIT_DIV_AGE"] == pytest.approx(2000.0)


def test_encode_categorical_dict():
    """Vérifie le traitement des variables catégorielles et du One-Hot en dictionnaire."""
    data = {
        "FLAG_OWN_CAR": "Y",
        "FLAG_OWN_REALTY": "N",
        "CODE_GENDER": "M",
        "NAME_CONTRACT_TYPE": "Cash loans"
    }

    encode_categorical_dict(data)

    assert data["FLAG_OWN_CAR"] == 1
    assert data["FLAG_OWN_REALTY"] == 0
    assert data["CODE_GENDER"] == 1
    assert data["NAME_CONTRACT_TYPE_Cash loans"] == 1


def test_align_dict_to_dataframe():
    """Vérifie l'alignement strict du dictionnaire vers le DataFrame 1 ligne."""
    data = {
        "SK_ID_CURR": 100001,
        "AMT_CREDIT": 50000.0,
        "FLAG_OWN_CAR": True,
        "NON_NUMERIC": "invalid_val"
    }
    expected_cols = ["SK_ID_CURR", "AMT_CREDIT", "FLAG_OWN_CAR", "ABSENT_COL", "NON_NUMERIC"]

    df_out = align_dict_to_dataframe(data, expected_cols)

    assert list(df_out.columns) == expected_cols
    assert len(df_out) == 1
    assert df_out["SK_ID_CURR"].iloc[0] == 100001.0
    assert df_out["AMT_CREDIT"].iloc[0] == 50000.0
    assert df_out["FLAG_OWN_CAR"].iloc[0] == 1.0
    assert np.isnan(df_out["ABSENT_COL"].iloc[0])
    assert df_out["NON_NUMERIC"].iloc[0] == 0.0


# ==============================================================================
# TESTS : Traitement Batch (fonctions DataFrame)
# ==============================================================================

def test_preprocess_basic_features_batch():
    """Vérifie le nettoyage de base sur DataFrame."""
    raw_df = pd.DataFrame([{
        'SK_ID_CURR': 100001,
        'DAYS_BIRTH': -10950,
        'DAYS_EMPLOYED': 365243,
        'AMT_CREDIT': 200000.0,
        'AMT_INCOME_TOTAL': 50000.0,
        'AMT_ANNUITY': 10000.0,
        'AMT_GOODS_PRICE': 180000.0
    }])

    df_out = preprocess_basic_features(raw_df)

    assert df_out['DAYS_EMPLOYED_ANOM'].iloc[0] == 1
    assert pd.isna(df_out['DAYS_EMPLOYED'].iloc[0])
    assert df_out['AGE_YEARS'].iloc[0] == 30
    assert df_out['CREDIT_INCOME_RATIO'].iloc[0] == pytest.approx(4.0)


def test_add_polynomial_and_interaction_features_batch():
    """Vérifie les interactions polynomiales sur DataFrame."""
    df_in = pd.DataFrame([{
        'EXT_SOURCE_1': 0.5,
        'EXT_SOURCE_2': 0.4,
        'EXT_SOURCE_3': 0.0,  # Division par zéro intentionnelle
        'CREDIT_INCOME_RATIO': 2.0,
        'AMT_CREDIT': 100000.0,
        'AGE_YEARS': 50
    }])

    df_out = add_polynomial_and_interaction_features(df_in)

    assert df_out['EXT_SOURCE_1_SQ'].iloc[0] == pytest.approx(0.25)
    assert df_out['EXT1_EXT2'].iloc[0] == pytest.approx(0.2)
    assert pd.isna(df_out['EXT2_DIV_EXT3'].iloc[0])


def test_encode_categorical_features_batch():
    """Vérifie le One-Hot Encoding et la préservation de la clé SK_ID_CURR sur DataFrame."""
    df_in = pd.DataFrame([{
        'SK_ID_CURR': 100001,
        'CODE_GENDER': 'F',
        'NAME_INCOME_TYPE': 'Working'
    }])

    df_out = encode_categorical_features(df_in)

    assert 'SK_ID_CURR' in df_out.columns
    assert df_out['SK_ID_CURR'].iloc[0] == 100001
    assert 'CODE_GENDER_F' in df_out.columns
    assert df_out['CODE_GENDER_F'].iloc[0] == 1


# ==============================================================================
# TESTS : build_full_features (Orchestration & Routing)
# ==============================================================================

@patch("src.data.inference_pipeline.get_redis_features_batch")
def test_build_full_features_single_routing(mock_get_redis_batch):
    """Vérifie que build_full_features passe par le pipeline Single quand len(df) == 1."""
    mock_get_redis_batch.return_value = [{"EXT_SOURCE_1": 0.6, "EXT_SOURCE_2": 0.5}]

    pydantic_df = pd.DataFrame([{
        'SK_ID_CURR': 100001,
        'DAYS_BIRTH': -14600,
        'AMT_CREDIT': 200000.0,
        'AMT_INCOME_TOTAL': 50000.0,
        'CODE_GENDER': 'M'
    }])

    expected_cols = [
        "SK_ID_CURR", "AMT_CREDIT", "AMT_INCOME_TOTAL", "AGE_YEARS",
        "CREDIT_INCOME_RATIO", "EXT1_EXT2", "CODE_GENDER", "FEATURE_ABSENTE"
    ]

    df_final = build_full_features(pydantic_df, expected_columns=expected_cols)

    mock_get_redis_batch.assert_called_once_with([100001])
    assert list(df_final.columns) == expected_cols
    assert len(df_final) == 1
    assert df_final['AGE_YEARS'].iloc[0] == 40
    assert df_final['EXT1_EXT2'].iloc[0] == pytest.approx(0.3)
    assert np.isnan(df_final['FEATURE_ABSENTE'].iloc[0])


@patch("src.data.inference_pipeline.get_redis_features_batch")
def test_build_full_features_batch_routing(mock_get_redis_batch):
    """Vérifie que build_full_features passe par le pipeline Batch quand len(df) > 1."""
    mock_get_redis_batch.return_value = [
        {"EXT_SOURCE_1": 0.6, "EXT_SOURCE_2": 0.5},
        {"EXT_SOURCE_1": 0.2, "EXT_SOURCE_2": 0.8}
    ]

    pydantic_df = pd.DataFrame([
        {
            'SK_ID_CURR': 100001,
            'DAYS_BIRTH': -14600,
            'AMT_CREDIT': 200000.0,
            'AMT_INCOME_TOTAL': 50000.0,
            'CODE_GENDER': 'M'
        },
        {
            'SK_ID_CURR': 100002,
            'DAYS_BIRTH': -10950,
            'AMT_CREDIT': 100000.0,
            'AMT_INCOME_TOTAL': 50000.0,
            'CODE_GENDER': 'F'
        }
    ])

    expected_cols = [
        "SK_ID_CURR", "AMT_CREDIT", "AMT_INCOME_TOTAL", "AGE_YEARS",
        "CREDIT_INCOME_RATIO", "EXT1_EXT2", "CODE_GENDER_M", "CODE_GENDER_F", "FEATURE_ABSENTE"
    ]

    df_final = build_full_features(pydantic_df, expected_columns=expected_cols)

    mock_get_redis_batch.assert_called_once_with([100001, 100002])
    assert list(df_final.columns) == expected_cols
    assert len(df_final) == 2

    # Client 1
    assert df_final['AGE_YEARS'].iloc[0] == 40
    assert df_final['EXT1_EXT2'].iloc[0] == pytest.approx(0.3)
    assert df_final['CODE_GENDER_M'].iloc[0] == 1

    # Client 2
    assert df_final['AGE_YEARS'].iloc[1] == 30
    assert df_final['EXT1_EXT2'].iloc[1] == pytest.approx(0.16)
    assert df_final['CODE_GENDER_F'].iloc[1] == 1

    # Les colonnes absentes sont complétées par 0 en mode batch
    assert (df_final['FEATURE_ABSENTE'] == 0).all()
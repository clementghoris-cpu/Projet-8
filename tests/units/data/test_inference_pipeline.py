import json
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest

# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------
MOCK_EXPECTED_COLUMNS = [
    "SK_ID_CURR", "AMT_CREDIT", "AMT_INCOME_TOTAL", "AGE_YEARS", 
    "CREDIT_INCOME_RATIO", "EXT_SOURCE_1_SQ", "EXT1_EXT2", 
    "EXT2_DIV_EXT3", "CODE_GENDER_F", "CODE_GENDER_M"
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
        add_polynomial_and_interaction_features,
        build_full_features,
        encode_categorical_features,
        get_redis_features_batch,
        preprocess_basic_features,
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
# TESTS : preprocess_basic_features
# ==============================================================================

def test_preprocess_basic_features_anomalies_and_ratios():
    """Vérifie la correction de DAYS_EMPLOYED, la conversion des âges et le calcul des ratios."""
    raw_df = pd.DataFrame([{
        'SK_ID_CURR': 100001,
        'DAYS_BIRTH': -10950,        # 30 ans (10950 / 365)
        'DAYS_EMPLOYED': 365243,      # Valeur anormale (Retraité/Sans emploi)
        'AMT_CREDIT': 200000.0,
        'AMT_INCOME_TOTAL': 50000.0,
        'AMT_ANNUITY': 10000.0,
        'AMT_GOODS_PRICE': 180000.0
    }])

    df_out = preprocess_basic_features(raw_df)

    # 1. Traitement des anomalies DAYS_EMPLOYED
    assert df_out['DAYS_EMPLOYED_ANOM'].iloc[0] == 1
    assert pd.isna(df_out['DAYS_EMPLOYED'].iloc[0])
    assert pd.isna(df_out['YEARS_EMPLOYED'].iloc[0])

    # 2. Age en années
    assert df_out['AGE_YEARS'].iloc[0] == 30

    # 3. Ratios
    assert df_out['CREDIT_INCOME_RATIO'].iloc[0] == pytest.approx(4.0)
    assert df_out['ANNUITY_INCOME_RATIO'].iloc[0] == pytest.approx(0.2)
    assert df_out['CREDIT_GOODS_RATIO'].iloc[0] == pytest.approx(200000.0 / 180000.0)


# ==============================================================================
# TESTS : add_polynomial_and_interaction_features
# ==============================================================================

def test_add_polynomial_and_interaction_features():
    """Vérifie la génération des colonnes polynomiales, multiplications et divisions."""
    df_in = pd.DataFrame([{
        'EXT_SOURCE_1': 0.5,
        'EXT_SOURCE_2': 0.4,
        'EXT_SOURCE_3': 0.2,
        'CREDIT_INCOME_RATIO': 2.0,
        'AMT_CREDIT': 100000.0,
        'AGE_YEARS': 50
    }])

    df_out = add_polynomial_and_interaction_features(df_in)

    # 1. Puissances
    assert df_out['EXT_SOURCE_1_SQ'].iloc[0] == pytest.approx(0.25)
    assert df_out['EXT_SOURCE_1_CUBIC'].iloc[0] == pytest.approx(0.125)

    # 2. Multiplications
    assert df_out['EXT1_EXT2'].iloc[0] == pytest.approx(0.2)

    # 3. Divisions
    assert df_out['EXT2_DIV_EXT3'].iloc[0] == pytest.approx(2.0)
    assert df_out['CREDIT_DIV_AGE'].iloc[0] == pytest.approx(2000.0)


def test_add_polynomial_division_by_zero():
    """Vérifie que la division par zéro produit NaN au lieu de inf/-inf."""
    df_in = pd.DataFrame([{
        'EXT_SOURCE_2': 0.5,
        'EXT_SOURCE_3': 0.0, # Division par zéro !
        'AMT_CREDIT': 100000.0,
        'AGE_YEARS': 0       # Division par zéro !
    }])

    df_out = add_polynomial_and_interaction_features(df_in)

    assert pd.isna(df_out['EXT2_DIV_EXT3'].iloc[0])
    assert pd.isna(df_out['CREDIT_DIV_AGE'].iloc[0])


# ==============================================================================
# TESTS : encode_categorical_features
# ==============================================================================

def test_encode_categorical_features():
    """Vérifie le One-Hot Encoding et la préservation de la clé SK_ID_CURR."""
    df_in = pd.DataFrame([{
        'SK_ID_CURR': 100001,
        'CODE_GENDER': 'F',
        'NAME_INCOME_TYPE': 'Working'
    }])

    df_out = encode_categorical_features(df_in)

    # La clé SK_ID_CURR doit toujours être présente et inchangée
    assert 'SK_ID_CURR' in df_out.columns
    assert df_out['SK_ID_CURR'].iloc[0] == 100001

    # Des colonnes créées par pd.get_dummies doivent exister
    assert 'CODE_GENDER_F' in df_out.columns
    assert df_out['CODE_GENDER_F'].iloc[0] == 1


# ==============================================================================
# TESTS : build_full_features (Pipeline complet en Batch)
# ==============================================================================

@patch("src.data.inference_pipeline.get_redis_features_batch")
def test_build_full_features_pipeline_batch(mock_get_redis_batch):
    """Vérifie l'orchestration complète du pipeline pour un batch de plusieurs lignes."""
    # Simulation des données Redis retournées pour 2 clients
    mock_get_redis_batch.return_value = [
        {"EXT_SOURCE_1": 0.6, "EXT_SOURCE_2": 0.5},
        {"EXT_SOURCE_1": 0.2, "EXT_SOURCE_2": 0.8}
    ]

    # DataFrame d'entrée simulant un batch de 2 requêtes
    pydantic_df = pd.DataFrame([
        {
            'SK_ID_CURR': 100001,
            'DAYS_BIRTH': -14600, # 40 ans
            'AMT_CREDIT': 200000.0,
            'AMT_INCOME_TOTAL': 50000.0,
            'CODE_GENDER': 'M'
        },
        {
            'SK_ID_CURR': 100002,
            'DAYS_BIRTH': -10950, # 30 ans
            'AMT_CREDIT': 100000.0,
            'AMT_INCOME_TOTAL': 50000.0,
            'CODE_GENDER': 'F'
        }
    ])

    expected_cols = [
        "SK_ID_CURR", "AMT_CREDIT", "AMT_INCOME_TOTAL", "AGE_YEARS",
        "CREDIT_INCOME_RATIO", "EXT1_EXT2", "CODE_GENDER_M", "CODE_GENDER_F", "FEATURE_NON_EXISTANTE"
    ]

    df_final = build_full_features(pydantic_df, expected_columns=expected_cols)

    # get_redis_features_batch a dû être appelé avec les 2 SK_ID_CURR
    mock_get_redis_batch.assert_called_once_with([100001, 100002])

    # Les colonnes finales doivent correspondre EXACTEMENT à expected_cols
    assert list(df_final.columns) == expected_cols
    assert len(df_final) == 2

    # Vérification des valeurs calculées pour le Client 1
    assert df_final['AGE_YEARS'].iloc[0] == 40
    assert df_final['CREDIT_INCOME_RATIO'].iloc[0] == pytest.approx(4.0)
    assert df_final['EXT1_EXT2'].iloc[0] == pytest.approx(0.3)
    assert df_final['CODE_GENDER_M'].iloc[0] == 1
    assert df_final['CODE_GENDER_F'].iloc[0] == 0

    # Vérification des valeurs calculées pour le Client 2
    assert df_final['AGE_YEARS'].iloc[1] == 30
    assert df_final['EXT1_EXT2'].iloc[1] == pytest.approx(0.16)
    assert df_final['CODE_GENDER_M'].iloc[1] == 0
    assert df_final['CODE_GENDER_F'].iloc[1] == 1

    # La colonne inexistante est bien complétée avec 0
    assert (df_final['FEATURE_NON_EXISTANTE'] == 0).all()
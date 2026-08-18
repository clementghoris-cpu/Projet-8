import json
from unittest.mock import MagicMock, patch

import fakeredis
import pandas as pd
import pytest

from src.data.cache.aggregation_to_cache_pipeline import load_csv_to_redis


@pytest.fixture
def mock_redis_client():
    """Fixture retournant un client Redis simulé en mémoire via fakeredis."""
    server = fakeredis.FakeServer()
    return fakeredis.FakeRedis(server=server, decode_responses=True)


@pytest.fixture
def sample_csv_data(tmp_path):
    """Crée un fichier CSV temporaire valide pour tester le chargement."""
    csv_file = tmp_path / "test_aggregation.csv"
    df = pd.DataFrame({
        'SK_ID_CURR': [100001, 100002, 100003],
        'POS_COUNT': [2.0, None, 5.0],
        'PREV_APPROVED_AMT_MEAN': [150000.0, 200000.0, None]
    })
    df.to_csv(csv_file, index=False)
    return str(csv_file)


# ==============================================================================
# TESTS UNITAIRES
# ==============================================================================

@patch('src.data.cache.aggregation_to_cache_pipeline.redis_client')
def test_load_csv_to_redis_functional(mock_client, sample_csv_data, mock_redis_client):
    """
    Vérifie l'injection réelle des clés/valeurs dans Redis avec gestion des NaN.
    """
    # On redirige le client de votre script vers FakeRedis
    mock_client.pipeline.side_effect = mock_redis_client.pipeline

    # Exécution
    load_csv_to_redis(sample_csv_data)

    # 1. Vérification des clés créées
    assert mock_redis_client.exists("client:100001") == 1
    assert mock_redis_client.exists("client:100002") == 1
    assert mock_redis_client.exists("client:100003") == 1

    # 2. Vérification du contenu JSON pour un client standard
    data_100001 = json.loads(mock_redis_client.get("client:100001"))
    assert data_100001["POS_COUNT"] == 2.0
    assert data_100001["PREV_APPROVED_AMT_MEAN"] == 150000.0
    assert "SK_ID_CURR" not in data_100001  # L'ID a bien été retiré du corps JSON

    # 3. Vérification de la conversion des NaN en null (None en JSON)
    data_100002 = json.loads(mock_redis_client.get("client:100002"))
    assert data_100002["POS_COUNT"] is None

@patch('src.data.cache.aggregation_to_cache_pipeline.BATCH_SIZE', 10)
@patch('src.data.cache.aggregation_to_cache_pipeline.redis_client')
def test_load_csv_to_redis_batch_execution(mock_client, tmp_path):
    """
    Vérifie le bon fonctionnement du découpage par batch (BATCH_SIZE).
    """
    # Création d'un CSV volumineux pour déclencher plusieurs batchs
    csv_file = tmp_path / "large_aggregation.csv"
    num_rows = 25  # Supposons BATCH_SIZE réduit pour le test ou simulation
    df = pd.DataFrame({
        'SK_ID_CURR': list(range(100, 100 + num_rows)),
        'FEATURE': [1.0] * num_rows
    })
    df.to_csv(csv_file, index=False)

    # Mock du pipeline Redis
    mock_pipeline = MagicMock()
    mock_client.pipeline.return_value = mock_pipeline

    load_csv_to_redis(str(csv_file))

    # Avec 25 lignes et un BATCH_SIZE de 10:
    # Batchs exécutés à idx=9, idx=19 et 1 fois à la fin -> total 3 appels à execute()
    assert mock_pipeline.execute.call_count == 3


@patch('pandas.read_csv', side_effect=FileNotFoundError("CSV introuvable"))
def test_load_csv_to_redis_file_not_found(mock_read_csv):
    """Vérifie le comportement si le fichier CSV est inexistant."""
    with pytest.raises(FileNotFoundError):
        load_csv_to_redis("path/invalid/file.csv")
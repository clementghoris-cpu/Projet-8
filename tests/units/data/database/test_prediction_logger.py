from unittest.mock import MagicMock, patch

import pytest

from src.database.prediction_logger import PredictionLog, log_to_postgres


@pytest.fixture
def mock_db_session():
    """Fixture qui simule la session SQLAlchemy."""
    session = MagicMock()
    return session


# ==============================================================================
# TESTS UNITAIRES (Mocks)
# ==============================================================================

@patch('src.database.prediction_logger.SessionLocal')
def test_log_to_postgres_success(mock_session_factory, mock_db_session):
    """Vérifie le comportement nominal d'un enregistrement réussi."""
    mock_session_factory.return_value.__enter__.return_value = mock_db_session

    inputs_payload = {"AMT_CREDIT": 100000.0, "AGE": 30}
    
    log_to_postgres(
        sk_id_curr=100001,
        inputs=inputs_payload,
        model_version="v1.0.0",
        execution_time_ms=45.2,
        status_code=200,
        probability=0.12,
        prediction=0,
        error_message=None
    )

    # 1. Vérifie qu'un objet PredictionLog a été ajouté
    assert mock_db_session.add.call_count == 1
    added_obj = mock_db_session.add.call_args[0][0]
    
    assert isinstance(added_obj, PredictionLog)
    assert added_obj.sk_id_curr == 100001
    assert added_obj.inputs == inputs_payload
    assert added_obj.model_version == "v1.0.0"
    assert added_obj.probability == 0.12
    assert added_obj.prediction == 0
    assert added_obj.execution_time_ms == 45.2
    assert added_obj.status_code == 200
    assert added_obj.error_message is None

    # 2. Vérifie la validation des transactions
    mock_db_session.commit.assert_called_once()
    mock_db_session.rollback.assert_not_called()


@patch('src.database.prediction_logger.SessionLocal')
def test_log_to_postgres_error_handling(mock_session_factory, mock_db_session):
    """Vérifie que la transaction fait un rollback et ferme la session en cas d'erreur BDD."""
    mock_session_factory.return_value.__enter__.return_value = mock_db_session
    
    # Simuler une exception SQLAlchemy lors du commit
    mock_db_session.commit.side_effect = Exception("Erreur de connexion BDD")

    # Exécution (ne doit pas lever d'exception non gérée grâce au try/except dans la fonction)
    log_to_postgres(
        sk_id_curr=100002,
        inputs={"FEATURE": 1},
        model_version="v1.0.0",
        execution_time_ms=10.0,
        status_code=500,
        error_message="Internal Server Error"
    )

    # Vérifications
    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()
    mock_db_session.rollback.assert_called_once()  # Rollback doit être appelé !


@patch('src.database.prediction_logger.SessionLocal')
def test_log_to_postgres_optional_fields_none(mock_session_factory, mock_db_session):
    """Vérifie l'enregistrement avec uniquement les champs obligatoires (valeurs optionnelles à None)."""
    mock_session_factory.return_value.__enter__.return_value = mock_db_session

    log_to_postgres(
        sk_id_curr=100003,
        inputs={"test": "data"},
        model_version="v2.1",
        execution_time_ms=12.5,
        status_code=500
    )

    added_obj = mock_db_session.add.call_args[0][0]
    
    assert added_obj.probability is None
    assert added_obj.prediction is None
    assert added_obj.error_message is None
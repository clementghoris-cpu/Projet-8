import enum
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Enum,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, sessionmaker

from config.project_config import Database_settings

DATABASE_URL = Database_settings.database_url

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class OptimizationStage(str, enum.Enum):
    BASELINE = "BASELINE"
    OPTI_1 = "OPTI_1"
    OPTI_2 = "OPTI_2"
    OPTI_3 = "OPTI_3"

class APIProfilingLog(Base):
    __tablename__ = "api_profiling_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)
    sk_id_curr = Column(Integer, nullable=True)
    optimization_stage = Column(
        Enum(OptimizationStage, name="opti_stage_enum"),
        default=OptimizationStage.BASELINE,
        nullable=False
    )
    model_version = Column(String(50), nullable=False)
    total_latency_ms = Column(Float, nullable=False)
    time_pydantic_to_df_ms = Column(Float, nullable=True)
    time_feature_engineering_ms = Column(Float, nullable=True)
    time_model_predict_ms = Column(Float, nullable=True)
    time_db_logging_ms = Column(Float, nullable=True)
    cprofile_top_functions = Column(JSON, nullable=True) # Captures détaillées de cProfile au format JSON (Top fonctions)


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    sk_id_curr = Column(Integer, index=True, nullable=False)
    model_version = Column(String(50), nullable=False)    
    inputs = Column(JSONB, nullable=False)    
    probability = Column(Float, nullable=True)
    prediction = Column(Integer, nullable=True)
    execution_time_ms = Column(Float, nullable=False)
    status_code = Column(Integer, nullable=False)
    error_message = Column(Text, nullable=True)

def init_db():
    '''Initialise les tables en base de données.'''
    Base.metadata.create_all(bind=engine)

def log_to_postgres(sk_id_curr: int, inputs: dict, model_version: str, execution_time_ms: float,
    status_code: int, probability: float | None = None, prediction: int | None = None, error_message: str | None = None):
    '''
    insère le log de la prédiction
    '''
    with SessionLocal() as db:
        try:
            db_log = PredictionLog(
                sk_id_curr=sk_id_curr,
                inputs=inputs,
                model_version=model_version,
                probability=probability,
                prediction=prediction,
                execution_time_ms=execution_time_ms,
                status_code=status_code,
                error_message=error_message
            )
            db.add(db_log)
            db.commit()
            # Le commentaire ajouté à droite ci-dessous est pour éviter que la librairie Ruff renvoi une erreur (utilisation d'une Exception générique) à la vérification du code
        except Exception as e: # noqa: BLE001
            db.rollback()
            print(f"Erreur lors du logging en BDD : {e}")

def fetch_inputs_and_prediction_from_log_data(limit: int = 1000) -> pd.DataFrame:
    '''
    Récupère les données inputs et prédiction de la table prediction_logs
    '''
    stmt = (
        select(PredictionLog.id, PredictionLog.inputs, PredictionLog.prediction)
        .where(PredictionLog.status_code == 200)
        .order_by(PredictionLog.timestamp.desc())
        .limit(limit)
    )

    with SessionLocal() as db:
        df_logs = pd.read_sql(stmt, db.bind)

    return df_logs

def log_profile_to_db(sk_id_curr: int | None, optimization_stage: OptimizationStage, model_version: str, total_latency_ms: float,
    metrics: dict[str, float], time_db_logging_ms: float, cprofile_report: str):
    '''
    Insère le log du profile d'une prédiction
    '''
    with SessionLocal() as db:
        try:
            db_profile_log = APIProfilingLog(
                sk_id_curr=sk_id_curr,
                optimization_stage=optimization_stage,
                model_version=model_version,
                total_latency_ms=round(total_latency_ms, 2),
                time_pydantic_to_df_ms=round(metrics.get("pydantic_to_df_ms", 0.0), 2),
                time_feature_engineering_ms=round(metrics.get("feature_engineering_ms", 0.0), 2),
                time_model_predict_ms=round(metrics.get("model_predict_ms", 0.0), 2),
                time_db_logging_ms=round(time_db_logging_ms, 2),
                cprofile_top_functions={"report": cprofile_report}
            )
            db.add(db_profile_log)
            db.commit()
        except Exception as e: # noqa : BLE001
            db.rollback()
            print(f"Erreur lors du logging du profilage en BDD : {e}")

def fetch_profiling_logs(limit: int = 1000) -> pd.DataFrame:
    """
    Récupère les métriques de profilage de la table api_profiling_logs sous forme de DataFrame Pandas.
    """
    stmt = (
        select(
            APIProfilingLog.id,
            APIProfilingLog.timestamp,
            APIProfilingLog.sk_id_curr,
            APIProfilingLog.optimization_stage,
            APIProfilingLog.model_version,
            APIProfilingLog.total_latency_ms,
            APIProfilingLog.time_pydantic_to_df_ms,
            APIProfilingLog.time_feature_engineering_ms,
            APIProfilingLog.time_model_predict_ms,
            APIProfilingLog.time_db_logging_ms,
            APIProfilingLog.cprofile_top_functions,
        )
        .order_by(APIProfilingLog.timestamp.desc())
        .limit(limit)
    )

    with SessionLocal() as db:
        df_profiling = pd.read_sql(stmt, db.bind)

    return df_profiling
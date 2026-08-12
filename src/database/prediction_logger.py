from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import JSONB 
from config.project_config import Database_settings

DATABASE_URL = Database_settings.database_url

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
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
    status_code: int, probability: float = None, prediction: int = None, error_message: str = None):
    '''
    Ouvre une session, insère le log de la prédiction, et referme la session proprement.
    '''
    db = SessionLocal()
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
    except Exception as e:
        db.rollback()
        print(f"Erreur lors du logging en BDD : {e}")
    finally:
        db.close()
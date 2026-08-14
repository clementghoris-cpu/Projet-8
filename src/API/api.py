import time
import skops.io as sio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from src.API.schemas import ClientApplicationInput
from src.data.inference_pipeline import build_full_features, EXPECTED_COLUMNS
from src.database.prediction_logger import init_db, log_to_postgres
from config.project_config import App_config

def load_model():
    """
    Charge le modèle ML depuis le fichier model.skops.
    """
    try:
        skops_path = App_config.paths.model
        unknown_types = sio.get_untrusted_types(file=skops_path)
        model = sio.load(skops_path, trusted=unknown_types)
        
        print(f"Modèle ML chargé avec succès depuis : {skops_path}")
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle skops : {e}")
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gère le cycle de vie de l'application (Démarrage / Extinction).
    """
    print("Démarrage de l'API : Initialisation de la BDD PostgreSQL et chargement du modèle...")
    init_db()
    
    # Stockage du modèle et de ses paramètres dans app.state
    app.state.model = load_model()
    app.state.model_version = App_config.model.version
    app.state.threshold = App_config.model.optimal_threshold

    yield

    print("Extinction de l'API : Nettoyage des ressources...")


app = FastAPI(
    title="API Scoring Crédit - Projet 8",
    description="API prédisant la probabilité de défaut de paiement d'un client.",
    version=App_config.model.version,
    lifespan=lifespan
)

@app.post("/predict")
async def predict_credit(client_data: ClientApplicationInput, request: Request):
    """
    Reçoit les données du formulaire, applique le pipeline ML et renvoie la décision.
    """
    start_time = time.time()
    
    # Récupération du modèle depuis app.state
    model = request.app.state.model
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Modèle ML non chargé sur le serveur. Vérifiez les logs de démarrage."
        )

    try:
        # Conversion Pydantic -> Pandas
        df_input = client_data.to_pandas()
        sk_id_curr = client_data.SK_ID_CURR
        
        df_features = build_full_features(df_input, EXPECTED_COLUMNS)
        
        prob_default = float(model.predict_proba(df_features)[:, 1][0])
        decision_class = 0 if prob_default >= request.app.state.threshold else 1
        decision_text = "ACCORDÉ" if decision_class == 1 else "REFUSÉ"
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        log_to_postgres(
            sk_id_curr=sk_id_curr,
            inputs=client_data.model_dump(),
            model_version=request.app.state.model_version,
            execution_time_ms=execution_time_ms,
            status_code=200,
            probability=prob_default,
            prediction=decision_class
        )
        
        return {
            "SK_ID_CURR": sk_id_curr,
            "decision": decision_text,
            "seuil": request.app.state.threshold,
            "probabilite": round(prob_default, 4),
            "execution_time_ms": round(execution_time_ms, 2)
        }

    except Exception as e:
        execution_time_ms = (time.time() - start_time) * 1000
        log_to_postgres(
            sk_id_curr=client_data.SK_ID_CURR,
            inputs=client_data.model_dump(),
            model_version=request.app.state.model_version,
            execution_time_ms=execution_time_ms,
            status_code=500,
            error_message=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Erreur interne lors de la prédiction : {str(e)}")

@app.get("/")
def healthcheck(request: Request):
    is_model_loaded = request.app.state.model is not None
    return {
        "status": "API de Scoring Crédit opérationnelle",
        "model_loaded": is_model_loaded,
        "model_version": App_config.model.version
    }
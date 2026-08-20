import time
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import skops.io as sio
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request

from config.project_config import Api_settings, App_config
from src.API.helper import run_pipeline
from src.API.schemas import ClientApplicationInput
from src.data.inference_pipeline import EXPECTED_COLUMNS, build_full_features
from src.database.database_access import (
    OptimizationStage,
    fetch_inputs_and_prediction_from_log_data,
    fetch_profiling_logs,
    init_db,
    log_profile_to_db,
    log_to_postgres,
)


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
    # Le commentaire ajouté à droite ci-dessous est pour éviter que la librairie Ruff renvoi une erreur (utilisation d'une Exception générique) à la vérification du code
    except Exception as e: # noqa: BLE001
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
    app.state.enable_cProfile = Api_settings.enable_cProfile

    yield

    print("Extinction de l'API : Nettoyage des ressources...")


app = FastAPI(
    title="API Scoring Crédit - Projet 8",
    description="API prédisant la probabilité de défaut de paiement d'un client.",
    version=App_config.model.version,
    lifespan=lifespan
)


@app.post("/predict")
async def predict_credit(client_data: ClientApplicationInput, request: Request, background_tasks: BackgroundTasks):
    """
    Reçoit les données du formulaire, applique le pipeline ML et renvoie la décision.
    """
    start_total_ns = time.perf_counter_ns()
    
    # Récupération du modèle depuis app.state
    model = request.app.state.model
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Modèle ML non chargé sur le serveur. Vérifiez les logs de démarrage."
        )

    try:
        sk_id_curr = client_data.SK_ID_CURR

        # Pipeline ML sous profilage cProfile & temps
        profile_res = run_pipeline(
            client_data=client_data,
            model=model,
            enable_cprofile=request.app.state.enable_cProfile
        )
                        
        prob_default = profile_res["prob_default"]
        decision_class = 0 if prob_default >= request.app.state.threshold else 1
        decision_text = "ACCORDÉ" if decision_class == 1 else "REFUSÉ"
        
        end_inference_ns = time.perf_counter_ns()
        inference_time_ms = (end_inference_ns - start_total_ns) / 1_000_000

        background_tasks.add_task(
            log_to_postgres,
            sk_id_curr=sk_id_curr,
            inputs=client_data.model_dump(),
            model_version=request.app.state.model_version,
            execution_time_ms=inference_time_ms,
            status_code=200,
            probability=prob_default,
            prediction=decision_class
        )
        
        background_tasks.add_task(
            log_profile_to_db,
            sk_id_curr=sk_id_curr,
            optimization_stage=OptimizationStage.OPTI_2,  
            model_version=request.app.state.model_version,
            total_latency_ms=inference_time_ms,
            metrics=profile_res["metrics"],
            time_db_logging_ms=0.0,  
            cprofile_report=profile_res["cprofile_report"]
        )
        
        return {
            "SK_ID_CURR": sk_id_curr,
            "decision": decision_text,
            "seuil": request.app.state.threshold,
            "probabilite": round(prob_default, 4),
            "execution_time_ms": round(inference_time_ms, 2)
        }

    # Le commentaire ajouté à droite ci-dessous est pour éviter que la librairie Ruff renvoi une erreur (utilisation d'une Exception générique) à la vérification du code
    except Exception as e: # noqa: BLE001
        execution_time_ms = (time.perf_counter_ns() - start_total_ns) / 1_000_000

        log_to_postgres(
            sk_id_curr=client_data.SK_ID_CURR,
            inputs=client_data.model_dump(),
            model_version=request.app.state.model_version,
            execution_time_ms=execution_time_ms,
            status_code=500,
            error_message=str(e)
        )

        raise HTTPException(status_code=500, detail=f"Erreur interne lors de la prédiction : {e!s}")


@app.get("/")
def healthcheck(request: Request):
    is_model_loaded = request.app.state.model is not None
    return {
        "status": "API de Scoring Crédit opérationnelle",
        "model_loaded": is_model_loaded,
        "model_version": App_config.model.version
    }


@app.get("/logs/inference-data")
def get_inference_logs(limit: int = 500):
    """
    Récupère les N derniers logs d'inférence, reconstruit les 324 features 
    via build_full_features et retourne le dataframe sous forme de JSON
    """
    try:
        raw_logs = fetch_inputs_and_prediction_from_log_data(limit)

        if raw_logs.empty:
            return { "status": "success", "count": 0, "data": [] }

        df_inputs = pd.DataFrame(raw_logs["inputs"].tolist())

        df_inputs = pd.concat([
                df_inputs.reset_index(drop=True),
                raw_logs[["id"]].reset_index(drop=True)
            ], axis=1)

        df_full = build_full_features(df_inputs, EXPECTED_COLUMNS)
        df_full["TARGET"] = raw_logs["prediction"].values

        # Conversion explicite du NaN Pandas en None JSON pour éviter les erreurs de sérialisation
        df_full = df_full.replace({np.nan: None})

        return { "status": "success", "count": df_full.shape[0], "data": df_full.to_dict(orient="records") }
    
    except Exception as e: # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs/profiling")
async def get_profiling_logs(limit: int = 1000):
    """
    Retourne l'historique des métriques de profilage sous forme de dictionnaire JSON.
    """
    try:
        df_logs = fetch_profiling_logs(limit=limit)
        return df_logs.to_dict(orient="records")
    except Exception as e: # noqa: BLE001
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur lors de la récupération des logs de profilage : {e!s}"
        )
import cProfile
import io
import pstats
import time
from typing import Any

from src.API.schemas import ClientApplicationInput
from src.data.inference_pipeline import EXPECTED_COLUMNS, build_full_features


def run_pipeline(client_data : ClientApplicationInput, model : Any, enable_cprofile: bool = False) -> dict[str, Any]:
    """
    Exécute le pipeline ML. Si enable_cprofile est True, utilise cProfile.
    Sinon, mesure uniquement les temps via perf_counter
    """
    if enable_cprofile:
        return profile_pipeline(client_data, model)

    t0 = time.perf_counter_ns()
    df_input = client_data.to_pandas()
    t1 = time.perf_counter_ns()
    
    df_features = build_full_features(df_input, EXPECTED_COLUMNS)
    t2 = time.perf_counter_ns()
    
    prob_default = float(model.predict_proba(df_features)[:, 1][0])
    t3 = time.perf_counter_ns()
    
    return {
        "df_features": df_features,
        "prob_default": prob_default,
        "metrics": {
            "pydantic_to_df_ms": (t1 - t0) / 1_000_000,
            "feature_engineering_ms": (t2 - t1) / 1_000_000,
            "model_predict_ms": (t3 - t2) / 1_000_000,
        },
        "cprofile_report": "DISABLED"  # Marker pour indiquer que cProfile était éteint
    }

    

def profile_pipeline(client_data : ClientApplicationInput, model : Any) -> dict[str, Any]:
    """
    Exécute le pipeline ML sous cProfile et mesure la durée de chaque étape
    """
    profiler = cProfile.Profile()
    profiler.enable()

    t0 = time.perf_counter_ns()

    df_inputs = client_data.to_pandas()
    t1 = time.perf_counter_ns()

    df_features = build_full_features(df_inputs, EXPECTED_COLUMNS)
    t2 = time.perf_counter_ns()

    prob_defaut = float(model.predict_proba(df_features)[:, 1][0])
    t3 = time.perf_counter_ns()

    profiler.disable()

    buffer = io.StringIO()
    ps = pstats.Stats(profiler, stream=buffer).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(10)

    cprofile_raw_report : str = buffer.getvalue()
    
    return {
        "df_features": df_features,
        "prob_default": prob_defaut,
        "metrics": {
            "pydantic_to_df_ms" : (t1 - t0) / 1_000_000,
            "feature_engineering_ms" : (t2 - t1) / 1_000_000,
            "model_predict_ms" : (t3 - t2) / 1_000_000
        },
        "cprofile_report": cprofile_raw_report
    }
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from config.project_config import Api_settings

st.set_page_config(page_title="Profilage & Performance API ML", layout="wide")
st.title("Analyse des goulots d'étranglement & profilage API")

PROFILING_ENDPOINT = f"{Api_settings.url}:{Api_settings.port}/{Api_settings.logs_profiling_route}"

@st.cache_data(ttl=30)
def load_data_from_api(limit: int = 1000) -> pd.DataFrame:
    response = requests.get(PROFILING_ENDPOINT, params={"limit": limit}, timeout=10)
    response.raise_for_status()
    data = response.json()
    return pd.DataFrame(data)

try:
    df = load_data_from_api()
except Exception as e: # noqa: BLE001
    st.error(f"Erreur lors de l'appel à l'API ({PROFILING_ENDPOINT}) : {e}")
    st.stop()

if df.empty:
    st.warning("Aucune donnée de profilage disponible pour le moment. Effectuez quelques requêtes /predict sur l'API.")
    st.stop()

# Nettoyage et conversion des types
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])

# --- BARRE DE FILTRES ---
st.sidebar.header("Filtres")
selected_stages = st.sidebar.multiselect(
    "Étape d'optimisation :",
    options=df["optimization_stage"].unique().tolist(),
    default=df["optimization_stage"].unique().tolist()
)

st.sidebar.divider()
with st.sidebar.expander("Guide des Optimisations", expanded=True):
    st.markdown("""
    **`BASELINE`**
    * Code initial synchrone.
    * Pandas + Scikit-Learn wrapper + Écritures BDD bloquantes.

    **`OPTI_1` (Background Tasks)**
    * Écritures BDD déportées en tâche de fond (`BackgroundTasks`).
    * Suppression du temps d'attente I/O BDD pour le client. 

    **`OPTI_2` (Fast Dict Features Engineering)**
    * Feature Engineering sur dictionnaires Python (sans Pandas).
    * Fallback Pandas conservé pour les logs batch.
    """)

df_filtered = df[df["optimization_stage"].isin(selected_stages)]

# --- METRIQUES GLOBALES (KPIs) ---
st.subheader("Vue d'ensemble")
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Requêtes analysées", len(df_filtered))
col2.metric("Latence moyenne", f"{df_filtered['total_latency_ms'].mean():.2f} ms")
col3.metric("Pydantic -> DF", f"{df_filtered['time_pydantic_to_df_ms'].mean():.2f} ms")
col4.metric("Feature Engineering", f"{df_filtered['time_feature_engineering_ms'].mean():.2f} ms")
col5.metric("Inférence Modèle", f"{df_filtered['time_model_predict_ms'].mean():.2f} ms")

st.divider()

# --- VISUALISATION DES GOULOTS D'ÉTRANBLEMENT ---
st.subheader("1. Répartition du temps par étape (Goulots d'Étranglement)")

# Agrégation moyenne par OptimizationStage
df_grouped = df_filtered.groupby("optimization_stage")[
    [
        "time_pydantic_to_df_ms",
        "time_feature_engineering_ms",
        "time_model_predict_ms",
        "time_db_logging_ms"
    ]
].mean().reset_index()

# Transformation pour Plotly (Stack Bar Chart)
df_melted = df_grouped.melt(
    id_vars=["optimization_stage"],
    var_name="Etape",
    value_name="Temps_ms"
)

# Libellés plus clairs
stage_mapping = {
    "time_pydantic_to_df_ms": "1. Pydantic -> Pandas",
    "time_feature_engineering_ms": "2. Feature Engineering",
    "time_model_predict_ms": "3. Inférence Modèle",
    "time_db_logging_ms": "4. Logging BDD"
}
df_melted["Etape"] = df_melted["Etape"].map(stage_mapping)

fig = px.bar(
    df_melted,
    x="optimization_stage",
    y="Temps_ms",
    color="Etape",
    title="Décomposition de la Latence Moyenne par Phase d'Optimisation (ms)",
    labels={
        "optimization_stage": "Phase d'optimisation",
        "Temps_ms": "Temps (ms)",
        "Etape": "Sous-composant"
    },
    barmode="stack",
    text_auto=".1f"
)

st.plotly_chart(fig, width='stretch')

# --- DETAILED CPROFILE INSPECTOR ---
st.subheader("2. Inspection Détaillée cProfile")

# Fonction pour vérifier si un vrai rapport cProfile existe
def is_valid_cprofile(val):
    if isinstance(val, dict):
        report = val.get("report")
        return report is not None and report != "DISABLED"
    return False

df_cprofile = df_filtered[df_filtered["cprofile_top_functions"].apply(is_valid_cprofile)].copy()

if not df_cprofile.empty:
    selected_id = st.selectbox(
        "Sélectionner l'ID d'une requête pour voir son rapport cProfile :",
        options=df_cprofile["id"].tolist(),
        format_func=lambda x: f"Requête #{x} | Stage: {df_cprofile[df_cprofile['id']==x]['optimization_stage'].values[0]} | Latence: {df_cprofile[df_cprofile['id']==x]['total_latency_ms'].values[0]} ms"
    )
    
    selected_row = df_cprofile[df_cprofile["id"] == selected_id].iloc[0]
    report_content = selected_row["cprofile_top_functions"]
    
    if isinstance(report_content, dict) and "report" in report_content:
        st.text_area("Rapport brut cProfile", value=report_content["report"], height=300)
    else:
        st.json(report_content)
else:
    st.info("Aucun rapport cProfile complet enregistré dans la sélection actuelle.")
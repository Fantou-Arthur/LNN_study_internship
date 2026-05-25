import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Configuration de la page
st.set_page_config(
    page_title="Trade Drift Analysis | GAT vs CfC",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style Custom
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("🌍 World Trade Drift Analysis Dashboard")
st.markdown("""
### Étude Comparative de la Résilience des Modèles : Graph Attention (GAT) vs Closed-form Continuous (CfC)
Ce tableau de bord analyse comment les modèles de Deep Learning vieillissent et se dégradent face aux chocs économiques mondiaux (2020-2023).
""")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    file_path = 'research_results.csv'
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

df = load_data()

if df is not None:
    # --- SIDEBAR ---
    st.sidebar.header("Paramètres d'Affichage")
    metric_choice = st.sidebar.selectbox("Choisir la métrique d'erreur", ["sMAPE_GAT", "sMAPE_CfC"])
    show_drift = st.sidebar.checkbox("Afficher la courbe de Drift (KS)", value=True)

    # --- METRICS ROW ---
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        latest_ks = df['KS'].iloc[-1]
        st.metric("Drift Final (KS)", f"{latest_ks:.4f}", delta=f"{latest_ks - df['KS'].iloc[0]:.4f}", delta_color="inverse")
    
    with col2:
        best_model = "CfC (Liquid)" if df['sMAPE_CfC'].mean() < df['sMAPE_GAT'].mean() else "GAT (Global)"
        st.metric("Modèle le plus Robuste", best_model)
        
    with col3:
        avg_error = df['sMAPE_CfC'].mean() if "CfC" in best_model else df['sMAPE_GAT'].mean()
        st.metric("Erreur Moyenne (%)", f"{avg_error:.2f}%")
        
    with col4:
        peak_year = df.loc[df['KS'].idxmax(), 'Year']
        st.metric("Pic de Dérive", int(peak_year))

    # --- MAIN CHARTS ---
    st.divider()
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("📈 Évolution de l'Erreur vs Drift Temporel")
        fig = go.Figure()
        
        # Erreur GAT
        fig.add_trace(go.Scatter(
            x=df['Year'], y=df['sMAPE_GAT'],
            name="Erreur GAT (%)",
            line=dict(color='#1f77b4', width=4, dash='dot'),
            marker=dict(size=10)
        ))
        
        # Erreur CfC
        fig.add_trace(go.Scatter(
            x=df['Year'], y=df['sMAPE_CfC'],
            name="Erreur CfC (%)",
            line=dict(color='#2ca02c', width=4),
            marker=dict(size=12, symbol='diamond')
        ))
        
        if show_drift:
            # Drift en fond
            fig.add_trace(go.Bar(
                x=df['Year'], y=df['KS'],
                name="Intensité Drift (KS)",
                yaxis='y2',
                marker_color='orange',
                opacity=0.2
            ))

        fig.update_layout(
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(title="Erreur sMAPE (%)", gridcolor='lightgray'),
            yaxis2=dict(title="Drift Statistique (KS)", overlaying='y', side='right', range=[0, max(df['KS'])*1.5]),
            margin=dict(l=20, r=20, t=50, b=20),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("🎯 Analyse de Corrélation")
        # Scatter plot Drift vs Erreur
        fig_corr = px.scatter(
            df, x="KS", y=metric_choice,
            text="Year",
            trendline="ols",
            color=metric_choice,
            color_continuous_scale="Viridis",
            labels={"KS": "Indice de Drift (Chaos)", metric_choice: "Erreur Modèle (%)"}
        )
        fig_corr.update_traces(textposition='top center', marker=dict(size=15))
        fig_corr.update_layout(template="plotly_white")
        st.plotly_chart(fig_corr, use_container_width=True)

    # --- RAW DATA ---
    with st.expander("📄 Voir les données brutes"):
        st.dataframe(df.style.highlight_max(axis=0, color='lightpink'))

else:
    st.error("### 🛑 Fichier de résultats introuvable !")
    st.info("""
    Le dashboard ne peut pas s'afficher car le fichier **`research_results.csv`** est absent.
    
    **Comment résoudre ce problème ?**
    1. Exécutez d'abord votre script de calcul : `python scientific_drift_benchmark.py`
    2. Attendez que l'entraînement et l'évaluation soient terminés.
    3. Une fois le fichier CSV généré, ce dashboard se mettra à jour automatiquement.
    """)
    
    # Image de démo si possible ou placeholder
    st.image("https://raw.githubusercontent.com/streamlit/dataset-metadata/master/images/drift.png", caption="Exemple de visualisation de Data Drift", use_container_width=True)

# --- FOOTER ---
st.divider()
st.caption("Projet de Recherche : Liquid Networks vs Graph Attention Networks in International Trade Forecasting (2017-2023)")

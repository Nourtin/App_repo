import streamlit as st

# MUST BE FIRST LINE
st.set_page_config(
    page_title="Call Center Dashboard",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd
import plotly.express as px
import json
import plotly.graph_objects as go

from google_selector import list_sheets, choisir_feuille
from ai_recommendation import GeminiAdvisor
from analyse import (
    kpi_globaux, appels_par_jour, appels_par_mois, appels_par_heure,
    repartition_classification, appels_par_fournisseur, classification_par_fournisseur,
    taux_remplissage_code_postal, comparer_codes_postaux, analyse_fiabilite_par_fournisseur,
    codes_postaux_non_correspondants, analyse_par_type_logement,
    comparer_types_logement, classification_detaillee_par_type, appels_par_piso_casa,
    analyse_par_type_logement
)
from ats_analysis import render_ats_tab

PALETTE = px.colors.qualitative.Set2
try:
    from google import genai
    # gspread is removed as we use google_selector
except Exception as e:
    genai = None

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'fichier' not in st.session_state:
    st.session_state.fichier = None
if 'sheets_list' not in st.session_state:
    st.session_state.sheets_list = None
if 'selected_sheets' not in st.session_state:
    st.session_state.selected_sheets = []

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("📞 Call Center")
    st.markdown("---")
    st.subheader("🔗 Connexion Google Sheet")

    with st.expander("📖 Comment obtenir l'URL ?"):
        st.markdown("""
        1. Ouvrez votre Google Sheet
        2. Cliquez sur **Partager**
        3. Dans **"Accès général"**, sélectionnez : **"Toute personne disposant du lien"**
        4. Copiez le lien
        5. Collez-le ci-dessous
        """)

    sheet_url = st.text_input(
        "URL du Google Sheet",
        placeholder="https://docs.google.com/spreadsheets/d/...",
    )

    if sheet_url:
        if st.button("📂 Charger les feuilles", type="primary"):
            try:
                with st.spinner("Chargement du fichier..."):
                    st.session_state.fichier, st.session_state.sheets_list = list_sheets(sheet_url)
                    st.success(f"{len(st.session_state.sheets_list)} feuille(s) trouvée(s)")
            except Exception as e:
                st.error(f"Erreur de chargement: {str(e)}")
                st.session_state.fichier = None
                st.session_state.sheets_list = None

    if st.session_state.sheets_list:
        st.markdown("---")
        st.subheader("📑 Sélection des feuilles")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Tout sélectionner"):
                st.session_state.selected_sheets = st.session_state.sheets_list.copy()
        with col2:
            if st.button("Effacer tout"):
                st.session_state.selected_sheets = []
                st.rerun()

        selected_sheets = st.multiselect(
            "Choisissez les feuilles à analyser",
            options=st.session_state.sheets_list,
            default=st.session_state.selected_sheets,
        )
        st.session_state.selected_sheets = selected_sheets

        if selected_sheets:
            st.success(f"📊 {len(selected_sheets)} feuille(s) sélectionnée(s)")

            if st.button("🔄 Charger les données", type="primary"):
                try:
                    all_dfs = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for i, sheet_name in enumerate(selected_sheets):
                        status_text.text(f"Chargement de '{sheet_name}'... ({i+1}/{len(selected_sheets)})")
                        df = choisir_feuille(st.session_state.fichier, sheet_name)
                        if df is not None and not df.empty:
                            df['_source_feuille'] = sheet_name
                            all_dfs.append(df)
                        progress_bar.progress((i + 1) / len(selected_sheets))

                    status_text.text("Finalisation...")

                    if all_dfs:
                        st.session_state.df_raw = pd.concat(all_dfs, ignore_index=True)
                        st.session_state.stats_chargement = {
                            "nb_feuilles": len(selected_sheets),
                            "total_lignes": len(st.session_state.df_raw),
                            "feuilles_details": [
                                {"nom": sheet_name, "lignes": len(df)}
                                for sheet_name, df in zip(selected_sheets, all_dfs)
                            ]
                        }
                        status_text.empty()
                        progress_bar.empty()
                        st.success(f"Données chargées : {len(st.session_state.df_raw):,} lignes")
                    else:
                        st.error("Aucune donnée valide chargée")

                except Exception as e:
                    st.error(f"Erreur : {str(e)}")
        else:
            st.warning("Veuillez sélectionner au moins une feuille")

    if st.session_state.df_raw is not None:
        st.markdown("---")
        st.subheader("Filtres")
        df_raw = st.session_state.df_raw
        if "list_name" in df_raw.columns:
            fournisseurs = ["Tous"] + sorted(df_raw["list_name"].dropna().unique().tolist())
            fourn_sel = st.selectbox("Fournisseur (list_name)", fournisseurs)
        else:
            fourn_sel = "Tous"
        if "Timestamp" in df_raw.columns:
            ts_all = pd.to_datetime(df_raw["Timestamp"], errors="coerce", dayfirst=True).dropna()
            if not ts_all.empty:
                date_min, date_max = ts_all.min().date(), ts_all.max().date()
                date_range = st.date_input("Période", value=(date_min, date_max), min_value=date_min, max_value=date_max)
            else: date_range = None
        else: date_range = None
    else:
        fourn_sel = "Tous"
        date_range = None

    api_key_input = st.text_input("🔑 Clé API Gemini", type="password", key="gemini_key_global")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if st.session_state.df_raw is None:
    st.info("👈 Commencez par entrer l'URL de votre Google Sheet dans la barre latérale")
    st.stop()

df = st.session_state.df_raw.copy()
if fourn_sel != "Tous" and "list_name" in df.columns:
    df = df[df["list_name"] == fourn_sel]
if date_range is not None and "Timestamp" in df.columns:
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        ts = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True)
        df = df[(ts.dt.date >= date_range[0]) & (ts.dt.date <= date_range[1])]

if df.empty:
    st.warning("Aucune donnée pour les filtres sélectionnés.")
    st.stop()

tab1, tab2, tab3, tab4, tab5, tab_ats = st.tabs([
    "📊 Analyse globale", "🏢 Par fournisseur", "📍 Codes postaux & Fiabilité", 
    "🏠 Logements", "AI Recommendations", "📋 Analyse des ATS par IA"
])

with tab1:
    st.header("Analyse globale des appels")
    kpis = kpi_globaux(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total appels", f"{kpis['total_appels']:,}")
    c2.metric("Appels utiles", f"{kpis['appels_utiles']:,}" if kpis['appels_utiles'] is not None else "—")
    c3.metric("Taux utiles", f"{kpis['taux_utiles_pct']}%" if kpis['taux_utiles_pct'] is not None else "—")
    c4.metric("Durée moyenne", f"{kpis['duree_moyenne_sec']:.0f}s" if kpis['duree_moyenne_sec'] is not None else "—")
    
    st.markdown("---")
    col_j, col_m = st.columns(2)
    with col_j:
        st.subheader("Appels par jour")
        df_jour = appels_par_jour(df)
        if not df_jour.empty:
            fig = px.bar(df_jour, x="date", y="nb_appels", color_discrete_sequence=[PALETTE[0]])
            st.plotly_chart(fig, use_container_width=True)
            
with tab2:
    st.header("Analyse par fournisseur")
    df_fourn = appels_par_fournisseur(df)
    if not df_fourn.empty:
        st.dataframe(df_fourn, use_container_width=True)

with tab_ats:
    render_ats_tab(api_key_input=api_key_input)
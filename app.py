import streamlit as st
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
    duree_par_classification
)
from ats_analysis import render_ats_tab

PALETTE = px.colors.qualitative.Set2

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

                        with st.expander("📋 Détail du chargement"):
                            st.write(f"**Total lignes :** {len(st.session_state.df_raw):,}")
                            st.write(f"**Colonnes :** {', '.join(st.session_state.df_raw.columns[:8])}")
                            for detail in st.session_state.stats_chargement["feuilles_details"]:
                                st.write(f"- {detail['nom']} : {detail['lignes']:,} lignes")
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
                date_min = ts_all.min().date()
                date_max = ts_all.max().date()
                date_range = st.date_input("Période", value=(date_min, date_max),
                                           min_value=date_min, max_value=date_max)
            else:
                date_range = None
        else:
            date_range = None

        st.markdown("---")
        if st.button("🔄 Actualiser"):
            st.cache_data.clear()
            st.rerun()
    else:
        fourn_sel = "Tous"
        date_range = None

    # Clé API Gemini (sidebar global)
    api_key_input = st.text_input(
        "🔑 Clé API Gemini",
        type="password",
        placeholder="AIza...",
        key="gemini_key_global"
    )

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if st.session_state.df_raw is None:
    st.info("👈 Commencez par entrer l'URL de votre Google Sheet dans la barre latérale")
    st.stop()

# FILTRES
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

# ─────────────────────────────────────────────
# ONGLETS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab_ats = st.tabs([
    "📊 Analyse globale",
    "🏢 Par fournisseur",
    "📍 Codes postaux & Fiabilité",
    "🏠 Logements",
    "🤖 AI Recommendations",
    "📋 Analyse des ATS par IA"
])

# ══════════════════════════════════════════════
# TAB 1 — ANALYSE GLOBALE
# ══════════════════════════════════════════════
with tab1:
    st.header("Analyse globale des appels")

    kpis = kpi_globaux(df)
    if "Classification" in df.columns:
        classifications_qualif = ["PEU INTERESSE", "INTERESSE", "TRES INTERESSE", "EDIFICIOS", "RDV LEADS", "WHATSAP"]
        qualif_mask = df["Classification"].astype(str).str.upper().str.strip().isin([c.upper() for c in classifications_qualif])
        appels_qualifies = qualif_mask.sum()
        taux_qualifie = round(appels_qualifies / len(df) * 100, 1) if len(df) > 0 else 0
    else:
        appels_qualifies = None
        taux_qualifie = None

    # Première ligne : 5 métriques
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total appels", f"{kpis['total_appels']:,}")
    c2.metric("Appels utiles", f"{kpis['appels_utiles']:,}" if kpis['appels_utiles'] is not None else "—")
    c3.metric("Taux utiles", f"{kpis['taux_utiles_pct']}%" if kpis['taux_utiles_pct'] is not None else "—")
    c4.metric("Durée moyenne globale", f"{kpis['duree_moyenne_sec']:.0f}s" if kpis['duree_moyenne_sec'] is not None else "—")
    c5.metric("Taux qualification", f"{taux_qualifie}%" if taux_qualifie is not None else "—")

    # Deuxième ligne : Durées par catégorie
    st.markdown("---")
    st.subheader("⏱️ Durées moyennes par catégorie")
    
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    
    with col_d1:
        duree_utiles = kpis.get('duree_moyenne_utiles_sec')
        st.metric(
            "📞 Appels utiles", 
            f"{duree_utiles:.0f}s" if duree_utiles is not None else "—",
            help="Durée moyenne des appels avec classification valide"
        )
    
    with col_d2:
        duree_rdv = kpis.get('duree_moyenne_rdv_leads_sec')
        st.metric(
            "🎯 RDV Leads", 
            f"{duree_rdv:.0f}s" if duree_rdv is not None else "—",
            help="Durée moyenne des appels classifiés 'RDV LEADS'"
        )
    
    with col_d3:
        duree_tres = kpis.get('duree_moyenne_tres_interesse_sec')
        st.metric(
            "🔥 Très intéressé", 
            f"{duree_tres:.0f}s" if duree_tres is not None else "—",
            help="Durée moyenne des appels classifiés 'TRES INTERESSE'"
        )
    
    with col_d4:
        duree_inter = kpis.get('duree_moyenne_interesse_sec')
        st.metric(
            "👍 Intéressé", 
            f"{duree_inter:.0f}s" if duree_inter is not None else "—",
            help="Durée moyenne des appels classifiés 'INTERESSE'"
        )

    st.markdown("---")
    
    # Graphique des durées par classification
    st.subheader("📊 Durée moyenne par classification")
    
    df_duree_classif = duree_par_classification(df)
    if not df_duree_classif.empty:
        # Filtrer pour garder les classifications avec assez d'échantillons
        df_duree_classif = df_duree_classif[df_duree_classif["count"] >= 5]
        
        if not df_duree_classif.empty:
            fig = px.bar(
                df_duree_classif.sort_values("duree_moy_sec", ascending=False),
                x="duree_moy_sec",
                y="Classification",
                orientation="h",
                text="duree_moy_sec",
                color="duree_moy_sec",
                color_continuous_scale="Viridis",
                title="Durée moyenne par type de classification"
            )
            fig.update_traces(texttemplate="%{text:.0f}s", textposition="outside")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Afficher aussi sous forme de tableau
            with st.expander("📑 Tableau détaillé des durées par classification"):
                st.dataframe(
                    df_duree_classif.rename(columns={
                        "Classification": "Classification",
                        "count": "Nombre d'appels",
                        "duree_moy_sec": "Durée moyenne (s)",
                        "duree_mediane_sec": "Durée médiane (s)",
                        "duree_min_sec": "Durée min (s)",
                        "duree_max_sec": "Durée max (s)"
                    }),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("Pas assez de données par classification")
    else:
        st.info("Colonne Duration_seconds non trouvée")

    st.markdown("---")
    col_j, col_m = st.columns(2)

    with col_j:
        st.subheader("Appels par jour")
        df_jour = appels_par_jour(df)
        if not df_jour.empty:
            df_jour["date"] = df_jour["date"].astype(str)
            fig = px.bar(df_jour, x="date", y="nb_appels", color_discrete_sequence=[PALETTE[0]])
            fig.update_layout(xaxis_title="", yaxis_title="Appels", xaxis=dict(type="category"), margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Colonne Timestamp absente.")

    with col_m:
        st.subheader("Appels par mois")
        df_mois = appels_par_mois(df)
        if not df_mois.empty:
            df_mois["mois"] = df_mois["mois"].astype(str)
            fig = px.bar(df_mois, x="mois", y="nb_appels", color_discrete_sequence=[PALETTE[1]], text="nb_appels")
            fig.update_traces(textposition="outside")
            fig.update_layout(xaxis_title="", yaxis_title="Appels", xaxis=dict(type="category"), margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Colonne Timestamp absente.")

    st.markdown("---")
    col_cl, col_h = st.columns(2)

    with col_cl:
        st.subheader("Répartition par classification")
        df_cls = repartition_classification(df)
        if not df_cls.empty:
            fig = px.pie(df_cls, names="Classification", values="count", color_discrete_sequence=PALETTE, hole=0.4)
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune donnée de classification valide.")

    with col_h:
        st.subheader("Appels par heure de la journée")
        df_heure = appels_par_heure(df)
        if not df_heure.empty:
            fig = px.line(df_heure, x="heure", y="nb_appels", markers=True, color_discrete_sequence=[PALETTE[2]])
            fig.update_traces(line=dict(width=2), marker=dict(size=7))
            fig.update_layout(xaxis=dict(tickmode="linear", dtick=1, title="Heure"), yaxis_title="Appels", margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Colonne Timestamp absente.")

# ══════════════════════════════════════════════
# TAB 2 — PAR FOURNISSEUR
# ══════════════════════════════════════════════
with tab2:
    st.header("Analyse par fournisseur (list_name)")
    df_fourn = appels_par_fournisseur(df)

    if df_fourn.empty:
        st.info("Colonne list_name absente.")
    else:
        st.subheader("📊 Récapitulatif fournisseurs")
        st.dataframe(
            df_fourn.rename(columns={
                "list_name": "Fournisseur", "nb_appels": "Total appels",
                "nb_utiles": "Appels classifiés", "taux_utiles_pct": "Taux classification (%)",
                "nb_qualifies": "Appels qualifiés", "taux_qualifies_pct": "Taux qualification (%)",
                "duree_moy_sec": "Durée moy. (s)",
            }),
            use_container_width=True, hide_index=True,
        )

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("📈 Nombre d'appels par fournisseur")
            fig = px.bar(df_fourn.sort_values("nb_appels"), x="nb_appels", y="list_name",
                         orientation="h", text="nb_appels", color_discrete_sequence=[PALETTE[0]])
            fig.update_traces(textposition="outside")
            fig.update_layout(yaxis_title="", xaxis_title="Appels", margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("Taux d'appels qualifiés (%)")
            fig = px.bar(df_fourn.sort_values("taux_qualifies_pct"), x="taux_qualifies_pct", y="list_name",
                         orientation="h", text="taux_qualifies_pct", color="taux_qualifies_pct",
                         color_continuous_scale="Greens")
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(yaxis_title="", xaxis_title="%", margin=dict(t=10), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("⏱️ Durée moyenne d'appel par fournisseur (secondes)")
        fig = px.bar(df_fourn.sort_values("duree_moy_sec"), x="duree_moy_sec", y="list_name",
                     orientation="h", text="duree_moy_sec", color_discrete_sequence=[PALETTE[3]])
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_title="", xaxis_title="Secondes", margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("📋 Répartition des classifications par fournisseur")
        df_cls_fourn = classification_par_fournisseur(df)

        if not df_cls_fourn.empty:
            fig = px.bar(df_cls_fourn, x="pct", y="list_name", color="Classification",
                         orientation="h", text="count", color_discrete_sequence=PALETTE, barmode="stack")
            fig.update_traces(textposition="inside", insidetextanchor="middle")
            fig.update_layout(yaxis_title="", xaxis_title="% des appels utiles",
                               legend_title="Classification", margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📑 Voir le tableau détaillé"):
                pivot = df_cls_fourn.pivot_table(index="list_name", columns="Classification", values="count", fill_value=0)
                st.dataframe(pivot, use_container_width=True)
        else:
            st.info("Aucune donnée de classification valide.")

        st.markdown("---")

        # Analyse logements par fournisseur
        st.subheader("🏠 Analyse des types de logement par fournisseur")

        if "tipo_vivienda" in df.columns:
            fournisseurs_list = sorted(df["list_name"].dropna().unique())
            selected_fournisseur = st.selectbox(
                "Choisissez un fournisseur pour voir le détail des logements",
                options=["Tous les fournisseurs"] + fournisseurs_list,
                key="logement_fournisseur_select"
            )

            df_logement_filter = df if selected_fournisseur == "Tous les fournisseurs" else df[df["list_name"] == selected_fournisseur]
            df_logement_clean = df_logement_filter.copy()
            df_logement_clean["tipo_vivienda"] = df_logement_clean["tipo_vivienda"].astype(str).str.strip()
            df_logement_clean = df_logement_clean[~df_logement_clean["tipo_vivienda"].isin(["", "nan", "None"])]

            if not df_logement_clean.empty:
                col_log1, col_log2, col_log3 = st.columns(3)
                col_log1.metric("Appels avec type logement", f"{len(df_logement_clean):,}")
                col_log2.metric("Types différents", df_logement_clean["tipo_vivienda"].nunique())
                col_log3.metric("Type le plus fréquent", df_logement_clean["tipo_vivienda"].mode().iloc[0])

                st.markdown("---")
                col_chart1, col_chart2 = st.columns(2)

                with col_chart1:
                    counts = df_logement_clean["tipo_vivienda"].value_counts()
                    fig_pie = px.pie(values=counts.values, names=counts.index,
                                     title="Répartition des logements", color_discrete_sequence=PALETTE, hole=0.3)
                    fig_pie.update_traces(textinfo="percent+label")
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col_chart2:
                    df_top = df_logement_clean["tipo_vivienda"].value_counts().head(10).reset_index()
                    df_top.columns = ["Type de logement", "Nombre d'appels"]
                    fig_bar = px.bar(df_top, x="Nombre d'appels", y="Type de logement", orientation="h",
                                     text="Nombre d'appels", color="Nombre d'appels", color_continuous_scale="Blues")
                    fig_bar.update_traces(textposition="outside")
                    fig_bar.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(fig_bar, use_container_width=True)

                if selected_fournisseur == "Tous les fournisseurs":
                    st.markdown("---")
                    st.subheader("Distribution des logements par fournisseur")
                    df_cross = pd.crosstab(df_logement_clean["list_name"], df_logement_clean["tipo_vivienda"])
                    top_types = df_cross.sum().sort_values(ascending=False).head(5).index
                    fig_stacked = px.bar(df_cross[top_types], x=df_cross[top_types].index, y=top_types,
                                         barmode="stack", color_discrete_sequence=PALETTE)
                    fig_stacked.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_stacked, use_container_width=True)

                    with st.expander("📑 Tableau détaillé"):
                        st.dataframe(df_cross, use_container_width=True)

                if "Classification" in df.columns:
                    st.subheader("Classification des appels par type de logement")
                    df_classif_log = df_logement_clean[df_logement_clean["Classification"].notna()]
                    df_classif_log = df_classif_log[~df_classif_log["Classification"].astype(str).str.lower().isin(["non trouvé", "non trouve", ""])]

                    if not df_classif_log.empty:
                        top_logements = df_classif_log["tipo_vivienda"].value_counts().head(5).index
                        cross_classif = pd.crosstab(
                            df_classif_log[df_classif_log["tipo_vivienda"].isin(top_logements)]["tipo_vivienda"],
                            df_classif_log[df_classif_log["tipo_vivienda"].isin(top_logements)]["Classification"]
                        )
                        fig_classif = px.bar(cross_classif, barmode="group", color_discrete_sequence=PALETTE)
                        st.plotly_chart(fig_classif, use_container_width=True)

                if selected_fournisseur == "Tous les fournisseurs":
                    st.markdown("---")
                    df_pivot = pd.crosstab(df_logement_clean["list_name"], df_logement_clean["tipo_vivienda"], normalize="index") * 100
                    st.dataframe(df_pivot.round(1).style.format("{:.1f}%"), use_container_width=True, height=400)
                    csv = df_pivot.to_csv().encode('utf-8')
                    st.download_button("📥 Exporter", data=csv, file_name="logement_par_fournisseur.csv", mime="text/csv")
            else:
                st.info("Aucune donnée sur les types de logement pour ce fournisseur")
        else:
            st.info("Colonne 'tipo_vivienda' non trouvée.")

# ══════════════════════════════════════════════
# TAB 3 — CODES POSTAUX & FIABILITÉ
# ══════════════════════════════════════════════
with tab3:
    st.header("Analyse des codes postaux et fiabilité des données")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Taux de remplissage")
        taux = taux_remplissage_code_postal(df)
        if taux:
            for colonne, stats in taux.items():
                st.metric(f"Colonne: {colonne}", f"{stats['taux_remplissage']}%",
                           f"{stats['nb_remplis']}/{stats['total_lignes']} lignes")
                st.progress(stats['taux_remplissage'] / 100)
        else:
            st.warning("Colonnes 'code_postal' et/ou 'codigo_postal' non trouvées")

    with col2:
        st.subheader("Comparaison Client vs Fournisseur")
        df_comp, stats = comparer_codes_postaux(df)
        if stats:
            st.metric("Total comparaisons", stats['total_comparaisons'])
            st.metric("Correspondances", f"{stats['nb_correspondances']} / {stats['total_comparaisons']}")
            st.metric("Taux de correspondance", f"{stats['taux_correspondance']}%")
            if stats['taux_correspondance'] >= 80:
                st.success(f"Bonne fiabilité : {stats['taux_correspondance']}%")
            elif stats['taux_correspondance'] >= 50:
                st.warning(f"Fiabilité moyenne : {stats['taux_correspondance']}%")
            else:
                st.error(f"Faible fiabilité : {stats['taux_correspondance']}%")
        else:
            st.info("Pas assez de données pour comparer les codes postaux")

    st.markdown("---")
    st.subheader("🏢 Fiabilité par fournisseur")
    df_fiabilite = analyse_fiabilite_par_fournisseur(df)

    if not df_fiabilite.empty:
        st.dataframe(
            df_fiabilite.rename(columns={
                "fournisseur": "Fournisseur", "total_appels": "Total appels",
                "taux_remplissage_client": "Taux remplissage client (%)",
                "taux_remplissage_fournisseur": "Taux remplissage fournisseur (%)",
                "nb_comparaisons": "Nb comparaisons", "taux_correspondance": "Taux correspondance (%)"
            }),
            use_container_width=True, hide_index=True,
        )

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            df_plot = df_fiabilite.melt(id_vars=["fournisseur"],
                                         value_vars=["taux_remplissage_client", "taux_remplissage_fournisseur"],
                                         var_name="source", value_name="taux")
            fig = px.bar(df_plot, x="fournisseur", y="taux", color="source", barmode="group")
            st.plotly_chart(fig, use_container_width=True)

        with col_g2:
            fig = px.bar(df_fiabilite, x="fournisseur", y="taux_correspondance",
                         color="taux_correspondance", color_continuous_scale="RdYlGn")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Données insuffisantes")

    st.markdown("---")
    st.subheader("🔍 Codes postaux non correspondants")
    df_non_corr = codes_postaux_non_correspondants(df)

    if not df_non_corr.empty:
        cols_afficher = ["list_name", "code_postal", "codigo_postal", "code_postal_clean", "codigo_postal_clean"]
        cols_disponibles = [col for col in cols_afficher if col in df_non_corr.columns]
        st.dataframe(df_non_corr[cols_disponibles].head(100), use_container_width=True, hide_index=True)
        csv = df_non_corr[cols_disponibles].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Exporter les non-correspondances", data=csv,
                           file_name="non_correspondances_codes_postaux.csv", mime="text/csv")
    else:
        st.success("Tous les codes postaux disponibles correspondent !")

# ══════════════════════════════════════════════
# TAB 4 — LOGEMENTS
# ══════════════════════════════════════════════
# ══════════════════════════════════════════════
# TAB 4 — LOGEMENTS (avec regroupement)
# ══════════════════════════════════════════════
with tab4:
    st.header("🏠 Analyse des logements")

    if "piso_casa" not in df.columns and "tipo_vivienda" not in df.columns:
        st.warning("Colonne 'piso_casa' ou 'tipo_vivienda' non trouvée")
    else:
        # Déterminer la colonne à utiliser
        col_logement = "tipo_vivienda" if "tipo_vivienda" in df.columns else "piso_casa"
        
        # Appliquer le regroupement
        df_log = df.copy()
        df_log["type_groupe"], df_log["type_detail"] = _mapper_type_logement(df_log[col_logement])
        
        st.subheader("📊 Vue d'ensemble par type de logement (regroupée)")
        
        # Option pour afficher regroupé ou détail
        view_mode = st.radio(
            "Mode d'affichage",
            options=["🏠 Vue regroupée", "📋 Vue détaillée (tous les types)"],
            horizontal=True
        )
        
        if view_mode == "🏠 Vue regroupée":
            # Analyse avec regroupement
            analyse_types = analyse_par_type_logement(df, regrouper=True)
            df_comparaison = comparer_types_logement(df, regrouper=True)
        else:
            # Vue détaillée sans regroupement
            analyse_types = analyse_par_type_logement(df, regrouper=False)
            df_comparaison = comparer_types_logement(df, regrouper=False)
        
        if "error" not in analyse_types and not df_comparaison.empty:
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Types de logement", len(df_comparaison))
            col_m2.metric("Total appels", f"{df_comparaison['total_appels'].sum():,}")
            meilleur_type = df_comparaison.loc[df_comparaison["taux_qualifies"].idxmax(), "type_logement"]
            meilleur_taux = df_comparaison["taux_qualifies"].max()
            col_m3.metric("🏆 Meilleur taux", f"{meilleur_taux}%", meilleur_type)
            pire_type = df_comparaison.loc[df_comparaison["taux_qualifies"].idxmin(), "type_logement"]
            pire_taux = df_comparaison["taux_qualifies"].min()
            col_m4.metric("Plus faible", f"{pire_taux}%", pire_type)
            
            st.markdown("---")
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                fig = px.bar(df_comparaison.sort_values("taux_qualifies"), x="taux_qualifies", y="type_logement",
                             orientation="h", text="taux_qualifies", color="taux_qualifies",
                             color_continuous_scale="RdYlGn")
                fig.update_traces(texttemplate="%{text}%", textposition="outside")
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col_g2:
                fig = px.bar(df_comparaison.sort_values("total_appels"), x="total_appels", y="type_logement",
                             orientation="h", text="total_appels", color="total_appels",
                             color_continuous_scale="Blues")
                fig.update_traces(textposition="outside")
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.dataframe(
                df_comparaison.rename(columns={
                    "type_logement": "Type de logement", "total_appels": "Total appels",
                    "appels_valides": "Appels valides", "taux_valides": "Taux valides (%)",
                    "appels_qualifies": "Appels qualifiés", "taux_qualifies": "Taux qualification (%)",
                    "pct_rdv_leads": "RDV Leads (%)", "pct_tres_interesse": "Très intéressé (%)",
                    "pct_interesse": "Intéressé (%)"
                }),
                use_container_width=True, hide_index=True
            )
        
        st.markdown("---")
        st.subheader("🔍 Drill-down par type de logement")
        
        # Liste des groupes disponibles (si mode regroupé)
        if view_mode == "🏠 Vue regroupée":
            types_list = sorted(df_log["type_groupe"].dropna().unique())
            types_list = [t for t in types_list if t not in ["", "nan", "None"]]
        else:
            types_list = sorted(df_log["type_detail"].dropna().unique())
            types_list = [t for t in types_list if t not in ["", "nan", "None"]]
        
        if types_list:
            selected_type = st.selectbox("Choisissez un type de logement", types_list)
            
            if view_mode == "🏠 Vue regroupée" and selected_type:
                # Afficher le détail du groupe (drill-down)
                st.markdown(f"**📋 Détail des sous-types pour '{selected_type}':**")
                details = get_detail_by_group(df_log, selected_type)
                if not details.empty:
                    st.dataframe(details, use_container_width=True, hide_index=True)
                    
                    # Graphique des sous-types
                    fig = px.bar(details, x="Type original", y="Nombre d'appels",
                                 text="%", color="Nombre d'appels",
                                 color_continuous_scale="Viridis")
                    fig.update_traces(texttemplate="%{text}%", textposition="outside")
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Analyse détaillée pour ce groupe
                if selected_type in analyse_types:
                    data = analyse_types[selected_type]
                    
                    st.markdown("---")
                    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
                    col_d1.metric("Total appels", data["total_appels"])
                    col_d2.metric("Appels utiles", data["appels_utiles"])
                    col_d3.metric("Taux utiles", f"{data['taux_utiles_pct']}%")
                    col_d4.metric("Taux qualification", f"{data['taux_qualifies_pct']}%")
                    
                    # Afficher les pourcentages prioritaires
                    if "pct_prioritaires" in data:
                        st.markdown("**📊 Classifications prioritaires:**")
                        cols = st.columns(3)
                        for i, (cat, values) in enumerate(data["pct_prioritaires"].items()):
                            with cols[i]:
                                st.metric(
                                    cat,
                                    f"{values['pct_du_total']}%",
                                    f"{values['count']} appels"
                                )
                    
                    if data["duree_moyenne_sec"]:
                        st.metric("Durée moyenne", f"{data['duree_moyenne_sec']}s")
                    
                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        if data["repartition_classifications"]:
                            df_repart = pd.DataFrame([{"Classification": k, "Nombre": v}
                                                       for k, v in data["repartition_classifications"].items()])
                            fig = px.pie(df_repart, names="Classification", values="Nombre",
                                         color_discrete_sequence=PALETTE, hole=0.3)
                            fig.update_traces(textinfo="percent+label")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col_r2:
                        st.subheader("🏆 Top 3 classifications")
                        if data["top_classifications"]:
                            for i, (classification, count) in enumerate(data["top_classifications"].items(), 1):
                                pct = round(count / data['total_appels'] * 100, 1)
                                st.markdown(f"**{i}. {classification}** → {count} appels ({pct}%)")
            else:
                # Vue détaillée - affichage direct
                if selected_type in analyse_types:
                    data = analyse_types[selected_type]
                    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
                    col_d1.metric("Total appels", data["total_appels"])
                    col_d2.metric("Appels utiles", data["appels_utiles"])
                    col_d3.metric("Taux utiles", f"{data['taux_utiles_pct']}%")
                    col_d4.metric("Taux qualification", f"{data['taux_qualifies_pct']}%")
                    
                    if data["duree_moyenne_sec"]:
                        st.metric("Durée moyenne", f"{data['duree_moyenne_sec']}s")
                    
                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        if data["repartition_classifications"]:
                            df_repart = pd.DataFrame([{"Classification": k, "Nombre": v}
                                                       for k, v in data["repartition_classifications"].items()])
                            fig = px.pie(df_repart, names="Classification", values="Nombre",
                                         color_discrete_sequence=PALETTE, hole=0.3)
                            fig.update_traces(textinfo="percent+label")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col_r2:
                        st.subheader("🏆 Top 3 classifications")
                        if data["top_classifications"]:
                            for i, (classification, count) in enumerate(data["top_classifications"].items(), 1):
                                pct = round(count / data['total_appels'] * 100, 1)
                                st.markdown(f"**{i}. {classification}** → {count} appels ({pct}%)")
        else:
            st.warning("Aucun type de logement valide trouvé")
        
        st.markdown("---")
        st.subheader("📊 Vue simplifiée")
        df_tipo = appels_par_piso_casa(df)
        
        if not df_tipo.empty:
            col_p, col_b = st.columns(2)
            with col_p:
                fig = px.pie(df_tipo, names="piso_casa", values="count",
                             color_discrete_sequence=PALETTE, hole=0.4)
                fig.update_traces(textinfo="percent+label")
                fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
            
            with col_b:
                fig = px.bar(df_tipo.sort_values("count"), x="count", y="piso_casa",
                             orientation="h", text="pct", color="count", color_continuous_scale="Purples")
                fig.update_traces(texttemplate="%{text}%", textposition="outside")
                fig.update_layout(yaxis_title="", xaxis_title="Appels", coloraxis_showscale=False, margin=dict(t=10))
                st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(df_tipo.rename(columns={"piso_casa": "Type de logement", "count": "Appels", "pct": "%"}),
                         use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
# TAB 5 — AI RECOMMENDATIONS
# ══════════════════════════════════════════════
with tab5:
    st.header("🤖 IA Décisionnelle - Recommandations Intelligentes")
    st.markdown("---")

    with st.expander("📊 Aperçu des données", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("Total appels", f"{len(df):,}")
        col2.metric("Fournisseurs", df["list_name"].nunique() if "list_name" in df.columns else "N/A")
        col3.metric("Types logement", df["tipo_vivienda"].nunique() if "tipo_vivienda" in df.columns else "N/A")

    st.markdown("---")

    if not api_key_input:
        st.info("👈 Entrez votre clé API Gemini dans la barre latérale pour activer les recommandations IA")

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analyse_btn = st.button("🔮 LANCER L'ANALYSE IA", type="primary", disabled=not api_key_input)

    if analyse_btn:
        advisor = GeminiAdvisor(api_key=api_key_input)
        if not advisor.is_configured:
            st.error("Clé API invalide ou erreur de connexion")
        else:
            st.success("IA Gemini connectée")
            with st.spinner("Gemini analyse vos données..."):
                resultat = advisor.analyser_tous_les_volets(df)
            if resultat:
                st.balloons()
                st.success("Analyse terminée !")
                st.session_state.analyse_ia_resultat = resultat
            else:
                st.error("Échec de l'analyse")

    st.markdown("---")
    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["⏰ Analyse Horaires", "🏢 Analyse Fournisseurs", "🏠 Analyse Logements"])

    with sub_tab1:
        st.subheader("⏰ Analyse des horaires")
        df_h_raw = appels_par_heure(df)

        if not df_h_raw.empty:
            col_h1, col_h2, col_h3 = st.columns(3)
            best = df_h_raw.loc[df_h_raw["nb_appels"].idxmax(), "heure"]
            col_h1.metric("📞 Heure de pointe", f"{best}h")
            col_h2.metric("Total appels", f"{df_h_raw['nb_appels'].sum():,}")
            col_h3.metric("Créneaux actifs", len(df_h_raw[df_h_raw["nb_appels"] > 0]))

            fig = px.line(df_h_raw, x="heure", y="nb_appels", markers=True)
            fig.add_vline(x=best, line_dash="dash", line_color="green")
            fig.add_hline(y=df_h_raw["nb_appels"].mean(), line_dash="dash", line_color="red")
            fig.update_layout(xaxis=dict(tickmode="linear", tick0=0, dtick=1))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_h_raw, use_container_width=True, hide_index=True)

        if "analyse_ia_resultat" in st.session_state:
            resultat = st.session_state.analyse_ia_resultat
            if "analyse_horaire" in resultat:
                h = resultat["analyse_horaire"]
                st.markdown("---")
                col_h1, col_h2, col_h3 = st.columns(3)
                col_h1.metric("🏆 Meilleure heure", f"{h.get('meilleure_heure', 'N/A')}h")
                col_h2.metric("📊 Taux", f"{h.get('meilleur_taux', 0)}%")
                col_h3.metric("📞 Heure max appels", f"{h.get('heure_plus_appels', 'N/A')}h")

                if "performance_par_heure" in h:
                    df_h_ia = pd.DataFrame([{"heure": heur, "taux": d.get("taux", 0)}
                                             for heur, d in h["performance_par_heure"].items()])
                    if not df_h_ia.empty:
                        fig = px.line(df_h_ia, x="heure", y="taux", markers=True)
                        fig.add_hline(y=50, line_dash="dash", line_color="red")
                        st.plotly_chart(fig, use_container_width=True)

            if "recommandations" in resultat and "horaires" in resultat["recommandations"]:
                st.info(f"💡 {resultat['recommandations']['horaires']}")

    with sub_tab2:
        st.subheader("🏢 Analyse fournisseurs")

        if "list_name" in df.columns:
            df_f_raw = (df.groupby("list_name").size().reset_index(name="appels")
                        .sort_values("appels", ascending=False))
            df_f_raw["part_%"] = (df_f_raw["appels"] / df_f_raw["appels"].sum() * 100).round(2)

            col_f1, col_f2, col_f3 = st.columns(3)
            col_f1.metric("Nombre fournisseurs", len(df_f_raw))
            col_f2.metric("🏆 Principal", df_f_raw.iloc[0]["list_name"])
            col_f3.metric("Total appels", f"{df_f_raw['appels'].sum():,}")

            st.dataframe(df_f_raw, use_container_width=True, hide_index=True)
            fig = px.bar(df_f_raw.sort_values("appels"), x="appels", y="list_name", orientation="h",
                         color="appels", color_continuous_scale="Blues")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        if "analyse_ia_resultat" in st.session_state:
            resultat = st.session_state.analyse_ia_resultat
            if "analyse_fournisseurs" in resultat:
                df_f_ia = pd.DataFrame(resultat["analyse_fournisseurs"])
                if not df_f_ia.empty:
                    st.markdown("---")
                    meilleur = df_f_ia.loc[df_f_ia["taux_classification"].idxmax()]
                    pire = df_f_ia.loc[df_f_ia["taux_classification"].idxmin()]
                    col_f1, col_f2, col_f3 = st.columns(3)
                    col_f1.metric("Fournisseurs analysés", len(df_f_ia))
                    col_f2.metric("🏆 Meilleur taux", f"{meilleur['taux_classification']}%", meilleur['nom'])
                    col_f3.metric("⚠️ À améliorer", f"{pire['taux_classification']}%", pire['nom'])
                    st.dataframe(df_f_ia.sort_values("taux_classification", ascending=False), use_container_width=True)
                    fig = px.bar(df_f_ia.sort_values("taux_classification"), x="taux_classification", y="nom",
                                 orientation="h", color="taux_classification", color_continuous_scale="RdYlGn")
                    st.plotly_chart(fig, use_container_width=True)

            if "recommandations" in resultat and "fournisseurs" in resultat["recommandations"]:
                st.success(f"💡 {resultat['recommandations']['fournisseurs']}")

    with sub_tab3:
        st.subheader("🏠 Analyse logements")

        if "tipo_vivienda" in df.columns:
            df_l_raw = (df.groupby("tipo_vivienda").size().reset_index(name="appels")
                        .rename(columns={"tipo_vivienda": "type"}).sort_values("appels", ascending=False))
            df_l_raw["part_%"] = (df_l_raw["appels"] / df_l_raw["appels"].sum() * 100).round(2)

            col_l1, col_l2, col_l3 = st.columns(3)
            col_l1.metric("Types logement", len(df_l_raw))
            col_l2.metric("🏆 Type principal", df_l_raw.iloc[0]["type"])
            col_l3.metric("Total appels", f"{df_l_raw['appels'].sum():,}")

            st.dataframe(df_l_raw, use_container_width=True, hide_index=True)
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                fig = px.bar(df_l_raw.head(10).sort_values("appels"), x="appels", y="type",
                             orientation="h", color="appels", color_continuous_scale="Greens")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            with col_g2:
                fig2 = px.pie(df_l_raw.head(10), values="appels", names="type")
                st.plotly_chart(fig2, use_container_width=True)

        if "analyse_ia_resultat" in st.session_state:
            resultat = st.session_state.analyse_ia_resultat
            if "analyse_logements" in resultat:
                df_l_ia = pd.DataFrame(resultat["analyse_logements"])
                if not df_l_ia.empty:
                    st.markdown("---")
                    meilleur = df_l_ia.loc[df_l_ia["taux_classification"].idxmax()]
                    col_l1, col_l2, col_l3 = st.columns(3)
                    col_l1.metric("Types analysés", len(df_l_ia))
                    col_l2.metric("🏆 Top", f"{meilleur['taux_classification']}%", meilleur['type'])
                    col_l3.metric("Total appels", f"{df_l_ia['appels'].sum():,}")
                    st.dataframe(df_l_ia.sort_values("taux_classification", ascending=False), use_container_width=True)
                    col_g1, col_g2 = st.columns(2)
                    with col_g1:
                        fig = px.bar(df_l_ia.head(10), x="taux_classification", y="type",
                                     orientation="h", color="taux_classification", color_continuous_scale="Greens")
                        st.plotly_chart(fig, use_container_width=True)
                    with col_g2:
                        fig2 = px.bar(df_l_ia.head(10), x="appels", y="type",
                                      orientation="h", color="appels", color_continuous_scale="Blues")
                        st.plotly_chart(fig2, use_container_width=True)

            if "recommandations" in resultat and "logements" in resultat["recommandations"]:
                st.info(f"💡 {resultat['recommandations']['logements']}")

    if "analyse_ia_resultat" in st.session_state:
        resultat = st.session_state.analyse_ia_resultat
        st.markdown("---")

        if "prediction" in resultat:
            st.subheader("🔮 Prédiction")
            st.info(resultat["prediction"])

        if "resume_executif" in resultat:
            st.subheader("📌 Résumé exécutif")
            st.info(resultat["resume_executif"])

        if "actions_prioritaires" in resultat:
            st.subheader("🚀 Actions prioritaires")
            for action in resultat["actions_prioritaires"]:
                st.markdown(f"**👉 {action['action']}**  \n📌 Pourquoi: {action['pourquoi']}  \n🎯 Impact: {action['impact']}")

        st.markdown("---")
        st.download_button("📥 Export JSON", data=json.dumps(resultat, ensure_ascii=False, indent=2),
                           file_name="analyse_ia.json", mime="application/json")

# ══════════════════════════════════════════════
# TAB ATS
# ══════════════════════════════════════════════
with tab_ats:
    render_ats_tab(api_key_input=api_key_input)

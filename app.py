import streamlit as st

st.set_page_config(
    page_title="Call Center Dashboard",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from google_selector import list_sheets, choisir_feuille
from ai_recommendation import GeminiAdvisor
from analyse import (
    kpi_globaux,
    appels_par_jour,
    appels_par_mois,
    appels_par_heure,
    repartition_classification,
    appels_par_fournisseur,
    classification_par_fournisseur,
    taux_remplissage_code_postal,
    comparer_codes_postaux,
    analyse_fiabilite_par_fournisseur,
    codes_postaux_non_correspondants,
    analyse_par_type_logement,
    comparer_types_logement,
    classification_detaillee_par_type,
    appels_par_piso_casa,
    duree_par_classification,
    _mapper_type_logement,
     get_detail_by_group,
    _est_utile, _est_qualifie,
    analyse_fiabilite_par_fournisseur
)
from ats_analysis import render_ats_tab

PALETTE = px.colors.qualitative.Set2

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

CLASSIFICATIONS_QUALIF = [
    "PEU INTERESSE", "INTERESSE", "TRES INTERESSE",
    "EDIFICIOS", "RDV LEADS", "WHATSAP",
]


def _taux_qualification(df: pd.DataFrame):
    """Retourne (nb_qualifiés, taux%) ou (None, None) si colonne absente."""
    if "Classification" not in df.columns or df.empty:
        return None, None
    mask = df["Classification"].astype(str).str.upper().str.strip().isin(
        [c.upper() for c in CLASSIFICATIONS_QUALIF]
    )
    nb = int(mask.sum())
    taux = round(nb / len(df) * 100, 1)
    return nb, taux


def _get_detail_by_group(df_log: pd.DataFrame, groupe: str) -> pd.DataFrame:
    """Retourne le détail des sous-types pour un groupe donné."""
    subset = df_log[df_log["type_groupe"] == groupe]
    if subset.empty:
        return pd.DataFrame()
    details = (
        subset["type_detail"]
        .value_counts()
        .reset_index()
    )
    details.columns = ["Type original", "Nombre d'appels"]
    details["%"] = (details["Nombre d'appels"] / details["Nombre d'appels"].sum() * 100).round(1)
    return details


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

for key, default in [
    ("df_raw", None),
    ("fichier", None),
    ("sheets_list", None),
    ("selected_sheets", []),
    ("stats_chargement", {}),
    ("analyse_ia_resultat", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

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
        4. Copiez le lien et collez-le ci-dessous
        """)

    sheet_url = st.text_input(
        "URL du Google Sheet",
        placeholder="https://docs.google.com/spreadsheets/d/...",
    )

    if sheet_url and st.button("📂 Charger les feuilles", type="primary"):
        try:
            with st.spinner("Chargement du fichier..."):
                st.session_state.fichier, st.session_state.sheets_list = list_sheets(sheet_url)
            st.success(f"{len(st.session_state.sheets_list)} feuille(s) trouvée(s)")
        except Exception as exc:
            st.error(f"Erreur de chargement : {exc}")
            st.session_state.fichier = None
            st.session_state.sheets_list = None

    if st.session_state.sheets_list:
        st.markdown("---")
        st.subheader("📑 Sélection des feuilles")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Tout sélectionner"):
                st.session_state.selected_sheets = st.session_state.sheets_list.copy()
                st.rerun()
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
                    total = len(selected_sheets)

                    for i, sheet_name in enumerate(selected_sheets):
                        status_text.text(f"Chargement de '{sheet_name}'... ({i + 1}/{total})")
                        df_sheet = choisir_feuille(st.session_state.fichier, sheet_name)
                        if df_sheet is not None and not df_sheet.empty:
                            df_sheet["_source_feuille"] = sheet_name
                            all_dfs.append(df_sheet)
                        progress_bar.progress((i + 1) / total)

                    status_text.empty()
                    progress_bar.empty()

                    if all_dfs:
                        st.session_state.df_raw = pd.concat(all_dfs, ignore_index=True)
                        st.session_state.stats_chargement = {
                            "nb_feuilles": total,
                            "total_lignes": len(st.session_state.df_raw),
                            "feuilles_details": [
                                {"nom": name, "lignes": len(d)}
                                for name, d in zip(selected_sheets, all_dfs)
                            ],
                        }
                        st.success(f"Données chargées : {len(st.session_state.df_raw):,} lignes")

                        with st.expander("📋 Détail du chargement"):
                            st.write(f"**Total lignes :** {len(st.session_state.df_raw):,}")
                            st.write(f"**Colonnes :** {', '.join(st.session_state.df_raw.columns[:8])}")
                            for detail in st.session_state.stats_chargement["feuilles_details"]:
                                st.write(f"- {detail['nom']} : {detail['lignes']:,} lignes")
                    else:
                        st.error("Aucune donnée valide chargée")

                except Exception as exc:
                    st.error(f"Erreur : {exc}")
        else:
            st.warning("Veuillez sélectionner au moins une feuille")

    # Filtres (uniquement si données chargées)
    fourn_sel = "Tous"
    date_range = None

    if st.session_state.df_raw is not None:
        st.markdown("---")
        st.subheader("Filtres")
        df_raw = st.session_state.df_raw

        if "list_name" in df_raw.columns:
            fournisseurs = ["Tous"] + sorted(df_raw["list_name"].dropna().unique().tolist())
            fourn_sel = st.selectbox("Fournisseur (list_name)", fournisseurs)

        if "Timestamp" in df_raw.columns:
            ts_all = pd.to_datetime(df_raw["Timestamp"], errors="coerce", dayfirst=True).dropna()
            if not ts_all.empty:
                date_min, date_max = ts_all.min().date(), ts_all.max().date()
                date_range = st.date_input(
                    "Période",
                    value=(date_min, date_max),
                    min_value=date_min,
                    max_value=date_max,
                )

        st.markdown("---")
        if st.button("🔄 Actualiser"):
            st.cache_data.clear()
            st.rerun()

    # Clé API Gemini
    api_key_input = st.text_input(
        "🔑 Clé API Gemini",
        type="password",
        placeholder="AIza...",
        key="gemini_key_global",
    )

# ─────────────────────────────────────────────
# GARDE : données obligatoires
# ─────────────────────────────────────────────

if st.session_state.df_raw is None:
    st.info("👈 Commencez par entrer l'URL de votre Google Sheet dans la barre latérale")
    st.stop()

# ─────────────────────────────────────────────
# APPLICATION DES FILTRES
# ─────────────────────────────────────────────

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
    "📋 Analyse des ATS par IA",
])

# ══════════════════════════════════════════════
# TAB 1 — ANALYSE GLOBALE
# ══════════════════════════════════════════════

with tab1:
    st.header("Analyse globale des appels")

    kpis = kpi_globaux(df)
    _, taux_qualifie = _taux_qualification(df)

    # Ligne 1 : KPIs principaux
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total appels", f"{kpis['total_appels']:,}")
    c2.metric("Appels utiles", f"{kpis['appels_utiles']:,}" if kpis["appels_utiles"] is not None else "—")
    c3.metric("Taux utiles", f"{kpis['taux_utiles_pct']}%" if kpis["taux_utiles_pct"] is not None else "—")
    c4.metric("Durée moy. globale", f"{kpis['duree_moyenne_sec']:.0f}s" if kpis["duree_moyenne_sec"] is not None else "—")
    c5.metric("Taux qualification", f"{taux_qualifie}%" if taux_qualifie is not None else "—")

    # Ligne 2 : Durées par catégorie
    st.markdown("---")
    st.subheader("⏱️ Durées moyennes par catégorie")
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    col_d1.metric("📞 Appels utiles", f"{kpis['duree_moyenne_utiles_sec']:.0f}s" if kpis.get("duree_moyenne_utiles_sec") else "—")
    col_d2.metric("🎯 RDV Leads", f"{kpis['duree_moyenne_rdv_leads_sec']:.0f}s" if kpis.get("duree_moyenne_rdv_leads_sec") else "—")
    col_d3.metric("🔥 Très intéressé", f"{kpis['duree_moyenne_tres_interesse_sec']:.0f}s" if kpis.get("duree_moyenne_tres_interesse_sec") else "—")
    col_d4.metric("👍 Intéressé", f"{kpis['duree_moyenne_interesse_sec']:.0f}s" if kpis.get("duree_moyenne_interesse_sec") else "—")

    st.markdown("---")

    # Graphique durée par classification
    st.subheader("📊 Durée moyenne par classification")
    df_duree_classif = duree_par_classification(df)
    if not df_duree_classif.empty:
        df_duree_classif = df_duree_classif[df_duree_classif["count"] >= 5]
        if not df_duree_classif.empty:
            fig = px.bar(
                df_duree_classif.sort_values("duree_moy_sec", ascending=False),
                x="duree_moy_sec", y="Classification", orientation="h",
                text="duree_moy_sec", color="duree_moy_sec",
                color_continuous_scale="Viridis",
                title="Durée moyenne par type de classification",
            )
            fig.update_traces(texttemplate="%{text:.0f}s", textposition="outside")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📑 Tableau détaillé des durées par classification"):
                st.dataframe(
                    df_duree_classif.rename(columns={
                        "count": "Nombre d'appels",
                        "duree_moy_sec": "Durée moyenne (s)",
                        "duree_mediane_sec": "Durée médiane (s)",
                        "duree_min_sec": "Durée min (s)",
                        "duree_max_sec": "Durée max (s)",
                    }),
                    use_container_width=True, hide_index=True,
                )
        else:
            st.info("Pas assez de données par classification (min 5 appels).")
    else:
        st.info("Colonne Duration_seconds non trouvée.")

    st.markdown("---")
    col_j, col_m = st.columns(2)

    with col_j:
        st.subheader("Appels par jour")
        df_jour = appels_par_jour(df)
        if not df_jour.empty:
            df_jour["date"] = df_jour["date"].astype(str)
            fig = px.bar(df_jour, x="date", y="nb_appels", color_discrete_sequence=[PALETTE[0]])
            fig.update_layout(xaxis=dict(type="category", title=""), yaxis_title="Appels", margin=dict(t=10))
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
            fig.update_layout(xaxis=dict(type="category", title=""), yaxis_title="Appels", margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Colonne Timestamp absente.")

    st.markdown("---")
    col_cl, col_h = st.columns(2)

    with col_cl:
        st.subheader("Répartition par classification")
        df_cls = repartition_classification(df)
        if not df_cls.empty:
            fig = px.pie(df_cls, names="Classification", values="count",
                         color_discrete_sequence=PALETTE, hole=0.4)
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune donnée de classification valide.")

    with col_h:
        st.subheader("Appels par heure de la journée")
        df_heure = appels_par_heure(df)
        if not df_heure.empty:
            fig = px.line(df_heure, x="heure", y="nb_appels", markers=True,
                          color_discrete_sequence=[PALETTE[2]])
            fig.update_traces(line=dict(width=2), marker=dict(size=7))
            fig.update_layout(xaxis=dict(tickmode="linear", dtick=1, title="Heure"),
                              yaxis_title="Appels", margin=dict(t=10))
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
                "list_name": "Fournisseur",
                "nb_appels": "Total appels",
                "nb_utiles": "Appels classifiés",
                "taux_utiles_pct": "Taux classification (%)",
                "nb_qualifies": "Appels qualifiés",
                "taux_qualifies_pct": "Taux qualification (%)",
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
                pivot = df_cls_fourn.pivot_table(
                    index="list_name", columns="Classification", values="count", fill_value=0
                )
                st.dataframe(pivot, use_container_width=True)
        else:
            st.info("Aucune donnée de classification valide.")

        # ── Analyse logements par fournisseur ──
        st.markdown("---")
        st.subheader("🏠 Analyse des types de logement par fournisseur")

        col_logement = next(
            (c for c in ["tipo_vivienda", "piso_casa"] if c in df.columns), None
        )

        if col_logement is None:
            st.info("Colonne 'tipo_vivienda' ou 'piso_casa' non trouvée.")
        else:
            use_grouping = st.checkbox(
                "📦 Regrouper les types similaires", value=True,
                key="grouping_checkbox_tab2",
                help="Regroupe automatiquement les types (PISO, CASA, etc.)"
            )

            df_log_temp = df.copy()
            df_log_temp["type_groupe"], df_log_temp["type_detail"] = _mapper_type_logement(df_log_temp[col_logement])
            col_analyse = "type_groupe" if use_grouping else "type_detail"

            # Nettoyage
            df_log_temp = df_log_temp[
                df_log_temp[col_analyse].notna() &
                (df_log_temp[col_analyse].astype(str).str.strip() != "") &
                (df_log_temp[col_analyse].astype(str) != "nan")
            ]

            if df_log_temp.empty:
                st.info("Aucune donnée sur les types de logement.")
            else:
                fournisseurs_list = sorted(df["list_name"].dropna().unique()) if "list_name" in df.columns else []
                selected_fournisseur = st.selectbox(
                    "Choisissez un fournisseur",
                    ["Tous les fournisseurs"] + fournisseurs_list,
                    key="logement_fournisseur_select_tab2",
                )

                df_filtered = (
                    df_log_temp if selected_fournisseur == "Tous les fournisseurs"
                    else df_log_temp[df_log_temp["list_name"] == selected_fournisseur]
                )

                # Métriques
                col_log1, col_log2, col_log3 = st.columns(3)
                col_log1.metric("Appels avec type logement", f"{len(df_filtered):,}")
                col_log2.metric("Types différents", df_filtered[col_analyse].nunique())
                top_type = df_filtered[col_analyse].mode().iloc[0] if not df_filtered.empty else "N/A"
                col_log3.metric("Type le plus fréquent", top_type)

                st.markdown("---")
                col_chart1, col_chart2 = st.columns(2)

                with col_chart1:
                    st.subheader("🥧 Répartition par type")
                    counts = df_filtered[col_analyse].value_counts()
                    fig_pie = px.pie(values=counts.values, names=counts.index,
                                    title="Répartition des logements",
                                    color_discrete_sequence=PALETTE, hole=0.3)
                    fig_pie.update_traces(textinfo="percent+label")
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col_chart2:
                    st.subheader("📊 Top types")
                    df_top = df_filtered[col_analyse].value_counts().head(10).reset_index()
                    df_top.columns = ["Type de logement", "Nombre d'appels"]
                    fig_bar = px.bar(df_top, x="Nombre d'appels", y="Type de logement",
                                    orientation="h", text="Nombre d'appels",
                                    color="Nombre d'appels", color_continuous_scale="Blues")
                    fig_bar.update_traces(textposition="outside")
                    fig_bar.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(fig_bar, use_container_width=True)

                # Drill-down (mode regroupé uniquement)
                if use_grouping:
                    st.markdown("---")
                    st.subheader("🔍 Drill-down par type")
                    groupes_disponibles = sorted(df_filtered[col_analyse].unique())
                    selected_groupe = st.selectbox(
                        "Choisissez un groupe pour voir les sous-types",
                        groupes_disponibles, key="drilldown_select_tab2",
                    )
                    if selected_groupe:
                        details = _get_detail_by_group(df_filtered, selected_groupe)
                        if not details.empty:
                            st.dataframe(details, use_container_width=True, hide_index=True)
                            fig = px.bar(details, x="Type original", y="Nombre d'appels",
                                        text="%", color="Nombre d'appels",
                                        color_continuous_scale="Viridis")
                            fig.update_traces(texttemplate="%{text}%", textposition="outside")
                            fig.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig, use_container_width=True)

                # Classification par type de logement
                if "Classification" in df.columns:
                    st.markdown("---")
                    st.subheader("🎯 Classification des appels par type de logement")
                    df_classif = df_filtered[
                        df_filtered["Classification"].notna() &
                        ~df_filtered["Classification"].astype(str).str.lower().isin(["non trouvé", "non trouve", ""])
                    ]
                    if not df_classif.empty:
                        top_types = df_classif[col_analyse].value_counts().head(5).index
                        cross_classif = pd.crosstab(
                            df_classif[df_classif[col_analyse].isin(top_types)][col_analyse],
                            df_classif[df_classif[col_analyse].isin(top_types)]["Classification"],
                        )
                        fig_classif = px.bar(cross_classif, barmode="group",
                                            title="Classifications par type de logement (Top 5)",
                                            color_discrete_sequence=PALETTE)
                        st.plotly_chart(fig_classif, use_container_width=True)
                        with st.expander("📑 Voir le tableau détaillé"):
                            st.dataframe(cross_classif, use_container_width=True)

                # Récapitulatif par fournisseur (tous)
                if selected_fournisseur == "Tous les fournisseurs" and "list_name" in df_filtered.columns:
                    st.markdown("---")
                    st.subheader("📋 Récapitulatif par fournisseur")
                    df_pivot = pd.crosstab(
                        df_filtered["list_name"], df_filtered[col_analyse], normalize="index"
                    ) * 100
                    st.dataframe(
                        df_pivot.round(1).style.format("{:.1f}%"),
                        use_container_width=True, height=400,
                    )
                    csv = df_pivot.to_csv().encode("utf-8")
                    st.download_button(
                        "📥 Exporter les données", data=csv,
                        file_name="logement_par_fournisseur.csv", mime="text/csv",
                    )

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
                st.metric(
                    f"Colonne : {colonne}",
                    f"{stats['taux_remplissage']}%",
                    f"{stats['nb_remplis']}/{stats['total_lignes']} lignes",
                )
                st.progress(stats["taux_remplissage"] / 100)
        else:
            st.warning("Colonnes 'code_postal' et/ou 'codigo_postal' non trouvées.")

    with col2:
        st.subheader("Comparaison Client vs Fournisseur")
        df_comp, stats = comparer_codes_postaux(df)
        if stats:
            st.metric("Total comparaisons", stats["total_comparaisons"])
            st.metric("Correspondances", f"{stats['nb_correspondances']} / {stats['total_comparaisons']}")
            taux_corr = stats["taux_correspondance"]
            st.metric("Taux de correspondance", f"{taux_corr}%")
            if taux_corr >= 80:
                st.success(f"Bonne fiabilité : {taux_corr}%")
            elif taux_corr >= 50:
                st.warning(f"Fiabilité moyenne : {taux_corr}%")
            else:
                st.error(f"Faible fiabilité : {taux_corr}%")
        else:
            st.info("Pas assez de données pour comparer les codes postaux.")

    st.subheader("🏢 Fiabilité par fournisseur")
df_fiabilite = analyse_fiabilite_par_fournisseur(df)

if not df_fiabilite.empty:
    # Nettoyer le DataFrame pour éviter les types non compatibles
    df_fiabilite_clean = df_fiabilite.copy()
    
    # Convertir les colonnes problématiques
    for col in df_fiabilite_clean.columns:
        # Si la colonne contient des dictionnaires ou listes, les convertir en string
        if df_fiabilite_clean[col].apply(lambda x: isinstance(x, (dict, list))).any():
            df_fiabilite_clean[col] = df_fiabilite_clean[col].astype(str)
    
    # Renommer les colonnes pour l'affichage
    col_rename = {}
    for col in df_fiabilite_clean.columns:
        if 'fournisseur' in col.lower():
            col_rename[col] = "Fournisseur"
        elif 'total_appels' in col.lower():
            col_rename[col] = "Total appels"
        elif 'taux_remplissage_client' in col.lower():
            col_rename[col] = "Taux client (%)"
        elif 'taux_remplissage_fournisseur' in col.lower():
            col_rename[col] = "Taux fournisseur (%)"
        elif 'nb_comparaisons' in col.lower():
            col_rename[col] = "Nb comparaisons"
        elif 'taux_correspondance' in col.lower():
            col_rename[col] = "Taux correspondance (%)"
    
    df_fiabilite_clean = df_fiabilite_clean.rename(columns=col_rename)
    
    # Afficher le tableau
    st.dataframe(df_fiabilite_clean, use_container_width=True, hide_index=True)

    # Graphiques
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        # Graphique des taux de remplissage
        if 'Fournisseur' in df_fiabilite_clean.columns:
            df_plot = pd.DataFrame()
            df_plot['Fournisseur'] = df_fiabilite_clean['Fournisseur']
            df_plot['Taux client'] = pd.to_numeric(df_fiabilite_clean.get('Taux client (%)', 0), errors='coerce').fillna(0)
            df_plot['Taux fournisseur'] = pd.to_numeric(df_fiabilite_clean.get('Taux fournisseur (%)', 0), errors='coerce').fillna(0)
            
            df_melt = df_plot.melt(id_vars=['Fournisseur'], var_name='Source', value_name='Taux (%)')
            fig = px.bar(df_melt, x='Fournisseur', y='Taux (%)', color='Source', barmode='group')
            st.plotly_chart(fig, use_container_width=True)
    
    with col_g2:
        # Graphique du taux de correspondance
        if 'Fournisseur' in df_fiabilite_clean.columns and 'Taux correspondance (%)' in df_fiabilite_clean.columns:
            df_corr = df_fiabilite_clean[['Fournisseur', 'Taux correspondance (%)']].copy()
            df_corr['Taux correspondance (%)'] = pd.to_numeric(df_corr['Taux correspondance (%)'], errors='coerce').fillna(0)
            
            fig = px.bar(
                df_corr,
                x='Fournisseur',
                y='Taux correspondance (%)',
                color='Taux correspondance (%)',
                color_continuous_scale='RdYlGn',
                title="Taux de correspondance par fournisseur"
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Données insuffisantes pour analyser la fiabilité par fournisseur")
    st.subheader("🔍 Codes postaux non correspondants")
    df_non_corr = codes_postaux_non_correspondants(df)

    if not df_non_corr.empty:
        cols_afficher = ["list_name", "code_postal", "codigo_postal", "code_postal_clean", "codigo_postal_clean"]
        cols_disponibles = [c for c in cols_afficher if c in df_non_corr.columns]
        st.dataframe(df_non_corr[cols_disponibles].head(100), use_container_width=True, hide_index=True)
        csv = df_non_corr[cols_disponibles].to_csv(index=False).encode("utf-8")
        st.download_button("📥 Exporter les non-correspondances", data=csv,
                           file_name="non_correspondances_codes_postaux.csv", mime="text/csv")
    else:
        st.success("Tous les codes postaux disponibles correspondent !")
# ══════════════════════════════════════════════
# TAB 4 — LOGEMENTS (avec regroupement et drill-down)
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
        
        # Option d'affichage
        st.subheader("📊 Vue d'ensemble")
        
        # Premier graphique : répartition générale
        col_vue1, col_vue2 = st.columns(2)
        
        with col_vue1:
            # Graphique en camembert des groupes
            group_counts = df_log["type_groupe"].value_counts().reset_index()
            group_counts.columns = ["Groupe", "Nombre d'appels"]
            group_counts["%"] = (group_counts["Nombre d'appels"] / group_counts["Nombre d'appels"].sum() * 100).round(1)
            
            fig = px.pie(
                group_counts,
                names="Groupe",
                values="Nombre d'appels",
                title="Répartition par groupe de logement",
                color_discrete_sequence=PALETTE,
                hole=0.3
            )
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_vue2:
            # Graphique en barres des groupes
            fig = px.bar(
                group_counts.sort_values("Nombre d'appels", ascending=True),
                x="Nombre d'appels",
                y="Groupe",
                orientation="h",
                text="%",
                color="Nombre d'appels",
                color_continuous_scale="Blues",
                title="Nombre d'appels par groupe"
            )
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Switch entre vue regroupée et vue détaillée
        st.markdown("---")
        view_mode = st.radio(
            "🔍 Mode d'affichage détaillé",
            options=["📦 Vue par groupe (regroupée)", "📋 Vue par type original (détaillée)"],
            horizontal=True,
            key="view_mode_tab4"
        )
        
        if view_mode == "📦 Vue par groupe (regroupée)":
            # ==========================================
            # VUE REGROUPÉE
            # ==========================================
            st.subheader("📊 Analyse par groupe de logement")
            
            # Statistiques par groupe
            st.dataframe(group_counts, use_container_width=True, hide_index=True)
            
            # Métriques par groupe
            st.markdown("---")
            st.subheader("📈 Métriques par groupe")
            
            # Calculer les métriques par groupe
            group_metrics = []
            for groupe in df_log["type_groupe"].unique():
                df_g = df_log[df_log["type_groupe"] == groupe]
                total = len(df_g)
                
                # Appels utiles
                if "Classification" in df.columns:
                    utiles = _est_utile(df_g["Classification"]).sum()
                    taux_utiles = round(utiles / total * 100, 1) if total > 0 else 0
                    
                    # Appels qualifiés
                    qualifies = _est_qualifie(df_g["Classification"]).sum()
                    taux_qualifies = round(qualifies / total * 100, 1) if total > 0 else 0
                    
                    # RDV Leads, Très intéressé, Intéressé
                    rdv = (df_g["Classification"].astype(str).str.upper() == "RDV LEADS").sum()
                    tres = (df_g["Classification"].astype(str).str.upper() == "TRES INTERESSE").sum()
                    inter = (df_g["Classification"].astype(str).str.upper() == "INTERESSE").sum()
                else:
                    utiles = 0
                    taux_utiles = 0
                    qualifies = 0
                    taux_qualifies = 0
                    rdv = tres = inter = 0
                
                group_metrics.append({
                    "Groupe": groupe,
                    "Total": total,
                    "Taux utiles": f"{taux_utiles}%",
                    "Taux qualifiés": f"{taux_qualifies}%",
                    "RDV Leads": rdv,
                    "Très intéressé": tres,
                    "Intéressé": inter
                })
            
            df_metrics = pd.DataFrame(group_metrics)
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)
            
            # Graphique comparatif des taux
            col_comp1, col_comp2 = st.columns(2)
            with col_comp1:
                fig = px.bar(
                    df_metrics,
                    x="Groupe",
                    y="Taux qualifiés",
                    title="Taux de qualification par groupe",
                    color="Taux qualifiés",
                    color_continuous_scale="RdYlGn",
                    text="Taux qualifiés"
                )
                fig.update_traces(texttemplate="%{text}%", textposition="outside")
                st.plotly_chart(fig, use_container_width=True)
            
            with col_comp2:
                # Graphique des RDV Leads
                fig = px.bar(
                    df_metrics,
                    x="Groupe",
                    y="RDV Leads",
                    title="Nombre de RDV Leads par groupe",
                    color="RDV Leads",
                    color_continuous_scale="Viridis",
                    text="RDV Leads"
                )
                fig.update_traces(texttemplate="%{text}", textposition="outside")
                st.plotly_chart(fig, use_container_width=True)
            
            # Drill-down
            st.markdown("---")
            st.subheader("🔍 Drill-down : voir les détails d'un groupe")
            
            groupes = sorted(df_log["type_groupe"].dropna().unique())
            selected_groupe = st.selectbox(
                "Choisissez un groupe pour voir ses sous-types",
                groupes,
                key="drilldown_tab4"
            )
            
            if selected_groupe:
                details = get_detail_by_group(df_log, selected_groupe)
                if not details.empty:
                    st.write(f"**📋 Détail des sous-types pour '{selected_groupe}' :**")
                    
                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        st.dataframe(details, use_container_width=True, hide_index=True)
                    
                    with col_d2:
                        fig = px.bar(
                            details,
                            x="Type original",
                            y="Nombre d'appels",
                            text="%",
                            color="Nombre d'appels",
                            color_continuous_scale="Viridis",
                            title=f"Sous-types de {selected_groupe}"
                        )
                        fig.update_traces(texttemplate="%{text}%", textposition="outside")
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
        
        else:
            # ==========================================
            # VUE DÉTAILLÉE (types originaux)
            # ==========================================
            st.subheader("📋 Analyse par type original (détaillé)")
            
            detail_counts = df_log["type_detail"].value_counts().reset_index()
            detail_counts.columns = ["Type de logement", "Nombre d'appels"]
            detail_counts["%"] = (detail_counts["Nombre d'appels"] / detail_counts["Nombre d'appels"].sum() * 100).round(1)
            
            # Top 15 pour la lisibilité
            top_n = 15
            detail_counts_top = detail_counts.head(top_n)
            autres = detail_counts.iloc[top_n:]
            
            if not autres.empty:
                autres_count = autres["Nombre d'appels"].sum()
                autres_pct = autres["%"].sum()
                detail_counts_top = pd.concat([
                    detail_counts_top,
                    pd.DataFrame({"Type de logement": ["AUTRES"], "Nombre d'appels": [autres_count], "%": [round(autres_pct, 1)]})
                ], ignore_index=True)
            
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                fig = px.pie(
                    detail_counts_top,
                    names="Type de logement",
                    values="Nombre d'appels",
                    title=f"Top {top_n} des types de logement",
                    color_discrete_sequence=PALETTE,
                    hole=0.3
                )
                fig.update_traces(textinfo="percent+label")
                st.plotly_chart(fig, use_container_width=True)
            
            with col_d2:
                fig = px.bar(
                    detail_counts_top.sort_values("Nombre d'appels", ascending=True),
                    x="Nombre d'appels",
                    y="Type de logement",
                    orientation="h",
                    text="%",
                    color="Nombre d'appels",
                    color_continuous_scale="Blues",
                    title="Nombre d'appels par type"
                )
                fig.update_traces(texttemplate="%{text}%", textposition="outside")
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Tableau complet
            with st.expander("📑 Voir tous les types détaillés"):
                st.dataframe(detail_counts, use_container_width=True, hide_index=True)
            
            # Métriques par type original (top 10)
            st.markdown("---")
            st.subheader("📈 Top 10 des types - Métriques de qualification")
            
            top_10_types = detail_counts.head(10)["Type de logement"].tolist()
            df_top_types = df_log[df_log["type_detail"].isin(top_10_types)]
            
            if "Classification" in df.columns and not df_top_types.empty:
                top_metrics = []
                for type_val in top_10_types:
                    df_t = df_top_types[df_top_types["type_detail"] == type_val]
                    total = len(df_t)
                    
                    utiles = _est_utile(df_t["Classification"]).sum()
                    taux_utiles = round(utiles / total * 100, 1) if total > 0 else 0
                    
                    qualifies = _est_qualifie(df_t["Classification"]).sum()
                    taux_qualifies = round(qualifies / total * 100, 1) if total > 0 else 0
                    
                    rdv = (df_t["Classification"].astype(str).str.upper() == "RDV LEADS").sum()
                    tres = (df_t["Classification"].astype(str).str.upper() == "TRES INTERESSE").sum()
                    inter = (df_t["Classification"].astype(str).str.upper() == "INTERESSE").sum()
                    
                    top_metrics.append({
                        "Type": type_val,
                        "Total": total,
                        "Taux utiles": f"{taux_utiles}%",
                        "Taux qualifiés": f"{taux_qualifies}%",
                        "RDV": rdv,
                        "Très int.": tres,
                        "Intéressé": inter
                    })
                
                df_top_metrics = pd.DataFrame(top_metrics)
                st.dataframe(df_top_metrics, use_container_width=True, hide_index=True)
        
        # ==========================================
        # SECTION COMMUNE : Classifications prioritaires
        # ==========================================
        st.markdown("---")
        st.subheader("🎯 Classifications prioritaires")
        
        if "Classification" in df.columns:
            # Filtrer les classifications valides
            df_classif = df_log[df_log["Classification"].notna()]
            df_classif = df_classif[~df_classif["Classification"].astype(str).str.lower().isin(["non trouvé", "non trouve", ""])]
            
            if not df_classif.empty:
                # Choix du niveau d'analyse
                level_choice = st.radio(
                    "Niveau d'analyse",
                    options=["Par groupe", "Par type détaillé"],
                    horizontal=True,
                    key="priority_level_tab4"
                )
                
                if level_choice == "Par groupe":
                    col_analyse = "type_groupe"
                else:
                    col_analyse = "type_detail"
                    # Limiter aux top 10 pour la lisibilité
                    top_types = df_classif[col_analyse].value_counts().head(10).index
                    df_classif = df_classif[df_classif[col_analyse].isin(top_types)]
                
                # Calculer les statistiques
                stats = []
                for type_val in df_classif[col_analyse].unique():
                    df_type = df_classif[df_classif[col_analyse] == type_val]
                    total = len(df_type)
                    
                    rdv = (df_type["Classification"].astype(str).str.upper() == "RDV LEADS").sum()
                    tres = (df_type["Classification"].astype(str).str.upper() == "TRES INTERESSE").sum()
                    inter = (df_type["Classification"].astype(str).str.upper() == "INTERESSE").sum()
                    
                    stats.append({
                        "Type": type_val,
                        "Total appels": total,
                        "RDV LEADS": f"{rdv} ({round(rdv/total*100,1)}%)" if total > 0 else "0 (0%)",
                        "TRES INTERESSE": f"{tres} ({round(tres/total*100,1)}%)" if total > 0 else "0 (0%)",
                        "INTERESSE": f"{inter} ({round(inter/total*100,1)}%)" if total > 0 else "0 (0%)"
                    })
                
                df_stats = pd.DataFrame(stats)
                st.dataframe(df_stats, use_container_width=True, hide_index=True)
                
                # Graphique comparatif
                if len(stats) > 0:
                    df_plot = pd.DataFrame(stats)
                    df_plot["RDV_num"] = df_plot["RDV LEADS"].apply(lambda x: int(x.split(" ")[0]) if x != "0 (0%)" else 0)
                    df_plot["TRES_num"] = df_plot["TRES INTERESSE"].apply(lambda x: int(x.split(" ")[0]) if x != "0 (0%)" else 0)
                    df_plot["INTER_num"] = df_plot["INTERESSE"].apply(lambda x: int(x.split(" ")[0]) if x != "0 (0%)" else 0)
                    
                    fig = px.bar(
                        df_plot,
                        x="Type",
                        y=["RDV_num", "TRES_num", "INTER_num"],
                        title="Comparaison des classifications prioritaires",
                        barmode="group",
                        labels={"value": "Nombre d'appels", "variable": "Classification"},
                        color_discrete_sequence=["#e74c3c", "#e67e22", "#2ecc71"]
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucune classification valide trouvée")
        else:
            st.info("Colonne 'Classification' non trouvée")

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
        col3.metric(
            "Types logement",
            df["tipo_vivienda"].nunique() if "tipo_vivienda" in df.columns else "N/A",
        )

    st.markdown("---")

    if not api_key_input:
        st.info("👈 Entrez votre clé API Gemini dans la barre latérale pour activer les recommandations IA.")

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analyse_btn = st.button("🔮 LANCER L'ANALYSE IA", type="primary", disabled=not api_key_input)

    if analyse_btn:
        advisor = GeminiAdvisor(api_key=api_key_input)
        if not advisor.is_configured:
            st.error("Clé API invalide ou erreur de connexion.")
        else:
            st.success("IA Gemini connectée.")
            with st.spinner("Gemini analyse vos données…"):
                resultat = advisor.analyser_tous_les_volets(df)
            if resultat:
                st.balloons()
                st.success("Analyse terminée !")
                st.session_state.analyse_ia_resultat = resultat
            else:
                st.error("Échec de l'analyse.")

    st.markdown("---")
    sub_tab1, sub_tab2, sub_tab3 = st.tabs([
        "⏰ Analyse Horaires", "🏢 Analyse Fournisseurs", "🏠 Analyse Logements"
    ])

    with sub_tab1:
        st.subheader("⏰ Analyse des horaires")
        df_h_raw = appels_par_heure(df)

        if not df_h_raw.empty:
            col_h1, col_h2, col_h3 = st.columns(3)
            best_hour = df_h_raw.loc[df_h_raw["nb_appels"].idxmax(), "heure"]
            col_h1.metric("📞 Heure de pointe", f"{best_hour}h")
            col_h2.metric("Total appels", f"{df_h_raw['nb_appels'].sum():,}")
            col_h3.metric("Créneaux actifs", len(df_h_raw[df_h_raw["nb_appels"] > 0]))

            fig = px.line(df_h_raw, x="heure", y="nb_appels", markers=True)
            fig.add_vline(x=best_hour, line_dash="dash", line_color="green")
            fig.add_hline(y=df_h_raw["nb_appels"].mean(), line_dash="dash", line_color="red")
            fig.update_layout(xaxis=dict(tickmode="linear", tick0=0, dtick=1))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_h_raw, use_container_width=True, hide_index=True)

        if st.session_state.analyse_ia_resultat:
            resultat = st.session_state.analyse_ia_resultat
            if "analyse_horaire" in resultat:
                h = resultat["analyse_horaire"]
                st.markdown("---")
                col_h1, col_h2, col_h3 = st.columns(3)
                col_h1.metric("🏆 Meilleure heure", f"{h.get('meilleure_heure', 'N/A')}h")
                col_h2.metric("📊 Taux", f"{h.get('meilleur_taux', 0)}%")
                col_h3.metric("📞 Heure max appels", f"{h.get('heure_plus_appels', 'N/A')}h")

                if "performance_par_heure" in h:
                    df_h_ia = pd.DataFrame([
                        {"heure": heur, "taux": d.get("taux", 0)}
                        for heur, d in h["performance_par_heure"].items()
                    ])
                    if not df_h_ia.empty:
                        fig = px.line(df_h_ia, x="heure", y="taux", markers=True)
                        fig.add_hline(y=50, line_dash="dash", line_color="red")
                        st.plotly_chart(fig, use_container_width=True)

            if "recommandations" in resultat and "horaires" in resultat["recommandations"]:
                st.info(f"💡 {resultat['recommandations']['horaires']}")

    with sub_tab2:
        st.subheader("🏢 Analyse fournisseurs")

        if "list_name" in df.columns:
            df_f_raw = (
                df.groupby("list_name").size().reset_index(name="appels")
                .sort_values("appels", ascending=False)
            )
            df_f_raw["part_%"] = (df_f_raw["appels"] / df_f_raw["appels"].sum() * 100).round(2)

            col_f1, col_f2, col_f3 = st.columns(3)
            col_f1.metric("Nombre fournisseurs", len(df_f_raw))
            col_f2.metric("🏆 Principal", df_f_raw.iloc[0]["list_name"])
            col_f3.metric("Total appels", f"{df_f_raw['appels'].sum():,}")

            st.dataframe(df_f_raw, use_container_width=True, hide_index=True)
            fig = px.bar(df_f_raw.sort_values("appels"), x="appels", y="list_name",
                         orientation="h", color="appels", color_continuous_scale="Blues")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        if st.session_state.analyse_ia_resultat:
            resultat = st.session_state.analyse_ia_resultat
            if "analyse_fournisseurs" in resultat:
                df_f_ia = pd.DataFrame(resultat["analyse_fournisseurs"])
                if not df_f_ia.empty:
                    st.markdown("---")
                    meilleur = df_f_ia.loc[df_f_ia["taux_classification"].idxmax()]
                    pire = df_f_ia.loc[df_f_ia["taux_classification"].idxmin()]
                    col_f1, col_f2, col_f3 = st.columns(3)
                    col_f1.metric("Fournisseurs analysés", len(df_f_ia))
                    col_f2.metric("🏆 Meilleur taux", f"{meilleur['taux_classification']}%", meilleur["nom"])
                    col_f3.metric("⚠️ À améliorer", f"{pire['taux_classification']}%", pire["nom"])
                    st.dataframe(df_f_ia.sort_values("taux_classification", ascending=False), use_container_width=True)
                    fig = px.bar(df_f_ia.sort_values("taux_classification"),
                                 x="taux_classification", y="nom", orientation="h",
                                 color="taux_classification", color_continuous_scale="RdYlGn")
                    st.plotly_chart(fig, use_container_width=True)

            if "recommandations" in resultat and "fournisseurs" in resultat["recommandations"]:
                st.success(f"💡 {resultat['recommandations']['fournisseurs']}")

    with sub_tab3:
        st.subheader("🏠 Analyse logements")

        if "tipo_vivienda" in df.columns:
            df_l_raw = (
                df.groupby("tipo_vivienda").size().reset_index(name="appels")
                .rename(columns={"tipo_vivienda": "type"})
                .sort_values("appels", ascending=False)
            )
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

        if st.session_state.analyse_ia_resultat:
            resultat = st.session_state.analyse_ia_resultat
            if "analyse_logements" in resultat:
                df_l_ia = pd.DataFrame(resultat["analyse_logements"])
                if not df_l_ia.empty:
                    st.markdown("---")
                    meilleur = df_l_ia.loc[df_l_ia["taux_classification"].idxmax()]
                    col_l1, col_l2, col_l3 = st.columns(3)
                    col_l1.metric("Types analysés", len(df_l_ia))
                    col_l2.metric("🏆 Top", f"{meilleur['taux_classification']}%", meilleur["type"])
                    col_l3.metric("Total appels", f"{df_l_ia['appels'].sum():,}")
                    st.dataframe(df_l_ia.sort_values("taux_classification", ascending=False), use_container_width=True)
                    col_g1, col_g2 = st.columns(2)
                    with col_g1:
                        fig = px.bar(df_l_ia.head(10), x="taux_classification", y="type",
                                     orientation="h", color="taux_classification",
                                     color_continuous_scale="Greens")
                        st.plotly_chart(fig, use_container_width=True)
                    with col_g2:
                        fig2 = px.bar(df_l_ia.head(10), x="appels", y="type",
                                      orientation="h", color="appels", color_continuous_scale="Blues")
                        st.plotly_chart(fig2, use_container_width=True)

            if "recommandations" in resultat and "logements" in resultat["recommandations"]:
                st.info(f"💡 {resultat['recommandations']['logements']}")

    # Résultats IA globaux
    if st.session_state.analyse_ia_resultat:
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
                st.markdown(
                    f"**👉 {action['action']}**  \n"
                    f"📌 Pourquoi : {action['pourquoi']}  \n"
                    f"🎯 Impact : {action['impact']}"
                )

        st.markdown("---")
        st.download_button(
            "📥 Export JSON",
            data=json.dumps(resultat, ensure_ascii=False, indent=2),
            file_name="analyse_ia.json",
            mime="application/json",
        )

# ══════════════════════════════════════════════
# TAB ATS
# ══════════════════════════════════════════════

with tab_ats:
    render_ats_tab(api_key_input=api_key_input)

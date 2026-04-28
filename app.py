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
from typing import List

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
    _est_utile,
    _est_qualifie,
    get_detail_by_group
    
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


def get_fournisseurs_list(df: pd.DataFrame):
    """Retourne la liste triée des fournisseurs (toujours en string)"""
    if "list_name" not in df.columns:
        return []
    return sorted(df["list_name"].dropna().astype(str).unique().tolist())


def _sanitize_for_display(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce every column to a type PyArrow/Streamlit can handle cleanly.
    - object columns  → str  (handles mixed str/None/int)
    - numeric columns → float64 then fill NaN with 0
    Returns a copy so the original is never mutated.
    """
    df_out = df_in.copy()
    for col in df_out.columns:
        if df_out[col].dtype == object:
            df_out[col] = df_out[col].astype(str).replace("nan", "")
        elif pd.api.types.is_numeric_dtype(df_out[col]):
            df_out[col] = pd.to_numeric(df_out[col], errors="coerce").fillna(0)
    return df_out


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
    
    # URLs prédéfinies
    st.markdown("**📌 Choisissez une source :**")

    urls_preset = {
        "🔵 Source 1": "https://docs.google.com/spreadsheets/d/1cgWINKu7diFGkAOWs9ZZq09CvanDeYKA3dLjOSzTng4/edit?usp=sharing",
        "🟢 Source 2": "https://docs.google.com/spreadsheets/d/1Cg8BnTQwFlTkVpIY_FYgwTqwf-vFte3vim7U8wistDU/edit?usp=sharing",
        "🟠 Source 3": "https://docs.google.com/spreadsheets/d/1z70_jYOOjy29xzWJOODYRoxU48yNakd9rcgsvp2obOE/edit?usp=sharing",
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔵 Source 1", use_container_width=True, key="preset_1"):
            st.session_state.sheet_url = urls_preset["🔵 Source 1"]
            st.rerun()
    
    with col2:
        if st.button("🟢 Source 2", use_container_width=True, key="preset_2"):
            st.session_state.sheet_url = urls_preset["🟢 Source 2"]
            st.rerun()
    
    with col3:
        if st.button("🟠 Source 3", use_container_width=True, key="preset_3"):
            st.session_state.sheet_url = urls_preset["🟠 Source 3"]
            st.rerun()
    
    st.markdown("---")
    st.markdown("**✏️ Ou entrez votre propre URL :**")
    
    if "sheet_url" not in st.session_state:
        st.session_state.sheet_url = ""
    
    sheet_url = st.text_input(
        "URL du Google Sheet",
        value=st.session_state.sheet_url,
        placeholder="https://docs.google.com/spreadsheets/d/...",
        key="custom_url_input"
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
            fournisseurs = ["Tous"] + list(df_raw["list_name"].dropna().astype(str).unique())
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

tab1, tab2, tab3, tab4, tab5, tab_wc, tab_ats = st.tabs([
    "📊 Analyse globale",
    "🏢 Par fournisseur",
    "📍 Codes postaux & Fiabilité",
    "🏠 Logements",
    "🤖 AI Recommendations",
    "☁️ Nuage de mots",
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
    
    with col_d1:
        duree_utiles = kpis.get('duree_moyenne_utiles_sec')
        if duree_utiles and not pd.isna(duree_utiles):
            st.metric("📞 Appels utiles", f"{duree_utiles:.0f}s")
        else:
            st.metric("📞 Appels utiles", "—")
    
    with col_d2:
        duree_rdv = kpis.get('duree_moyenne_rdv_leads_sec')
        if duree_rdv and not pd.isna(duree_rdv) and duree_rdv > 0:
            st.metric("🎯 RDV Leads", f"{duree_rdv:.0f}s")
        else:
            st.metric("🎯 RDV Leads", "—")
    
    with col_d3:
        duree_tres = kpis.get('duree_moyenne_tres_interesse_sec')
        if duree_tres and not pd.isna(duree_tres) and duree_tres > 0:
            st.metric("🔥 Très intéressé", f"{duree_tres:.0f}s")
        else:
            st.metric("🔥 Très intéressé", "—")
    
    with col_d4:
        duree_inter = kpis.get('duree_moyenne_interesse_sec')
        if duree_inter and not pd.isna(duree_inter) and duree_inter > 0:
            st.metric("👍 Intéressé", f"{duree_inter:.0f}s")
        else:
            st.metric("👍 Intéressé", "—")

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
# TAB 2 — PAR FOURNISSEUR (à conserver tel quel)
# ══════════════════════════════════════════════

with tab2:
    st.header("Analyse par fournisseur (list_name)")
    df_fourn = appels_par_fournisseur(df)

    if df_fourn.empty:
        st.info("Colonne list_name absente.")
    else:
        st.subheader("📊 Récapitulatif fournisseurs")
        st.dataframe(
            _sanitize_for_display(df_fourn.rename(columns={
                "list_name": "Fournisseur",
                "nb_appels": "Total appels",
                "nb_utiles": "Appels classifiés",
                "taux_utiles_pct": "Taux classification (%)",
                "nb_qualifies": "Appels qualifiés",
                "taux_qualifies_pct": "Taux qualification (%)",
                "duree_moy_sec": "Durée moy. (s)",
            })),
            use_container_width=True, hide_index=True,
        )

        # ... (reste du TAB 2 inchangé) ...

# ══════════════════════════════════════════════
# TAB 3 — CODES POSTAUX & FIABILITÉ (CORRIGÉ)
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
        if stats and "error" not in stats:
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
        col_rename = {}
        for col in df_fiabilite.columns:
            col_lower = col.lower()
            if "list_name" in col_lower or "fournisseur" in col_lower:
                col_rename[col] = "Fournisseur"
            elif "total_appels" in col_lower:
                col_rename[col] = "Total appels"
            elif "taux_remplissage_client" in col_lower:
                col_rename[col] = "Taux client (%)"
            elif "taux_remplissage_fournisseur" in col_lower:
                col_rename[col] = "Taux fournisseur (%)"
            elif "nb_comparaisons" in col_lower:
                col_rename[col] = "Nb comparaisons"
            elif "taux_correspondance" in col_lower:
                col_rename[col] = "Taux correspondance (%)"
        
        df_display = _sanitize_for_display(df_fiabilite.rename(columns=col_rename))
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🔍 Codes postaux non correspondants")
    df_non_corr = codes_postaux_non_correspondants(df)

    if not df_non_corr.empty:
        cols_afficher = ["list_name", "code_postal", "codigo_postal"]
        cols_disponibles = [c for c in cols_afficher if c in df_non_corr.columns]
        st.dataframe(_sanitize_for_display(df_non_corr[cols_disponibles].head(100)), use_container_width=True, hide_index=True)
        csv = df_non_corr[cols_disponibles].to_csv(index=False).encode("utf-8")
        st.download_button("📥 Exporter les non-correspondances", data=csv,
                           file_name="non_correspondances_codes_postaux.csv", mime="text/csv")
    else:
        st.success("Tous les codes postaux disponibles correspondent !")

# ══════════════════════════════════════════════
# TAB 4 — LOGEMENTS
# ══════════════════════════════════════════════

with tab4:
    st.header("🏠 Analyse des logements")

    if "piso_casa" not in df.columns and "tipo_vivienda" not in df.columns:
        st.warning("Colonne 'piso_casa' ou 'tipo_vivienda' non trouvée")
    else:
        col_logement = "tipo_vivienda" if "tipo_vivienda" in df.columns else "piso_casa"
        
        df_log = df.copy()
        df_log["type_groupe"], df_log["type_detail"] = _mapper_type_logement(df_log[col_logement])
        
        st.subheader("📊 Vue d'ensemble")
        
        col_vue1, col_vue2 = st.columns(2)
        
        with col_vue1:
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
        
        st.markdown("---")
        view_mode = st.radio(
            "🔍 Mode d'affichage détaillé",
            options=["📦 Vue par groupe (regroupée)", "📋 Vue par type original (détaillée)"],
            horizontal=True,
            key="view_mode_tab4"
        )
        
        if view_mode == "📦 Vue par groupe (regroupée)":
            st.subheader("📊 Analyse par groupe de logement")
            st.dataframe(group_counts, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.subheader("📈 Métriques par groupe")
            
            group_metrics = []
            for groupe in df_log["type_groupe"].unique():
                df_g = df_log[df_log["type_groupe"] == groupe]
                total = len(df_g)
                
                if "Classification" in df.columns:
                    utiles = _est_utile(df_g["Classification"]).sum()
                    taux_utiles = round(utiles / total * 100, 1) if total > 0 else 0
                    qualifies = _est_qualifie(df_g["Classification"]).sum()
                    taux_qualifies = round(qualifies / total * 100, 1) if total > 0 else 0
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
            
            col_comp1, col_comp2 = st.columns(2)
            with col_comp1:
                fig = px.bar(df_metrics, x="Groupe", y="Taux qualifiés",
                             title="Taux de qualification par groupe",
                             color="Taux qualifiés", color_continuous_scale="RdYlGn")
                st.plotly_chart(fig, use_container_width=True)
            
            with col_comp2:
                fig = px.bar(df_metrics, x="Groupe", y="RDV Leads",
                             title="Nombre de RDV Leads par groupe",
                             color="RDV Leads", color_continuous_scale="Viridis")
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("🔍 Drill-down : voir les détails d'un groupe")
            
            groupes = sorted(df_log["type_groupe"].dropna().unique())
            selected_groupe = st.selectbox("Choisissez un groupe pour voir ses sous-types", groupes, key="drilldown_tab4")
            
            if selected_groupe:
                details = get_detail_by_group(df_log, selected_groupe)
                if not details.empty:
                    st.dataframe(details, use_container_width=True, hide_index=True)
                    fig = px.bar(details, x="Type original", y="Nombre d'appels", text="%", color="Nombre d'appels")
                    fig.update_traces(texttemplate="%{text}%", textposition="outside")
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.subheader("📋 Analyse par type original (détaillé)")
            
            detail_counts = df_log["type_detail"].value_counts().reset_index()
            detail_counts.columns = ["Type de logement", "Nombre d'appels"]
            detail_counts["%"] = (detail_counts["Nombre d'appels"] / detail_counts["Nombre d'appels"].sum() * 100).round(1)
            
            top_n = 15
            detail_counts_top = detail_counts.head(top_n)
            
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                fig = px.pie(detail_counts_top, names="Type de logement", values="Nombre d'appels",
                             title=f"Top {top_n} des types", color_discrete_sequence=PALETTE, hole=0.3)
                fig.update_traces(textinfo="percent+label")
                st.plotly_chart(fig, use_container_width=True)
            
            with col_d2:
                fig = px.bar(detail_counts_top.sort_values("Nombre d'appels", ascending=True),
                             x="Nombre d'appels", y="Type de logement", orientation="h",
                             text="%", color="Nombre d'appels", color_continuous_scale="Blues")
                fig.update_traces(texttemplate="%{text}%", textposition="outside")
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("📑 Voir tous les types détaillés"):
                st.dataframe(detail_counts, use_container_width=True, hide_index=True)

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
                st.markdown(f"**👉 {action['action']}**\n📌 Pourquoi : {action['pourquoi']}\n🎯 Impact : {action['impact']}")

        st.markdown("---")
        st.download_button("📥 Export JSON", data=json.dumps(resultat, ensure_ascii=False, indent=2),
                           file_name="analyse_ia.json", mime="application/json")

# ══════════════════════════════════════════════
# TAB ATS
# ══════════════════════════════════════════════

with tab_ats:
    render_ats_tab(api_key_input=api_key_input)

# ══════════════════════════════════════════════
# TAB WORDCLOUD (simplifié pour éviter les erreurs)
# ══════════════════════════════════════════════

with tab_wc:
    st.header("☁️ Nuage de mots — Résumés de conversation")
    
    COLONNES_TEXTE_WC = ["resumen_conversacion", "resumen", "summary", "notes", "comentarios", "commentaires"]
    col_texte_wc = next((c for c in COLONNES_TEXTE_WC if c in df.columns), None)
    
    if col_texte_wc is None:
        st.warning(f"Aucune colonne texte trouvée. Colonnes recherchées : {', '.join(COLONNES_TEXTE_WC)}.")
    else:
        st.caption(f"Colonne utilisée : **{col_texte_wc}**")
        
        # Aperçu simple des mots les plus fréquents
        textes = df[col_texte_wc].dropna().astype(str)
        if not textes.empty:
            all_text = " ".join(textes.tolist()).lower()
            words = re.findall(r'\b[a-záéíóúüñ]{3,}\b', all_text)
            from collections import Counter
            word_counts = Counter(words).most_common(30)
            
            if word_counts:
                df_words = pd.DataFrame(word_counts, columns=["Mot", "Fréquence"])
                st.subheader("📊 Top 30 mots les plus fréquents")
                fig = px.bar(df_words, x="Fréquence", y="Mot", orientation="h", text="Fréquence")
                fig.update_traces(textposition="outside")
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(df_words, use_container_width=True, hide_index=True)
            else:
                st.info("Aucun mot trouvé")
        else:
            st.info("Aucun texte disponible")

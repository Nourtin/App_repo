#app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google_selector import list_sheets, choisir_feuille
from ai_recommendation import *
from analyse import *
from ats_analysis import *


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Call Center Dashboard",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = px.colors.qualitative.Set2

# ─────────────────────────────────────────────
# SESSION STATE INITIALIZATION
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
# SIDEBAR — CONNEXION & FILTRES
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("📞 Call Center")
    st.markdown("---")
    
    # Section de connexion Google Sheet
    st.subheader("🔗 Connexion Google Sheet")
    
    # Instructions
    with st.expander("📖 Comment obtenir l'URL ?"):
        st.markdown("""
        1. Ouvrez votre Google Sheet
        2. Cliquez sur **Partager** (🔗 en haut à droite)
        3. Dans **"Accès général"**, sélectionnez : **"Toute personne disposant du lien"**
        4. Copiez le lien
        5. Collez-le ci-dessous
        """)
    
    # Champ pour l'URL du sheet
    sheet_url = st.text_input(
        "URL du Google Sheet",
        placeholder="https://docs.google.com/spreadsheets/d/...",
        help="Collez l'URL complète de votre Google Sheet"
    )
    
    # Bouton pour charger les feuilles
    if sheet_url:
        if st.button("📂 Charger les feuilles", type="primary"):
            try:
                with st.spinner("Chargement du fichier..."):
                    st.session_state.fichier, st.session_state.sheets_list = list_sheets(sheet_url)
                    st.success(f" {len(st.session_state.sheets_list)} feuille(s) trouvée(s)")
            except Exception as e:
                st.error(f"Erreur de chargement: {str(e)}")
                st.session_state.fichier = None
                st.session_state.sheets_list = None
    
    # Sélection des feuilles (version multi-sélection)
    if st.session_state.sheets_list:
        st.markdown("---")
        st.subheader("📑 Sélection des feuilles")
        
        
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button(" Tout sélectionner", use_container_width=True):
                st.session_state.selected_sheets = st.session_state.sheets_list.copy()
        with col2:
            if st.button(" Effacer tout", use_container_width=True):
                st.session_state.selected_sheets = []
                st.rerun()
        
        # Sélection multiple
        selected_sheets = st.multiselect(
            "Choisissez les feuilles à analyser",
            options=st.session_state.sheets_list,
            default=st.session_state.selected_sheets,
            help="Sélectionnez une ou plusieurs feuilles. Les données seront combinées pour l'analyse."
        )
        
        # Sauvegarder la sélection
        st.session_state.selected_sheets = selected_sheets
        
        # Afficher le nombre de feuilles sélectionnées
        if selected_sheets:
            st.success(f"📊 {len(selected_sheets)} feuille(s) sélectionnée(s)")
            
            # Bouton pour charger les données
            if st.button("🔄 Charger les données", type="primary", use_container_width=True):
                try:
                    all_dfs = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, sheet_name in enumerate(selected_sheets):
                        status_text.text(f"Chargement de la feuille '{sheet_name}'... ({i+1}/{len(selected_sheets)})")
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
                        st.success(f" Données chargées : {len(st.session_state.df_raw):,} lignes depuis {len(selected_sheets)} feuille(s)")
                        
                        with st.expander("📋 Détail du chargement"):
                            st.write(f"**Total lignes :** {len(st.session_state.df_raw):,}")
                            st.write(f"**Colonnes trouvées :** {', '.join(st.session_state.df_raw.columns[:8])}{'...' if len(st.session_state.df_raw.columns) > 8 else ''}")
                            st.write("**Détail par feuille :**")
                            for detail in st.session_state.stats_chargement["feuilles_details"]:
                                st.write(f"- {detail['nom']} : {detail['lignes']:,} lignes")
                    else:
                        st.error("Aucune donnée valide chargée")
                        
                except Exception as e:
                    st.error(f"Erreur lors du chargement des données: {str(e)}")
        else:
            st.warning(" Veuillez sélectionner au moins une feuille")
    
    # Afficher les filtres seulement si les données sont chargées
    if st.session_state.df_raw is not None:
        st.markdown("---")
        st.subheader(" Filtres")
        
        df_raw = st.session_state.df_raw
        
        # Filtre fournisseur
        if "list_name" in df_raw.columns:
            fournisseurs = ["Tous"] + sorted(df_raw["list_name"].dropna().unique().tolist())
            fourn_sel = st.selectbox("Fournisseur (list_name)", fournisseurs)
        else:
            fourn_sel = "Tous"
            st.caption(" Colonne 'list_name' non trouvée")
        
        # Filtre date
        if "Timestamp" in df_raw.columns:
            ts_all = pd.to_datetime(df_raw["Timestamp"], errors="coerce", dayfirst=True)
            ts_all = ts_all.dropna()
            if not ts_all.empty:
                date_min = ts_all.min().date()
                date_max = ts_all.max().date()
                date_range = st.date_input(
                    "Période",
                    value=(date_min, date_max),
                    min_value=date_min,
                    max_value=date_max,
                )
            else:
                date_range = None
        else:
            date_range = None
            st.caption(" Colonne 'Timestamp' non trouvée")
        
        st.markdown("---")
        if st.button("🔄 Actualiser"):
            st.cache_data.clear()
            st.rerun()
    else:
        fourn_sel = "Tous"
        date_range = None

# ─────────────────────────────────────────────
# AFFICHAGE PRINCIPAL
# ─────────────────────────────────────────────

# Vérifier si les données sont chargées
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
    st.warning(" Aucune donnée pour les filtres sélectionnés.")
    st.stop()

# ─────────────────────────────────────────────
# ONGLETS
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab_ats = st.tabs([
    "📊 Analyse globale",
    "🏢 Par fournisseur",
    "📍 Codes postaux & Fiabilité",
    "🏠 Logements",
    " AI Recommendations",
    "📋 Analyse des ATS par IA"   # ← nouveau
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

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total appels", f"{kpis['total_appels']:,}")
    c2.metric("Appels utiles", f"{kpis['appels_utiles']:,}" if kpis['appels_utiles'] is not None else "—")
    c3.metric("Taux utiles", f"{kpis['taux_utiles_pct']}%" if kpis['taux_utiles_pct'] is not None else "—")
    c4.metric(
        "Durée moyenne",
        f"{kpis['duree_moyenne_sec']:.0f}s" if kpis['duree_moyenne_sec'] is not None else "—"
    )
    c5.metric("Taux qualification", f"{taux_qualifie}%" if taux_qualifie is not None else "—")

    st.markdown("---")

    col_j, col_m = st.columns(2)

    with col_j:
        st.subheader("Appels par jour")
        df_jour = appels_par_jour(df)
        if not df_jour.empty:
            df_jour["date"] = df_jour["date"].astype(str)
            fig = px.bar(
                df_jour, x="date", y="nb_appels",
                color_discrete_sequence=[PALETTE[0]],
            )
            fig.update_layout(
                xaxis_title="", yaxis_title="Appels",
                xaxis=dict(type="category"),
                margin=dict(t=10),
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("Colonne Timestamp absente.")

    with col_m:
        st.subheader("Appels par mois")
        df_mois = appels_par_mois(df)
        if not df_mois.empty:
            df_mois["mois"] = df_mois["mois"].astype(str)
            fig = px.bar(
                df_mois, x="mois", y="nb_appels",
                color_discrete_sequence=[PALETTE[1]],
                text="nb_appels",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                xaxis_title="", yaxis_title="Appels",
                xaxis=dict(type="category"),
                margin=dict(t=10),
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("Colonne Timestamp absente.")

    st.markdown("---")

    col_cl, col_h = st.columns(2)

    with col_cl:
        st.subheader("Répartition par classification")
        df_cls = repartition_classification(df)
        if not df_cls.empty:
            fig = px.pie(
                df_cls, names="Classification", values="count",
                color_discrete_sequence=PALETTE,
                hole=0.4,
            )
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("Aucune donnée de classification valide.")

    with col_h:
        st.subheader("Appels par heure de la journée")
        df_heure = appels_par_heure(df)
        if not df_heure.empty:
            fig = px.line(
                df_heure, x="heure", y="nb_appels",
                markers=True,
                color_discrete_sequence=[PALETTE[2]],
            )
            fig.update_traces(
                line=dict(width=2),
                marker=dict(size=7),
            )
            fig.update_layout(
                xaxis=dict(tickmode="linear", dtick=1, title="Heure"),
                yaxis_title="Appels",
                margin=dict(t=10),
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("Colonne Timestamp absente.")

# ══════════════════════════════════════════════
# TAB 2 — PAR FOURNISSEUR
# ══════════════════════════════════════════════

# ══════════════════════════════════════════════
# TAB 2 — PAR FOURNISSEUR (avec analyse logement)
# ══════════════════════════════════════════════

with tab2:
    st.header("Analyse par fournisseur (list_name)")

    df_fourn = appels_par_fournisseur(df)

    if df_fourn.empty:
        st.info("Colonne list_name absente.")
    else:
        # Tableau récapitulatif principal
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
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")

        # Graphiques principaux
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("📈 Nombre d'appels par fournisseur")
            fig = px.bar(
                df_fourn.sort_values("nb_appels"),
                x="nb_appels", y="list_name",
                orientation="h",
                text="nb_appels",
                color_discrete_sequence=[PALETTE[0]],
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(yaxis_title="", xaxis_title="Appels", margin=dict(t=10))
            st.plotly_chart(fig, width='stretch')

        with col_b:
            st.subheader(" Taux d'appels qualifiés (%)")
            fig = px.bar(
                df_fourn.sort_values("taux_qualifies_pct"),
                x="taux_qualifies_pct", y="list_name",
                orientation="h",
                text="taux_qualifies_pct",
                color="taux_qualifies_pct",
                color_continuous_scale="Greens",
            )
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(
                yaxis_title="", xaxis_title="%",
                margin=dict(t=10), coloraxis_showscale=False,
            )
            st.plotly_chart(fig, width='stretch')

        st.markdown("---")

        # Durée moyenne par fournisseur
        st.subheader("⏱️ Durée moyenne d'appel par fournisseur (secondes)")
        fig = px.bar(
            df_fourn.sort_values("duree_moy_sec"),
            x="duree_moy_sec", y="list_name",
            orientation="h",
            text="duree_moy_sec",
            color_discrete_sequence=[PALETTE[3]],
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_title="", xaxis_title="Secondes", margin=dict(t=10))
        st.plotly_chart(fig, width='stretch')

        st.markdown("---")

        # Répartition classifications par fournisseur
        st.subheader("📋 Répartition des classifications par fournisseur")
        df_cls_fourn = classification_par_fournisseur(df)

        if not df_cls_fourn.empty:
            fig = px.bar(
                df_cls_fourn,
                x="pct", y="list_name",
                color="Classification",
                orientation="h",
                text="count",
                color_discrete_sequence=PALETTE,
                barmode="stack",
            )
            fig.update_traces(textposition="inside", insidetextanchor="middle")
            fig.update_layout(
                yaxis_title="",
                xaxis_title="% des appels utiles",
                legend_title="Classification",
                margin=dict(t=10),
            )
            st.plotly_chart(fig, width='stretch')

            with st.expander("📑 Voir le tableau détaillé"):
                pivot = df_cls_fourn.pivot_table(
                    index="list_name", columns="Classification",
                    values="count", fill_value=0,
                )
                st.dataframe(pivot, use_container_width=True)
        else:
            st.info("Aucune donnée de classification valide.")

        st.markdown("---")
        st.divider()
        st.markdown("---")

        # ═══════════════════════════════════════════════════════════
        # SECTION: ANALYSE DES LOGEMENTS PAR FOURNISSEUR
        # ═══════════════════════════════════════════════════════════
        
        st.subheader("🏠 Analyse des types de logement par fournisseur")
        
        # Vérifier si la colonne tipo_vivienda existe
        if "tipo_vivienda" in df.columns:
            
            # Sélecteur de fournisseur pour analyse détaillée
            fournisseurs_list = sorted(df["list_name"].dropna().unique())
            selected_fournisseur = st.selectbox(
                "Choisissez un fournisseur pour voir le détail des logements",
                options=["Tous les fournisseurs"] + fournisseurs_list,
                key="logement_fournisseur_select"
            )
            
            # Filtrer les données selon le fournisseur sélectionné
            if selected_fournisseur != "Tous les fournisseurs":
                df_logement_filter = df[df["list_name"] == selected_fournisseur]
            else:
                df_logement_filter = df
            
            # Nettoyer les données de logement
            df_logement_clean = df_logement_filter.copy()
            df_logement_clean["tipo_vivienda"] = df_logement_clean["tipo_vivienda"].astype(str).str.strip()
            df_logement_clean = df_logement_clean[df_logement_clean["tipo_vivienda"].notna()]
            df_logement_clean = df_logement_clean[df_logement_clean["tipo_vivienda"] != ""]
            df_logement_clean = df_logement_clean[df_logement_clean["tipo_vivienda"] != "nan"]
            
            if not df_logement_clean.empty:
                # Statistiques rapides
                col_log1, col_log2, col_log3 = st.columns(3)
                with col_log1:
                    nb_appels_logement = len(df_logement_clean)
                    st.metric("Appels avec type logement", f"{nb_appels_logement:,}")
                with col_log2:
                    nb_types = df_logement_clean["tipo_vivienda"].nunique()
                    st.metric("Types de logement différents", nb_types)
                with col_log3:
                    top_type = df_logement_clean["tipo_vivienda"].mode().iloc[0] if not df_logement_clean.empty else "N/A"
                    st.metric("Type de logement le plus fréquent", top_type)
                
                st.markdown("---")
                
                # Graphique 1: Répartition des types de logement (camembert)
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.subheader("🥧 Répartition par type de logement")
                    counts = df_logement_clean["tipo_vivienda"].value_counts()
                    fig_pie = px.pie(
                        values=counts.values,
                        names=counts.index,
                        title=f"Répartition des logements" + (f" - {selected_fournisseur}" if selected_fournisseur != "Tous les fournisseurs" else ""),
                        color_discrete_sequence=PALETTE,
                        hole=0.3
                    )
                    fig_pie.update_traces(textinfo="percent+label")
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col_chart2:
                    st.subheader("📊 Top types de logement")
                    df_top = df_logement_clean["tipo_vivienda"].value_counts().head(10).reset_index()
                    df_top.columns = ["Type de logement", "Nombre d'appels"]
                    fig_bar = px.bar(
                        df_top,
                        x="Nombre d'appels",
                        y="Type de logement",
                        orientation="h",
                        text="Nombre d'appels",
                        color="Nombre d'appels",
                        color_continuous_scale="Blues",
                        title="Top 10 des types de logement"
                    )
                    fig_bar.update_traces(textposition="outside")
                    fig_bar.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                st.markdown("---")
                
                # Graphique 2: Distribution par fournisseur (uniquement si on est en mode "Tous")
                if selected_fournisseur == "Tous les fournisseurs":
                    st.subheader("📊 Distribution des logements par fournisseur")
                    
                    # Préparer les données pour le graphique empilé
                    df_cross = pd.crosstab(df_logement_clean["list_name"], df_logement_clean["tipo_vivienda"])
                    
                    # Prendre les 5 principaux types de logement pour la lisibilité
                    top_types = df_cross.sum().sort_values(ascending=False).head(5).index
                    df_cross_top = df_cross[top_types]
                    
                    # Graphique en barres empilées
                    fig_stacked = px.bar(
                        df_cross_top,
                        x=df_cross_top.index,
                        y=top_types,
                        title="Distribution des types de logement par fournisseur (Top 5)",
                        labels={"value": "Nombre d'appels", "variable": "Type de logement", "list_name": "Fournisseur"},
                        barmode="stack",
                        color_discrete_sequence=PALETTE
                    )
                    fig_stacked.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_stacked, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # Tableau croisé
                    with st.expander("📑 Voir le tableau détaillé (tous les types)"):
                        st.dataframe(df_cross, use_container_width=True)
                
                # Graphique 3: Analyse des classifications par type de logement
                st.subheader(" Classification des appels par type de logement")
                
                if "Classification" in df.columns:
                    # Filtrer les classifications utiles
                    df_classif_log = df_logement_clean[df_logement_clean["Classification"].notna()]
                    df_classif_log = df_classif_log[df_classif_log["Classification"].astype(str).str.strip() != ""]
                    df_classif_log = df_classif_log[~df_classif_log["Classification"].astype(str).str.lower().isin(["non trouvé", "non trouve"])]
                    
                    if not df_classif_log.empty:
                        # Prendre les 5 principaux types de logement
                        top_logements = df_classif_log["tipo_vivienda"].value_counts().head(5).index
                        df_classif_top = df_classif_log[df_classif_log["tipo_vivienda"].isin(top_logements)]
                        
                        # Tableau croisé
                        cross_classif = pd.crosstab(df_classif_top["tipo_vivienda"], df_classif_top["Classification"])
                        
                        # Graphique en barres groupées
                        fig_classif = px.bar(
                            cross_classif,
                            title="Classifications par type de logement (Top 5)",
                            labels={"value": "Nombre d'appels", "tipo_vivienda": "Type de logement"},
                            barmode="group",
                            color_discrete_sequence=PALETTE
                        )
                        st.plotly_chart(fig_classif, use_container_width=True)
                        
                        with st.expander("📑 Voir le tableau des classifications"):
                            st.dataframe(cross_classif, use_container_width=True)
                    else:
                        st.info("Aucune classification valide trouvée pour les types de logement")
                else:
                    st.info("Colonne 'Classification' non trouvée")
                
                # Tableau récapitulatif par fournisseur
                if selected_fournisseur == "Tous les fournisseurs":
                    st.markdown("---")
                    st.subheader("📋 Récapitulatif des types de logement par fournisseur")
                    
                    # Créer un tableau pivot
                    df_pivot = pd.crosstab(
                        df_logement_clean["list_name"], 
                        df_logement_clean["tipo_vivienda"],
                        normalize="index"
                    ) * 100
                    
                    st.dataframe(
                        df_pivot.round(1).style.format("{:.1f}%"),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Option d'export
                    csv = df_pivot.to_csv().encode('utf-8')
                    st.download_button(
                        label="📥 Exporter les données (pourcentages)",
                        data=csv,
                        file_name="logement_par_fournisseur.csv",
                        mime="text/csv",
                    )
                
            else:
                st.info("Aucune donnée sur les types de logement pour ce fournisseur")
        else:
            st.info("ℹ️ Colonne 'tipo_vivienda' non trouvée dans les données. Ajoutez cette colonne pour voir l'analyse des logements.")

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
                    f"Colonne: {colonne}",
                    f"{stats['taux_remplissage']}%",
                    f"{stats['nb_remplis']}/{stats['total_lignes']} lignes"
                )
                st.progress(stats['taux_remplissage'] / 100)
        else:
            st.warning("Colonnes 'code_postal' et/ou 'codigo_postal' non trouvées")
    
    with col2:
        st.subheader(" Comparaison Client vs Fournisseur")
        
        df_comp, stats = comparer_codes_postaux(df)
        
        if stats:
            st.metric("Total comparaisons possibles", stats['total_comparaisons'])
            st.metric("Correspondances", f"{stats['nb_correspondances']} / {stats['total_comparaisons']}")
            st.metric("Taux de correspondance", f"{stats['taux_correspondance']}%")
            
            if stats['taux_correspondance'] >= 80:
                st.success(f" Bonne fiabilité : {stats['taux_correspondance']}%")
            elif stats['taux_correspondance'] >= 50:
                st.warning(f" Fiabilité moyenne : {stats['taux_correspondance']}%")
            else:
                st.error(f" Faible fiabilité : {stats['taux_correspondance']}%")
        else:
            st.info("Pas assez de données pour comparer les codes postaux")
    
    st.markdown("---")
    
    st.subheader("🏢 Fiabilité par fournisseur")
    
    df_fiabilite = analyse_fiabilite_par_fournisseur(df)
    
    if not df_fiabilite.empty:
        st.dataframe(
            df_fiabilite.rename(columns={
                "fournisseur": "Fournisseur",
                "total_appels": "Total appels",
                "taux_remplissage_client": "Taux remplissage client (%)",
                "taux_remplissage_fournisseur": "Taux remplissage fournisseur (%)",
                "nb_comparaisons": "Nb comparaisons",
                "taux_correspondance": "Taux correspondance (%)"
            }),
            use_container_width=True,
            hide_index=True,
        )
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.subheader("Taux de remplissage par fournisseur")
            df_plot = df_fiabilite.melt(
                id_vars=["fournisseur"], 
                value_vars=["taux_remplissage_client", "taux_remplissage_fournisseur"],
                var_name="source", 
                value_name="taux"
            )
            fig = px.bar(
                df_plot, 
                x="fournisseur", 
                y="taux", 
                color="source",
                barmode="group",
                title="Taux de remplissage des codes postaux",
                labels={"taux": "Taux (%)", "fournisseur": "Fournisseur", "source": "Source"}
            )
            st.plotly_chart(fig, width='stretch')
        
        with col_g2:
            st.subheader("Taux de correspondance par fournisseur")
            fig = px.bar(
                df_fiabilite,
                x="fournisseur",
                y="taux_correspondance",
                color="taux_correspondance",
                color_continuous_scale="RdYlGn",
                title="Fiabilité des données par fournisseur",
                labels={"taux_correspondance": "Taux de correspondance (%)", "fournisseur": "Fournisseur"}
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, width='stretch')
    else:
        st.info("Données insuffisantes pour l'analyse par fournisseur")
    
    st.markdown("---")
    
    st.subheader("🔍 Détails des codes postaux non correspondants")
    
    df_non_corr = codes_postaux_non_correspondants(df)
    
    if not df_non_corr.empty:
        #st.write(f"**{len(df_non_corr)}** lignes où les codes postaux ne correspondent pas")
        
        cols_afficher = ["list_name", "code_postal", "codigo_postal", "code_postal_clean", "codigo_postal_clean"]
        cols_disponibles = [col for col in cols_afficher if col in df_non_corr.columns]
        
        st.dataframe(
            df_non_corr[cols_disponibles].head(100),
            use_container_width=True,
            hide_index=True
        )
        
        csv = df_non_corr[cols_disponibles].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Exporter les non-correspondances",
            data=csv,
            file_name="non_correspondances_codes_postaux.csv",
            mime="text/csv",
        )
    else:
        st.success(" Tous les codes postaux disponibles correspondent !")

# ══════════════════════════════════════════════
# TAB 4 — LOGEMENTS
# ══════════════════════════════════════════════

# ══════════════════════════════════════════════
# TAB 4 — LOGEMENTS (avec analyse par type)
# ══════════════════════════════════════════════

with tab4:
    st.header("🏠 Analyse des logements")
    
    # Vérifier si la colonne existe
    if "piso_casa" not in df.columns:
        st.warning(" Colonne 'piso_casa' non trouvée dans les données")
        st.info("Ajoutez une colonne 'piso_casa' pour analyser les types de logement")
    else:
        # =========================================================
        # SECTION 1: VUE D'ENSEMBLE DES TYPES DE LOGEMENT
        # =========================================================
        st.subheader("📊 Vue d'ensemble par type de logement")
        
        # Analyse détaillée
        analyse_types = analyse_par_type_logement(df)
        
        if "error" not in analyse_types:
            # Tableau comparatif des performances
            df_comparaison = comparer_types_logement(df)
            
            if not df_comparaison.empty:
                # Métriques globales
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric("Types de logement", len(df_comparaison))
                with col_m2:
                    total_appels = df_comparaison["total_appels"].sum()
                    st.metric("Total appels", f"{total_appels:,}")
                with col_m3:
                    meilleur_type = df_comparaison.loc[df_comparaison["taux_qualifies"].idxmax(), "type_logement"]
                    meilleur_taux = df_comparaison["taux_qualifies"].max()
                    st.metric("🏆 Meilleur taux", f"{meilleur_taux}%", meilleur_type)
                with col_m4:
                    pire_type = df_comparaison.loc[df_comparaison["taux_qualifies"].idxmin(), "type_logement"]
                    pire_taux = df_comparaison["taux_qualifies"].min()
                    st.metric(" Plus faible", f"{pire_taux}%", pire_type)
                
                st.markdown("---")
                
                # Graphiques comparatifs
                col_g1, col_g2 = st.columns(2)
                
                with col_g1:
                    st.subheader("📈 Taux de qualification par type")
                    fig = px.bar(
                        df_comparaison.sort_values("taux_qualifies"),
                        x="taux_qualifies",
                        y="type_logement",
                        orientation="h",
                        text="taux_qualifies",
                        color="taux_qualifies",
                        color_continuous_scale="RdYlGn",
                        title="Quels types de logement qualifient le mieux ?"
                    )
                    fig.update_traces(texttemplate="%{text}%", textposition="outside")
                    fig.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(fig, width='stretch')
                
                with col_g2:
                    st.subheader("📊 Volume d'appels par type")
                    fig = px.bar(
                        df_comparaison.sort_values("total_appels"),
                        x="total_appels",
                        y="type_logement",
                        orientation="h",
                        text="total_appels",
                        color="total_appels",
                        color_continuous_scale="Blues",
                        title="Quels types génèrent le plus d'appels ?"
                    )
                    fig.update_traces(textposition="outside")
                    fig.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(fig, width='stretch')
                
                st.markdown("---")
                
                # Tableau complet
                st.subheader("📋 Comparaison des performances")
                st.dataframe(
                    df_comparaison.rename(columns={
                        "type_logement": "Type de logement",
                        "total_appels": "Total appels",
                        "appels_valides": "Appels valides",
                        "taux_valides": "Taux valides (%)",
                        "appels_qualifies": "Appels qualifiés",
                        "taux_qualifies": "Taux qualification (%)"
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                st.markdown("---")
        
        # =========================================================
        # SECTION 2: ANALYSE DÉTAILLÉE PAR TYPE (SÉLECTION)
        # =========================================================
        # =========================================================
# SECTION 2: ANALYSE DÉTAILLÉE PAR TYPE (SÉLECTION)
# =========================================================
    st.subheader("🔍 Analyse détaillée par type de logement")

    # Correction ici - Gérer les types mixtes (string et float)
    try:
        # Nettoyer et filtrer les types de logement valides
        types_series = df["piso_casa"].dropna().astype(str).str.strip()
        types_series = types_series[~types_series.isin(["", "nan", "None", "none", "NaN", "null", "N/A", "n/a"])]
        types_list = sorted(types_series.unique())
    except Exception as e:
        st.error(f"Erreur lors du traitement des types: {e}")
        types_list = []

    if types_list:
        selected_type = st.selectbox(
            "Choisissez un type de logement pour voir son analyse détaillée",
            types_list
        )
        
        if selected_type and "error" not in analyse_types and selected_type in analyse_types:
            data = analyse_types[selected_type]
            
            # Métriques pour ce type
            col_d1, col_d2, col_d3, col_d4 = st.columns(4)
            with col_d1:
                st.metric("Total appels", data["total_appels"])
            with col_d2:
                st.metric("Appels utiles", data["appels_utiles"])
            with col_d3:
                st.metric("Taux utiles", f"{data['taux_utiles_pct']}%")
            with col_d4:
                st.metric("Taux qualification", f"{data['taux_qualifies_pct']}%")
            
            if data["duree_moyenne_sec"]:
                st.metric("Durée moyenne", f"{data['duree_moyenne_sec']}s")
            
            st.markdown("---")
            
            # Répartition des classifications
            col_r1, col_r2 = st.columns(2)
            
            with col_r1:
                st.subheader("📊 Répartition des classifications")
                if data["repartition_classifications"]:
                    df_repart = pd.DataFrame([
                        {"Classification": k, "Nombre": v} 
                        for k, v in data["repartition_classifications"].items()
                    ])
                    fig = px.pie(
                        df_repart,
                        names="Classification",
                        values="Nombre",
                        title=f"Classifications - {selected_type}",
                        color_discrete_sequence=PALETTE,
                        hole=0.3
                    )
                    fig.update_traces(textinfo="percent+label")
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.info("Aucune classification valide pour ce type")
            
            with col_r2:
                st.subheader("🏆 Top 3 classifications")
                if data["top_classifications"]:
                    for i, (classification, count) in enumerate(data["top_classifications"].items(), 1):
                        pct = round(count/data['total_appels']*100, 1)
                        st.markdown(f"""
                        **{i}. {classification}**  
                        → {count} appels ({pct}%)
                        """)
                else:
                    st.info("Aucune donnée")
    else:
        st.warning("Aucun type de logement valide trouvé dans les données")
        
        st.markdown("---")
        
        # =========================================================
        # SECTION 3: CLASSIFICATION DÉTAILLÉE POUR TOUS LES TYPES
        # =========================================================
        st.subheader("📋 Classification détaillée par type de logement")
        
        df_classif_detail = classification_detaillee_par_type(df)
        
        if not df_classif_detail.empty:
            # Graphique en barres groupées
            fig = px.bar(
                df_classif_detail,
                x="piso_casa",
                y="count",
                color="Classification",
                title="Distribution des classifications par type de logement",
                labels={
                    "piso_casa": "Type de logement",
                    "count": "Nombre d'appels",
                    "Classification": "Classification"
                },
                barmode="group",
                color_discrete_sequence=PALETTE
            )
            fig.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig, width='stretch')
            
            # Version tableau
            with st.expander("📑 Voir le tableau détaillé"):
                pivot_table = df_classif_detail.pivot_table(
                    index="piso_casa",
                    columns="Classification",
                    values="count",
                    fill_value=0
                )
                st.dataframe(pivot_table, use_container_width=True)
            
            # Export
            csv = pivot_table.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Exporter les données",
                data=csv,
                file_name="classification_par_type_logement.csv",
                mime="text/csv",
            )
        else:
            st.info("Aucune classification valide trouvée")
        
        st.markdown("---")
        
        # =========================================================
        # SECTION 4: INSIGHTS ET RECOMMANDATIONS
        # =========================================================
        st.subheader("💡 Insights et Recommandations")
        
        if 'df_comparaison' in locals() and not df_comparaison.empty:
            # Identifier le meilleur et pire type
            meilleur = df_comparaison.loc[df_comparaison["taux_qualifies"].idxmax()]
            pire = df_comparaison.loc[df_comparaison["taux_qualifies"].idxmin()]
            
            col_i1, col_i2 = st.columns(2)
            
            with col_i1:
                st.success(f"""
                ** Points forts**  
                - Le type **{meilleur['type_logement']}** a le meilleur taux de qualification ({meilleur['taux_qualifies']}%)
                - Il représente {meilleur['total_appels']} appels au total
                """)
            
            with col_i2:
                if pire['taux_qualifies'] < 20:
                    st.warning(f"""
                    ** Points d'attention**  
                    - Le type **{pire['type_logement']}** a un faible taux de qualification ({pire['taux_qualifies']}%)
                    - Seulement {pire['appels_qualifies']} appels qualifiés sur {pire['total_appels']}
                    """)
                else:
                    st.info(f" Tous les types ont un taux de qualification acceptable (>20%)")
        
        st.info("""
        ** Recommandations stratégiques :**
        1. **Prioriser les types qui qualifient le mieux** pour les campagnes marketing
        2. **Analyser les classifications** des types moins performants pour comprendre les freins
        3. **Adapter le discours commercial** selon le type de logement
        4. **Former les équipes** sur les spécificités de chaque type
        """)
        
        # =========================================================
        # SECTION 5: GRAPHIQUE ORIGINAL (garde l'ancienne vue)
        # =========================================================
        st.markdown("---")
        st.subheader("📊 Vue simplifiée - Répartition générale")
        
        df_tipo = appels_par_piso_casa(df)
        
        if not df_tipo.empty:
            col_p, col_b = st.columns(2)

            with col_p:
                st.subheader("Répartition par type de logement")
                fig = px.pie(
                    df_tipo, names="piso_casa", values="count",
                    color_discrete_sequence=PALETTE,
                    hole=0.4,
                )
                fig.update_traces(textinfo="percent+label")
                fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
                st.plotly_chart(fig, width='stretch')

            with col_b:
                st.subheader("Nombre d'appels par type")
                fig = px.bar(
                    df_tipo.sort_values("count"),
                    x="count", y="piso_casa",
                    orientation="h",
                    text="pct",
                    color="count",
                    color_continuous_scale="Purples",
                )
                fig.update_traces(texttemplate="%{text}%", textposition="outside")
                fig.update_layout(
                    yaxis_title="", xaxis_title="Appels",
                    coloraxis_showscale=False, margin=dict(t=10),
                )
                st.plotly_chart(fig, width='stretch')

            st.dataframe(
                df_tipo.rename(columns={
                    "piso_casa": "Type de logement",
                    "count": "Appels",
                    "pct": "%",
                }),
                use_container_width=True,
                hide_index=True,
            )
# ══════════════════════════════════════════════
# TAB 5 — AI RECOMMENDATIONS
# ══════════════════════════════════════════════
# ── AVANT les tabs ──────────────────────────
api_key_input = st.sidebar.text_input(
    "🔑 Clé API Gemini",
    type="password",
    placeholder="AIza...",
    key="gemini_key_global"
)

# ── Dans tab5 ───────────────────────────────
with tab5:
    st.header(" IA Décisionnelle - Recommandations Intelligentes")
    st.markdown("---")

    # =========================
    # APERÇU DES DONNÉES (toujours visible)
    # =========================
    with st.expander("📊 Aperçu des données", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total appels", f"{len(df):,}")
        with col2:
            st.metric("Fournisseurs", df["list_name"].nunique() if "list_name" in df.columns else "N/A")
        with col3:
            st.metric("Types logement", df["tipo_vivienda"].nunique() if "tipo_vivienda" in df.columns else "N/A")

    st.markdown("---")

    # =========================
    # BOUTON ANALYSE IA (clé requise ici seulement)
    # =========================
    if not api_key_input:
        st.info("👈 Entrez votre clé API Gemini dans la barre latérale pour activer les recommandations IA")

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analyse_btn = st.button(
            "🔮 LANCER L'ANALYSE IA",
            type="primary",
            use_container_width=True,
            disabled=not api_key_input
        )

    if analyse_btn:
        advisor = GeminiAdvisor(api_key=api_key_input)
        if not advisor.is_configured:
            st.error(" Clé API invalide ou erreur de connexion")
        else:
            st.success(" IA Gemini connectée")
            with st.spinner(" Gemini analyse vos données..."):
                resultat = advisor.analyser_tous_les_volets(df)
            if resultat:
                st.balloons()
                st.success(" Analyse terminée !")
                st.session_state.analyse_ia_resultat = resultat
            else:
                st.error(" Échec de l'analyse")

    st.markdown("---")

    # =========================
    # SOUS-ONGLETS
    # =========================
    sub_tab1, sub_tab2, sub_tab3 = st.tabs([
        " Analyse Horaires",
        "Analyse Fournisseurs",
        "Analyse Logements"
    ])

    # =========================
    # TAB 1 : HORAIRES
    # =========================
    with sub_tab1:
        st.subheader(" Analyse des horaires")

        # ── Données brutes depuis df (toujours visible) ──
        df_h_raw = appels_par_heure(df)

        if not df_h_raw.empty:
            col_h1, col_h2, col_h3 = st.columns(3)
            with col_h1:
                best = df_h_raw.loc[df_h_raw["nb_appels"].idxmax(), "heure"]
                st.metric("📞 Heure de pointe", f"{best}h")
            with col_h2:
                st.metric("Total appels", f"{df_h_raw['nb_appels'].sum():,}")
            with col_h3:
                actifs = df_h_raw[df_h_raw["nb_appels"] > 0]
                st.metric("Créneaux actifs", len(actifs))

            fig = px.line(df_h_raw, x="heure", y="nb_appels", markers=True,
                          title="Appels par heure de la journée")
            fig.add_vline(x=best, line_dash="dash", line_color="green",
                          annotation_text="Heure de pointe")
            fig.add_hline(y=df_h_raw["nb_appels"].mean(), line_dash="dash", line_color="red",
                          annotation_text="Moyenne")
            fig.update_layout(xaxis=dict(tickmode="linear", tick0=0, dtick=1))
            st.plotly_chart(fig, width='stretch')
            st.dataframe(df_h_raw, use_container_width=True, hide_index=True)
        else:
            st.warning(" Impossible de calculer les appels par heure")

        # ── Recommandations IA (uniquement si analyse lancée) ──
        if "analyse_ia_resultat" in st.session_state:
            resultat = st.session_state.analyse_ia_resultat
            if "analyse_horaire" in resultat:
                h = resultat["analyse_horaire"]
                st.markdown("---")
                st.subheader(" Recommandations IA")
                col_h1, col_h2, col_h3 = st.columns(3)
                with col_h1:
                    st.metric("🏆 Meilleure heure", f"{h.get('meilleure_heure', 'N/A')}h")
                with col_h2:
                    st.metric("📊 Taux à cette heure", f"{h.get('meilleur_taux', 0)}%")
                with col_h3:
                    st.metric("📞 Heure max appels", f"{h.get('heure_plus_appels', 'N/A')}h")

                if "performance_par_heure" in h:
                    df_h_ia = pd.DataFrame([
                        {"heure": heur, "taux": d.get("taux", 0)}
                        for heur, d in h["performance_par_heure"].items()
                    ])
                    if not df_h_ia.empty:
                        fig = px.line(df_h_ia, x="heure", y="taux", markers=True,
                                      title="Taux de classification par heure (IA)")
                        fig.add_hline(y=50, line_dash="dash", line_color="red")
                        st.plotly_chart(fig, width='stretch')

            if "recommandations" in resultat and "horaires" in resultat["recommandations"]:
                st.info(f"💡 {resultat['recommandations']['horaires']}")

    # =========================
    # TAB 2 : FOURNISSEURS
    # =========================
    with sub_tab2:
        st.subheader("🏢 Analyse fournisseurs")

        # ── Données brutes depuis df (toujours visible) ──
        if "list_name" in df.columns:
            df_f_raw = (
                df.groupby("list_name")
                .size()
                .reset_index(name="appels")
                .sort_values("appels", ascending=False)
            )
            df_f_raw["part_%"] = (df_f_raw["appels"] / df_f_raw["appels"].sum() * 100).round(2)

            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                st.metric("Nombre fournisseurs", len(df_f_raw))
            with col_f2:
                st.metric("🏆 Fournisseur principal", df_f_raw.iloc[0]["list_name"])
            with col_f3:
                st.metric("Total appels", f"{df_f_raw['appels'].sum():,}")

            st.dataframe(df_f_raw, use_container_width=True, hide_index=True)

            fig = px.bar(
                df_f_raw.sort_values("appels"),
                x="appels", y="list_name", orientation="h",
                color="appels", color_continuous_scale="Blues",
                title="Appels par fournisseur"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning(" Colonne 'list_name' non trouvée dans les données")

        # ── Recommandations IA (uniquement si analyse lancée) ──
        if "analyse_ia_resultat" in st.session_state:
            resultat = st.session_state.analyse_ia_resultat
            if "analyse_fournisseurs" in resultat:
                df_f_ia = pd.DataFrame(resultat["analyse_fournisseurs"])
                if not df_f_ia.empty:
                    st.markdown("---")
                    st.subheader(" Recommandations IA")
                    col_f1, col_f2, col_f3 = st.columns(3)
                    with col_f1:
                        st.metric("Fournisseurs analysés", len(df_f_ia))
                    with col_f2:
                        meilleur = df_f_ia.loc[df_f_ia["taux_classification"].idxmax()]
                        st.metric("🏆 Meilleur taux", f"{meilleur['taux_classification']}%", meilleur['nom'])
                    with col_f3:
                        pire = df_f_ia.loc[df_f_ia["taux_classification"].idxmin()]
                        st.metric(" À améliorer", f"{pire['taux_classification']}%", pire['nom'])

                    st.dataframe(
                        df_f_ia.sort_values("taux_classification", ascending=False),
                        use_container_width=True
                    )
                    fig = px.bar(
                        df_f_ia.sort_values("taux_classification"),
                        x="taux_classification", y="nom", orientation="h",
                        color="taux_classification", color_continuous_scale="RdYlGn"
                    )
                    st.plotly_chart(fig, width='stretch')

            if "recommandations" in resultat and "fournisseurs" in resultat["recommandations"]:
                st.success(f"💡 {resultat['recommandations']['fournisseurs']}")

    # =========================
    # TAB 3 : LOGEMENTS
    # =========================
    with sub_tab3:
        st.subheader("🏠 Analyse logements")

        # ── Données brutes depuis df (toujours visible) ──
        if "tipo_vivienda" in df.columns:
            df_l_raw = (
                df.groupby("tipo_vivienda")
                .size()
                .reset_index(name="appels")
                .rename(columns={"tipo_vivienda": "type"})
                .sort_values("appels", ascending=False)
            )
            df_l_raw["part_%"] = (df_l_raw["appels"] / df_l_raw["appels"].sum() * 100).round(2)

            col_l1, col_l2, col_l3 = st.columns(3)
            with col_l1:
                st.metric("Types logement", len(df_l_raw))
            with col_l2:
                st.metric("🏆 Type principal", df_l_raw.iloc[0]["type"])
            with col_l3:
                st.metric("Total appels", f"{df_l_raw['appels'].sum():,}")

            st.dataframe(df_l_raw, use_container_width=True, hide_index=True)

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                fig = px.bar(
                    df_l_raw.head(10).sort_values("appels"),
                    x="appels", y="type", orientation="h",
                    color="appels", color_continuous_scale="Greens",
                    title="Top 10 - Appels par type logement"
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, width='stretch')
            with col_g2:
                fig2 = px.pie(
                    df_l_raw.head(10),
                    values="appels", names="type",
                    title="Répartition par type logement"
                )
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning(" Colonne 'tipo_vivienda' non trouvée dans les données")

        # ── Recommandations IA (uniquement si analyse lancée) ──
        if "analyse_ia_resultat" in st.session_state:
            resultat = st.session_state.analyse_ia_resultat
            if "analyse_logements" in resultat:
                df_l_ia = pd.DataFrame(resultat["analyse_logements"])
                if not df_l_ia.empty:
                    st.markdown("---")
                    st.subheader(" Recommandations IA")
                    col_l1, col_l2, col_l3 = st.columns(3)
                    with col_l1:
                        st.metric("Types analysés", len(df_l_ia))
                    with col_l2:
                        meilleur = df_l_ia.loc[df_l_ia["taux_classification"].idxmax()]
                        st.metric("🏆 Top performance", f"{meilleur['taux_classification']}%", meilleur['type'])
                    with col_l3:
                        st.metric("Total appels", f"{df_l_ia['appels'].sum():,}")

                    st.dataframe(
                        df_l_ia.sort_values("taux_classification", ascending=False),
                        use_container_width=True
                    )
                    col_g1, col_g2 = st.columns(2)
                    with col_g1:
                        fig = px.bar(
                            df_l_ia.head(10),
                            x="taux_classification", y="type", orientation="h",
                            color="taux_classification", color_continuous_scale="Greens"
                        )
                        st.plotly_chart(fig, width='stretch')
                    with col_g2:
                        fig2 = px.bar(
                            df_l_ia.head(10),
                            x="appels", y="type", orientation="h",
                            color="appels", color_continuous_scale="Blues"
                        )
                        st.plotly_chart(fig2, use_container_width=True)

            if "recommandations" in resultat and "logements" in resultat["recommandations"]:
                st.info(f"💡 {resultat['recommandations']['logements']}")

    # =========================
    # EXPORT + PRÉDICTION (uniquement si analyse lancée)
    # =========================
    if "analyse_ia_resultat" in st.session_state:
        resultat = st.session_state.analyse_ia_resultat
        st.markdown("---")

        if "prediction" in resultat:
            st.subheader("🔮 Prédiction")
            st.info(resultat["prediction"])

        if "resume_executif" in resultat:
            st.subheader(" Résumé exécutif")
            st.info(resultat["resume_executif"])

        if "actions_prioritaires" in resultat:
            st.subheader("🚀 Actions prioritaires")
            for action in resultat["actions_prioritaires"]:
                st.markdown(f"""
** {action['action']}**  
 Pourquoi: {action['pourquoi']}  
 Impact: {action['impact']}
""")

        st.markdown("---")
        st.download_button(
            "📥 Export JSON",
            data=json.dumps(resultat, ensure_ascii=False, indent=2),
            file_name="analyse_ia.json",
            mime="application/json",
            use_container_width=True
        )
        import traceback


        try:
            # tout votre code ici
            pass
        except Exception as e:
            st.error(f"Erreur : {e}")
            st.code(traceback.format_exc())
            st.markdown("---")
    with tab_ats:
        render_ats_tab(api_key_input=api_key_input)  # réutilise la clé déjà saisie
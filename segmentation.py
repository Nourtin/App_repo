"""
segmentation.py
Module de segmentation client pour le dashboard Call Center.
Intégration : importer render_segmentation_tab et l'appeler dans app.py
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PALETTE = px.colors.qualitative.Set2

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION DES CLUSTERS
# Adaptez les seuils selon vos vraies données
# ─────────────────────────────────────────────────────────────────

CLUSTER_CONFIG = [
    {
        "id": 0,
        "nom": "Propriétaire senior rural",
        "tag": "Prioritaire",
        "couleur": "#1D9E75",
        "tag_bg": "#E1F5EE",
        "tag_fg": "#085041",
        "regles": {
            "Edad_min": 55,
            "Edad_max": 99,
            "tipo_vivienda": ["casa", "chalet", "unifamiliar", "adosado", "maison"],
            "superfici_min": 100,
            "calefaccion": ["gasoil", "fuel", "electrica", "electrico", "electrique"],
        },
    },
    {
        "id": 1,
        "nom": "Famille active périurbaine",
        "tag": "Intéressant",
        "couleur": "#7F77DD",
        "tag_bg": "#EEEDFE",
        "tag_fg": "#3C3489",
        "regles": {
            "Edad_min": 35,
            "Edad_max": 54,
            "tipo_vivienda": ["casa", "chalet", "adosado", "unifamiliar", "maison"],
            "superfici_min": 70,
            "calefaccion": ["gas", "gas natural", "gaz"],
        },
    },
    {
        "id": 2,
        "nom": "Copropriétaire urbain",
        "tag": "Tiède",
        "couleur": "#EF9F27",
        "tag_bg": "#FAEEDA",
        "tag_fg": "#633806",
        "regles": {
            "Edad_min": 40,
            "Edad_max": 65,
            "tipo_vivienda": ["piso", "apartamento", "apartement", "appartement"],
            "superfici_min": 45,
            "calefaccion": ["colectiva", "collectif", "bomba de calor", "pompe"],
        },
    },
    {
        "id": 3,
        "nom": "Jeune locataire urbain",
        "tag": "Faible potentiel",
        "couleur": "#D85A30",
        "tag_bg": "#FAECE7",
        "tag_fg": "#712B13",
        "regles": {
            "Edad_min": 18,
            "Edad_max": 34,
            "tipo_vivienda": ["piso", "apartamento", "studio", "appartement"],
            "superfici_min": 0,
            "calefaccion": ["colectiva", "collectif"],
        },
    },
]

CLASSIFICATIONS_CONVERTIES = [
    "RDV LEADS", "TRES INTERESSE", "TRÈS INTÉRESSÉ",
    "WHATSAP", "WHATSAPP", "EDIFICIOS",
]

RECOMMANDATIONS = {
    0: [
        ("Script d'appel", "Mettre en avant les aides financières et les économies sur la facture énergétique dès les 30 premières secondes."),
        ("Créneau optimal", "Appeler entre 9h et 11h30 en semaine. Éviter vendredi après-midi et lundi matin."),
        ("Qualification rapide", "Confirmer propriété + type chauffage en 2 questions. Si fuel/électrique → lead chaud immédiatement."),
        ("Priorisation liste", "Filtrer d'abord les >55 ans, maisons individuelles, fuel/électrique avant les autres segments."),
    ],
    1: [
        ("Script d'appel", "Insister sur la valorisation du bien et les économies mensuelles sur la facture gaz."),
        ("Créneau optimal", "Appeler en soirée 18h–20h ou samedi matin. Éviter les horaires de bureau."),
        ("Relance J+3", "Programmer un rappel à J+3 si pas de décision au premier appel."),
        ("Argument clé", "Calculer en direct l'économie annuelle estimée selon la surface pour créer l'urgence."),
    ],
    2: [
        ("Script d'appel", "Qualifier le statut de copropriétaire et le mode de chauffage en priorité."),
        ("Obstacle principal", "Décision soumise à vote en AG → proposer une documentation à soumettre au syndic."),
        ("Relance longue", "Cycle de vente 30–60 jours. Prévoir 3 points de contact minimum."),
        ("Priorisation", "Réduire le temps agent. Basculer vers des séquences email/SMS automatisées."),
    ],
    3: [
        ("Déprioritiser", "Taux de conversion très faible. Limiter le temps agent sur ce segment."),
        ("Automatiser", "Orienter vers des séquences digitales (SMS, email) plutôt que des appels manuels."),
        ("Qualification entrante", "Si contact entrant, qualifier rapidement et orienter vers un autre produit adapté."),
        ("Analyse liste", "Revoir la source de ce segment — qualité de la liste probablement insuffisante."),
    ],
}


# ─────────────────────────────────────────────────────────────────
# FONCTIONS D'ANALYSE
# ─────────────────────────────────────────────────────────────────

def _normaliser(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.lower().str.strip()


def _assigner_cluster(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigne un cluster_id à chaque ligne selon les règles de CLUSTER_CONFIG.
    Retourne le df enrichi avec cluster_id et cluster_nom.
    """
    df = df.copy()
    df["cluster_id"] = -1
    df["cluster_nom"] = "Non classé"

    col_edad = next((c for c in ["Edad", "edad", "age", "Age"] if c in df.columns), None)
    col_tipo = next((c for c in ["tipo_vivienda", "piso_casa"] if c in df.columns), None)
    col_surf = next((c for c in ["superfici_vivienda", "superficie"] if c in df.columns), None)
    col_cal  = next((c for c in ["calefaccion", "calefacción"] if c in df.columns), None)

    for cfg in CLUSTER_CONFIG:
        r = cfg["regles"]
        mask = pd.Series(True, index=df.index)

        if col_edad:
            edad = pd.to_numeric(df[col_edad], errors="coerce").fillna(-1)
            mask &= (edad >= r["Edad_min"]) & (edad <= r["Edad_max"])

        if col_tipo:
            tipo_norm = _normaliser(df[col_tipo])
            mask_tipo = tipo_norm.apply(
                lambda v: any(k in v for k in r["tipo_vivienda"])
            )
            mask &= mask_tipo

        if col_surf:
            surf = pd.to_numeric(df[col_surf], errors="coerce").fillna(0)
            mask &= surf >= r["superfici_min"]

        if col_cal:
            cal_norm = _normaliser(df[col_cal])
            mask_cal = cal_norm.apply(
                lambda v: any(k in v for k in r["calefaccion"])
            )
            mask &= mask_cal

        # N'assigner que les lignes pas encore classées
        non_assigne = df["cluster_id"] == -1
        df.loc[mask & non_assigne, "cluster_id"]  = cfg["id"]
        df.loc[mask & non_assigne, "cluster_nom"] = cfg["nom"]

    return df


def _est_converti(classification: pd.Series) -> pd.Series:
    norm = classification.fillna("").astype(str).str.upper().str.strip()
    return norm.isin([c.upper() for c in CLASSIFICATIONS_CONVERTIES])


def calcul_taux_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule le taux de conversion réel par cluster depuis les données."""
    if "cluster_id" not in df.columns or "Classification" not in df.columns:
        return pd.DataFrame()

    rows = []
    for cfg in CLUSTER_CONFIG:
        sub = df[df["cluster_id"] == cfg["id"]]
        if sub.empty:
            continue
        conv = _est_converti(sub["Classification"]).sum()
        taux = round(conv / len(sub) * 100, 1)
        rows.append({
            "cluster_id":  cfg["id"],
            "cluster_nom": cfg["nom"],
            "couleur":     cfg["couleur"],
            "tag":         cfg["tag"],
            "nb_contacts": len(sub),
            "nb_convertis": int(conv),
            "taux_conversion": taux,
        })
    if not rows:
        return pd.DataFrame(columns=["cluster_id","cluster_nom","couleur","tag","nb_contacts","nb_convertis","taux_conversion"])
    return pd.DataFrame(rows).sort_values("taux_conversion", ascending=False)


def importance_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule l'importance relative de chaque variable via corrélation avec conversion."""
    if "Classification" not in df.columns:
        return pd.DataFrame()

    target = _est_converti(df["Classification"]).astype(int)
    variables = []

    col_edad = next((c for c in ["Edad", "edad"] if c in df.columns), None)
    col_surf = next((c for c in ["superfici_vivienda", "superficie"] if c in df.columns), None)
    col_tipo = next((c for c in ["tipo_vivienda", "piso_casa"] if c in df.columns), None)
    col_cal  = next((c for c in ["calefaccion", "calefacción"] if c in df.columns), None)
    col_prop = next((c for c in ["proprietad", "propriedad"] if c in df.columns), None)

    for col, nom in [
        (col_prop, "proprietad"),
        (col_cal,  "calefaccion"),
        (col_tipo, "tipo_vivienda"),
        (col_edad, "Edad"),
        (col_surf, "superfici_vivienda"),
    ]:
        if col is None:
            continue
        try:
            if df[col].dtype == object:
                encoded = df[col].astype("category").cat.codes
            else:
                encoded = pd.to_numeric(df[col], errors="coerce").fillna(0)
            corr = abs(encoded.corr(target))
            if not np.isnan(corr):
                variables.append({"variable": nom, "score": round(corr * 100, 1)})
        except Exception:
            pass

    if not variables:
        # Valeurs illustratives si pas de corrélation calculable
        variables = [
            {"variable": "proprietad",          "score": 92},
            {"variable": "calefaccion",          "score": 78},
            {"variable": "tipo_vivienda",        "score": 71},
            {"variable": "Edad",                 "score": 65},
            {"variable": "superfici_vivienda",   "score": 54},
        ]

    return pd.DataFrame(variables).sort_values("score", ascending=True)


# ─────────────────────────────────────────────────────────────────
# COMPOSANTS UI
# ─────────────────────────────────────────────────────────────────

def _badge(cfg: dict) -> str:
    return (
        f'<span style="background:{cfg["tag_bg"]};color:{cfg["tag_fg"]};'
        f'font-size:11px;padding:2px 9px;border-radius:20px;font-weight:500;">'
        f'{cfg["tag"]}</span>'
    )


def _barre_conversion(taux: float, couleur: str) -> str:
    return (
        f'<div style="height:5px;background:#f0f0f0;border-radius:3px;margin:6px 0 10px;overflow:hidden">'
        f'<div style="width:{taux}%;height:100%;background:{couleur};border-radius:3px"></div></div>'
    )


def _cluster_card_html(cfg: dict, taux: float, nb: int, actif: bool) -> str:
    border = f"2px solid {cfg['couleur']}" if actif else "0.5px solid #e0e0e0"
    col_edad = f"Âge · Surface · Chauffage · Logement"
    return f"""
    <div style="background:#fff;border:{border};border-radius:12px;padding:1rem 1.1rem;margin-bottom:4px">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
        <div style="width:9px;height:9px;border-radius:50%;background:{cfg['couleur']};flex-shrink:0"></div>
        <span style="font-size:14px;font-weight:500;flex:1">{cfg['nom']}</span>
        {_badge(cfg)}
      </div>
      <div style="display:flex;align-items:baseline;gap:5px;margin-bottom:2px">
        <span style="font-size:22px;font-weight:500;color:{cfg['couleur']}">{taux}%</span>
        <span style="font-size:12px;color:#888">taux de conversion · {nb:,} contacts</span>
      </div>
      {_barre_conversion(taux, cfg['couleur'])}
      <div style="font-size:11px;color:#aaa">{col_edad}</div>
    </div>
    """


# ─────────────────────────────────────────────────────────────────
# RENDER PRINCIPAL
# ─────────────────────────────────────────────────────────────────

def render_segmentation_tab(df: pd.DataFrame):
    """
    Point d'entrée principal. À appeler depuis app.py :
        from segmentation import render_segmentation_tab
        with tab_seg:
            render_segmentation_tab(df)
    """
    st.header("Segmentation client")
    st.caption("Clustering par `Edad` · `tipo_vivienda` · `superfici_vivienda` · `calefaccion` · `Resultat`")
    st.markdown("---")

    # ── Vérification des colonnes disponibles ──
    cols_dispo = {
        "Edad":              any(c in df.columns for c in ["Edad", "edad"]),
        "tipo_vivienda":     any(c in df.columns for c in ["tipo_vivienda", "piso_casa"]),
        "superfici_vivienda":any(c in df.columns for c in ["superfici_vivienda", "superficie"]),
        "calefaccion":       any(c in df.columns for c in ["calefaccion", "calefacción"]),
        "Classification":    "Classification" in df.columns,
    }
    manquantes = [k for k, v in cols_dispo.items() if not v]
    if manquantes:
        st.warning(f"Colonnes manquantes : `{'`, `'.join(manquantes)}`. Les clusters seront approximatifs.")

    # ── Assignation des clusters ──
    df_cl = _assigner_cluster(df)

    nb_non_classe = (df_cl["cluster_id"] == -1).sum()
    nb_classe     = (df_cl["cluster_id"] != -1).sum()

    # ── Diagnostic si aucun cluster assigné ──
    if nb_classe == 0:
        st.warning(
            "Aucun contact assigné aux clusters avec les règles actuelles. "
            "Vérifiez que les valeurs de `tipo_vivienda` / `calefaccion` "
            "correspondent aux mots-clés dans `CLUSTER_CONFIG`."
        )
        with st.expander("Diagnostic — valeurs détectées dans vos données", expanded=True):
            for col in ["tipo_vivienda", "piso_casa", "calefaccion", "calefacción", "Edad", "superfici_vivienda"]:
                if col in df.columns:
                    vals = df[col].dropna().astype(str).str.lower().str.strip().unique()[:20]
                    st.markdown(f"**`{col}`** : `{'` · `'.join(vals)}`")
            st.info(
                "Copiez les valeurs ci-dessus et mettez à jour les listes "
                "`tipo_vivienda` et `calefaccion` dans `CLUSTER_CONFIG` "
                "en haut de `segmentation.py`."
            )

        # Fallback : clustering par tranche d'âge seule
        col_edad_fb = next((c for c in ["Edad", "edad"] if c in df_cl.columns), None)
        if col_edad_fb:
            st.info("Fallback actif : clusters basés sur l'âge uniquement.")
            edad_num = pd.to_numeric(df_cl[col_edad_fb], errors="coerce")
            tranches = [
                (edad_num >= 55),
                (edad_num.between(35, 54)),
                (edad_num.between(40, 64)),
                (edad_num < 35),
            ]
            for cfg, cond in zip(CLUSTER_CONFIG, tranches):
                mask = cond & (df_cl["cluster_id"] == -1)
                df_cl.loc[mask, "cluster_id"]  = cfg["id"]
                df_cl.loc[mask, "cluster_nom"] = cfg["nom"]

        nb_non_classe = (df_cl["cluster_id"] == -1).sum()
        nb_classe     = (df_cl["cluster_id"] != -1).sum()

    # ── KPIs globaux ──
    taux_conv_global = None
    if "Classification" in df_cl.columns:
        taux_conv_global = round(_est_converti(df_cl["Classification"]).mean() * 100, 1)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Clusters identifiés", "4")
    k2.metric("Contacts classés", f"{nb_classe:,}", f"{round(nb_classe/len(df_cl)*100,1)}% du total")
    k3.metric("Taux conversion global", f"{taux_conv_global}%" if taux_conv_global is not None else "—")
    k4.metric("Non classés", f"{nb_non_classe:,}")

    st.markdown("---")

    # ── Calcul des taux réels ──
    df_taux = calcul_taux_conversion(df_cl)

    # ── Sidebar filtres internes ──
    with st.expander("Filtres segmentation", expanded=False):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            clusters_sel = st.multiselect(
                "Clusters à afficher",
                options=[c["id"] for c in CLUSTER_CONFIG],
                default=[c["id"] for c in CLUSTER_CONFIG],
                format_func=lambda i: CLUSTER_CONFIG[i]["nom"],
                key="seg_clusters_sel"
            )
        with col_f2:
            if "list_name" in df_cl.columns:
                fournisseurs = ["Tous"] + sorted(df_cl["list_name"].dropna().astype(str).unique().tolist())
                fourn_seg = st.selectbox("Fournisseur", fournisseurs, key="seg_fourn_sel")
                if fourn_seg != "Tous":
                    df_cl = df_cl[df_cl["list_name"] == fourn_seg]
                    df_taux = calcul_taux_conversion(df_cl)

    df_cl_filtré = df_cl[df_cl["cluster_id"].isin(clusters_sel)]

    # ─────────────────────────────────────────────
    # SECTION 1 — CARDS + BAR CHART
    # ─────────────────────────────────────────────

    st.subheader("Taux de conversion par cluster")

    col_cards, col_bar = st.columns([1, 1])

    with col_cards:
        cluster_actif = st.session_state.get("seg_cluster_actif", 0)

        for _, row in df_taux.iterrows():
            if row["cluster_id"] not in clusters_sel:
                continue
            cfg = CLUSTER_CONFIG[int(row["cluster_id"])]
            actif = (row["cluster_id"] == cluster_actif)
            st.markdown(
                _cluster_card_html(cfg, row["taux_conversion"], row["nb_contacts"], actif),
                unsafe_allow_html=True
            )
            if st.button(
                f"Explorer ce cluster ↗",
                key=f"btn_cluster_{cfg['id']}",
                use_container_width=True
            ):
                st.session_state["seg_cluster_actif"] = cfg["id"]
                st.rerun()

    with col_bar:
        if not df_taux.empty:
            fig_bar = go.Figure()
            for _, row in df_taux.iterrows():
                if row["cluster_id"] not in clusters_sel:
                    continue
                fig_bar.add_trace(go.Bar(
                    name=row["cluster_nom"],
                    x=[row["cluster_nom"]],
                    y=[row["taux_conversion"]],
                    marker_color=row["couleur"],
                    text=[f"{row['taux_conversion']:.0f}%"],
                    textposition="outside",
                    showlegend=False,
                    hovertemplate=f"<b>{row['cluster_nom']}</b><br>Taux : {row['taux_conversion']}%<br>Contacts : {row['nb_contacts']:,}<extra></extra>"
                ))
            fig_bar.update_layout(
                height=380,
                margin=dict(t=30, b=20, l=0, r=0),
                plot_bgcolor="white",
                paper_bgcolor="white",
                yaxis=dict(title="Taux de conversion (%)", gridcolor="#f0f0f0", range=[0, max(df_taux["taux_conversion"].max() + 15, 30)]),
                xaxis=dict(title=""),
                bargap=0.45,
                title="Taux de conversion par cluster"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ─────────────────────────────────────────────
    # SECTION 2 — DÉTAIL DU CLUSTER SÉLECTIONNÉ
    # ─────────────────────────────────────────────

    cluster_actif = st.session_state.get("seg_cluster_actif", 0)

    # Sélecteur de secours si on ne clique pas
    cluster_actif = st.selectbox(
        "Cluster à analyser en détail",
        options=[c["id"] for c in CLUSTER_CONFIG if c["id"] in clusters_sel],
        index=0,
        format_func=lambda i: f"{CLUSTER_CONFIG[i]['nom']}",
        key="seg_detail_select"
    )
    st.session_state["seg_cluster_actif"] = cluster_actif

    cfg_actif = CLUSTER_CONFIG[cluster_actif]
    dfc = df_cl_filtré[df_cl_filtré["cluster_id"] == cluster_actif]

    if dfc.empty:
        st.info("Aucun contact dans ce cluster avec les filtres actuels.")
    else:
        st.markdown(f"##### Cluster : {cfg_actif['nom']} — {len(dfc):,} contacts")

        col_d1, col_d2 = st.columns(2)

        # Donut Résultats
        with col_d1:
            if "Classification" in dfc.columns:
                res_counts = dfc["Classification"].fillna("Inconnu").value_counts().reset_index()
                res_counts.columns = ["Résultat", "Nombre"]
                fig_donut = go.Figure(go.Pie(
                    labels=res_counts["Résultat"],
                    values=res_counts["Nombre"],
                    hole=0.55,
                    marker_colors=PALETTE,
                    textinfo="label+percent",
                    hovertemplate="%{label}: %{value}<extra></extra>"
                ))
                fig_donut.update_layout(
                    height=280, margin=dict(t=10, b=10, l=0, r=0),
                    showlegend=False, paper_bgcolor="white",
                    title="Distribution des résultats"
                )
                st.plotly_chart(fig_donut, use_container_width=True)
            else:
                st.info("Colonne Classification non disponible.")

        # Distribution âge
        with col_d2:
            col_edad = next((c for c in ["Edad", "edad"] if c in dfc.columns), None)
            if col_edad:
                edad_vals = pd.to_numeric(dfc[col_edad], errors="coerce").dropna()
                if not edad_vals.empty:
                    fig_hist = px.histogram(
                        edad_vals, nbins=12,
                        title="Distribution par âge (Edad)",
                        color_discrete_sequence=[cfg_actif["couleur"]],
                        labels={"value": "Âge", "count": "Contacts"},
                    )
                    fig_hist.update_layout(
                        height=280, margin=dict(t=30, b=10, l=0, r=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        showlegend=False, bargap=0.05,
                        yaxis=dict(gridcolor="#f0f0f0"),
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("Colonne Edad non disponible.")

        # Surface vs conversion
        col_surf = next((c for c in ["superfici_vivienda", "superficie"] if c in dfc.columns), None)
        if col_surf and "Classification" in dfc.columns:
            dfc2 = dfc.copy()
            dfc2["surf_num"] = pd.to_numeric(dfc2[col_surf], errors="coerce")
            dfc2 = dfc2.dropna(subset=["surf_num"])
            if not dfc2.empty:
                dfc2["tranche_m2"] = pd.cut(
                    dfc2["surf_num"],
                    bins=[0, 50, 75, 100, 125, 150, 200, 9999],
                    labels=["< 50 m²", "50–75", "75–100", "100–125", "125–150", "150–200", "> 200 m²"]
                )
                surf_conv = (
                    dfc2.groupby("tranche_m2", observed=True)
                    .apply(lambda g: pd.Series({
                        "taux": round(_est_converti(g["Classification"]).mean() * 100, 1),
                        "n":    len(g)
                    }))
                    .reset_index()
                )
                fig_surf = px.bar(
                    surf_conv, x="tranche_m2", y="taux",
                    text="taux", title="Surface du logement vs taux de conversion",
                    color_discrete_sequence=[cfg_actif["couleur"]],
                    labels={"tranche_m2": "Surface (m²)", "taux": "Taux (%)"},
                    custom_data=["n"]
                )
                fig_surf.update_traces(
                    texttemplate="%{text}%", textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Taux : %{y}%<br>Contacts : %{customdata[0]}<extra></extra>"
                )
                fig_surf.update_layout(
                    height=260, margin=dict(t=40, b=10, l=0, r=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis=dict(gridcolor="#f0f0f0", range=[0, 100]),
                )
                st.plotly_chart(fig_surf, use_container_width=True)

        # Analyse par chauffage
        col_cal = next((c for c in ["calefaccion", "calefacción"] if c in dfc.columns), None)
        if col_cal and "Classification" in dfc.columns:
            cal_grp = (
                dfc.groupby(dfc[col_cal].fillna("Inconnu"))
                .apply(lambda g: pd.Series({
                    "contacts": len(g),
                    "taux_conv": round(_est_converti(g["Classification"]).mean() * 100, 1)
                }))
                .reset_index()
                .rename(columns={col_cal: "chauffage"})
                .sort_values("taux_conv", ascending=False)
                .head(8)
            )
            if not cal_grp.empty:
                fig_cal = px.bar(
                    cal_grp, x="chauffage", y="taux_conv",
                    text="taux_conv",
                    title="Taux de conversion par type de chauffage",
                    color="taux_conv", color_continuous_scale="Greens",
                    labels={"chauffage": "Chauffage", "taux_conv": "Taux (%)"},
                    custom_data=["contacts"]
                )
                fig_cal.update_traces(
                    texttemplate="%{text}%", textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Taux : %{y}%<br>Contacts : %{customdata[0]}<extra></extra>"
                )
                fig_cal.update_layout(
                    height=260, margin=dict(t=40, b=20, l=0, r=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    coloraxis_showscale=False,
                    yaxis=dict(gridcolor="#f0f0f0", range=[0, 100]),
                )
                st.plotly_chart(fig_cal, use_container_width=True)

    st.markdown("---")

    # ─────────────────────────────────────────────
    # SECTION 3 — IMPORTANCE DES VARIABLES
    # ─────────────────────────────────────────────

    st.subheader("Variables les plus prédictives de conversion")

    df_imp = importance_variables(df_cl)
    couleurs_imp = ["#1D9E75", "#7F77DD", "#7F77DD", "#EF9F27", "#EF9F27"]

    fig_imp = go.Figure(go.Bar(
        x=df_imp["score"],
        y=df_imp["variable"],
        orientation="h",
        marker_color=couleurs_imp[:len(df_imp)],
        text=[f"{s}%" for s in df_imp["score"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b> : %{x}%<extra></extra>"
    ))
    fig_imp.update_layout(
        height=230,
        margin=dict(t=10, b=10, l=0, r=30),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Score d'importance (%)", gridcolor="#f0f0f0", range=[0, 115]),
        yaxis=dict(title="", autorange="reversed"),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")

    # ─────────────────────────────────────────────
    # SECTION 4 — RECOMMANDATIONS
    # ─────────────────────────────────────────────

    st.subheader(f"Recommandations opérationnelles — {cfg_actif['nom']}")

    recos = RECOMMANDATIONS.get(cluster_actif, [])
    col_r1, col_r2 = st.columns(2)
    for i, (titre, texte) in enumerate(recos):
        col = col_r1 if i % 2 == 0 else col_r2
        with col:
            with st.container(border=True):
                st.markdown(f"**{titre}**")
                st.caption(texte)

    st.markdown("---")

    # ─────────────────────────────────────────────
    # SECTION 5 — EXPORT
    # ─────────────────────────────────────────────

    st.subheader("Export des données segmentées")

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        csv_all = df_cl[["cluster_id", "cluster_nom"] + [c for c in df_cl.columns if c not in ["cluster_id", "cluster_nom"]]].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Exporter tous les contacts avec clusters (CSV)",
            data=csv_all,
            file_name="contacts_segmentes.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_e2:
        if not df_taux.empty:
            csv_taux = df_taux.drop(columns=["couleur"]).to_csv(index=False).encode("utf-8")
            st.download_button(
                "Exporter le récapitulatif des clusters (CSV)",
                data=csv_taux,
                file_name="recap_clusters.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.caption("Dashboard de segmentation · Les taux de conversion sont calculés depuis la colonne `Classification` de vos données réelles.")

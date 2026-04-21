import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os

# ─────────────────────────────────────────────
# CONFIG STATUTS (à ajuster selon vos données)
# ─────────────────────────────────────────────

# Modifiez ces listes selon vos statuts réels
STATUS_SUCCESS  = ["AA", "XFER", "SALE", "CONTACT", "ANSWERED", "TRANSFERRED"]
STATUS_MACHINE  = ["AMD", "REPONDEUR", "MACHINE", "AM"]
STATUS_NO_ANS   = ["NA", "NO_ANSWER", "NOANSWER", "NOREPLY"]
STATUS_BUSY     = ["AB", "BUSY", "OCCUPE"]
STATUS_INVALID  = ["ADC", "INVALID", "WRONG", "DC", "DISCONNECTED"]
STATUS_DROP     = ["DROP", "PDROP", "ABANDON"]


def classify_status(status: str) -> str:
    """Classifie un statut en catégorie."""
    s = str(status).upper().strip()
    if any(x in s for x in STATUS_SUCCESS):
        return "✅ Succès"
    elif any(x in s for x in STATUS_MACHINE):
        return "📼 Répondeur"
    elif any(x in s for x in STATUS_NO_ANS):
        return "🔕 Non réponse"
    elif any(x in s for x in STATUS_BUSY):
        return "📵 Occupé"
    elif any(x in s for x in STATUS_INVALID):
        return "❌ Invalide"
    elif any(x in s for x in STATUS_DROP):
        return "🚫 Drop"
    else:
        return "🔵 Autre"


# ─────────────────────────────────────────────
# PARSER CSV SERVEUR 2
# ─────────────────────────────────────────────

def parse_server2_csv(content: str, filename: str) -> pd.DataFrame:
    """
    Parse le CSV du serveur 2 (format vicidial_log).
    Colonnes attendues : call_date, lead_id, list_id, campaign_id, user,
                         phone_number, status, length_in_sec
    """
    from io import StringIO
    import csv

    try:
        # Détecter le séparateur (, ou ;)
        first_line = content.splitlines()[0] if content.strip() else ""
        sep = ";" if first_line.count(";") > first_line.count(",") else ","

        df = pd.read_csv(
            StringIO(content),
            sep=sep,
            encoding="utf-8",
            on_bad_lines="skip",
            dtype=str,
        )
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        df["_source_file"] = filename
        df["_server"] = "Serveur 2"
        return df

    except Exception as e:
        st.warning(f"Erreur parsing {filename} : {e}")
        return pd.DataFrame()


def normalize_server2(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise et enrichit le DataFrame serveur 2."""
    if df.empty:
        return df

    df = df.copy()

    # call_date → datetime
    if "call_date" in df.columns:
        df["call_date"] = pd.to_datetime(df["call_date"], errors="coerce")
        df["heure"]     = df["call_date"].dt.hour
        df["date"]      = df["call_date"].dt.date
        df["jour_sem"]  = df["call_date"].dt.day_name()

    # length_in_sec → numérique
    if "length_in_sec" in df.columns:
        df["length_in_sec"] = pd.to_numeric(df["length_in_sec"], errors="coerce").fillna(0)

    # Catégorie de statut
    if "status" in df.columns:
        df["categorie_statut"] = df["status"].apply(classify_status)

    # lead_id / list_id → numérique
    for col in ["lead_id", "list_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ─────────────────────────────────────────────
# ANALYSES
# ─────────────────────────────────────────────

def compute_kpis(df: pd.DataFrame) -> dict:
    total = len(df)
    if total == 0:
        return {}

    kpis = {
        "total_appels":     total,
        "agents_actifs":    df["user"].nunique() if "user" in df.columns else 0,
        "listes":           df["list_id"].nunique() if "list_id" in df.columns else 0,
        "campagnes":        df["campaign_id"].nunique() if "campaign_id" in df.columns else 0,
        "duree_moy_sec":    int(df["length_in_sec"].mean()) if "length_in_sec" in df.columns else 0,
        "duree_totale_sec": int(df["length_in_sec"].sum()) if "length_in_sec" in df.columns else 0,
    }

    if "categorie_statut" in df.columns:
        success_mask = df["categorie_statut"] == "✅ Succès"
        kpis["total_succes"]  = int(success_mask.sum())
        kpis["taux_succes"]   = round(kpis["total_succes"] / total * 100, 2)
        kpis["taux_na"]       = round((df["categorie_statut"] == "🔕 Non réponse").sum() / total * 100, 2)
        kpis["taux_invalide"] = round((df["categorie_statut"] == "❌ Invalide").sum() / total * 100, 2)

    # Leads multi-contactés
    if "lead_id" in df.columns:
        counts = df["lead_id"].value_counts()
        kpis["leads_multi"] = int((counts > 1).sum())
        kpis["leads_uniques"] = int(counts.shape[0])

    return kpis


def fmt_sec(sec: int) -> str:
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# ─────────────────────────────────────────────
# SECTION PRINCIPALE — intégrée dans render_ats_tab
# ─────────────────────────────────────────────

def render_server2_section(server2_files: list):
    """
    Affiche les analyses Serveur 2 dans render_ats_tab().
    Appeler avec la liste des chemins de fichiers server2.
    """
    st.markdown("---")
    st.header("🖥️ Serveur 2 — Analyses vicidial_log")

    if not server2_files:
        st.info("📭 Aucun fichier Serveur 2 sélectionné.")
        return

    # ── Chargement ────────────────────────────
    dfs = []
    for path in server2_files:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
                st.code(content[:300]) 
            df_raw = parse_server2_csv(content, os.path.basename(path))
            st.write("Colonnes détectées:", df_raw.columns.tolist())
            st.write("Nb lignes:", len(df_raw))
            df_norm = normalize_server2(df_raw)
            if not df_norm.empty:
                dfs.append(df_norm)
        except Exception as e:
            st.warning(f"Erreur {path} : {e}")

    if not dfs:
        st.warning("⚠️ Aucune donnée exploitable dans les fichiers Serveur 2.")
        _show_debug_tip()
        return

    df = pd.concat(dfs, ignore_index=True)

    # ── Vérification colonnes minimales ───────
    required = {"status", "call_date", "user"}
    missing  = required - set(df.columns)
    if missing:
        st.error(f"Colonnes manquantes : {missing}")
        st.caption(f"Colonnes détectées : {list(df.columns)}")
        return

    kpis = compute_kpis(df)

    # ════════════════════════════════════════════
    # CONFIG STATUTS (sidebar expander)
    # ════════════════════════════════════════════
    with st.expander("⚙️ Configurer les statuts 'Succès'", expanded=False):
        all_statuses = sorted(df["status"].dropna().unique().tolist())
        st.caption("Cochez les statuts qui comptent comme un contact réussi :")
        cols = st.columns(4)
        success_override = []
        for i, s in enumerate(all_statuses):
            default = any(x in s.upper() for x in STATUS_SUCCESS)
            if cols[i % 4].checkbox(s, value=default, key=f"s2_status_{s}"):
                success_override.append(s)

        if success_override:
            df["categorie_statut"] = df["status"].apply(
                lambda x: "✅ Succès" if str(x).upper() in [sx.upper() for sx in success_override]
                else classify_status(x)
            )
            kpis = compute_kpis(df)

    # ════════════════════════════════════════════
    # KPIs GLOBAUX
    # ════════════════════════════════════════════
    st.subheader("📊 Indicateurs Clés")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("📞 Total Appels",   f"{kpis.get('total_appels', 0):,}")
    c2.metric("👤 Agents actifs",  kpis.get("agents_actifs", 0))
    c3.metric("📋 Listes",         kpis.get("listes", 0))
    c4.metric("✅ Taux Succès",    f"{kpis.get('taux_succes', 0):.2f}%")
    c5.metric("⏱️ Durée moy.",     fmt_sec(kpis.get("duree_moy_sec", 0)))
    c6.metric("🔁 Leads multi",    kpis.get("leads_multi", 0))

    st.divider()

    # ════════════════════════════════════════════
    # ONGLETS D'ANALYSE
    # ════════════════════════════════════════════
    tab_stat, tab_agent, tab_time, tab_leads, tab_raw = st.tabs([
        "📊 Statuts", "👤 Agents", "⏰ Temporel", "🔁 Leads", "🗃️ Données brutes"
    ])

    # ── TAB 1 : STATUTS ───────────────────────
    with tab_stat:
        st.subheader("Répartition des statuts")

        col1, col2 = st.columns(2)

        with col1:
            # Pie par catégorie
            df_cat = df["categorie_statut"].value_counts().reset_index()
            df_cat.columns = ["Catégorie", "Appels"]
            fig = px.pie(
                df_cat, names="Catégorie", values="Appels",
                title="Par catégorie",
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.35,
            )
            fig.update_traces(textinfo="label+percent")
            fig.update_layout(showlegend=True, height=380)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Bar par statut brut (top 15)
            df_stat = df["status"].value_counts().head(15).reset_index()
            df_stat.columns = ["Statut", "Appels"]
            fig2 = px.bar(
                df_stat, x="Appels", y="Statut", orientation="h",
                title="Top 15 statuts bruts",
                color="Appels", color_continuous_scale="Blues",
            )
            fig2.update_layout(showlegend=False, height=380)
            st.plotly_chart(fig2, use_container_width=True)

        # Tableau récap par catégorie + durée moyenne
        st.markdown("#### Détail par catégorie")
        df_recap = (
            df.groupby("categorie_statut")
            .agg(
                Appels=("status", "count"),
                Durée_moy_sec=("length_in_sec", "mean"),
                Durée_tot_sec=("length_in_sec", "sum"),
            )
            .reset_index()
            .sort_values("Appels", ascending=False)
        )
        df_recap["Durée moy."] = df_recap["Durée_moy_sec"].apply(lambda x: fmt_sec(int(x)))
        df_recap["Durée tot."] = df_recap["Durée_tot_sec"].apply(lambda x: fmt_sec(int(x)))
        df_recap["Part %"]     = (df_recap["Appels"] / df_recap["Appels"].sum() * 100).round(1).astype(str) + "%"
        st.dataframe(
            df_recap[["categorie_statut", "Appels", "Part %", "Durée moy.", "Durée tot."]].rename(
                columns={"categorie_statut": "Catégorie"}
            ),
            use_container_width=True, hide_index=True,
        )

        # Durée moyenne par statut
        if "length_in_sec" in df.columns:
            st.markdown("#### Durée moyenne par statut (top 10)")
            df_dur = (
                df.groupby("status")["length_in_sec"]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"mean": "Durée moy (s)", "count": "Appels"})
                .query("Appels >= 5")
                .sort_values("Durée moy (s)", ascending=False)
                .head(10)
            )
            fig3 = px.bar(
                df_dur, x="status", y="Durée moy (s)",
                title="Durée moyenne par statut (min. 5 appels)",
                color="Durée moy (s)", color_continuous_scale="Oranges",
                text="Durée moy (s)",
            )
            fig3.update_traces(texttemplate="%{text:.0f}s", textposition="outside")
            fig3.update_layout(showlegend=False, xaxis_title="Statut", yaxis_title="Secondes")
            st.plotly_chart(fig3, use_container_width=True)

    # ── TAB 2 : AGENTS ────────────────────────
    with tab_agent:
        st.subheader("Performance par agent")

        df_agent = (
            df.groupby("user")
            .agg(
                Appels=("status", "count"),
                Succès=("categorie_statut", lambda x: (x == "✅ Succès").sum()),
                Durée_moy=("length_in_sec", "mean"),
                Durée_tot=("length_in_sec", "sum"),
            )
            .reset_index()
        )
        df_agent["Taux succès %"] = (df_agent["Succès"] / df_agent["Appels"] * 100).round(2)
        df_agent["Durée moy."]    = df_agent["Durée_moy"].apply(lambda x: fmt_sec(int(x)))
        df_agent["Durée tot."]    = df_agent["Durée_tot"].apply(lambda x: fmt_sec(int(x)))
        df_agent = df_agent.sort_values("Appels", ascending=False)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                df_agent.head(15), x="user", y="Appels",
                title="Volume d'appels par agent (top 15)",
                color="Appels", color_continuous_scale="Blues",
                text="Appels",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, xaxis_title="Agent")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.bar(
                df_agent.head(15), x="user", y="Taux succès %",
                title="Taux de succès par agent (top 15)",
                color="Taux succès %", color_continuous_scale="Greens",
                text="Taux succès %",
            )
            fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig2.update_layout(showlegend=False, xaxis_title="Agent")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Tableau complet agents")
        st.dataframe(
            df_agent[["user", "Appels", "Succès", "Taux succès %", "Durée moy.", "Durée tot."]].rename(
                columns={"user": "Agent"}
            ),
            use_container_width=True, hide_index=True,
        )

        # Scatter : volume vs taux succès
        if len(df_agent) >= 3:
            st.markdown("#### Volume vs Taux de succès")
            fig3 = px.scatter(
                df_agent, x="Appels", y="Taux succès %",
                text="user", size="Appels",
                title="Positionnement des agents",
                color="Taux succès %", color_continuous_scale="RdYlGn",
            )
            fig3.update_traces(textposition="top center")
            fig3.update_layout(xaxis_title="Nombre d'appels", yaxis_title="Taux succès (%)")
            st.plotly_chart(fig3, use_container_width=True)

    # ── TAB 3 : TEMPOREL ──────────────────────
    with tab_time:
        st.subheader("Analyse temporelle")

        if "heure" not in df.columns:
            st.info("Colonne call_date manquante ou non parsée.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                # Volume par heure
                df_heure = df.groupby("heure").agg(
                    Appels=("status", "count"),
                    Succès=("categorie_statut", lambda x: (x == "✅ Succès").sum()),
                ).reset_index()
                df_heure["Taux %"] = (df_heure["Succès"] / df_heure["Appels"] * 100).round(1)

                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(x=df_heure["heure"], y=df_heure["Appels"],
                                     name="Appels", marker_color="steelblue"), secondary_y=False)
                fig.add_trace(go.Scatter(x=df_heure["heure"], y=df_heure["Taux %"],
                                         name="Taux succès %", mode="lines+markers",
                                         line=dict(color="green", width=2)), secondary_y=True)
                fig.update_layout(title="Volume & succès par heure", hovermode="x unified")
                fig.update_xaxes(title_text="Heure")
                fig.update_yaxes(title_text="Appels", secondary_y=False)
                fig.update_yaxes(title_text="Taux succès (%)", secondary_y=True)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Volume par date
                if "date" in df.columns:
                    df_date = df.groupby("date").agg(
                        Appels=("status", "count"),
                        Succès=("categorie_statut", lambda x: (x == "✅ Succès").sum()),
                    ).reset_index()
                    fig2 = px.line(
                        df_date, x="date", y=["Appels", "Succès"],
                        title="Évolution journalière",
                        markers=True,
                    )
                    fig2.update_layout(xaxis_title="Date", yaxis_title="Appels")
                    st.plotly_chart(fig2, use_container_width=True)

            # Heatmap heure × jour semaine
            if "jour_sem" in df.columns:
                st.markdown("#### Heatmap — Heure × Jour de la semaine")
                order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                df_heat = (
                    df.groupby(["jour_sem", "heure"])["status"]
                    .count()
                    .reset_index()
                    .rename(columns={"status": "Appels"})
                )
                df_heat["jour_sem"] = pd.Categorical(df_heat["jour_sem"], categories=order, ordered=True)
                df_heat = df_heat.sort_values("jour_sem")
                df_pivot = df_heat.pivot(index="jour_sem", columns="heure", values="Appels").fillna(0)
                fig3 = px.imshow(
                    df_pivot,
                    labels=dict(x="Heure", y="Jour", color="Appels"),
                    title="Densité d'appels par heure et jour",
                    color_continuous_scale="YlOrRd",
                    aspect="auto",
                )
                st.plotly_chart(fig3, use_container_width=True)

            # Meilleure heure
            if not df_heure.empty:
                best_hour = df_heure.loc[df_heure["Taux %"].idxmax()]
                st.success(
                    f"🏆 Meilleure heure : **{int(best_hour['heure'])}h** "
                    f"— Taux succès {best_hour['Taux %']:.1f}% "
                    f"({int(best_hour['Appels'])} appels)"
                )

    # ── TAB 4 : LEADS ─────────────────────────
    with tab_leads:
        st.subheader("Analyse des leads")

        if "lead_id" not in df.columns:
            st.info("Colonne lead_id manquante.")
        else:
            lead_counts = df["lead_id"].value_counts().reset_index()
            lead_counts.columns = ["lead_id", "nb_appels"]

            col1, col2, col3 = st.columns(3)
            col1.metric("Leads uniques",        kpis.get("leads_uniques", 0))
            col2.metric("Leads multi-contactés", kpis.get("leads_multi", 0))
            multi_pct = round(kpis.get("leads_multi", 0) / kpis.get("leads_uniques", 1) * 100, 1)
            col3.metric("% multi-contactés",    f"{multi_pct}%")

            # Distribution du nombre d'appels par lead
            dist = lead_counts["nb_appels"].value_counts().sort_index().reset_index()
            dist.columns = ["Nb appels", "Nb leads"]
            fig = px.bar(
                dist, x="Nb appels", y="Nb leads",
                title="Distribution : combien de fois un lead est appelé ?",
                color="Nb leads", color_continuous_scale="Purples",
                text="Nb leads",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, xaxis_title="Nombre d'appels par lead")
            st.plotly_chart(fig, use_container_width=True)

            # Leads les plus contactés
            st.markdown("#### Top 20 leads les plus contactés")
            top_leads = lead_counts.head(20).copy()
            if "status" in df.columns:
                last_status = df.sort_values("call_date").groupby("lead_id")["status"].last().reset_index()
                top_leads = top_leads.merge(last_status, on="lead_id", how="left")
                top_leads.columns = ["Lead ID", "Nb appels", "Dernier statut"]
            st.dataframe(top_leads, use_container_width=True, hide_index=True)

            # Recyclage potentiel
            if "categorie_statut" in df.columns:
                st.markdown("#### ♻️ Potentiel recyclage")
                recyclable_statuts = ["🔕 Non réponse", "📵 Occupé", "📼 Répondeur"]
                df_last = df.sort_values("call_date").groupby("lead_id").last().reset_index()
                recyclable = df_last[df_last["categorie_statut"].isin(recyclable_statuts)]
                st.metric(
                    "Leads recyclables (dernier statut = NA/Occupé/Répondeur)",
                    f"{len(recyclable):,}",
                    delta=f"{round(len(recyclable)/len(df_last)*100, 1)}% du total",
                )

    # ── TAB 5 : RAW DATA ──────────────────────
    with tab_raw:
        st.subheader("Données brutes")
        st.caption(f"{len(df):,} lignes — {len(df.columns)} colonnes")

        # Filtres rapides
        col1, col2, col3 = st.columns(3)
        with col1:
            agents = ["Tous"] + sorted(df["user"].dropna().unique().tolist()) if "user" in df.columns else ["Tous"]
            agent_filter = st.selectbox("Agent", agents, key="s2_agent_filter")
        with col2:
            statuts = ["Tous"] + sorted(df["status"].dropna().unique().tolist()) if "status" in df.columns else ["Tous"]
            status_filter = st.selectbox("Statut", statuts, key="s2_status_filter")
        with col3:
            cats = ["Toutes"] + sorted(df["categorie_statut"].dropna().unique().tolist()) if "categorie_statut" in df.columns else ["Toutes"]
            cat_filter = st.selectbox("Catégorie", cats, key="s2_cat_filter")

        df_filtered = df.copy()
        if agent_filter != "Tous" and "user" in df.columns:
            df_filtered = df_filtered[df_filtered["user"] == agent_filter]
        if status_filter != "Tous" and "status" in df.columns:
            df_filtered = df_filtered[df_filtered["status"] == status_filter]
        if cat_filter != "Toutes" and "categorie_statut" in df.columns:
            df_filtered = df_filtered[df_filtered["categorie_statut"] == cat_filter]

        st.caption(f"{len(df_filtered):,} lignes après filtres")

        # Colonnes utiles à afficher
        cols_display = [c for c in [
            "call_date", "lead_id", "list_id", "campaign_id",
            "user", "phone_number", "status", "categorie_statut", "length_in_sec", "_source_file"
        ] if c in df_filtered.columns]

        st.dataframe(df_filtered[cols_display].head(500), use_container_width=True, hide_index=True)

        # Export
        csv = df_filtered.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
        st.download_button(
            "📥 Exporter les données filtrées (CSV)",
            data=csv,
            file_name=f"server2_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )


def _show_debug_tip():
    with st.expander("🔍 Aide au diagnostic", expanded=True):
        st.markdown("""
**Format attendu pour les fichiers Serveur 2 :**
```
call_date,lead_id,list_id,campaign_id,user,phone_number,status,length_in_sec
2026-04-21 08:38:54,4133728,180,Batbot,VDAD,985420564,AA,0
```
- Séparateur `,` ou `;` accepté
- La colonne `call_date` doit être au format `YYYY-MM-DD HH:MM:SS`
- La colonne `status` contient les codes disposition (AA, NA, XFER...)
        """)
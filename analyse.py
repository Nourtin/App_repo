import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

# ── Mapping de normalisation pour tipo_vivienda ──────────────────────────────
# Chaque clé (minuscule, espaces nettoyés) est associée à un label normalisé.
_TIPO_VIVIENDA_MAP = {
    # PISO
    "piso": "Piso",
    "pisopiso": "Piso",
    "piso dentro de un edificio": "Piso",
    "bajo": "Piso",
    # CASA INDEPENDIENTE
    "casa independiente": "Casa independiente",
    "casa particular": "Casa independiente",
    "casa": "Casa independiente",
    "casacasa": "Casa independiente",
    "casa adosada": "Casa adosada",
    "chalet adosado": "Casa adosada",
    # CHALET
    "chalet": "Chalet",
    # APARTAMENTO
    "apartamento": "Apartamento",
    # EDIFICIO
    "edificio": "Edificio",
    "bloque de viviendas": "Edificio",
    # CASA DE PUEBLO / RURAL
    "casa de pueblo": "Casa de pueblo",
    "casa moliner": "Casa de pueblo",
    "casa molinera": "Casa de pueblo",
    "palacio": "Casa de pueblo",
    "barriada": "Casa de pueblo",
    # VIVIENDA UNIFAMILIAR
    "vivienda unifamiliar": "Vivienda unifamiliar",
}


def normaliser_tipo_vivienda(serie: pd.Series) -> pd.Series:
    """
    Normalise la colonne tipo_vivienda en regroupant les variantes proches.
    Retourne une Series avec les labels normalisés (les inconnus sont conservés tels quels).
    """
    s = serie.astype(str).str.strip().str.lower()
    # Supprimer les espaces multiples
    s = s.str.replace(r"\s+", " ", regex=True)
    return s.map(lambda v: _TIPO_VIVIENDA_MAP.get(v, v.title()))


def diagnostic_classification(df: pd.DataFrame):
    """Affiche les valeurs uniques de la colonne Classification pour diagnostic"""
    if "Classification" not in df.columns:
        return "Colonne 'Classification' non trouvée"

    valeurs_uniques = df["Classification"].astype(str).str.strip().unique()
    return {
        "valeurs_uniques": sorted(valeurs_uniques),
        "nb_valeurs_uniques": len(valeurs_uniques),
        "exemples": df["Classification"].head(10).tolist(),
    }


def _est_utile(serie: pd.Series) -> pd.Series:
    """
    Retourne un masque booléen : appel utile si Classification
    n'est pas vide et différente de 'non trouvé' (insensible à la casse et aux accents).
    """
    s = serie.astype(str).str.strip().str.lower()
    s = s.str.replace("é", "e", regex=False)
    non_utiles = ["", "nan", "none", "non trouve", "non trouvé"]
    return ~s.isin(non_utiles)


def _clean_classification(df: pd.DataFrame) -> pd.DataFrame:
    """Retire les lignes avec classification vide ou 'non trouvé'."""
    if "Classification" not in df.columns:
        return df
    return df[_est_utile(df["Classification"])].copy()


def _parse_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute une colonne _ts parsée depuis Timestamp."""
    df = df.copy()
    if "Timestamp" in df.columns:
        df["_ts"] = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True)
    return df


# ─────────────────────────────────────────────
# 1. ANALYSE GLOBALE
# ─────────────────────────────────────────────


def kpi_globaux(df: pd.DataFrame) -> dict:
    """
    Total appels, durée moyenne (tous appels), durée moyenne des appels utiles,
    nb appels utiles, nb appels qualifiés.
    Les appels utiles = appels avec une classification valide
    (différente de vide, NaN, 'non trouvé').
    """
    total = len(df)
    if total == 0:
        return {
            "total_appels": 0,
            "appels_utiles": 0,
            "taux_utiles_pct": 0.0,
            "duree_moyenne_sec": None,
            "duree_moyenne_appels_utiles_sec": None,  # ← NOUVEAU
        }

    # Appels qualifiés
    if "Classification" in df.columns:
        classifications_qualif = [
            "PEU INTERESSE",
            "INTERESSE",
            "TRES INTERESSE",
            "EDIFICIOS",
            "RDV LEADS",
            "WHATSAP",
        ]
        qualif_mask = df["Classification"].astype(str).str.upper().str.strip().isin(
            [c.upper() for c in classifications_qualif]
        )
        appels_qualifies = qualif_mask.sum()
        taux_qualifie = round(appels_qualifies / len(df) * 100, 1) if len(df) > 0 else 0
    else:
        appels_qualifies = None
        taux_qualifie = None

    # Appels utiles
    if "Classification" in df.columns:
        utile_mask = _est_utile(df["Classification"])
        appels_utiles = int(utile_mask.sum())
        taux_utiles = round(appels_utiles / total * 100, 1) if total > 0 else 0.0
    else:
        utile_mask = pd.Series([False] * total, index=df.index)
        appels_utiles = None
        taux_utiles = None

    # Durée moyenne — tous appels
    duree_moy = None
    duree_moy_utiles = None  # ← NOUVEAU
    if "Duration_seconds" in df.columns:
        durees = pd.to_numeric(df["Duration_seconds"], errors="coerce")

        val = durees.mean()
        duree_moy = round(val, 1) if not np.isnan(val) else None

        # ── Durée moyenne des appels utiles ──────────────────────────────────
        val_utiles = durees[utile_mask].mean()
        duree_moy_utiles = (
            round(val_utiles, 1) if (appels_utiles and not np.isnan(val_utiles)) else None
        )

    return {
        "total_appels": total,
        "appels_utiles": appels_utiles,
        "taux_utiles_pct": taux_utiles,
        "duree_moyenne_sec": duree_moy,
        "duree_moyenne_appels_utiles_sec": duree_moy_utiles,  # ← NOUVEAU
    }


def appels_par_jour(df: pd.DataFrame) -> pd.DataFrame:
    """Nombre d'appels par jour."""
    df = _parse_ts(df)
    if "_ts" not in df.columns:
        return pd.DataFrame()
    df["_date"] = df["_ts"].dt.date
    out = df.groupby("_date").size().reset_index(name="nb_appels")
    out.columns = ["date", "nb_appels"]
    return out


def appels_par_mois(df: pd.DataFrame) -> pd.DataFrame:
    """Nombre d'appels par mois (YYYY-MM)."""
    df = _parse_ts(df)
    if "_ts" not in df.columns:
        return pd.DataFrame()
    df["_mois"] = df["_ts"].dt.to_period("M").astype(str)
    out = df.groupby("_mois").size().reset_index(name="nb_appels")
    out.columns = ["mois", "nb_appels"]
    return out


def appels_par_heure(df: pd.DataFrame) -> pd.DataFrame:
    """Nombre d'appels par heure de la journée (0-23)."""
    df = _parse_ts(df)
    if "_ts" not in df.columns:
        return pd.DataFrame()
    df["_heure"] = df["_ts"].dt.hour
    out = (
        df.groupby("_heure")
        .size()
        .reindex(range(24), fill_value=0)
        .reset_index()
    )
    out.columns = ["heure", "nb_appels"]
    return out


def repartition_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Répartition par Classification — exclut les vides et 'non trouvé'.
    Retourne un DataFrame trié par count desc, prêt pour un bar chart horizontal.
    Colonnes : Classification, count, pct
    """
    if "Classification" not in df.columns:
        return pd.DataFrame()
    df_clean = _clean_classification(df)
    if df_clean.empty:
        return pd.DataFrame()
    counts = df_clean["Classification"].value_counts(dropna=True).reset_index()
    counts.columns = ["Classification", "count"]
    counts["pct"] = (counts["count"] / counts["count"].sum() * 100).round(1)
    # ── Déjà trié par count desc par value_counts ────────────────────────────
    return counts


# ─────────────────────────────────────────────
# 2. ANALYSE PAR FOURNISSEUR (list_name)
# ─────────────────────────────────────────────


def appels_par_fournisseur(df: pd.DataFrame) -> pd.DataFrame:
    """
    Par list_name :
      - nb total d'appels
      - nb appels utiles (classification valide != non trouvé/vide)
      - taux appels utiles (%)
      - nb appels qualifies (PEU INTERESSE/INTERESSE/TRES INTERESSE/EDIFICIOS/RDV LEADS/WHATSAP)
      - taux appels qualifies (%)
      - durée moyenne (sec)
    """
    if "list_name" not in df.columns:
        return pd.DataFrame()

    df = df.copy()

    if "Classification" not in df.columns:
        df["_utile"] = False
        df["_qualifie"] = False
    else:
        classifications_raw = df["Classification"].fillna("").astype(str).str.strip()
        non_utiles = ["", "nan", "none", "non trouve", "non trouvé"]
        df["_utile"] = ~classifications_raw.str.lower().str.replace("é", "e", regex=False).isin(
            non_utiles
        )
        classifications_qualif = [
            "PEU INTERESSE",
            "INTERESSE",
            "TRES INTERESSE",
            "EDIFICIOS",
            "RDV LEADS",
            "WHATSAP",
        ]
        qualif_normalized = [c.upper().strip() for c in classifications_qualif]
        df["_qualifie"] = classifications_raw.str.upper().isin(qualif_normalized)

    df["_duree"] = pd.to_numeric(
        df["Duration_seconds"] if "Duration_seconds" in df.columns else pd.Series(dtype=float),
        errors="coerce",
    )

    agg = df.groupby("list_name").agg(
        nb_appels=("list_name", "count"),
        nb_utiles=("_utile", "sum"),
        nb_qualifies=("_qualifie", "sum"),
        duree_moy_sec=("_duree", "mean"),
    ).reset_index()

    agg["taux_utiles_pct"] = (agg["nb_utiles"] / agg["nb_appels"] * 100).round(1)
    agg["taux_qualifies_pct"] = (agg["nb_qualifies"] / agg["nb_appels"] * 100).round(1)
    agg["nb_utiles"] = agg["nb_utiles"].astype(int)
    agg["nb_qualifies"] = agg["nb_qualifies"].astype(int)
    agg["duree_moy_sec"] = agg["duree_moy_sec"].round(1)
    agg.sort_values("nb_appels", ascending=False, inplace=True)

    return agg.reset_index(drop=True)


def classification_par_fournisseur(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format long : list_name × Classification (exclut vides et non trouvé).
    Colonnes : list_name, Classification, count, pct
    """
    if "list_name" not in df.columns or "Classification" not in df.columns:
        return pd.DataFrame()

    df_clean = _clean_classification(df)
    if df_clean.empty:
        return pd.DataFrame()

    cross = pd.crosstab(df_clean["list_name"], df_clean["Classification"])
    long = cross.reset_index().melt(
        id_vars="list_name", var_name="Classification", value_name="count"
    )
    long = long[long["count"] > 0].copy()
    totaux = long.groupby("list_name")["count"].transform("sum")
    long["pct"] = (long["count"] / totaux * 100).round(1)
    return long.reset_index(drop=True)


# ─────────────────────────────────────────────
# 3. ANALYSE GÉOGRAPHIQUE
# ─────────────────────────────────────────────


def appels_par_ville(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Nombre total d'appels par Ciudad."""
    if "Ciudad" not in df.columns:
        return pd.DataFrame()
    counts = (
        df["Ciudad"]
        .astype(str)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA})
        .value_counts(dropna=True)
        .head(top_n)
        .reset_index()
    )
    counts.columns = ["Ciudad", "nb_appels"]
    return counts


def appels_utiles_par_ville(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Appels utiles par Ciudad avec taux."""
    if "Ciudad" not in df.columns or "Classification" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["_utile"] = _est_utile(df["Classification"])
    df["_ciudad"] = (
        df["Ciudad"].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA})
    )
    df = df.dropna(subset=["_ciudad"])

    agg = df.groupby("_ciudad").agg(
        nb_appels=("_ciudad", "count"),
        nb_utiles=("_utile", "sum"),
    ).reset_index()
    agg.columns = ["Ciudad", "nb_appels", "nb_utiles"]
    agg["taux_utiles_pct"] = (agg["nb_utiles"] / agg["nb_appels"] * 100).round(1)
    agg["nb_utiles"] = agg["nb_utiles"].astype(int)
    agg.sort_values("nb_utiles", ascending=False, inplace=True)
    return agg.head(top_n).reset_index(drop=True)


# ─────────────────────────────────────────────
# 5. ANALYSE GÉOGRAPHIQUE - CODES POSTAUX
# ─────────────────────────────────────────────


def taux_remplissage_code_postal(df: pd.DataFrame) -> dict:
    resultats = {}
    colonnes_disponibles = []
    if "code_postal" in df.columns:
        colonnes_disponibles.append("code_postal")
    if "codigo_postal" in df.columns:
        colonnes_disponibles.append("codigo_postal")

    for col in colonnes_disponibles:
        non_vide = (
            df[col].astype(str).str.strip().notna()
            & (df[col].astype(str).str.strip() != "")
            & (df[col].astype(str).str.strip() != "nan")
        )
        nb_remplis = non_vide.sum()
        taux = round(nb_remplis / len(df) * 100, 1)
        resultats[col] = {
            "total_lignes": len(df),
            "nb_remplis": nb_remplis,
            "nb_vides": len(df) - nb_remplis,
            "taux_remplissage": taux,
        }
    return resultats


def comparer_codes_postaux(df: pd.DataFrame):
    if "code_postal" not in df.columns or "codigo_postal" not in df.columns:
        return pd.DataFrame()

    df_comp = df.copy()
    df_comp["code_postal_clean"] = (
        df_comp["code_postal"].astype(str).str.replace(r"\D", "", regex=True).str.strip()
    )
    df_comp["codigo_postal_clean"] = (
        df_comp["codigo_postal"].astype(str).str.replace(r"\D", "", regex=True).str.strip()
    )

    masque_disponibles = (
        (df_comp["code_postal_clean"] != "")
        & (df_comp["code_postal_clean"] != "nan")
        & (df_comp["codigo_postal_clean"] != "")
        & (df_comp["codigo_postal_clean"] != "nan")
    )

    df_disponibles = df_comp[masque_disponibles].copy()
    if len(df_disponibles) == 0:
        return pd.DataFrame()

    df_disponibles["correspond"] = (
        df_disponibles["code_postal_clean"] == df_disponibles["codigo_postal_clean"]
    )
    stats = {
        "total_comparaisons": len(df_disponibles),
        "nb_correspondances": df_disponibles["correspond"].sum(),
        "nb_differences": (~df_disponibles["correspond"]).sum(),
        "taux_correspondance": round(
            df_disponibles["correspond"].sum() / len(df_disponibles) * 100, 1
        ),
    }
    return df_disponibles, stats


def analyse_fiabilite_par_fournisseur(df: pd.DataFrame) -> pd.DataFrame:
    if "list_name" not in df.columns:
        return pd.DataFrame()
    if "code_postal" not in df.columns or "codigo_postal" not in df.columns:
        return pd.DataFrame()

    resultats = []
    for fournisseur in df["list_name"].unique():
        df_fourn = df[df["list_name"] == fournisseur].copy()
        df_fourn["code_postal_clean"] = (
            df_fourn["code_postal"].astype(str).str.replace(r"\D", "", regex=True).str.strip()
        )
        df_fourn["codigo_postal_clean"] = (
            df_fourn["codigo_postal"].astype(str).str.replace(r"\D", "", regex=True).str.strip()
        )

        client_rempli = (df_fourn["code_postal_clean"] != "") & (
            df_fourn["code_postal_clean"] != "nan"
        )
        fournisseur_rempli = (df_fourn["codigo_postal_clean"] != "") & (
            df_fourn["codigo_postal_clean"] != "nan"
        )

        taux_client = (
            round(client_rempli.sum() / len(df_fourn) * 100, 1) if len(df_fourn) > 0 else 0.0
        )
        taux_fournisseur = (
            round(fournisseur_rempli.sum() / len(df_fourn) * 100, 1) if len(df_fourn) > 0 else 0.0
        )

        les_deux_remplis = client_rempli & fournisseur_rempli
        if les_deux_remplis.sum() > 0:
            correspondance = (
                df_fourn["code_postal_clean"] == df_fourn["codigo_postal_clean"]
            )[les_deux_remplis].sum()
            taux_correspondance = round(correspondance / les_deux_remplis.sum() * 100, 1)
        else:
            taux_correspondance = 0

        resultats.append(
            {
                "fournisseur": fournisseur,
                "total_appels": len(df_fourn),
                "taux_remplissage_client": taux_client,
                "taux_remplissage_fournisseur": taux_fournisseur,
                "nb_comparaisons": les_deux_remplis.sum(),
                "taux_correspondance": taux_correspondance,
            }
        )

    df_resultat = pd.DataFrame(resultats)
    df_resultat = df_resultat.sort_values("total_appels", ascending=False).reset_index(drop=True)
    return df_resultat


def codes_postaux_non_correspondants(df: pd.DataFrame) -> pd.DataFrame:
    if "code_postal" not in df.columns or "codigo_postal" not in df.columns:
        return pd.DataFrame()

    df_comp = df.copy()
    df_comp["code_postal_clean"] = (
        df_comp["code_postal"].astype(str).str.replace(r"\D", "", regex=True).str.strip()
    )
    df_comp["codigo_postal_clean"] = (
        df_comp["codigo_postal"].astype(str).str.replace(r"\D", "", regex=True).str.strip()
    )

    masque = (
        (df_comp["code_postal_clean"] != "")
        & (df_comp["code_postal_clean"] != "nan")
        & (df_comp["codigo_postal_clean"] != "")
        & (df_comp["codigo_postal_clean"] != "nan")
        & (df_comp["code_postal_clean"] != df_comp["codigo_postal_clean"])
    )
    return df_comp[masque].copy()


# ─────────────────────────────────────────────
# 6. ANALYSE LOGEMENT PAR FOURNISSEUR
# ─────────────────────────────────────────────


def logement_par_fournisseur(
    df: pd.DataFrame, regrouper: bool = True
) -> pd.DataFrame:
    """
    Analyse du type de logement par fournisseur.

    Parameters
    ----------
    regrouper : bool
        Si True, normalise tipo_vivienda via `normaliser_tipo_vivienda()` avant
        de calculer le tableau croisé (bouton "Regrouper" dans l'UI).
        Si False, conserve les valeurs brutes (bouton "Déregrouper").
    """
    if "list_name" not in df.columns or "piso_casa" not in df.columns:
        return pd.DataFrame()

    df_clean = df.copy()
    df_clean["piso_casa"] = df_clean["piso_casa"].astype(str).str.strip()
    df_clean = df_clean[
        df_clean["piso_casa"].notna()
        & (df_clean["piso_casa"] != "")
        & (df_clean["piso_casa"] != "nan")
    ]

    if regrouper:
        df_clean["piso_casa"] = normaliser_tipo_vivienda(df_clean["piso_casa"])

    if df_clean.empty:
        return pd.DataFrame()

    cross = pd.crosstab(df_clean["list_name"], df_clean["piso_casa"])
    cross["total_appels"] = cross.sum(axis=1)
    cross_pct = cross.div(cross["total_appels"], axis=0) * 100
    cross_pct = cross_pct.rename(
        columns={col: f"{col}_pct" for col in cross_pct.columns if col != "total_appels"}
    )
    resultat = cross.join(cross_pct)
    return resultat.sort_values("total_appels", ascending=False).reset_index()


def repartition_tipo_vivienda(
    df: pd.DataFrame, regrouper: bool = True
) -> pd.DataFrame:
    """
    Répartition globale des types de logement (piso_casa).

    Parameters
    ----------
    regrouper : bool
        Si True, normalise les valeurs. Si False, conserve les valeurs brutes.

    Returns
    -------
    DataFrame avec colonnes : type_logement, count, pct
    """
    if "piso_casa" not in df.columns:
        return pd.DataFrame()

    s = df["piso_casa"].astype(str).str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA}).dropna()

    if regrouper:
        s = normaliser_tipo_vivienda(s)

    counts = s.value_counts().reset_index()
    counts.columns = ["type_logement", "count"]
    counts["pct"] = (counts["count"] / counts["count"].sum() * 100).round(1)
    return counts


def top_logement_par_fournisseur(
    df: pd.DataFrame, top_n: int = 3, regrouper: bool = True
) -> pd.DataFrame:
    """
    Pour chaque fournisseur, retourne les top N types de logement les plus fréquents.

    Parameters
    ----------
    regrouper : bool
        Normalise tipo_vivienda si True.
    """
    if "list_name" not in df.columns or "piso_casa" not in df.columns:
        return pd.DataFrame()

    df_clean = df.copy()
    df_clean["piso_casa"] = df_clean["piso_casa"].astype(str).str.strip()
    df_clean = df_clean[
        df_clean["piso_casa"].notna()
        & (df_clean["piso_casa"] != "")
        & (df_clean["piso_casa"] != "nan")
    ]

    if regrouper:
        df_clean["piso_casa"] = normaliser_tipo_vivienda(df_clean["piso_casa"])

    if df_clean.empty:
        return pd.DataFrame()

    grouped = (
        df_clean.groupby(["list_name", "piso_casa"]).size().reset_index(name="count")
    )

    resultats = []
    for fournisseur in grouped["list_name"].unique():
        df_fourn = (
            grouped[grouped["list_name"] == fournisseur]
            .sort_values("count", ascending=False)
            .head(top_n)
        )
        total = df_fourn["count"].sum()
        df_fourn = df_fourn.copy()
        df_fourn["pct_du_fournisseur"] = (df_fourn["count"] / total * 100).round(1)
        resultats.append(df_fourn)

    return pd.concat(resultats, ignore_index=True)


def classification_par_type_logement(
    df: pd.DataFrame, regrouper: bool = True
):
    """
    Analyse la classification des appels par type de logement.

    Parameters
    ----------
    regrouper : bool
        Normalise tipo_vivienda si True.
    """
    if "piso_casa" not in df.columns or "Classification" not in df.columns:
        return pd.DataFrame()

    df_clean = df.copy()
    df_clean["piso_casa"] = df_clean["piso_casa"].astype(str).str.strip()
    df_clean = df_clean[
        df_clean["piso_casa"].notna()
        & (df_clean["piso_casa"] != "")
        & (df_clean["piso_casa"] != "nan")
    ]

    if regrouper:
        df_clean["piso_casa"] = normaliser_tipo_vivienda(df_clean["piso_casa"])

    df_clean = df_clean[_est_utile(df_clean["Classification"])]

    if df_clean.empty:
        return pd.DataFrame()

    cross = pd.crosstab(df_clean["piso_casa"], df_clean["Classification"])
    cross["total"] = cross.sum(axis=1)
    cross_pct = cross.div(cross["total"], axis=0) * 100
    return cross, cross_pct


# ─────────────────────────────────────────────
# 4. ANALYSE MÉTIER — LOGEMENT
# ─────────────────────────────────────────────


def appels_par_piso_casa(
    df: pd.DataFrame, regrouper: bool = True
) -> pd.DataFrame:
    """
    Répartition des appels par piso_casa.

    Parameters
    ----------
    regrouper : bool
        Normalise tipo_vivienda si True.
    """
    if "piso_casa" not in df.columns:
        return pd.DataFrame()

    s = (
        df["piso_casa"]
        .astype(str)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )
    s = s.dropna()

    if regrouper:
        s = normaliser_tipo_vivienda(s)

    counts = s.value_counts(dropna=True).reset_index()
    counts.columns = ["piso_casa", "count"]
    counts["pct"] = (counts["count"] / counts["count"].sum() * 100).round(1)
    return counts


# ─────────────────────────────────────────────
# 5. FONCTIONS SUPPLÉMENTAIRES UTILES
# ─────────────────────────────────────────────


def details_appels_non_utiles(df: pd.DataFrame) -> pd.DataFrame:
    """Retourne les détails des appels non utiles."""
    if "Classification" not in df.columns:
        return pd.DataFrame()
    mask_non_utile = ~_est_utile(df["Classification"])
    return df[mask_non_utile].copy()


def statistiques_classification(df: pd.DataFrame) -> dict:
    """Statistiques détaillées sur la classification."""
    if "Classification" not in df.columns:
        return {}

    total = len(df)
    classifications = df["Classification"].astype(str).str.strip().str.lower()

    non_trouve = (classifications == "non trouvé").sum() + (classifications == "non trouve").sum()
    vides = (
        (classifications == "").sum()
        + (classifications == "nan").sum()
        + (classifications == "none").sum()
    )
    valides = total - non_trouve - vides

    return {
        "total_appels": total,
        "classifications_valides": valides,
        "classifications_non_trouve": non_trouve,
        "classifications_vides": vides,
        "taux_valides_pct": round(valides / total * 100, 1) if total > 0 else 0,
    }


# ─────────────────────────────────────────────
# 7. ANALYSE DÉTAILLÉE PAR TYPE DE LOGEMENT
# ─────────────────────────────────────────────


def analyse_par_type_logement(
    df: pd.DataFrame, regrouper: bool = True
) -> dict:
    """
    Analyse détaillée pour chaque type de logement (piso_casa).

    Parameters
    ----------
    regrouper : bool
        Normalise tipo_vivienda si True.
    """
    if "piso_casa" not in df.columns:
        return {"error": "Colonne 'piso_casa' non trouvée"}
    if "Classification" not in df.columns:
        return {"error": "Colonne 'Classification' non trouvée"}

    df_clean = df.copy()
    df_clean["piso_casa"] = df_clean["piso_casa"].astype(str).str.strip()
    df_clean["Classification"] = df_clean["Classification"].astype(str).str.strip()
    df_clean = df_clean[
        df_clean["piso_casa"].notna()
        & (df_clean["piso_casa"] != "")
        & (df_clean["piso_casa"] != "nan")
    ]

    if regrouper:
        df_clean["piso_casa"] = normaliser_tipo_vivienda(df_clean["piso_casa"])

    classifications_qualif = [
        "PEU INTERESSE",
        "INTERESSE",
        "TRES INTERESSE",
        "EDIFICIOS",
        "RDV LEADS",
        "WHATSAP",
    ]

    resultats = {}
    for type_log in df_clean["piso_casa"].unique():
        df_type = df_clean[df_clean["piso_casa"] == type_log]
        total_appels = len(df_type)

        classifications_valides = df_type[
            ~df_type["Classification"]
            .str.lower()
            .isin(["", "nan", "none", "non trouve", "non trouvé"])
        ]
        appels_utiles = len(classifications_valides)
        taux_utiles = round(appels_utiles / total_appels * 100, 1) if total_appels > 0 else 0

        appels_qualifies = classifications_valides[
            classifications_valides["Classification"]
            .str.upper()
            .isin([c.upper() for c in classifications_qualif])
        ].shape[0]
        taux_qualifies = (
            round(appels_qualifies / total_appels * 100, 1) if total_appels > 0 else 0
        )

        repartition = classifications_valides["Classification"].value_counts().to_dict()
        top_classifications = (
            classifications_valides["Classification"].value_counts().head(3).to_dict()
        )

        duree_moyenne = None
        if "Duration_seconds" in df_type.columns:
            duree_moyenne = round(
                pd.to_numeric(df_type["Duration_seconds"], errors="coerce").mean(), 1
            )

        resultats[type_log] = {
            "total_appels": total_appels,
            "appels_utiles": appels_utiles,
            "taux_utiles_pct": taux_utiles,
            "appels_qualifies": appels_qualifies,
            "taux_qualifies_pct": taux_qualifies,
            "duree_moyenne_sec": duree_moyenne,
            "repartition_classifications": repartition,
            "top_classifications": top_classifications,
        }

    return resultats


def comparer_types_logement(
    df: pd.DataFrame, regrouper: bool = True
) -> pd.DataFrame:
    """
    Compare les performances entre différents types de logement.

    Parameters
    ----------
    regrouper : bool
        Normalise tipo_vivienda si True.
    """
    if "piso_casa" not in df.columns or "Classification" not in df.columns:
        return pd.DataFrame()

    df_clean = df.copy()
    df_clean["piso_casa"] = df_clean["piso_casa"].astype(str).str.strip()
    df_clean["Classification"] = df_clean["Classification"].astype(str).str.strip()
    df_clean = df_clean[
        df_clean["piso_casa"].notna()
        & (df_clean["piso_casa"] != "")
        & (df_clean["piso_casa"] != "nan")
    ]

    if regrouper:
        df_clean["piso_casa"] = normaliser_tipo_vivienda(df_clean["piso_casa"])

    classifications_qualif = [
        "PEU INTERESSE",
        "INTERESSE",
        "TRES INTERESSE",
        "EDIFICIOS",
        "RDV LEADS",
        "WHATSAP",
    ]

    resultats = []
    for type_log in df_clean["piso_casa"].unique():
        df_type = df_clean[df_clean["piso_casa"] == type_log]
        total = len(df_type)

        valides = df_type[
            ~df_type["Classification"]
            .str.lower()
            .isin(["", "nan", "none", "non trouve", "non trouvé"])
        ]
        nb_valides = len(valides)

        qualifies = valides[
            valides["Classification"]
            .str.upper()
            .isin([c.upper() for c in classifications_qualif])
        ]
        nb_qualifies = len(qualifies)

        resultats.append(
            {
                "type_logement": type_log,
                "total_appels": total,
                "appels_valides": nb_valides,
                "taux_valides": round(nb_valides / total * 100, 1) if total > 0 else 0,
                "appels_qualifies": nb_qualifies,
                "taux_qualifies": round(nb_qualifies / total * 100, 1) if total > 0 else 0,
            }
        )

    df_resultat = pd.DataFrame(resultats)
    return df_resultat.sort_values("taux_qualifies", ascending=False).reset_index(drop=True)


def classification_detaillee_par_type(
    df: pd.DataFrame, regrouper: bool = True
) -> pd.DataFrame:
    """
    Classification détaillée pour chaque type de logement.
    Format : type_logement, classification, count, pct_du_type

    Parameters
    ----------
    regrouper : bool
        Normalise tipo_vivienda si True.
    """
    if "piso_casa" not in df.columns or "Classification" not in df.columns:
        return pd.DataFrame()

    df_clean = df.copy()
    df_clean["piso_casa"] = df_clean["piso_casa"].astype(str).str.strip()
    df_clean["Classification"] = df_clean["Classification"].astype(str).str.strip()
    df_clean = df_clean[
        df_clean["piso_casa"].notna()
        & (df_clean["piso_casa"] != "")
        & (df_clean["piso_casa"] != "nan")
        & (
            ~df_clean["Classification"]
            .str.lower()
            .isin(["", "nan", "none", "non trouve", "non trouvé"])
        )
    ]

    if regrouper:
        df_clean["piso_casa"] = normaliser_tipo_vivienda(df_clean["piso_casa"])

    if df_clean.empty:
        return pd.DataFrame()

    cross = pd.crosstab(df_clean["piso_casa"], df_clean["Classification"])
    cross["total"] = cross.sum(axis=1)

    long_df = cross.reset_index().melt(
        id_vars=["piso_casa", "total"], var_name="Classification", value_name="count"
    )
    long_df = long_df[long_df["count"] > 0].copy()
    long_df["pct_du_type"] = (long_df["count"] / long_df["total"] * 100).round(1)
    return long_df.reset_index(drop=True)

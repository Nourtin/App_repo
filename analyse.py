# analyse.py
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def diagnostic_classification(df: pd.DataFrame):
    """Affiche les valeurs uniques de la colonne Classification pour diagnostic"""
    if "Classification" not in df.columns:
        return "Colonne 'Classification' non trouvée"

    valeurs_uniques = df["Classification"].astype(str).str.strip().unique()
    return {
        "valeurs_uniques": sorted(valeurs_uniques),
        "nb_valeurs_uniques": len(valeurs_uniques),
        "exemples": df["Classification"].head(10).tolist()
    }

def _est_utile(serie: pd.Series) -> pd.Series:
    """
    Retourne un masque booléen : appel utile si Classification
    n'est pas vide et différente de 'non trouvé' (insensible à la casse et aux accents).
    """
    s = serie.astype(str).str.strip().str.lower()
    # Normaliser les accents pour "non trouvé"
    s = s.str.replace('é', 'e', regex=False)

    # Liste des valeurs non utiles
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
# CLASSIFICATIONS IMPORTANTES (constantes)
# ─────────────────────────────────────────────

CLASSIFICATIONS_QUALIFIEES = [
    "PEU INTERESSE", "INTERESSE", "TRES INTERESSE", 
    "EDIFICIOS", "RDV LEADS", "WHATSAP"
]

CLASSIFICATIONS_PRIORITAIRES = [
    "RDV LEADS", "TRES INTERESSE", "INTERESSE"
]


def _normaliser_classification(series: pd.Series) -> pd.Series:
    """Normalise les classifications pour comparaison (majuscules, strip)."""
    return series.astype(str).str.strip().str.upper()


def _est_qualifie(series: pd.Series) -> pd.Series:
    """Masque booléen pour les classifications qualifiées."""
    s = _normaliser_classification(series)
    return s.isin([c.upper() for c in CLASSIFICATIONS_QUALIFIEES])


def _est_prioritaire(series: pd.Series, classification: str) -> pd.Series:
    """Masque booléen pour une classification prioritaire spécifique."""
    s = _normaliser_classification(series)
    return s == classification.upper()


# ─────────────────────────────────────────────
# MAPPING TYPES DE LOGEMENT (regroupement intelligent)
# ─────────────────────────────────────────────

def _mapper_type_logement(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Regroupe automatiquement les types de logements similaires.

    Retourne:
        - series_groupee : la série avec les catégories regroupées
        - series_detail : la série avec le détail original (pour drill-down)
    """
    detail = series.astype(str).str.strip()

    # Nettoyer les valeurs vides
    detail = detail.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NONE": pd.NA})

    groupee = detail.copy()

    # Patterns pour regroupement (insensible à la casse)
    mappings = {
        # PISO (tous les types de piso)
        "PISO": [
            r'^piso\b', r'^apartamento', r'^ático', r'^atico', 
            r'^bajo', r'^entresuelo', r'^duplex', r'^dúplex',
            r'^estudio', r'^loft', r'^planta', r'^ático', r'^atico'
        ],
        # CASA (tous les types de maison)
        "CASA": [
            r'^casa\b', r'^chalet', r'^villa', r'^bungalow',
            r'^finca', r'^masia', r'^torre', r'^adosado',
            r'^pareado', r'^unifamiliar'
        ],
        # EDIFICIO / LOCAL COMMERCIAL
        "EDIFICIO/LOCAL": [
            r'^edificio', r'^local', r'^nave', r'^oficina',
            r'^comercial', r'^industrial', r'^bodega', r'^almacén'
        ],
        # PARKING / TRASTERO
        "PARKING/TRASTERO": [
            r'^parking', r'^garaje', r'^trastero', r'^plaza',
            r'^cochera'
        ],
        # TERRENO
        "TERRENO": [
            r'^terreno', r'^parcela', r'^solar', r'^finca rustica',
            r'^finca rústica'
        ],
    }

    # Pour chaque valeur, trouver le groupe correspondant
    for idx, val in groupee.items():
        if pd.isna(val):
            continue
        val_lower = str(val).lower()
        trouve = False
        for groupe, patterns in mappings.items():
            for pattern in patterns:
                if re.search(pattern, val_lower):
                    groupee.at[idx] = groupe
                    trouve = True
                    break
            if trouve:
                break
        if not trouve:
            # Si pas de match, garder la valeur originale (nettoyée)
            pass

    return groupee, detail


# ─────────────────────────────────────────────
# 1. ANALYSE GLOBALE
# ─────────────────────────────────────────────

def kpi_globaux(df: pd.DataFrame) -> dict:
    """
    Total appels, durée moyenne, nb appels utiles.
    Les appels utiles = appels avec une classification valide 
    (différente de vide, NaN, 'non trouvé')
    """
    total = len(df)
    if total == 0:
        return {
            "total_appels": 0,
            "appels_utiles": 0,
            "taux_utiles_pct": 0.0,
            "duree_moyenne_sec": None,
            "duree_moyenne_utiles_sec": None,
            "duree_moyenne_tres_interesse_sec": None,
            "duree_moyenne_interesse_sec": None,
            "duree_moyenne_rdv_leads_sec": None,
        }

    # Calcul des appels utiles
    if "Classification" in df.columns:
        appels_utiles = int(_est_utile(df["Classification"]).sum())
        taux_utiles = round(appels_utiles / total * 100, 1) if total > 0 else 0.0

        # Appels qualifiés
        qualif_mask = _est_qualifie(df["Classification"])
        appels_qualifies = int(qualif_mask.sum())
        taux_qualifie = round(appels_qualifies / total * 100, 1) if total > 0 else 0
    else:
        appels_utiles = None
        taux_utiles = None
        appels_qualifies = None
        taux_qualifie = None

    # Calcul de la durée moyenne globale
    duree_moy = None
    if "Duration_seconds" in df.columns:
        duree_moy = pd.to_numeric(df["Duration_seconds"], errors="coerce").mean()
        if not np.isnan(duree_moy):
            duree_moy = round(duree_moy, 1)

    # Durée moyenne des appels utiles
    duree_moy_utiles = _duree_moyenne_par_masque(df, _est_utile(df["Classification"]) if "Classification" in df.columns else pd.Series([False]*len(df)))

    # Durée moyenne par catégorie prioritaire
    duree_tres_interesse = _duree_moyenne_par_classification(df, "TRES INTERESSE")
    duree_interesse = _duree_moyenne_par_classification(df, "INTERESSE")
    duree_rdv_leads = _duree_moyenne_par_classification(df, "RDV LEADS")

    return {
        "total_appels": total,
        "appels_utiles": appels_utiles,
        "taux_utiles_pct": taux_utiles,
        "appels_qualifies": appels_qualifies,
        "taux_qualifies_pct": taux_qualifie,
        "duree_moyenne_sec": duree_moy,
        "duree_moyenne_utiles_sec": duree_moy_utiles,
        "duree_moyenne_tres_interesse_sec": duree_tres_interesse,
        "duree_moyenne_interesse_sec": duree_interesse,
        "duree_moyenne_rdv_leads_sec": duree_rdv_leads,
    }


def _duree_moyenne_par_masque(df: pd.DataFrame, masque: pd.Series) -> Optional[float]:
    """Calcule la durée moyenne pour un masque booléen donné."""
    if "Duration_seconds" not in df.columns:
        return None
    durees = pd.to_numeric(df.loc[masque, "Duration_seconds"], errors="coerce")
    moy = durees.mean()
    return round(moy, 1) if not np.isnan(moy) else None


def _duree_moyenne_par_classification(df: pd.DataFrame, classification: str) -> Optional[float]:
    """Calcule la durée moyenne pour une classification spécifique."""
    if "Classification" not in df.columns or "Duration_seconds" not in df.columns:
        return None
    masque = _est_prioritaire(df["Classification"], classification)
    return _duree_moyenne_par_masque(df, masque)


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


def repartition_classification(df: pd.DataFrame, max_categories: int = 12) -> pd.DataFrame:
    """
    Répartition par Classification — exclut les vides et 'non trouvé'.
    Limite le nombre de catégories affichées pour éviter le surchargement visuel.
    Les catégories au-delà de max_categories sont regroupées dans 'AUTRES'.
    """
    if "Classification" not in df.columns:
        return pd.DataFrame()
    df_clean = _clean_classification(df)
    if df_clean.empty:
        return pd.DataFrame()

    counts = df_clean["Classification"].value_counts(dropna=True).reset_index()
    counts.columns = ["Classification", "count"]

    # Si trop de catégories, regrouper les moins fréquentes
    if len(counts) > max_categories:
        top = counts.head(max_categories).copy()
        autres = counts.tail(len(counts) - max_categories)
        autres_count = autres["count"].sum()
        top = pd.concat([
            top,
            pd.DataFrame({"Classification": ["AUTRES"], "count": [autres_count]})
        ], ignore_index=True)
        counts = top

    counts["pct"] = (counts["count"] / counts["count"].sum() * 100).round(1)
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
      - durée moyenne utiles (sec)
      - durée moyenne par catégorie prioritaire
    """
    if "list_name" not in df.columns:
        return pd.DataFrame()

    df = df.copy()

    # Vérifier si la colonne Classification existe
    if "Classification" not in df.columns:
        df["_utile"] = False
        df["_qualifie"] = False
        for cat in CLASSIFICATIONS_PRIORITAIRES:
            df[f"_{cat.lower().replace(' ', '_')}"] = False
    else:
        classifications_raw = df["Classification"].fillna("").astype(str).str.strip()

        # Appels utiles
        non_utiles = ["", "nan", "none", "non trouve", "non trouvé"]
        df["_utile"] = ~classifications_raw.str.lower().str.replace('é', 'e', regex=False).isin(non_utiles)

        # Appels qualifiés
        qualif_normalized = [c.upper().strip() for c in CLASSIFICATIONS_QUALIFIEES]
        df["_qualifie"] = classifications_raw.str.upper().isin(qualif_normalized)

        # Masques pour chaque catégorie prioritaire
        for cat in CLASSIFICATIONS_PRIORITAIRES:
            df[f"_{cat.lower().replace(' ', '_')}"] = classifications_raw.str.upper() == cat.upper()

    df["_duree"] = pd.to_numeric(
        df["Duration_seconds"] if "Duration_seconds" in df.columns else pd.Series(dtype=float),
        errors="coerce"
    )

    # Agrégation
    agg_dict = {
        "nb_appels": ("list_name", "count"),
        "nb_utiles": ("_utile", "sum"),
        "nb_qualifies": ("_qualifie", "sum"),
        "duree_moy_sec": ("_duree", "mean"),
    }

    # Ajouter les durées moyennes par catégorie
    for cat in CLASSIFICATIONS_PRIORITAIRES:
        col_mask = f"_{cat.lower().replace(' ', '_')}"
        # Durée moyenne conditionnelle : on filtre d'abord
        pass  # On le fera après le groupby pour plus de clarté

    agg = df.groupby("list_name").agg(
        nb_appels=("list_name", "count"),
        nb_utiles=("_utile", "sum"),
        nb_qualifies=("_qualifie", "sum"),
        duree_moy_sec=("_duree", "mean"),
    ).reset_index()

    # Calculer les durées moyennes par catégorie pour chaque fournisseur
    for cat in CLASSIFICATIONS_PRIORITAIRES:
        col_mask = f"_{cat.lower().replace(' ', '_')}"
        durees_par_fournisseur = []
        for fournisseur in agg["list_name"]:
            df_fourn = df[df["list_name"] == fournisseur]
            masque = df_fourn[col_mask]
            durees = pd.to_numeric(df_fourn.loc[masque, "Duration_seconds"], errors="coerce")
            moy = durees.mean()
            durees_par_fournisseur.append(round(moy, 1) if not np.isnan(moy) else None)
        agg[f"duree_moy_{cat.lower().replace(' ', '_')}_sec"] = durees_par_fournisseur

    # Durée moyenne des appels utiles par fournisseur
    durees_utiles = []
    for fournisseur in agg["list_name"]:
        df_fourn = df[df["list_name"] == fournisseur]
        masque = df_fourn["_utile"]
        durees = pd.to_numeric(df_fourn.loc[masque, "Duration_seconds"], errors="coerce")
        moy = durees.mean()
        durees_utiles.append(round(moy, 1) if not np.isnan(moy) else None)
    agg["duree_moy_utiles_sec"] = durees_utiles

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
        df["Ciudad"].astype(str).str.strip()
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
    df["_ciudad"] = df["Ciudad"].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA})
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
# 4. ANALYSE CODES POSTAUX (CORRIGÉE)
# ─────────────────────────────────────────────

def _nettoyer_code_postal(series: pd.Series) -> pd.Series:
    """
    Nettoie une série de codes postaux : garde uniquement les chiffres,
    supprime les espaces, convertit en string.
    """
    s = series.astype(str).str.strip()
    # Remplacer les valeurs vides/NaN par chaîne vide
    s = s.replace({"nan": "", "None": "", "NONE": "", "<NA>": ""})
    # Garder uniquement les chiffres
    s = s.str.replace(r'\D', '', regex=True)
    return s


def _est_code_postal_valide(series: pd.Series) -> pd.Series:
    """Vérifie si un code postal est valide (non vide et contient des chiffres)."""
    s = series.astype(str).str.strip()
    s = s.replace({"nan": "", "None": "", "NONE": "", "<NA>": ""})
    s = s.str.replace(r'\D', '', regex=True)
    return (s != "") & (s.str.len() > 0)


def taux_remplissage_code_postal(df: pd.DataFrame) -> dict:
    """
    Calcule le taux de remplissage de la colonne code_postal
    et compare client vs fournisseur
    """
    resultats = {}

    # Vérifier les colonnes disponibles (différentes orthographes possibles)
    colonnes_possibles = {
        "code_postal": ["code_postal", "Code_postal", "CODE_POSTAL", "codigo_postal_client"],
        "codigo_postal": ["codigo_postal", "Codigo_postal", "CODIGO_POSTAL", "codigo_postal_fournisseur"]
    }

    colonnes_trouvees = {}
    for categorie, noms in colonnes_possibles.items():
        for nom in noms:
            if nom in df.columns:
                colonnes_trouvees[categorie] = nom
                break

    for categorie, col in colonnes_trouvees.items():
        valide = _est_code_postal_valide(df[col])
        nb_remplis = valide.sum()
        taux = round(nb_remplis / len(df) * 100, 1)

        resultats[categorie] = {
            "colonne_source": col,
            "total_lignes": len(df),
            "nb_remplis": int(nb_remplis),
            "nb_vides": int(len(df) - nb_remplis),
            "taux_remplissage": taux
        }

    return resultats


def comparer_codes_postaux(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Compare les codes postaux donnés par le client vs fournisseur.
    Retourne les lignes où les deux sont disponibles et le taux de correspondance.
    CORRECTION : utilise les bonnes colonnes pour le nettoyage.
    """
    # Détecter les colonnes
    col_client = None
    col_fournisseur = None

    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ["code_postal", "codepostal", "cp_client", "codigo_postal_client"]:
            col_client = col
        elif col_lower in ["codigo_postal", "codigopostal", "cp_fournisseur", "codigo_postal_fournisseur"]:
            col_fournisseur = col

    if col_client is None or col_fournisseur is None:
        return pd.DataFrame(), {"error": f"Colonnes manquantes: client={col_client}, fournisseur={col_fournisseur}"}

    df_comp = df.copy()

    # Nettoyer les codes postaux (garder seulement les chiffres) - CORRECTION ICI
    df_comp["code_postal_clean"] = _nettoyer_code_postal(df_comp[col_client])
    df_comp["codigo_postal_clean"] = _nettoyer_code_postal(df_comp[col_fournisseur])

    # Filtrer les lignes où les deux sont disponibles
    masque_disponibles = (
        _est_code_postal_valide(df_comp[col_client]) & 
        _est_code_postal_valide(df_comp[col_fournisseur])
    )

    df_disponibles = df_comp[masque_disponibles].copy()

    if len(df_disponibles) == 0:
        return pd.DataFrame(), {
            "total_comparaisons": 0,
            "nb_correspondances": 0,
            "nb_differences": 0,
            "taux_correspondance": 0.0,
            "message": "Aucune ligne avec les deux codes postaux disponibles"
        }

    # Vérifier la correspondance
    df_disponibles["correspond"] = df_disponibles["code_postal_clean"] == df_disponibles["codigo_postal_clean"]

    # Calcul des statistiques
    stats = {
        "total_comparaisons": int(len(df_disponibles)),
        "nb_correspondances": int(df_disponibles["correspond"].sum()),
        "nb_differences": int((~df_disponibles["correspond"]).sum()),
        "taux_correspondance": round(df_disponibles["correspond"].sum() / len(df_disponibles) * 100, 1),
        "colonne_client": col_client,
        "colonne_fournisseur": col_fournisseur
    }

    return df_disponibles, stats


def analyse_fiabilite_par_fournisseur(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse la fiabilité des données par fournisseur.
    Compare code_postal (client) vs codigo_postal (fournisseur).
    CORRECTION : logique de calcul revue pour éviter les scores à 0 incohérents.
    """
    if "list_name" not in df.columns:
        return pd.DataFrame()

    # Détecter les colonnes
    col_client = None
    col_fournisseur = None

    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ["code_postal", "codepostal", "cp_client"]:
            col_client = col
        elif col_lower in ["codigo_postal", "codigopostal", "cp_fournisseur"]:
            col_fournisseur = col

    if col_client is None or col_fournisseur is None:
        return pd.DataFrame({"error": [f"Colonnes CP manquantes: client={col_client}, fourn={col_fournisseur}"]})

    resultats = []

    for fournisseur in df["list_name"].dropna().unique():
        df_fourn = df[df["list_name"] == fournisseur].copy()
        total = len(df_fourn)

        if total == 0:
            continue

        # Nettoyer les codes postaux
        df_fourn["cp_client_clean"] = _nettoyer_code_postal(df_fourn[col_client])
        df_fourn["cp_fourn_clean"] = _nettoyer_code_postal(df_fourn[col_fournisseur])

        # Taux de remplissage
        client_rempli = _est_code_postal_valide(df_fourn[col_client])
        fournisseur_rempli = _est_code_postal_valide(df_fourn[col_fournisseur])

        taux_client = round(client_rempli.sum() / total * 100, 1)
        taux_fournisseur = round(fournisseur_rempli.sum() / total * 100, 1)

        # Comparaisons : quand les deux sont remplis
        les_deux_remplis = client_rempli & fournisseur_rempli
        nb_comparaisons = int(les_deux_remplis.sum())

        if nb_comparaisons > 0:
            correspondance = (df_fourn.loc[les_deux_remplis, "cp_client_clean"] == 
                            df_fourn.loc[les_deux_remplis, "cp_fourn_clean"]).sum()
            taux_correspondance = round(correspondance / nb_comparaisons * 100, 1)
        else:
            correspondance = 0
            taux_correspondance = None  # Pas de comparaison possible, pas 0

        resultats.append({
            "fournisseur": fournisseur,
            "total_appels": total,
            "taux_remplissage_client_pct": taux_client,
            "taux_remplissage_fournisseur_pct": taux_fournisseur,
            "nb_comparaisons": nb_comparaisons,
            "nb_correspondances": int(correspondance),
            "taux_correspondance_pct": taux_correspondance,
        })

    df_resultat = pd.DataFrame(resultats)
    if not df_resultat.empty:
        df_resultat = df_resultat.sort_values("total_appels", ascending=False).reset_index(drop=True)

    return df_resultat


def codes_postaux_non_correspondants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne les lignes où les codes postaux client et fournisseur ne correspondent pas.
    """
    # Détecter les colonnes
    col_client = None
    col_fournisseur = None

    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ["code_postal", "codepostal", "cp_client"]:
            col_client = col
        elif col_lower in ["codigo_postal", "codigopostal", "cp_fournisseur"]:
            col_fournisseur = col

    if col_client is None or col_fournisseur is None:
        return pd.DataFrame()

    df_comp = df.copy()
    df_comp["code_postal_clean"] = _nettoyer_code_postal(df_comp[col_client])
    df_comp["codigo_postal_clean"] = _nettoyer_code_postal(df_comp[col_fournisseur])

    # Filtrer où les deux sont disponibles et différents
    masque = (
        _est_code_postal_valide(df_comp[col_client]) & 
        _est_code_postal_valide(df_comp[col_fournisseur]) &
        (df_comp["code_postal_clean"] != df_comp["codigo_postal_clean"])
    )

    return df_comp[masque].copy()


# ─────────────────────────────────────────────
# 5. ANALYSE LOGEMENT PAR FOURNISSEUR (AVEC REGROUPEMENT)
# ─────────────────────────────────────────────

def logement_par_fournisseur(df: pd.DataFrame, regrouper: bool = True) -> pd.DataFrame:
    """
    Analyse du type de logement par fournisseur.

    Paramètres:
        regrouper: Si True, regroupe les catégories similaires (PISO, CASA, etc.)
                   Si False, affiche le détail original.

    Retourne pour chaque fournisseur la répartition des types de logement
    avec option drill-down (colonne 'detail' si regroupement actif).
    """
    if "list_name" not in df.columns:
        return pd.DataFrame()

    # Détecter la colonne de logement (piso_casa ou tipo_vivienda)
    col_logement = None
    for col in df.columns:
        if col.lower() in ["piso_casa", "tipo_vivienda", "tipo_vivienda", "type_logement"]:
            col_logement = col
            break

    if col_logement is None:
        return pd.DataFrame()

    df_clean = df.copy()

    # Appliquer le mapping si demandé
    if regrouper:
        df_clean["type_groupe"], df_clean["type_detail"] = _mapper_type_logement(df_clean[col_logement])
        col_analyse = "type_groupe"
    else:
        df_clean["type_detail"] = df_clean[col_logement].astype(str).str.strip()
        df_clean["type_groupe"] = df_clean["type_detail"]
        col_analyse = "type_detail"

    # Nettoyer
    df_clean = df_clean[df_clean[col_analyse].notna()]
    df_clean = df_clean[df_clean[col_analyse] != ""]
    df_clean = df_clean[df_clean[col_analyse] != "nan"]

    if df_clean.empty:
        return pd.DataFrame()

    # Créer un tableau croisé
    cross = pd.crosstab(df_clean["list_name"], df_clean[col_analyse])

    # Ajouter le total par fournisseur
    cross["total_appels"] = cross.sum(axis=1)

    # Ajouter les pourcentages
    cross_pct = cross.div(cross["total_appels"], axis=0) * 100
    cross_pct = cross_pct.rename(columns={col: f"{col}_pct" for col in cross_pct.columns})

    # Combiner
    resultat = cross.join(cross_pct)
    resultat = resultat.sort_values("total_appels", ascending=False)

    # Si regroupement actif, ajouter une colonne avec les détails (sous-catégories)
    if regrouper:
        details = df_clean.groupby("list_name")["type_detail"].apply(
            lambda x: x.value_counts().head(5).to_dict()
        ).reset_index()
        details.columns = ["list_name", "top_details"]
        resultat = resultat.reset_index().merge(details, on="list_name", how="left").set_index("list_name")

    return resultat.reset_index()


def top_logement_par_fournisseur(df: pd.DataFrame, top_n: int = 3, regrouper: bool = True) -> pd.DataFrame:
    """
    Pour chaque fournisseur, retourne le(s) type(s) de logement le(s) plus fréquent(s).

    Paramètres:
        top_n: Nombre de top types à retourner
        regrouper: Si True, utilise les catégories regroupées
    """
    if "list_name" not in df.columns:
        return pd.DataFrame()

    # Détecter la colonne de logement
    col_logement = None
    for col in df.columns:
        if col.lower() in ["piso_casa", "tipo_vivienda", "tipo_vivienda", "type_logement"]:
            col_logement = col
            break

    if col_logement is None:
        return pd.DataFrame()

    df_clean = df.copy()

    if regrouper:
        df_clean["type_groupe"], df_clean["type_detail"] = _mapper_type_logement(df_clean[col_logement])
        col_analyse = "type_groupe"
    else:
        df_clean["type_groupe"] = df_clean[col_logement].astype(str).str.strip()
        col_analyse = "type_groupe"

    # Nettoyer
    df_clean = df_clean[df_clean[col_analyse].notna()]
    df_clean = df_clean[df_clean[col_analyse] != ""]
    df_clean = df_clean[df_clean[col_analyse] != "nan"]

    if df_clean.empty:
        return pd.DataFrame()

    # Grouper par fournisseur et type
    grouped = df_clean.groupby(["list_name", col_analyse]).size().reset_index(name="count")

    # Pour chaque fournisseur, prendre les top N
    resultats = []
    for fournisseur in grouped["list_name"].unique():
        df_fourn = grouped[grouped["list_name"] == fournisseur].sort_values("count", ascending=False).head(top_n)
        total_fourn = df_clean[df_clean["list_name"] == fournisseur].shape[0]
        df_fourn["pct_du_fournisseur"] = (df_fourn["count"] / total_fourn * 100).round(1)
        df_fourn["total_fournisseur"] = total_fourn

        # Si regroupement, ajouter les sous-catégories principales
        if regrouper:
            details = df_clean[df_clean["list_name"] == fournisseur]["type_detail"].value_counts().head(3).to_dict()
            df_fourn["sous_categories"] = str(details)

        resultats.append(df_fourn)

    return pd.concat(resultats, ignore_index=True)


def classification_par_type_logement(df: pd.DataFrame, regrouper: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyse la classification des appels par type de logement.

    Retourne:
        - cross: tableau croisé avec counts
        - cross_pct: tableau croisé avec pourcentages
    """
    # Détecter la colonne de logement
    col_logement = None
    for col in df.columns:
        if col.lower() in ["piso_casa", "tipo_vivienda", "tipo_vivienda", "type_logement"]:
            col_logement = col
            break

    if col_logement is None or "Classification" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    df_clean = df.copy()

    if regrouper:
        df_clean["type_groupe"], df_clean["type_detail"] = _mapper_type_logement(df_clean[col_logement])
        col_analyse = "type_groupe"
    else:
        df_clean["type_groupe"] = df_clean[col_logement].astype(str).str.strip()
        col_analyse = "type_groupe"

    # Nettoyer
    df_clean = df_clean[df_clean[col_analyse].notna()]
    df_clean = df_clean[df_clean[col_analyse] != ""]
    df_clean = df_clean[df_clean[col_analyse] != "nan"]

    # Exclure les classifications non utiles
    df_clean = df_clean[_est_utile(df_clean["Classification"])]

    if df_clean.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Tableau croisé
    cross = pd.crosstab(df_clean[col_analyse], df_clean["Classification"])
    cross["total"] = cross.sum(axis=1)

    # Pourcentages
    cross_pct = cross.div(cross["total"], axis=0) * 100

    return cross, cross_pct


# ─────────────────────────────────────────────
# 6. ANALYSE MÉTIER — LOGEMENT (AMÉLIORÉE)
# ─────────────────────────────────────────────

def appels_par_piso_casa(df: pd.DataFrame, regrouper: bool = True) -> pd.DataFrame:
    """
    Répartition des appels par type de logement.
    Option regrouper pour fusionner les catégories similaires.
    """
    # Détecter la colonne
    col_logement = None
    for col in df.columns:
        if col.lower() in ["piso_casa", "tipo_vivienda", "tipo_vivienda", "type_logement"]:
            col_logement = col
            break

    if col_logement is None:
        return pd.DataFrame()

    if regrouper:
        groupee, detail = _mapper_type_logement(df[col_logement])
        series = groupee
    else:
        series = df[col_logement].astype(str).str.strip()

    counts = (
        series
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        .value_counts(dropna=True)
        .reset_index()
    )
    counts.columns = ["type_logement", "count"]
    counts["pct"] = (counts["count"] / counts["count"].sum() * 100).round(1)
    return counts


# ─────────────────────────────────────────────
# 7. FONCTIONS SUPPLÉMENTAIRES UTILES
# ─────────────────────────────────────────────

def details_appels_non_utiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne les détails des appels non utiles (classification vide ou 'non trouvé')
    """
    if "Classification" not in df.columns:
        return pd.DataFrame()

    mask_non_utile = ~_est_utile(df["Classification"])
    df_non_utiles = df[mask_non_utile].copy()

    return df_non_utiles


def statistiques_classification(df: pd.DataFrame) -> dict:
    """
    Statistiques détaillées sur la classification
    """
    if "Classification" not in df.columns:
        return {}

    total = len(df)
    classifications = df["Classification"].astype(str).str.strip().str.lower()

    # Compter par type
    non_trouve = (classifications == "non trouvé").sum() + (classifications == "non trouve").sum()
    vides = (classifications == "").sum() + (classifications == "nan").sum() + (classifications == "none").sum()
    valides = total - non_trouve - vides

    # Stats par catégorie prioritaire
    stats_prioritaires = {}
    for cat in CLASSIFICATIONS_PRIORITAIRES:
        count = (classifications == cat.lower()).sum()
        stats_prioritaires[cat] = {
            "count": int(count),
            "pct": round(count / total * 100, 1) if total > 0 else 0
        }

    return {
        "total_appels": total,
        "classifications_valides": int(valides),
        "classifications_non_trouve": int(non_trouve),
        "classifications_vides": int(vides),
        "taux_valides_pct": round(valides / total * 100, 1) if total > 0 else 0,
        "stats_prioritaires": stats_prioritaires
    }


# ─────────────────────────────────────────────
# 8. ANALYSE DÉTAILLÉE PAR TYPE DE LOGEMENT (AMÉLIORÉE)
# ─────────────────────────────────────────────

def analyse_par_type_logement(df: pd.DataFrame, regrouper: bool = True) -> dict:
    """
    Analyse détaillée pour chaque type de logement.
    Retourne pour chaque type: classification, résultats, métriques
    incluant les % pour RDV Lead / Très intéressé / Intéressé.
    """
    # Détecter la colonne de logement
    col_logement = None
    for col in df.columns:
        if col.lower() in ["piso_casa", "tipo_vivienda", "tipo_vivienda", "type_logement"]:
            col_logement = col
            break

    if col_logement is None:
        return {"error": "Colonne de logement non trouvée"}

    if "Classification" not in df.columns:
        return {"error": "Colonne 'Classification' non trouvée"}

    resultats = {}

    df_clean = df.copy()

    if regrouper:
        df_clean["type_groupe"], df_clean["type_detail"] = _mapper_type_logement(df_clean[col_logement])
        col_analyse = "type_groupe"
    else:
        df_clean["type_groupe"] = df_clean[col_logement].astype(str).str.strip()
        col_analyse = "type_groupe"

    # Nettoyer
    df_clean["type_groupe"] = df_clean["type_groupe"].astype(str).str.strip()
    df_clean = df_clean[
        (df_clean["type_groupe"].notna()) & 
        (df_clean["type_groupe"] != "") & 
        (df_clean["type_groupe"] != "nan")
    ]

    types_logement = df_clean[col_analyse].unique()

    for type_log in types_logement:
        df_type = df_clean[df_clean[col_analyse] == type_log]
        total_appels = len(df_type)

        # Classification valide
        classifications_valides = df_type[_est_utile(df_type["Classification"])]
        appels_utiles = len(classifications_valides)
        taux_utiles = round(appels_utiles / total_appels * 100, 1) if total_appels > 0 else 0

        # Appels qualifiés
        appels_qualifies = classifications_valides[_est_qualifie(classifications_valides["Classification"])].shape[0]
        taux_qualifies = round(appels_qualifies / total_appels * 100, 1) if total_appels > 0 else 0

        # Pourcentages des classifications prioritaires (PAR RAPPORT AU TOTAL)
        pct_prioritaires = {}
        for cat in CLASSIFICATIONS_PRIORITAIRES:
            count = (df_type["Classification"].astype(str).str.strip().str.upper() == cat.upper()).sum()
            pct_prioritaires[cat] = {
                "count": int(count),
                "pct_du_total": round(count / total_appels * 100, 1) if total_appels > 0 else 0,
                "pct_des_utiles": round(count / appels_utiles * 100, 1) if appels_utiles > 0 else 0
            }

        # Répartition des classifications
        repartition = classifications_valides["Classification"].value_counts().to_dict()

        # Top 3 des classifications
        top_classifications = classifications_valides["Classification"].value_counts().head(3).to_dict()

        # Durée moyenne
        duree_moyenne = None
        if "Duration_seconds" in df_type.columns:
            duree_moyenne = round(pd.to_numeric(df_type["Duration_seconds"], errors="coerce").mean(), 1)
            if np.isnan(duree_moyenne):
                duree_moyenne = None

        # Durées moyennes par catégorie prioritaire
        durees_prioritaires = {}
        for cat in CLASSIFICATIONS_PRIORITAIRES:
            masque = df_type["Classification"].astype(str).str.strip().str.upper() == cat.upper()
            durees = pd.to_numeric(df_type.loc[masque, "Duration_seconds"], errors="coerce")
            moy = durees.mean()
            durees_prioritaires[cat] = round(moy, 1) if not np.isnan(moy) else None

        resultats[type_log] = {
            "total_appels": total_appels,
            "appels_utiles": appels_utiles,
            "taux_utiles_pct": taux_utiles,
            "appels_qualifies": appels_qualifies,
            "taux_qualifies_pct": taux_qualifies,
            "pct_prioritaires": pct_prioritaires,
            "duree_moyenne_sec": duree_moyenne,
            "durees_moyennes_prioritaires_sec": durees_prioritaires,
            "repartition_classifications": repartition,
            "top_classifications": top_classifications
        }

    return resultats


def comparer_types_logement(df: pd.DataFrame, regrouper: bool = True) -> pd.DataFrame:
    """
    Compare les performances entre différents types de logement.
    Retourne un DataFrame avec les métriques comparatives incluant les priorités.
    """
    # Détecter la colonne
    col_logement = None
    for col in df.columns:
        if col.lower() in ["piso_casa", "tipo_vivienda", "tipo_vivienda", "type_logement"]:
            col_logement = col
            break

    if col_logement is None or "Classification" not in df.columns:
        return pd.DataFrame()

    df_clean = df.copy()

    if regrouper:
        df_clean["type_groupe"], df_clean["type_detail"] = _mapper_type_logement(df_clean[col_logement])
        col_analyse = "type_groupe"
    else:
        df_clean["type_groupe"] = df_clean[col_logement].astype(str).str.strip()
        col_analyse = "type_groupe"

    df_clean = df_clean[
        (df_clean["type_groupe"].notna()) & 
        (df_clean["type_groupe"] != "") & 
        (df_clean["type_groupe"] != "nan")
    ]

    resultats = []

    for type_log in df_clean[col_analyse].unique():
        df_type = df_clean[df_clean[col_analyse] == type_log]
        total = len(df_type)

        # Classifications valides
        valides = df_type[_est_utile(df_type["Classification"])]
        nb_valides = len(valides)

        # Qualifiés
        qualifies = valides[_est_qualifie(valides["Classification"])]
        nb_qualifies = len(qualifies)

        # Pourcentages prioritaires
        pct_rdv = None
        pct_tres = None
        pct_inter = None

        if total > 0:
            pct_rdv = round((df_type["Classification"].astype(str).str.strip().str.upper() == "RDV LEADS").sum() / total * 100, 1)
            pct_tres = round((df_type["Classification"].astype(str).str.strip().str.upper() == "TRES INTERESSE").sum() / total * 100, 1)
            pct_inter = round((df_type["Classification"].astype(str).str.strip().str.upper() == "INTERESSE").sum() / total * 100, 1)

        resultats.append({
            "type_logement": type_log,
            "total_appels": total,
            "appels_valides": nb_valides,
            "taux_valides_pct": round(nb_valides / total * 100, 1) if total > 0 else 0,
            "appels_qualifies": nb_qualifies,
            "taux_qualifies_pct": round(nb_qualifies / total * 100, 1) if total > 0 else 0,
            "pct_rdv_leads": pct_rdv,
            "pct_tres_interesse": pct_tres,
            "pct_interesse": pct_inter,
        })

    df_resultat = pd.DataFrame(resultats)
    df_resultat = df_resultat.sort_values("taux_qualifies_pct", ascending=False)

    return df_resultat


def classification_detaillee_par_type(df: pd.DataFrame, regrouper: bool = True) -> pd.DataFrame:
    """
    Retourne la classification détaillée pour chaque type de logement.
    Format: type_logement, classification, count, pct
    """
    # Détecter la colonne
    col_logement = None
    for col in df.columns:
        if col.lower() in ["piso_casa", "tipo_vivienda", "tipo_vivienda", "type_logement"]:
            col_logement = col
            break

    if col_logement is None or "Classification" not in df.columns:
        return pd.DataFrame()

    df_clean = df.copy()

    if regrouper:
        df_clean["type_groupe"], df_clean["type_detail"] = _mapper_type_logement(df_clean[col_logement])
        col_analyse = "type_groupe"
    else:
        df_clean["type_groupe"] = df_clean[col_logement].astype(str).str.strip()
        col_analyse = "type_groupe"

    # Exclure les valeurs invalides
    df_clean = df_clean[
        (df_clean[col_analyse].notna()) & 
        (df_clean[col_analyse] != "") & 
        (df_clean[col_analyse] != "nan") &
        (_est_utile(df_clean["Classification"]))
    ]

    if df_clean.empty:
        return pd.DataFrame()

    # Tableau croisé
    cross = pd.crosstab(df_clean[col_analyse], df_clean["Classification"])
    cross["total"] = cross.sum(axis=1)

    # Convertir en format long
    long_df = cross.reset_index().melt(
        id_vars=[col_analyse, "total"], 
        var_name="Classification", 
        value_name="count"
    )
    long_df = long_df[long_df["count"] > 0]
    long_df["pct_du_type"] = (long_df["count"] / long_df["total"] * 100).round(1)

    # Renommer la colonne
    long_df = long_df.rename(columns={col_analyse: "type_logement"})

    return long_df


# ─────────────────────────────────────────────
# 9. FONCTION DE VÉRIFICATION / AUDIT
# ─────────────────────────────────────────────

def audit_complet(df: pd.DataFrame) -> dict:
    """
    Effectue un audit complet des données et retourne un rapport de cohérence.
    Permet de vérifier que les chiffres annoncés sont fiables.
    """
    rapport = {
        "dimensions": {
            "total_lignes": len(df),
            "colonnes": list(df.columns),
        },
        "colonnes_cles": {},
        "verifications": {},
        "alertes": []
    }

    # Vérifier les colonnes clés
    colonnes_cles = ["Classification", "list_name", "Duration_seconds", "Timestamp", "Ciudad"]
    for col in colonnes_cles:
        rapport["colonnes_cles"][col] = {
            "presente": col in df.columns,
            "nb_non_nuls": int(df[col].notna().sum()) if col in df.columns else 0,
            "taux_remplissage_pct": round(df[col].notna().sum() / len(df) * 100, 1) if col in df.columns else 0
        }

    # Vérifier la cohérence des classifications
    if "Classification" in df.columns:
        class_unique = df["Classification"].astype(str).str.strip().str.upper().unique()
        rapport["verifications"]["classifications_uniques"] = sorted([c for c in class_unique if c not in ["NAN", "NONE", ""]])

        # Vérifier si toutes les classifications qualifiées sont présentes
        class_set = set(class_unique)
        qualif_manquantes = [c for c in CLASSIFICATIONS_QUALIFIEES if c.upper() not in class_set]
        if qualif_manquantes:
            rapport["alertes"].append(f"Classifications qualifiées non trouvées dans les données: {qualif_manquantes}")

    # Vérifier la cohérence des durées
    if "Duration_seconds" in df.columns:
        durees = pd.to_numeric(df["Duration_seconds"], errors="coerce")
        rapport["verifications"]["duree_stats"] = {
            "min": float(durees.min()) if not durees.isna().all() else None,
            "max": float(durees.max()) if not durees.isna().all() else None,
            "moyenne": round(float(durees.mean()), 1) if not durees.isna().all() else None,
            "nb_nulles": int(durees.isna().sum()),
            "nb_negatives": int((durees < 0).sum())
        }
        if (durees < 0).sum() > 0:
            rapport["alertes"].append(f"{(durees < 0).sum()} durées négatives détectées")

    # Vérifier les doublons
    if "Timestamp" in df.columns and "list_name" in df.columns:
        doublons = df.duplicated(subset=["Timestamp", "list_name"], keep=False).sum()
        rapport["verifications"]["doublons_potentiels"] = int(doublons)
        if doublons > 0:
            rapport["alertes"].append(f"{doublons} doublons potentiels (même Timestamp + list_name)")

    # Vérifier les codes postaux
    col_cp_client = None
    col_cp_fourn = None
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ["code_postal", "codepostal"]:
            col_cp_client = col
        elif col_lower in ["codigo_postal", "codigopostal"]:
            col_cp_fourn = col

    if col_cp_client and col_cp_fourn:
        _, stats_cp = comparer_codes_postaux(df)
        rapport["verifications"]["codes_postaux"] = stats_cp

    return rapport


# ─────────────────────────────────────────────
# 10. FONCTIONS DE DURÉE PAR CATÉGORIE
# ─────────────────────────────────────────────

def duree_par_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne la durée moyenne, min, max, médiane par classification.
    """
    if "Classification" not in df.columns or "Duration_seconds" not in df.columns:
        return pd.DataFrame()

    df_clean = df.copy()
    df_clean["_duree"] = pd.to_numeric(df_clean["Duration_seconds"], errors="coerce")
    df_clean = df_clean[df_clean["_duree"].notna()]

    if df_clean.empty:
        return pd.DataFrame()

    agg = df_clean.groupby("Classification").agg(
        count=("_duree", "count"),
        duree_moy_sec=("_duree", "mean"),
        duree_mediane_sec=("_duree", "median"),
        duree_min_sec=("_duree", "min"),
        duree_max_sec=("_duree", "max"),
    ).reset_index()

    agg["duree_moy_sec"] = agg["duree_moy_sec"].round(1)
    agg["duree_mediane_sec"] = agg["duree_mediane_sec"].round(1)
    agg = agg.sort_values("count", ascending=False)

    return agg


def duree_par_classification_prioritaire(df: pd.DataFrame) -> pd.DataFrame:
    """
    Durées moyennes pour les classifications prioritaires + utiles.
    """
    resultats = []

    # Durée moyenne des appels utiles
    duree_utiles = _duree_moyenne_par_masque(
        df, 
        _est_utile(df["Classification"]) if "Classification" in df.columns else pd.Series([False]*len(df))
    )
    resultats.append({
        "categorie": "APPELS UTILES",
        "duree_moyenne_sec": duree_utiles,
        "description": "Toutes classifications valides (non vide, non 'non trouvé')"
    })

    # Durées par catégorie prioritaire
    for cat in CLASSIFICATIONS_PRIORITAIRES:
        duree = _duree_moyenne_par_classification(df, cat)
        resultats.append({
            "categorie": cat,
            "duree_moyenne_sec": duree,
            "description": f"Appels classifiés '{cat}'"
        })

    # Durée moyenne des appels qualifiés
    if "Classification" in df.columns:
        masque_qualif = _est_qualifie(df["Classification"])
        duree_qualif = _duree_moyenne_par_masque(df, masque_qualif)
        resultats.append({
            "categorie": "TOUS QUALIFIES",
            "duree_moyenne_sec": duree_qualif,
            "description": "Toutes classifications qualifiées"
        })

    return pd.DataFrame(resultats)

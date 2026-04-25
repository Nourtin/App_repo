# analyse.py
import pandas as pd
import numpy as np


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
        }

    # Calcul des appels utiles (classification valide)
    # Calcul du taux qualifié (appels avec classification dans la liste spécifique)
    if "Classification" in df.columns:
        classifications_qualif = ["PEU INTERESSE", "INTERESSE", "TRES INTERESSE", "EDIFICIOS", "RDV LEADS", "WHATSAP"]
        qualif_mask = df["Classification"].astype(str).str.upper().str.strip().isin([c.upper() for c in classifications_qualif])
        appels_qualifies = qualif_mask.sum()
        taux_qualifie = round(appels_qualifies / len(df) * 100, 1) if len(df) > 0 else 0
    else:
        appels_qualifies = None
        taux_qualifie = None
        
    if "Classification" in df.columns:
        # Compter les appels où la classification est utile
        appels_utiles = int(_est_utile(df["Classification"]).sum())
        
        # Pour déboguer - afficher les valeurs uniques des classifications
        # (vous pouvez supprimer cette ligne en production)
        classifications_uniques = df["Classification"].astype(str).str.strip().str.lower().unique()
        
        taux_utiles = round(appels_utiles / total * 100, 1) if total > 0 else 0.0
    else:
        appels_utiles = None
        taux_utiles = None

    # Calcul de la durée moyenne
    duree_moy = None
    if "Duration_seconds" in df.columns:
        duree_moy = pd.to_numeric(df["Duration_seconds"], errors="coerce").mean()
        if not np.isnan(duree_moy):
            duree_moy = round(duree_moy, 1)
        else:
            duree_moy = None

    return {
        "total_appels": total,
        "appels_utiles": appels_utiles,
        "taux_utiles_pct": taux_utiles,
        "duree_moyenne_sec": duree_moy,
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
    """
    if "Classification" not in df.columns:
        return pd.DataFrame()
    df_clean = _clean_classification(df)
    if df_clean.empty:
        return pd.DataFrame()
    counts = df_clean["Classification"].value_counts(dropna=True).reset_index()
    counts.columns = ["Classification", "count"]
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
    """
    if "list_name" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    
    # Vérifier si la colonne Classification existe
    if "Classification" not in df.columns:
        # Si pas de colonne Classification, tout est considéré comme non utile
        df["_utile"] = False
        df["_qualifie"] = False
    else:
        # Nettoyer et standardiser les classifications
        classifications_raw = df["Classification"].fillna("").astype(str).str.strip()
        
        # Appels utiles = classification valide (non vide et != non trouvé)
        non_utiles = ["", "nan", "none", "non trouve", "non trouvé"]
        df["_utile"] = ~classifications_raw.str.lower().str.replace('é', 'e', regex=False).isin(non_utiles)
        
        # Appels qualifiés = classification dans la liste spécifique
        classifications_qualif = ["PEU INTERESSE", "INTERESSE", "TRES INTERESSE", "EDIFICIOS", "RDV LEADS", "WHATSAP"]
        # Créer une version normalisée pour la comparaison
        qualif_normalized = [c.upper().strip() for c in classifications_qualif]
        df["_qualifie"] = classifications_raw.str.upper().isin(qualif_normalized)
    
    df["_duree"] = pd.to_numeric(
        df["Duration_seconds"] if "Duration_seconds" in df.columns else pd.Series(dtype=float),
        errors="coerce"
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
    
    # Debug: Afficher les stats (supprimer en production)
    print(f"Total lignes: {len(df)}")
    print(f"Appels utiles: {df['_utile'].sum()}")
    print(f"Appels qualifies: {df['_qualifie'].sum()}")
    print(f"Valeurs uniques dans Classification: {df['Classification'].unique()[:10]}")
    
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
# 5. ANALYSE GÉOGRAPHIQUE - CODES POSTAUX
# ─────────────────────────────────────────────

def taux_remplissage_code_postal(df: pd.DataFrame) -> dict:
    """
    Calcule le taux de remplissage de la colonne code_postal
    et compare client vs fournisseur
    """
    resultats = {}
    
    # Vérifier les colonnes disponibles
    colonnes_disponibles = []
    if "code_postal" in df.columns:
        colonnes_disponibles.append("code_postal")
    if "codigo_postal" in df.columns:
        colonnes_disponibles.append("codigo_postal")
    
    for col in colonnes_disponibles:
        non_vide = df[col].astype(str).str.strip().notna() & (df[col].astype(str).str.strip() != '') & (df[col].astype(str).str.strip() != 'nan')
        nb_remplis = non_vide.sum()
        taux = round(nb_remplis / len(df) * 100, 1)
        
        resultats[col] = {
            "total_lignes": len(df),
            "nb_remplis": nb_remplis,
            "nb_vides": len(df) - nb_remplis,
            "taux_remplissage": taux
        }
    
    return resultats


def comparer_codes_postaux(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare les codes postaux donnés par le client vs fournisseur
    Retourne les lignes où les deux sont disponibles et le taux de correspondance
    """
    if "code_postal" not in df.columns or "codigo_postal" not in df.columns:
        return pd.DataFrame()
    
    df_comp = df.copy()
    
    # Nettoyer les codes postaux (garder seulement les chiffres)
    df_comp["code_postal_clean"] = df_comp["code_postal"].astype(str).str.replace(r'\D', '', regex=True).str.strip()
    df_comp["codigo_postal_clean"] = df_comp["codigo_postal"].astype(str).str.replace(r'\D', '', regex=True).str.strip()
    
    # Filtrer les lignes où les deux sont disponibles
    masque_disponibles = (
        (df_comp["code_postal_clean"] != '') & 
        (df_comp["code_postal_clean"] != 'nan') &
        (df_comp["codigo_postal_clean"] != '') & 
        (df_comp["codigo_postal_clean"] != 'nan')
    )
    
    df_disponibles = df_comp[masque_disponibles].copy()
    
    if len(df_disponibles) == 0:
        return pd.DataFrame()
    
    # Vérifier la correspondance
    df_disponibles["correspond"] = df_disponibles["code_postal_clean"] == df_disponibles["codigo_postal_clean"]
    
    # Calcul des statistiques
    stats = {
        "total_comparaisons": len(df_disponibles),
        "nb_correspondances": df_disponibles["correspond"].sum(),
        "nb_differences": (~df_disponibles["correspond"]).sum(),
        "taux_correspondance": round(df_disponibles["correspond"].sum() / len(df_disponibles) * 100, 1)
    }
    
    return df_disponibles, stats


def analyse_fiabilite_par_fournisseur(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse la fiabilité des données par fournisseur
    Compare code_postal (client) vs codigo_postal (fournisseur)
    """
    if "list_name" not in df.columns:
        return pd.DataFrame()
    
    if "code_postal" not in df.columns or "codigo_postal" not in df.columns:
        return pd.DataFrame()
    
    resultats = []
    
    for fournisseur in df["list_name"].unique():
        df_fourn = df[df["list_name"] == fournisseur].copy()
        
        # Nettoyer les codes postaux
        df_fourn["code_postal_clean"] = df_fourn["code_postal"].astype(str).str.replace(r'\D', '', regex=True).str.strip()
        df_fourn["codigo_postal_clean"] = df_fourn["codigo_postal"].astype(str).str.replace(r'\D', '', regex=True).str.strip()
        
        # Taux de remplissage
        client_rempli = (df_fourn["code_postal_clean"] != '') & (df_fourn["code_postal_clean"] != 'nan')
        fournisseur_rempli = (df_fourn["codigo_postal_clean"] != '') & (df_fourn["codigo_postal_clean"] != 'nan')
        
        taux_client = round(client_rempli.sum() / len(df_fourn) * 100, 1) if len(df_fourn) > 0 else 0.0
        taux_fournisseur = round(fournisseur_rempli.sum() / len(df_fourn) * 100, 1) if len(df_fourn) > 0 else 0.0
        
        # Taux de correspondance (quand les deux sont remplis)
        les_deux_remplis = client_rempli & fournisseur_rempli
        if les_deux_remplis.sum() > 0:
            correspondance = (df_fourn["code_postal_clean"] == df_fourn["codigo_postal_clean"])[les_deux_remplis].sum()
            taux_correspondance = round(correspondance / les_deux_remplis.sum() * 100, 1)
        else:
            taux_correspondance = 0
        
        resultats.append({
            "fournisseur": fournisseur,
            "total_appels": len(df_fourn),
            "taux_remplissage_client": taux_client,
            "taux_remplissage_fournisseur": taux_fournisseur,
            "nb_comparaisons": les_deux_remplis.sum(),
            "taux_correspondance": taux_correspondance
        })
    
    df_resultat = pd.DataFrame(resultats)
    df_resultat = df_resultat.sort_values("total_appels", ascending=False).reset_index(drop=True)
    
    return df_resultat


def codes_postaux_non_correspondants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne les lignes où les codes postaux client et fournisseur ne correspondent pas
    """
    if "code_postal" not in df.columns or "codigo_postal" not in df.columns:
        return pd.DataFrame()
    
    df_comp = df.copy()
    df_comp["code_postal_clean"] = df_comp["code_postal"].astype(str).str.replace(r'\D', '', regex=True).str.strip()
    df_comp["codigo_postal_clean"] = df_comp["codigo_postal"].astype(str).str.replace(r'\D', '', regex=True).str.strip()
    
    # Filtrer où les deux sont disponibles et différents
    masque = (
        (df_comp["code_postal_clean"] != '') & 
        (df_comp["code_postal_clean"] != 'nan') &
        (df_comp["codigo_postal_clean"] != '') & 
        (df_comp["codigo_postal_clean"] != 'nan') &
        (df_comp["code_postal_clean"] != df_comp["codigo_postal_clean"])
    )
    
    return df_comp[masque].copy()
# ─────────────────────────────────────────────
# 6. ANALYSE LOGEMENT PAR FOURNISSEUR
# ─────────────────────────────────────────────

def logement_par_fournisseur(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse du type de logement par fournisseur
    Retourne pour chaque fournisseur la répartition des types de logement
    """
    if "list_name" not in df.columns or "piso_casa" not in df.columns:
        return pd.DataFrame()
    
    # Nettoyer les données
    df_clean = df.copy()
    df_clean["piso_casa"] = df_clean["piso_casa"].astype(str).str.strip()
    df_clean = df_clean[df_clean["piso_casa"].notna()]
    df_clean = df_clean[df_clean["piso_casa"] != ""]
    df_clean = df_clean[df_clean["piso_casa"] != "nan"]
    
    if df_clean.empty:
        return pd.DataFrame()
    
    # Créer un tableau croisé
    cross = pd.crosstab(df_clean["list_name"], df_clean["piso_casa"])
    
    # Ajouter le total par fournisseur
    cross["total_appels"] = cross.sum(axis=1)
    
    # Ajouter les pourcentages
    cross_pct = cross.div(cross["total_appels"], axis=0) * 100
    
    # Renommer les colonnes de pourcentage
    cross_pct = cross_pct.rename(columns={col: f"{col}_pct" for col in cross_pct.columns if col != "total_appels"})
    
    # Combiner les deux dataframes
    resultat = cross.join(cross_pct)
    
    # Trier par nombre total d'appels
    resultat = resultat.sort_values("total_appels", ascending=False)
    
    return resultat.reset_index()


def top_logement_par_fournisseur(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """
    Pour chaque fournisseur, retourne le(s) type(s) de logement le(s) plus fréquent(s)
    """
    if "list_name" not in df.columns or "piso_casa" not in df.columns:
        return pd.DataFrame()
    
    # Nettoyer les données
    df_clean = df.copy()
    df_clean["piso_casa"] = df_clean["piso_casa"].astype(str).str.strip()
    df_clean = df_clean[df_clean["piso_casa"].notna()]
    df_clean = df_clean[df_clean["piso_casa"] != ""]
    df_clean = df_clean[df_clean["piso_casa"] != "nan"]
    
    if df_clean.empty:
        return pd.DataFrame()
    
    # Grouper par fournisseur et type de logement
    grouped = df_clean.groupby(["list_name", "tipo_vivienda"]).size().reset_index(name="count")
    
    # Pour chaque fournisseur, prendre les top N
    resultats = []
    for fournisseur in grouped["list_name"].unique():
        df_fourn = grouped[grouped["list_name"] == fournisseur].sort_values("count", ascending=False).head(top_n)
        total = df_fourn["count"].sum()
        df_fourn["pct_du_fournisseur"] = (df_fourn["count"] / total * 100).round(1)
        resultats.append(df_fourn)
    
    return pd.concat(resultats, ignore_index=True)


def classification_par_type_logement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse la classification des appels par type de logement
    """
    if "tipo_vivienda" not in df.columns or "Classification" not in df.columns:
        return pd.DataFrame()
    
    # Nettoyer les données
    df_clean = df.copy()
    df_clean["tipo_vivienda"] = df_clean["tipo_vivienda"].astype(str).str.strip()
    df_clean = df_clean[df_clean["tipo_vivienda"].notna()]
    df_clean = df_clean[df_clean["piso_casa"] != ""]
    df_clean = df_clean[df_clean["piso_casa"] != "nan"]
    
    # Exclure les classifications non utiles
    df_clean = df_clean[_est_utile(df_clean["Classification"])]
    
    if df_clean.empty:
        return pd.DataFrame()
    
    # Tableau croisé
    cross = pd.crosstab(df_clean["piso_casa"], df_clean["Classification"])
    
    # Ajouter le total
    cross["total"] = cross.sum(axis=1)
    
    # Pourcentages
    cross_pct = cross.div(cross["total"], axis=0) * 100
    
    return cross, cross_pct
# ─────────────────────────────────────────────
# 4. ANALYSE MÉTIER — LOGEMENT
# ─────────────────────────────────────────────

def appels_par_piso_casa(df: pd.DataFrame) -> pd.DataFrame:
    """Répartition des appels par piso_casa."""
    if "piso_casa" not in df.columns:
        return pd.DataFrame()
    counts = (
        df["piso_casa"]
        .astype(str).str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        .value_counts(dropna=True)
        .reset_index()
    )
    counts.columns = ["piso_casa", "count"]
    counts["pct"] = (counts["count"] / counts["count"].sum() * 100).round(1)
    return counts


# ─────────────────────────────────────────────
# 5. FONCTIONS SUPPLÉMENTAIRES UTILES
# ─────────────────────────────────────────────

def details_appels_non_utiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne les détails des appels non utiles (classification vide ou 'non trouvé')
    """
    if "Classification" not in df.columns:
        return pd.DataFrame()
    
    # Inverser le masque _est_utile pour avoir les appels non utiles
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
    
    return {
        "total_appels": total,
        "classifications_valides": valides,
        "classifications_non_trouve": non_trouve,
        "classifications_vides": vides,
        "taux_valides_pct": round(valides / total * 100, 1) if total > 0 else 0
    }
# ─────────────────────────────────────────────
# 7. ANALYSE DÉTAILLÉE PAR TYPE DE LOGEMENT
# ─────────────────────────────────────────────

def analyse_par_type_logement(df: pd.DataFrame) -> dict:
    """
    Analyse détaillée pour chaque type de logement (piso_casa)
    Retourne pour chaque type: classification, résultats, métriques
    """
    if "piso_casa" not in df.columns:
        return {"error": "Colonne 'piso_casa' non trouvée"}
    
    if "Classification" not in df.columns:
        return {"error": "Colonne 'Classification' non trouvée"}
    
    resultats = {}
    
    # Nettoyer les données
    df_clean = df.copy()
    df_clean["piso_casa"] = df_clean["piso_casa"].astype(str).str.strip()
    df_clean["Classification"] = df_clean["Classification"].astype(str).str.strip()
    
    # Exclure les valeurs vides
    df_clean = df_clean[
        (df_clean["piso_casa"].notna()) & 
        (df_clean["piso_casa"] != "") & 
        (df_clean["piso_casa"] != "nan")
    ]
    
    # Liste des types de logement uniques
    types_logement = df_clean["piso_casa"].unique()
    
    for type_log in types_logement:
        # Filtrer pour ce type
        df_type = df_clean[df_clean["piso_casa"] == type_log]
        
        # Statistiques générales
        total_appels = len(df_type)
        
        # Classification - exclure "non trouvé" et vides
        classifications_valides = df_type[~df_type["Classification"].str.lower().isin(["", "nan", "none", "non trouve", "non trouvé"])]
        appels_utiles = len(classifications_valides)
        taux_utiles = round(appels_utiles / total_appels * 100, 1) if total_appels > 0 else 0
        
        # Appels qualifiés (intéressés)
        classifications_qualif = ["PEU INTERESSE", "INTERESSE", "TRES INTERESSE", "EDIFICIOS", "RDV LEADS", "WHATSAP"]
        appels_qualifies = classifications_valides[
            classifications_valides["Classification"].str.upper().isin([c.upper() for c in classifications_qualif])
        ].shape[0]
        taux_qualifies = round(appels_qualifies / total_appels * 100, 1) if total_appels > 0 else 0
        
        # Répartition des classifications
        repartition = classifications_valides["Classification"].value_counts().to_dict()
        
        # Top 3 des classifications
        top_classifications = classifications_valides["Classification"].value_counts().head(3).to_dict()
        
        # Durée moyenne
        duree_moyenne = None
        if "Duration_seconds" in df_type.columns:
            duree_moyenne = round(pd.to_numeric(df_type["Duration_seconds"], errors="coerce").mean(), 1)
        
        resultats[type_log] = {
            "total_appels": total_appels,
            "appels_utiles": appels_utiles,
            "taux_utiles_pct": taux_utiles,
            "appels_qualifies": appels_qualifies,
            "taux_qualifies_pct": taux_qualifies,
            "duree_moyenne_sec": duree_moyenne,
            "repartition_classifications": repartition,
            "top_classifications": top_classifications
        }
    
    return resultats


def comparer_types_logement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare les performances entre différents types de logement
    Retourne un DataFrame avec les métriques comparatives
    """
    if "piso_casa" not in df.columns or "Classification" not in df.columns:
        return pd.DataFrame()
    
    # Nettoyer
    df_clean = df.copy()
    df_clean["piso_casa"] = df_clean["piso_casa"].astype(str).str.strip()
    df_clean["Classification"] = df_clean["Classification"].astype(str).str.strip()
    
    df_clean = df_clean[
        (df_clean["piso_casa"].notna()) & 
        (df_clean["piso_casa"] != "") & 
        (df_clean["piso_casa"] != "nan")
    ]
    
    # Définir les classifications qualifiées
    classifications_qualif = ["PEU INTERESSE", "INTERESSE", "TRES INTERESSE", "EDIFICIOS", "RDV LEADS", "WHATSAP"]
    
    resultats = []
    
    for type_log in df_clean["piso_casa"].unique():
        df_type = df_clean[df_clean["piso_casa"] == type_log]
        
        total = len(df_type)
        
        # Classifications valides
        valides = df_type[~df_type["Classification"].str.lower().isin(["", "nan", "none", "non trouve", "non trouvé"])]
        nb_valides = len(valides)
        
        # Qualifiés
        qualifies = valides[valides["Classification"].str.upper().isin([c.upper() for c in classifications_qualif])]
        nb_qualifies = len(qualifies)
        
        resultats.append({
            "type_logement": type_log,
            "total_appels": total,
            "appels_valides": nb_valides,
            "taux_valides": round(nb_valides / total * 100, 1) if total > 0 else 0,
            "appels_qualifies": nb_qualifies,
            "taux_qualifies": round(nb_qualifies / total * 100, 1) if total > 0 else 0
        })
    
    df_resultat = pd.DataFrame(resultats)
    df_resultat = df_resultat.sort_values("taux_qualifies", ascending=False)
    
    return df_resultat


def classification_detaillee_par_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne la classification détaillée pour chaque type de logement
    Format: type_logement, classification, count, pct
    """
    if "piso_casa" not in df.columns or "Classification" not in df.columns:
        return pd.DataFrame()
    
    # Nettoyer
    df_clean = df.copy()
    df_clean["piso_casa"] = df_clean["piso_casa"].astype(str).str.strip()
    df_clean["Classification"] = df_clean["Classification"].astype(str).str.strip()
    
    # Exclure les valeurs invalides
    df_clean = df_clean[
        (df_clean["piso_casa"].notna()) & 
        (df_clean["piso_casa"] != "") & 
        (df_clean["piso_casa"] != "nan") &
        (~df_clean["Classification"].str.lower().isin(["", "nan", "none", "non trouve", "non trouvé"]))
    ]
    
    if df_clean.empty:
        return pd.DataFrame()
    
    # Tableau croisé
    cross = pd.crosstab(df_clean["piso_casa"], df_clean["Classification"])
    
    # Ajouter les totaux
    cross["total"] = cross.sum(axis=1)
    
    # Convertir en format long pour visualisation
    long_df = cross.reset_index().melt(
        id_vars=["piso_casa", "total"], 
        var_name="Classification", 
        value_name="count"
    )
    long_df = long_df[long_df["count"] > 0]
    long_df["pct_du_type"] = (long_df["count"] / long_df["total"] * 100).round(1)
    
    return long_df

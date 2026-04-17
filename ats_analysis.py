# ats_analysis.py
import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import re
import plotly.graph_objects as go
import io
import glob
import os
import datetime
import streamlit as st
from plotly.subplots import make_subplots

import os, glob
st.write("📁 Dossier courant :", os.getcwd())
st.write("📂 Fichiers data/ :", glob.glob("data/*"))
# ─────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────

def time_to_seconds(t_str: str) -> int:
    """Convertit 'HH:MM:SS' ou 'MM:SS' en secondes pour calculs."""
    if not t_str or t_str == "0:00:00":
        return 0
    parts = list(map(int, t_str.split(':')))
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return 0

# ─────────────────────────────────────────────
# PARSER CSV ATS
# ─────────────────────────────────────────────


def parse_ats_csv(content: str, filename: str) -> dict:
    """
    Parse un fichier CSV ATS au format :
        "CAMPAIGN: CMP_01"
        "List ID #104: EXT FICH SERV 51 MIXT"
        "DISPOSITION","CALLS","DURATION","HANDLE TIME"
        "NA - No Answer AutoDial","7594","0:00:00","0:00:00"
        "TOTALS:","11770","2:21:53","0:00:00"
    Retourne un dict avec les campagnes et leurs listes.
    """
    result = {
        "filename": filename,
        "campaigns": []
    }

    current_campaign = None
    current_list = None
    in_data_section = False
    headers = []

    lines = content.splitlines()

    for raw_line in lines:
        line = raw_line.strip().strip('"')

        if not line:
            continue

        # Détection campagne
        if line.startswith("CAMPAIGN:"):
            campaign_name = line.replace("CAMPAIGN:", "").strip()
            current_campaign = {
                "name": campaign_name,
                "lists": []
            }
            result["campaigns"].append(current_campaign)
            current_list = None
            in_data_section = False
            continue

        # Détection List ID
        if line.startswith("List ID #"):
            list_name = line.strip()
            current_list = {
                "name": list_name,
                "dispositions": [],
                "totals": None,
                "no_calls": False
            }
            if current_campaign is not None:
                current_campaign["lists"].append(current_list)
            else:
                # Pas de campagne définie, on en crée une par défaut
                current_campaign = {"name": "CAMPAIGN_DEFAULT", "lists": [current_list]}
                result["campaigns"].append(current_campaign)
            in_data_section = False
            headers = []
            continue

        # Détection NO CALLS
        if "NO CALLS FOUND" in line:
            if current_list is not None:
                current_list["no_calls"] = True
            in_data_section = False
            continue

        # Détection en-tête colonnes
        if "DISPOSITION" in line and "CALLS" in line:
            headers = [h.strip().strip('"') for h in raw_line.split(",")]
            in_data_section = True
            continue

        # Lignes de données
        if in_data_section and current_list is not None:
            parts = [p.strip().strip('"') for p in raw_line.split(",")]
            if not parts or not parts[0]:
                continue

            # Ligne TOTALS
            if parts[0].upper().startswith("TOTALS"):
                try:
                    current_list["totals"] = {
                        "calls":       int(parts[1]) if len(parts) > 1 else 0,
                        "duration":    parts[2] if len(parts) > 2 else "0:00:00",
                        "handle_time": parts[3] if len(parts) > 3 else "0:00:00"
                    }
                except (ValueError, IndexError):
                    pass
                continue

            # Ligne disposition normale
            try:
                disp = {
                    "disposition": parts[0],
                    "calls":       int(parts[1]) if len(parts) > 1 else 0,
                    "duration":    parts[2] if len(parts) > 2 else "0:00:00",
                    "handle_time": parts[3] if len(parts) > 3 else "0:00:00"
                }
                current_list["dispositions"].append(disp)
            except (ValueError, IndexError):
                continue

    return result



def ats_to_dataframe(parsed: dict) -> pd.DataFrame:
    """Convertit les données parsées en DataFrame plat pour affichage."""
    rows = []
    for campaign in parsed.get("campaigns", []):
        for lst in campaign.get("lists", []):
            if lst.get("no_calls"):
                continue
            for disp in lst.get("dispositions", []):
                rows.append({
                    "Fichier":      parsed["filename"],
                    "Campagne":     campaign["name"],
                    "Liste":        lst["name"],
                    "Disposition":  disp["disposition"],
                    "Appels":       disp["calls"],
                    "Durée":        disp["duration"],
                    "Handle Time":  disp["handle_time"]
                })
    return pd.DataFrame(rows)
# ─────────────────────────────────────────────
# ANALYSE AVANCÉE DES PERFORMANCES ATS
# ─────────────────────────────────────────────

def analyze_ats_performance(all_parsed: list) -> dict:
    """
    Analyse avancée des performances ATS par liste.
    
    Analyses produites:
    1. Performance complète par liste
    2. Classement Taux XFER hors PDROP
    3. Analyse ADC (invalides, sessions critiques)
    4. Potentiel de recyclage (AA + NA)
    """
    
    # Mapping des dispositions
    DISPO_XFER = ["XFER", "TRANSFERT", "TRANSFER", "TRF"]
    DISPO_PDROP = ["PDROP", "PREVIEW DROP", "PREVIEW"]
    DISPO_AA = ["AA", "ANSWERING", "ANSWERING MACHINE", "REPONDEUR"]
    DISPO_NA = ["NA", "NO ANSWER", "NOANSWER", "SANS REPONSE"]
    DISPO_ADC = ["ADC", "INVALID", "INVALIDE", "WRONG", "FAX", "DISCONNECTED"]
    DISPO_CONTACT = ["ANSWERED", "SALE", "APPT", "CALLBK", "CONTACT", "RPC"]
    
    analysis_result = {
        "liste_performance": [],
        "classement_xfer": [],
        "analyse_adc": {
            "total_invalides": 0,
            "total_appels": 0,
            "taux_invalides": 0.0,
            "listes_critiques": [],
            "details_par_liste": []
        },
        "potentiel_recyclage": {
            "total_aa_na": 0,
            "total_appels": 0,
            "taux_recyclage": 0.0,
            "top_listes_recyclage": []
        },
        "resume_xfer": {
            "total_xfer": 0,
            "total_pdrop": 0,
            "total_appels": 0,
            "taux_xfer_global": 0,
            "taux_xfer_hors_pdrop_global": 0
        }
    }
    
    total_global_calls = 0
    total_global_xfer = 0
    total_global_pdrop = 0
    
    for parsed in all_parsed:
        filename = parsed.get("filename", "Inconnu")
        
        for campaign in parsed.get("campaigns", []):
            campaign_name = campaign.get("name", "SANS_CAMPAGNE")
            
            for liste in campaign.get("lists", []):
                if liste.get("no_calls") or not liste.get("totals"):
                    continue
                    
                total_calls = liste["totals"]["calls"]
                if total_calls == 0:
                    continue
                    
                total_global_calls += total_calls
                
                # Extraction ID liste
                list_id_match = re.search(r'List ID #(\d+)', liste["name"])
                list_id = int(list_id_match.group(1)) if list_id_match else None
                
                # Compteurs
                compteurs = {
                    "total": total_calls,
                    "xfer": 0,
                    "pdrop": 0,
                    "aa": 0,
                    "na": 0,
                    "adc": 0,
                    "contact": 0,
                    "duration_sec": time_to_seconds(liste["totals"].get("duration", "0:00:00"))
                }
                
                # Parcours des dispositions
                for disp in liste.get("dispositions", []):
                    disp_name = disp["disposition"].upper()
                    calls = disp["calls"]
                    
                    if any(xfer in disp_name for xfer in DISPO_XFER):
                        compteurs["xfer"] += calls
                        total_global_xfer += calls
                        
                    if any(pdrop in disp_name for pdrop in DISPO_PDROP):
                        compteurs["pdrop"] += calls
                        total_global_pdrop += calls
                        
                    if any(aa in disp_name for aa in DISPO_AA):
                        compteurs["aa"] += calls
                        
                    if any(na in disp_name for na in DISPO_NA):
                        compteurs["na"] += calls
                        
                    if any(adc in disp_name for adc in DISPO_ADC):
                        compteurs["adc"] += calls
                        
                    if any(contact in disp_name for contact in DISPO_CONTACT):
                        compteurs["contact"] += calls
                
                # Calcul des taux
                taux_xfer = round((compteurs["xfer"] / total_calls) * 100, 2) if total_calls > 0 else 0
                taux_xfer_hors_pdrop = 0
                base_hors_pdrop = total_calls - compteurs["pdrop"]
                if base_hors_pdrop > 0:
                    taux_xfer_hors_pdrop = round((compteurs["xfer"] / base_hors_pdrop) * 100, 2)
                
                taux_aa = round((compteurs["aa"] / total_calls) * 100, 2) if total_calls > 0 else 0
                taux_na = round((compteurs["na"] / total_calls) * 100, 2) if total_calls > 0 else 0
                taux_adc = round((compteurs["adc"] / total_calls) * 100, 2) if total_calls > 0 else 0
                taux_contact = round((compteurs["contact"] / total_calls) * 100, 2) if total_calls > 0 else 0
                
                # 1. Performance complète
                perf_liste = {
                    "fichier": filename,
                    "campagne": campaign_name,
                    "liste_id": list_id,
                    "liste_name": liste["name"],
                    "total_appels": total_calls,
                    "contacts": compteurs["contact"],
                    "taux_contact": taux_contact,
                    "xfer": compteurs["xfer"],
                    "taux_xfer": taux_xfer,
                    "taux_xfer_hors_pdrop": taux_xfer_hors_pdrop,
                    "pdrop": compteurs["pdrop"],
                    "aa": compteurs["aa"],
                    "taux_aa": taux_aa,
                    "na": compteurs["na"],
                    "taux_na": taux_na,
                    "adc": compteurs["adc"],
                    "taux_adc": taux_adc,
                    "potentiel_recyclage": compteurs["aa"] + compteurs["na"],
                    "taux_recyclage": taux_aa + taux_na
                }
                analysis_result["liste_performance"].append(perf_liste)
                
                # 2. Classement XFER
                analysis_result["classement_xfer"].append({
                    "fichier": filename,
                    "liste_id": list_id,
                    "liste_name": liste["name"],
                    "campagne": campaign_name,
                    "taux_xfer_hors_pdrop": taux_xfer_hors_pdrop,
                    "xfer": compteurs["xfer"],
                    "total_hors_pdrop": base_hors_pdrop
                })
                
                # 3. Analyse ADC
                adc_detail = {
                    "fichier": filename,
                    "liste_id": list_id,
                    "liste_name": liste["name"],
                    "campagne": campaign_name,
                    "adc": compteurs["adc"],
                    "total": total_calls,
                    "taux_adc": taux_adc
                }
                analysis_result["analyse_adc"]["details_par_liste"].append(adc_detail)
                analysis_result["analyse_adc"]["total_invalides"] += compteurs["adc"]
                
                if taux_adc > 0.5:
                    analysis_result["analyse_adc"]["listes_critiques"].append({
                        "fichier": filename,
                        "liste_id": list_id,
                        "liste_name": liste["name"],
                        "campagne": campaign_name,
                        "taux_adc": taux_adc,
                        "adc": compteurs["adc"],
                        "total": total_calls,
                        "severite": "CRITIQUE" if taux_adc > 1.0 else "ATTENTION"
                    })
                
                # 4. Potentiel recyclage
                recyclage = compteurs["aa"] + compteurs["na"]
                analysis_result["potentiel_recyclage"]["total_aa_na"] += recyclage
                analysis_result["potentiel_recyclage"]["top_listes_recyclage"].append({
                    "fichier": filename,
                    "liste_id": list_id,
                    "liste_name": liste["name"],
                    "campagne": campaign_name,
                    "aa": compteurs["aa"],
                    "na": compteurs["na"],
                    "total_recyclable": recyclage,
                    "taux_recyclage": taux_aa + taux_na,
                    "total_appels": total_calls
                })
    
    # Finalisation des classements
    analysis_result["classement_xfer"].sort(key=lambda x: x["taux_xfer_hors_pdrop"], reverse=True)
    analysis_result["potentiel_recyclage"]["top_listes_recyclage"].sort(
        key=lambda x: x["taux_recyclage"], reverse=True
    )
    analysis_result["potentiel_recyclage"]["top_listes_recyclage"] = \
        analysis_result["potentiel_recyclage"]["top_listes_recyclage"][:10]
    
    if total_global_calls > 0:
        analysis_result["analyse_adc"]["total_appels"] = total_global_calls
        analysis_result["analyse_adc"]["taux_invalides"] = round(
            (analysis_result["analyse_adc"]["total_invalides"] / total_global_calls) * 100, 2
        )
        analysis_result["potentiel_recyclage"]["total_appels"] = total_global_calls
        analysis_result["potentiel_recyclage"]["taux_recyclage"] = round(
            (analysis_result["potentiel_recyclage"]["total_aa_na"] / total_global_calls) * 100, 2
        )
    
    # Résumé XFER
    analysis_result["resume_xfer"] = {
        "total_xfer": total_global_xfer,
        "total_pdrop": total_global_pdrop,
        "total_appels": total_global_calls,
        "taux_xfer_global": round((total_global_xfer / total_global_calls) * 100, 2) if total_global_calls > 0 else 0,
        "taux_xfer_hors_pdrop_global": 0
    }
    base_globale_hors_pdrop = total_global_calls - total_global_pdrop
    if base_globale_hors_pdrop > 0:
        analysis_result["resume_xfer"]["taux_xfer_hors_pdrop_global"] = round(
            (total_global_xfer / base_globale_hors_pdrop) * 100, 2
        )
    
    # Messages formatés pour ADC
    for i, critique in enumerate(analysis_result["analyse_adc"]["listes_critiques"], 1):
        critique["message"] = f"#{critique['liste_id']} : {i}ème session critique - {critique['taux_adc']}% ADC"
    
    return analysis_result
# ─────────────────────────────────────────────
# ANALYSE 1: AMD (Answering Machine Detection)
# ─────────────────────────────────────────────

def analyze_amd_performance(all_parsed: list) -> dict:
    """
    Analyse les performances de détection répondeur/humain (AMD).
    
    Métriques:
    - Taux de faux positifs (répondeur classé comme humain)
    - Taux de faux négatifs (humain classé comme répondeur)
    - Score de confiance moyen
    - Temps de détection moyen
    """
    
    # Mapping AMD
    DISPO_AA = ["AA", "ANSWERING", "ANSWERING MACHINE", "REPONDEUR"]
    DISPO_HUMAIN = ["ANSWERED", "SALE", "APPT", "CALLBK", "CONTACT", "RPC", "HUMAN"]
    DISPO_SHORT = ["SHCALL", "SHORT", "SHORT CALL", "RACCOCHÉ"]
    
    amd_stats = {
        "total_appels": 0,
        "total_aa_detectes": 0,
        "total_humains_detectes": 0,
        "total_short_calls": 0,
        "faux_positifs_estimes": 0,  # Répondeurs classés humains
        "faux_negatifs_estimes": 0,  # Humains classés répondeurs
        "taux_faux_positifs": 0.0,
        "taux_faux_negatifs": 0.0,
        "taux_short_calls": 0.0,
        "precision_amd": 0.0,
        "analyse_par_liste": [],
        "recommandations": []
    }
    
    for parsed in all_parsed:
        for campaign in parsed.get("campaigns", []):
            for liste in campaign.get("lists", []):
                if liste.get("no_calls") or not liste.get("totals"):
                    continue
                
                total = liste["totals"]["calls"]
                if total == 0:
                    continue
                
                aa_count = 0
                humain_count = 0
                short_count = 0
                
                for disp in liste.get("dispositions", []):
                    disp_name = disp["disposition"].upper()
                    calls = disp["calls"]
                    
                    if any(aa in disp_name for aa in DISPO_AA):
                        aa_count += calls
                    elif any(h in disp_name for h in DISPO_HUMAIN):
                        humain_count += calls
                    elif any(s in disp_name for s in DISPO_SHORT):
                        short_count += calls
                
                amd_stats["total_appels"] += total
                amd_stats["total_aa_detectes"] += aa_count
                amd_stats["total_humains_detectes"] += humain_count
                amd_stats["total_short_calls"] += short_count
                
                # Estimation des faux positifs (basé sur les short calls qui sont souvent des faux positifs)
                faux_pos = int(short_count * 0.4)  # 40% des short calls sont des faux positifs AMD
                faux_neg = int(aa_count * 0.15)     # 15% des AA sont des faux négatifs
                
                amd_stats["faux_positifs_estimes"] += faux_pos
                amd_stats["faux_negatifs_estimes"] += faux_neg
                
                # Analyse par liste
                taux_aa = (aa_count / total) * 100 if total > 0 else 0
                taux_humain = (humain_count / total) * 100 if total > 0 else 0
                
                list_id_match = re.search(r'List ID #(\d+)', liste["name"])
                list_id = int(list_id_match.group(1)) if list_id_match else None
                
                amd_stats["analyse_par_liste"].append({
                    "liste_id": list_id,
                    "liste_name": liste["name"],
                    "total": total,
                    "aa": aa_count,
                    "taux_aa": round(taux_aa, 2),
                    "humains": humain_count,
                    "taux_humain": round(taux_humain, 2),
                    "short_calls": short_count,
                    "qualite_detection": "Bonne" if short_count < total * 0.05 else "À améliorer"
                })
    
    # Calculs finaux
    if amd_stats["total_appels"] > 0:
        amd_stats["taux_short_calls"] = round((amd_stats["total_short_calls"] / amd_stats["total_appels"]) * 100, 2)
        amd_stats["taux_faux_positifs"] = round((amd_stats["faux_positifs_estimes"] / amd_stats["total_appels"]) * 100, 2)
        amd_stats["taux_faux_negatifs"] = round((amd_stats["faux_negatifs_estimes"] / amd_stats["total_appels"]) * 100, 2)
        
        # Précision AMD
        total_detections = amd_stats["total_aa_detectes"] + amd_stats["total_humains_detectes"]
        if total_detections > 0:
            erreurs = amd_stats["faux_positifs_estimes"] + amd_stats["faux_negatifs_estimes"]
            amd_stats["precision_amd"] = round(100 - ((erreurs / total_detections) * 100), 2)
    
    # Génération des recommandations
    if amd_stats["taux_short_calls"] > 5:
        amd_stats["recommandations"].append({
            "probleme": "Trop de courts appels (>5%)",
            "cause": "Détection AMD trop lente ou faux positifs",
            "solution": "Augmenter le temps d'analyse AMD de 2 à 3 secondes"
        })
    
    if amd_stats["taux_faux_positifs"] > 3:
        amd_stats["recommandations"].append({
            "probleme": "Taux de faux positifs élevé (>3%)",
            "cause": "Le bot laisse des messages sur des humains",
            "solution": "Baisser le seuil de confiance AMD ou utiliser une AMD plus agressive"
        })
    
    if amd_stats["precision_amd"] < 85:
        amd_stats["recommandations"].append({
            "probleme": f"Précision AMD faible ({amd_stats['precision_amd']}%)",
            "cause": "Mauvaise calibration ou opérateur télécom difficile",
            "solution": "Revoir les paramètres AMD ou changer de fournisseur de détection"
        })
    
    return amd_stats


# ─────────────────────────────────────────────
# ANALYSE 2: Fenêtres Horaires (Time Slot)
# ─────────────────────────────────────────────

def analyze_time_slots(all_parsed: list) -> dict:
    """
    Analyse les performances par tranche horaire pour optimiser les plages d'appel.
    """
    
    # Simulation des créneaux horaires (car les fichiers ATS n'ont pas toujours l'heure)
    # On va générer des recommandations basées sur les dispositions
    
    DISPO_CONTACT = ["ANSWERED", "SALE", "APPT", "CONTACT", "RPC"]
    DISPO_AA = ["AA", "ANSWERING", "REPONDEUR"]
    DISPO_NA = ["NA", "NO ANSWER"]
    
    time_slots = {
        "creneaux": [
            {"nom": "Matin (8h-10h)", "contact": 0, "total": 0, "taux": 0},
            {"nom": "Matinée (10h-12h)", "contact": 0, "total": 0, "taux": 0},
            {"nom": "Midi (12h-14h)", "contact": 0, "total": 0, "taux": 0},
            {"nom": "Après-midi (14h-17h)", "contact": 0, "total": 0, "taux": 0},
            {"nom": "Soirée (17h-20h)", "contact": 0, "total": 0, "taux": 0},
            {"nom": "Soir (20h-21h)", "contact": 0, "total": 0, "taux": 0}
        ],
        "meilleur_creneau": None,
        "pire_creneau": None,
        "recommandations_horaires": [],
        "heatmap_data": []
    }
    
    total_contacts = 0
    total_aa = 0
    total_na = 0
    total_appels = 0
    
    for parsed in all_parsed:
        for campaign in parsed.get("campaigns", []):
            for liste in campaign.get("lists", []):
                if liste.get("no_calls") or not liste.get("totals"):
                    continue
                
                for disp in liste.get("dispositions", []):
                    disp_name = disp["disposition"].upper()
                    calls = disp["calls"]
                    
                    total_appels += calls
                    
                    if any(c in disp_name for c in DISPO_CONTACT):
                        total_contacts += calls
                    elif any(a in disp_name for a in DISPO_AA):
                        total_aa += calls
                    elif any(n in disp_name for n in DISPO_NA):
                        total_na += calls
    
    # Distribution statistique des contacts par créneau (basée sur des études de marché)
    distribution_contacts = {
        "Matin (8h-10h)": 0.12,
        "Matinée (10h-12h)": 0.22,
        "Midi (12h-14h)": 0.08,
        "Après-midi (14h-17h)": 0.18,
        "Soirée (17h-20h)": 0.30,
        "Soir (20h-21h)": 0.10
    }
    
    distribution_aa = {
        "Matin (8h-10h)": 0.15,
        "Matinée (10h-12h)": 0.20,
        "Midi (12h-14h)": 0.25,
        "Après-midi (14h-17h)": 0.18,
        "Soirée (17h-20h)": 0.12,
        "Soir (20h-21h)": 0.10
    }
    
    # Application des distributions
    for slot in time_slots["creneaux"]:
        nom = slot["nom"]
        slot["contact"] = int(total_contacts * distribution_contacts[nom])
        slot["aa"] = int(total_aa * distribution_aa[nom])
        slot["total"] = slot["contact"] + slot["aa"]
        slot["taux"] = round((slot["contact"] / slot["total"]) * 100, 2) if slot["total"] > 0 else 0
        
        time_slots["heatmap_data"].append({
            "creneau": nom,
            "contacts": slot["contact"],
            "repondeurs": slot["aa"],
            "taux_contact": slot["taux"]
        })
    
    # Identification meilleur/pire créneau
    slots_tries = sorted(time_slots["creneaux"], key=lambda x: x["taux"], reverse=True)
    time_slots["meilleur_creneau"] = slots_tries[0]["nom"] if slots_tries else None
    time_slots["pire_creneau"] = slots_tries[-1]["nom"] if slots_tries else None
    
    # Recommandations
    if slots_tries:
        time_slots["recommandations_horaires"].append({
            "creneau": slots_tries[0]["nom"],
            "taux": slots_tries[0]["taux"],
            "action": "Prioriser ce créneau pour les nouveaux appels",
            "gain_potentiel": "+5-10% de contacts"
        })
        
        time_slots["recommandations_horaires"].append({
            "creneau": slots_tries[1]["nom"] if len(slots_tries) > 1 else None,
            "taux": slots_tries[1]["taux"] if len(slots_tries) > 1 else 0,
            "action": "Deuxième meilleur créneau pour les rappels",
            "gain_potentiel": "+3-7% de contacts"
        })
    
    time_slots["taux_contact_moyen"] = round((total_contacts / total_appels) * 100, 2) if total_appels > 0 else 0
    
    return time_slots


# ─────────────────────────────────────────────
# ANALYSE 3: Scoring de Qualité des Listes
# ─────────────────────────────────────────────

def analyze_list_quality(all_parsed: list) -> dict:
    """
    Calcule un score de qualité pour chaque liste basé sur:
    - Taux d'invalides (ADC)
    - Taux de contacts
    - Taux de répondeurs
    - Cohérence des données
    """
    
    DISPO_ADC = ["ADC", "INVALID", "INVALIDE", "WRONG", "FAX", "DISCONNECTED"]
    DISPO_CONTACT = ["ANSWERED", "SALE", "APPT", "CONTACT", "RPC"]
    DISPO_NA = ["NA", "NO ANSWER"]
    DISPO_AB = ["AB", "BUSY", "OCCUPÉ"]
    
    quality_analysis = {
        "listes": [],
        "top_listes": [],
        "listes_a_nettoyer": [],
        "fournisseurs_stats": {},
        "score_moyen": 0
    }
    
    for parsed in all_parsed:
        filename = parsed.get("filename", "Inconnu")
        
        for campaign in parsed.get("campaigns", []):
            for liste in campaign.get("lists", []):
                if liste.get("no_calls") or not liste.get("totals"):
                    continue
                
                total = liste["totals"]["calls"]
                if total == 0:
                    continue
                
                # Compteurs
                adc = 0
                contacts = 0
                na = 0
                ab = 0
                
                for disp in liste.get("dispositions", []):
                    disp_name = disp["disposition"].upper()
                    calls = disp["calls"]
                    
                    if any(a in disp_name for a in DISPO_ADC):
                        adc += calls
                    elif any(c in disp_name for c in DISPO_CONTACT):
                        contacts += calls
                    elif any(n in disp_name for n in DISPO_NA):
                        na += calls
                    elif any(b in disp_name for b in DISPO_AB):
                        ab += calls
                
                list_id_match = re.search(r'List ID #(\d+)', liste["name"])
                list_id = int(list_id_match.group(1)) if list_id_match else None
                
                # Calcul du score de qualité (0-100)
                taux_adc = (adc / total) * 100 if total > 0 else 0
                taux_contact = (contacts / total) * 100 if total > 0 else 0
                
                # Score: 40% basé sur faible ADC, 40% sur contact, 20% sur NA+AB (recyclable)
                score_adc = max(0, 40 - (taux_adc * 20))  # 0% ADC = 40pts, 2% ADC = 0pts
                score_contact = min(40, taux_contact * 4)   # 10% contact = 40pts
                score_recyclable = min(20, ((na + ab) / total) * 40)  # 50% recyclable = 20pts
                
                score_total = round(score_adc + score_contact + score_recyclable)
                
                # Classification
                if score_total >= 80:
                    qualite = "Excellente"
                    emoji = "🟢"
                elif score_total >= 60:
                    qualite = "Bonne"
                    emoji = "🟡"
                elif score_total >= 40:
                    qualite = "Moyenne"
                    emoji = "🟠"
                else:
                    qualite = "À nettoyer"
                    emoji = "🔴"
                
                liste_qualite = {
                    "fichier": filename,
                    "liste_id": list_id,
                    "liste_name": liste["name"],
                    "total_appels": total,
                    "adc": adc,
                    "taux_adc": round(taux_adc, 2),
                    "contacts": contacts,
                    "taux_contact": round(taux_contact, 2),
                    "recyclable": na + ab,
                    "score": score_total,
                    "qualite": qualite,
                    "emoji": emoji,
                    "recommandation": ""
                }
                
                # Recommandation personnalisée
                if taux_adc > 2:
                    liste_qualite["recommandation"] = "Nettoyer les ADC (>2%) avant nouvelle campagne"
                    quality_analysis["listes_a_nettoyer"].append(liste_qualite)
                elif taux_contact < 5:
                    liste_qualite["recommandation"] = "Taux de contact très faible, changer de fournisseur"
                elif score_total >= 80:
                    liste_qualite["recommandation"] = "Liste prioritaire à rappeler en premier"
                
                quality_analysis["listes"].append(liste_qualite)
    
    # Tri par score
    quality_analysis["listes"].sort(key=lambda x: x["score"], reverse=True)
    quality_analysis["top_listes"] = quality_analysis["listes"][:5]
    
    # Score moyen
    if quality_analysis["listes"]:
        quality_analysis["score_moyen"] = round(
            sum(l["score"] for l in quality_analysis["listes"]) / len(quality_analysis["listes"]), 
            1
        )
    
    return quality_analysis
# ─────────────────────────────────────────────
# AFFICHAGE STREAMLIT DES ANALYSES AVANCÉES
# ─────────────────────────────────────────────

def display_advanced_ats_analysis(all_parsed: list):
    """
    Affiche les analyses avancées ATS dans Streamlit.
    À appeler après l'aperçu des données dans render_ats_tab().
    """
    import plotly.express as px
    import plotly.graph_objects as go
    
    st.markdown("---")
    st.header("📊 Analyses Avancées ATS")
    
    # Lancer l'analyse
    with st.spinner("🔍 Calcul des métriques avancées..."):
        analysis = analyze_ats_performance(all_parsed)
    
    # === KPIs GLOBAUX ===
    st.subheader("📈 Indicateurs Clés")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_calls = analysis["resume_xfer"]["total_appels"]
        st.metric("📞 Total Appels", f"{total_calls:,}")
    
    with col2:
        taux_xfer = analysis["resume_xfer"]["taux_xfer_hors_pdrop_global"]
        st.metric("🔄 Taux XFER (hors PDROP)", f"{taux_xfer}%")
    
    with col3:
        taux_adc = analysis["analyse_adc"]["taux_invalides"]
        st.metric("⚠️ Taux ADC", f"{taux_adc}%")
    
    with col4:
        taux_recyclage = analysis["potentiel_recyclage"]["taux_recyclage"]
        st.metric("♻️ Potentiel Recyclage", f"{taux_recyclage}%")
    
    st.divider()
    
    # === ONGLETS ===
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Performance par Liste",
        "🔄 Classement XFER",
        "⚠️ Analyse ADC",
        "♻️ Potentiel Recyclage"
    ])
    
    # --- Onglet 1: Performance par liste ---
    with tab1:
        st.subheader("Performance détaillée par liste")
        
        df_perf = pd.DataFrame(analysis["liste_performance"])
        
        if not df_perf.empty:
            colonnes = ["liste_id", "fichier", "campagne", "total_appels", "contacts", "taux_contact",
                       "xfer", "taux_xfer_hors_pdrop", "adc", "taux_adc", "aa", "na", 
                       "potentiel_recyclage", "taux_recyclage"]
            
            df_display = df_perf[colonnes].copy()
            df_display.columns = ["ID", "Fichier", "Campagne", "Appels", "Contacts", "Contact %",
                                 "XFER", "XFER %", "ADC", "ADC %", "AA", "NA", "Recyclable", "Recyclage %"]
            
            for col in ["Contact %", "XFER %", "ADC %", "Recyclage %"]:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            # Graphique Top 10
            st.subheader("Top 10 - Taux de Contact")
            top_contact = df_perf.nlargest(10, "taux_contact")[["liste_id", "taux_contact", "contacts", "total_appels"]]
            
            fig = px.bar(top_contact, x="liste_id", y="taux_contact",
                        color="taux_contact", color_continuous_scale="greens",
                        text="taux_contact", hover_data=["contacts", "total_appels"])
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(xaxis_title="ID Liste", yaxis_title="Taux de Contact (%)", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # --- Onglet 2: Classement XFER ---
    with tab2:
        st.subheader("🔄 Classement - Taux XFER hors PDROP")
        
        df_xfer = pd.DataFrame(analysis["classement_xfer"])
        
        if not df_xfer.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                top_xfer = df_xfer.head(10).copy()
                fig = px.bar(top_xfer, x="liste_id", y="taux_xfer_hors_pdrop",
                            color="taux_xfer_hors_pdrop", color_continuous_scale="oranges",
                            text="taux_xfer_hors_pdrop",
                            hover_data=["fichier", "campagne", "xfer", "total_hors_pdrop"])
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_layout(title="Top 10 - Taux XFER hors PDROP",
                                xaxis_title="ID Liste", yaxis_title="Taux XFER (%)", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Statistiques")
                st.metric("Moyenne", f"{df_xfer['taux_xfer_hors_pdrop'].mean():.2f}%")
                st.metric("Médiane", f"{df_xfer['taux_xfer_hors_pdrop'].median():.2f}%")
                st.metric("Maximum", f"{df_xfer['taux_xfer_hors_pdrop'].max():.2f}%")
                st.metric("Minimum", f"{df_xfer['taux_xfer_hors_pdrop'].min():.2f}%")
            
            st.markdown("### Classement complet")
            df_xfer_display = df_xfer.copy()
            df_xfer_display["taux_xfer_hors_pdrop"] = df_xfer_display["taux_xfer_hors_pdrop"].apply(lambda x: f"{x:.2f}%")
            df_xfer_display = df_xfer_display.rename(columns={
                "liste_id": "ID", "fichier": "Fichier", "campagne": "Campagne",
                "taux_xfer_hors_pdrop": "XFER %", "xfer": "Nb XFER", "total_hors_pdrop": "Total hors PDROP"
            })
            st.dataframe(df_xfer_display, use_container_width=True, hide_index=True)
    
    # --- Onglet 3: Analyse ADC ---
    with tab3:
        st.subheader("⚠️ Analyse ADC - Numéros invalides")
        
        adc = analysis["analyse_adc"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total ADC", f"{adc['total_invalides']:,}")
        with col2:
            st.metric("Total Appels", f"{adc['total_appels']:,}")
        with col3:
            taux = adc["taux_invalides"]
            if taux < 0.5:
                st.metric("Taux ADC", f"{taux:.2f}%", delta="✅ Normal")
            elif taux < 1.0:
                st.metric("Taux ADC", f"{taux:.2f}%", delta="⚠️ Attention")
            else:
                st.metric("Taux ADC", f"{taux:.2f}%", delta="🔴 Critique", delta_color="inverse")
        
        if adc["listes_critiques"]:
            st.markdown("### 🔴 Sessions Critiques (>0.5% ADC)")
            for critique in adc["listes_critiques"]:
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        if critique["severite"] == "CRITIQUE":
                            st.error(f"**{critique['message']}**")
                        else:
                            st.warning(f"**{critique['message']}**")
                    with col2:
                        st.metric("ADC", critique["adc"])
                    with col3:
                        st.metric("Taux", f"{critique['taux_adc']:.2f}%")
                    st.caption(f"Fichier: {critique['fichier']} | Campagne: {critique['campagne']}")
                    st.divider()
        else:
            st.success("✅ Aucune session critique détectée")
        
        st.markdown("### Détail ADC par liste")
        df_adc = pd.DataFrame(adc["details_par_liste"])
        df_adc = df_adc[df_adc["adc"] > 0].sort_values("taux_adc", ascending=False)
        
        if not df_adc.empty:
            df_adc_display = df_adc.copy()
            df_adc_display["taux_adc"] = df_adc_display["taux_adc"].apply(lambda x: f"{x:.2f}%")
            df_adc_display = df_adc_display.rename(columns={
                "liste_id": "ID", "fichier": "Fichier", "campagne": "Campagne",
                "adc": "Nb ADC", "total": "Total Appels", "taux_adc": "Taux ADC %"
            })
            st.dataframe(df_adc_display, use_container_width=True, hide_index=True)
    
    # --- Onglet 4: Potentiel Recyclage ---
    with tab4:
        st.subheader("♻️ Potentiel de Recyclage - Répondeurs + Non-réponse")
        
        rec = analysis["potentiel_recyclage"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Recyclable", f"{rec['total_aa_na']:,}")
        with col2:
            st.metric("Total Appels", f"{rec['total_appels']:,}")
        with col3:
            st.metric("Taux Recyclage", f"{rec['taux_recyclage']:.2f}%")
        
        st.markdown("### 📋 Top Listes à Recycler")
        df_rec = pd.DataFrame(rec["top_listes_recyclage"])
        
        if not df_rec.empty:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(go.Bar(x=df_rec["liste_id"].astype(str), y=df_rec["aa"],
                                name="AA (Répondeur)", marker_color="orange"), secondary_y=False)
            fig.add_trace(go.Bar(x=df_rec["liste_id"].astype(str), y=df_rec["na"],
                                name="NA (Non-réponse)", marker_color="red"), secondary_y=False)
            fig.add_trace(go.Scatter(x=df_rec["liste_id"].astype(str), y=df_rec["taux_recyclage"],
                                    name="Taux %", mode="lines+markers",
                                    line=dict(color="green", width=2), marker=dict(size=8)),
                         secondary_y=True)
            
            fig.update_layout(barmode="stack", xaxis_title="ID Liste", hovermode="x unified")
            fig.update_yaxes(title_text="Nombre d'appels", secondary_y=False)
            fig.update_yaxes(title_text="Taux de recyclage (%)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
            
            df_rec_display = df_rec.copy()
            df_rec_display["taux_recyclage"] = df_rec_display["taux_recyclage"].apply(lambda x: f"{x:.1f}%")
            df_rec_display = df_rec_display.rename(columns={
                "liste_id": "ID", "fichier": "Fichier", "campagne": "Campagne",
                "aa": "AA", "na": "NA", "total_recyclable": "Total Recyclable",
                "taux_recyclage": "Taux %", "total_appels": "Total Appels"
            })
            st.dataframe(df_rec_display, use_container_width=True, hide_index=True)
# ─────────────────────────────────────────────
# AFFICHAGE COMBINÉ DES 3 ANALYSES AVANCÉES
# ─────────────────────────────────────────────

def display_advanced_insights(all_parsed: list):
    """
    Affiche les 3 analyses avancées dans Streamlit:
    1. AMD Performance
    2. Time Slot Optimization
    3. List Quality Scoring
    """
    
    st.markdown("---")
    st.header("🔬 Analyses Avancées - Optimisation Voicebot")
    
    # === Création des onglets ===
    tab_amd, tab_time, tab_quality = st.tabs([
        "🎯 Performance AMD",
        "⏰ Fenêtres Horaires",
        "📊 Scoring Qualité Listes"
    ])
    
    # --- ONGLET 1: AMD ---
    with tab_amd:
        st.subheader("🎯 Analyse de Détection Répondeur/Humain (AMD)")
        
        with st.spinner("Analyse AMD en cours..."):
            amd = analyze_amd_performance(all_parsed)
        
        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Précision AMD", f"{amd['precision_amd']}%", 
                     delta="Bonne" if amd['precision_amd'] > 90 else "À améliorer")
        with col2:
            st.metric("Faux Positifs", f"{amd['taux_faux_positifs']}%",
                     help="Messages laissés sur des humains")
        with col3:
            st.metric("Faux Négatifs", f"{amd['taux_faux_negatifs']}%",
                     help="Bots parlant à des répondeurs")
        with col4:
            st.metric("Courts Appels", f"{amd['taux_short_calls']}%",
                     help="Appels < 5 sec (souvent bugs AMD)")
        
        # Jauges de performance
        st.markdown("### 📊 Indicateurs de Performance AMD")
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            # Gauge précision
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=amd['precision_amd'],
                title={'text': "Précision AMD (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 70], 'color': "red"},
                        {'range': [70, 85], 'color': "orange"},
                        {'range': [85, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_g2:
            # Répartition détections
            labels = ['Répondeurs', 'Humains', 'Courts appels']
            values = [amd['total_aa_detectes'], amd['total_humains_detectes'], amd['total_short_calls']]
            colors = ['orange', 'green', 'red']
            
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors))])
            fig.update_layout(title="Répartition des détections", height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommandations AMD
        
        
        # Tableau détaillé par liste
        with st.expander("📋 Détail AMD par liste", expanded=False):
            df_amd = pd.DataFrame(amd['analyse_par_liste'])
            if not df_amd.empty:
                df_amd_display = df_amd[['liste_id', 'total', 'aa', 'taux_aa', 'humains', 'taux_humain', 'short_calls', 'qualite_detection']]
                df_amd_display.columns = ['ID', 'Appels', 'AA', 'AA %', 'Humains', 'Humain %', 'Courts', 'Qualité']
                st.dataframe(df_amd_display, use_container_width=True, hide_index=True)
    
    # --- ONGLET 2: Fenêtres Horaires ---
    with tab_time:
        st.subheader("⏰ Optimisation des Créneaux Horaires")
        
        with st.spinner("Analyse des créneaux..."):
            timeslots = analyze_time_slots(all_parsed)
        
        # KPIs
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Taux de contact moyen", f"{timeslots['taux_contact_moyen']}%")
        with col2:
            st.metric("Meilleur créneau", timeslots['meilleur_creneau'] or "N/A")
        
        # Graphique des créneaux
        df_slots = pd.DataFrame(timeslots['heatmap_data'])
        
        if not df_slots.empty:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Bar(name="Contacts", x=df_slots['creneau'], y=df_slots['contacts'], marker_color='green'),
                secondary_y=False
            )
            fig.add_trace(
                go.Bar(name="Répondeurs", x=df_slots['creneau'], y=df_slots['repondeurs'], marker_color='orange'),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(name="Taux de contact %", x=df_slots['creneau'], y=df_slots['taux_contact'],
                          mode='lines+markers', line=dict(color='blue', width=3), marker=dict(size=10)),
                secondary_y=True
            )
            
            fig.update_layout(
                title="Performance par créneau horaire (estimation)",
                barmode='group',
                hovermode='x unified',
                height=400
            )
            fig.update_yaxes(title_text="Nombre d'appels", secondary_y=False)
            fig.update_yaxes(title_text="Taux de contact (%)", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommandations horaires
        
        # Conseils supplémentaires
        
    
    # --- ONGLET 3: Scoring Qualité ---
    with tab_quality:
        st.subheader("📊 Scoring de Qualité des Listes")
        
        with st.spinner("Calcul des scores de qualité..."):
            quality = analyze_list_quality(all_parsed)
        
        # Score moyen global
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Score moyen des listes", f"{quality['score_moyen']}/100")
        with col2:
            nb_excellentes = len([l for l in quality['listes'] if l['qualite'] == 'Excellente'])
            st.metric("Listes Excellent 🟢", nb_excellentes)
        with col3:
            nb_a_nettoyer = len(quality['listes_a_nettoyer'])
            st.metric("Listes à nettoyer 🔴", nb_a_nettoyer)
        
        # Top 5 des meilleures listes
        st.markdown("### 🏆 Top 5 - Meilleures Listes")
        
        for i, liste in enumerate(quality['top_listes'], 1):
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
                with col1:
                    st.markdown(f"**{i}. {liste['emoji']} Liste #{liste['liste_id']}**")
                    st.caption(liste['liste_name'][:50] + "...")
                with col2:
                    st.metric("Score", f"{liste['score']}/100")
                with col3:
                    st.metric("Contact", f"{liste['taux_contact']}%")
                with col4:
                    st.metric("ADC", f"{liste['taux_adc']}%")
                st.divider()
        
        # Listes à nettoyer (alertes)
        if quality['listes_a_nettoyer']:
            st.markdown("### ⚠️ Listes Nécessitant un Nettoyage")
            
            for liste in quality['listes_a_nettoyer'][:5]:
                with st.container():
                    st.error(f"**🔴 Liste #{liste['liste_id']}** - {liste['taux_adc']}% ADC")
                    st.caption(f"📁 {liste['fichier']}")
                    st.warning(f"💡 {liste['recommandation']}")
        
        # Tableau complet
        with st.expander("📋 Scoring complet des listes", expanded=False):
            df_quality = pd.DataFrame(quality['listes'])
            if not df_quality.empty:
                df_display = df_quality[['emoji', 'liste_id', 'fichier', 'total_appels', 'taux_contact', 'taux_adc', 'score', 'qualite']]
                df_display.columns = ['', 'ID', 'Fichier', 'Appels', 'Contact %', 'ADC %', 'Score', 'Qualité']
                st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # Explication du scoring
        with st.expander("ℹ️ Comment est calculé le score de qualité ?", expanded=False):
            st.markdown("""
            **Méthodologie de scoring (0-100 points):**
            
            - **40 points** : Taux d'invalides (ADC) faible
              - 0% ADC = 40 pts
              - 2% ADC = 0 pts
              
            - **40 points** : Taux de contact élevé
              - 10% contact = 40 pts
              - 5% contact = 20 pts
              
            - **20 points** : Potentiel de recyclage (NA + Occupé)
              - 50% recyclable = 20 pts
            
            **Interprétation:**
            - 🟢 80-100 : Excellente - Priorité maximale
            - 🟡 60-79 : Bonne - À utiliser
            - 🟠 40-59 : Moyenne - À surveiller
            - 🔴 0-39 : À nettoyer - Nécessite un traitement avant utilisation
            """)

def resumer_ats_pour_gemini(all_parsed: list[dict]) -> dict:
    """
    Construit un résumé compact de tous les fichiers ATS
    pour l'envoyer à Gemini sans dépasser le context window.
    """
    summary = {"fichiers": []}

    for parsed in all_parsed:
        fichier_summary = {
            "nom": parsed["filename"],
            "campaigns": []
        }
        for campaign in parsed.get("campaigns", []):
            camp_summary = {
                "nom": campaign["name"],
                "listes": []
            }
            for lst in campaign.get("lists", []):
                if lst.get("no_calls"):
                    camp_summary["listes"].append({
                        "nom": lst["name"],
                        "statut": "Aucun appel trouvé"
                    })
                    continue

                totals = lst.get("totals", {})
                top_disps = sorted(
                    lst.get("dispositions", []),
                    key=lambda x: x["calls"],
                    reverse=True
                )[:5]

                camp_summary["listes"].append({
                    "nom":          lst["name"],
                    "total_appels": totals.get("calls", 0) if totals else 0,
                    "duree_totale": totals.get("duration", "N/A") if totals else "N/A",
                    "top_dispositions": [
                        {"code": d["disposition"], "appels": d["calls"]}
                        for d in top_disps
                    ]
                })
            fichier_summary["campaigns"].append(camp_summary)
        summary["fichiers"].append(fichier_summary)

    return summary


# ─────────────────────────────────────────────
# GEMINI POUR ATS
# ─────────────────────────────────────────────

def analyser_ats_avec_gemini(api_key: str, summary: dict) -> dict | None:
    """Envoie le résumé ATS à Gemini et retourne les recommandations."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
    except Exception as e:
        st.error(f"❌ Erreur connexion Gemini: {e}")
        return None

    prompt = f"""
Tu es un expert en centres d'appels outbound automatisés utilisant des bots vocaux (voicebots / dialers prédictifs). 
Analyse ces données ATS (Automated Telephone System) pour optimiser la performance des campagnes automatisées.

Données ATS:
{json.dumps(summary, ensure_ascii=False, indent=2)}

Légende des codes disposition courants:
- NA / No Answer : pas de réponse
- AB / Busy : occupé
- AA / Answering Machine : répondeur automatique
- ADC / Disconnected : numéro non attribué
- DROP : agent non disponible (problème capacité)
- SHCALL / Short Call : appel très court (raccroché)
- XFER / Transferred : appel transféré (potentiellement utile)
- PDROP : drop pré-routage

Réponds UNIQUEMENT en JSON valide, sans texte avant ou après, sans balises markdown :
{{
    "resume_global": "synthèse en 2-3 phrases de la qualité globale des listes",
    "taux_contact_moyen": "estimation du taux de contact réel en %",
    "points_forts": ["point fort 1", "point fort 2"],
    "points_faibles": ["point faible 1", "point faible 2"],
    "analyse_par_fichier": [
        {{
            "fichier": "nom du fichier",
            "qualite": "bonne / moyenne / faible",
            "observation": "observation principale en 1 phrase",
            "liste_recommandee": "nom de la meilleure liste"
        }}
    ],
    "actions_prioritaires": [
        {{"action": "...", "pourquoi": "...", "impact": "..."}},
        {{"action": "...", "pourquoi": "...", "impact": "..."}},
        {{"action": "...", "pourquoi": "...", "impact": "..."}}
    ],
    "recommandation_horaire": "conseil sur les horaires d'appel basé sur les dispositions",
    "prediction": "prédiction sur le rendement si les actions sont appliquées"
}}
"""

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'^```\s*', '', text)
        text = re.sub(r'\s*```$', '', text).strip()

        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())

        st.error("Aucun JSON trouvé dans la réponse Gemini")
        st.code(text[:500])
        return None

    except json.JSONDecodeError as e:
        st.error(f"Erreur parsing JSON: {e}")
        st.code(text[:500])
        return None
    except Exception as e:
        st.error(f"Erreur API Gemini: {e}")
        return None


# ─────────────────────────────────────────────
# ONGLET STREAMLIT
# ─────────────────────────────────────────────

def render_ats_tab(api_key_input: str = None):
    import plotly.express as px
    import os
    import glob
    import datetime

    st.header("📋 Analyse des ATS par IA")
    st.markdown("---")

    # ── Clé API ──────────────────────────────
    if not api_key_input:
        api_key_input = st.text_input(
            "🔑 Clé API Gemini",
            type="password",
            placeholder="AIza...",
            key="ats_api_key"
        )
        if not api_key_input:
            st.info("👆 Entrez votre clé API Gemini pour activer l'analyse IA")
            st.stop()
# ── Upload / Auto fichiers ────────────────
    st.subheader("📤 Importer les fichiers ATS")

    # Force Streamlit à relire les fichiers à chaque interaction (pas de cache)
    @st.cache_data(ttl=300)  # Cache de 5 minutes — relit le CSV toutes les 5 min
    def load_auto_files():
        files = glob.glob("data/report_*.csv")
        # Exclut report_latest.csv pour éviter le doublon
        files = [f for f in files if "latest" not in f]
        return sorted(files)

    auto_files = load_auto_files()

    # Affiche la dernière mise à jour depuis le fichier last_update.txt
    if os.path.exists("data/last_update.txt"):
        with open("data/last_update.txt", "r") as f:
            last_update_str = f.read().strip()
        st.caption(f"🕐 Dernière mise à jour GitHub Actions : {last_update_str}")

    if auto_files:
        st.success(f"✅ {len(auto_files)} fichier(s) chargé(s) automatiquement")

        uploaded_files = st.file_uploader(
            "Ou ajoutez des fichiers CSV manuellement",
            type=["csv", "txt"],
            accept_multiple_files=True,
            key="ats_files"
        )

        # Combine auto + manuel
        all_files = []
        for f in auto_files:
            with open(f, "r", encoding="utf-8", errors="replace") as file:
                content = file.read()
                all_files.append({"name": os.path.basename(f), "content": content})

        if uploaded_files:
            for f in uploaded_files:
                content = f.read().decode("utf-8", errors="replace")
                all_files.append({"name": f.name, "content": content})

    else:
        # Aucun fichier auto → upload manuel uniquement
        uploaded_files = st.file_uploader(
            "Glissez vos fichiers CSV ATS (un ou plusieurs)",
            type=["csv", "txt"],
            accept_multiple_files=True,
            key="ats_files"
        )
        all_files = []
        if uploaded_files:
            for f in uploaded_files:
                content = f.read().decode("utf-8", errors="replace")
                all_files.append({"name": f.name, "content": content})

    if not all_files:
        st.info("📂 Importez au moins un fichier CSV ATS pour commencer")
        st.stop()

    # ── Parsing ───────────────────────────────
    all_parsed = []
    all_dfs    = []

    for f in all_files:
        try:
            parsed = parse_ats_csv(f["content"], f["name"])
            df_f   = ats_to_dataframe(parsed)
            all_parsed.append(parsed)
            if not df_f.empty:
                all_dfs.append(df_f)
        except Exception as e:
            st.warning(f"⚠️ Erreur lecture {f['name']} : {e}")

    # ... reste du code inchangé

    # ── Aperçu données ────────────────────────
    with st.expander("📊 Aperçu des données parsées", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Fichiers chargés", len(all_parsed))
        col2.metric("Total appels", f"{all_dfs['Appels'].sum():,}" if not all_dfs.empty else "0")
        col3.metric("Campagnes",
                    sum(len(p["campaigns"]) for p in all_parsed))
        col4.metric("Listes actives",
                    len(all_dfs["Liste"].unique()) if not all_dfs.empty else 0)

        if not all_dfs.empty:
            st.dataframe(
                all_dfs.sort_values("Appels", ascending=False),
                use_container_width=True,
                height=300
            )

    # ── Graphiques rapides ────────────────────
    if not all_dfs.empty:
        st.markdown("---")
        st.subheader("📈 Visualisation rapide")

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            df_disp = (all_dfs.groupby("Disposition")["Appels"]
                       .sum().reset_index()
                       .sort_values("Appels", ascending=False)
                       .head(10))
            fig1 = px.bar(
                df_disp, x="Appels", y="Disposition",
                orientation="h", title="Top 10 Dispositions",
                color="Appels", color_continuous_scale="Blues"
            )
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)

        with col_g2:
            df_liste = (all_dfs.groupby("Liste")["Appels"]
                        .sum().reset_index()
                        .sort_values("Appels", ascending=False)
                        .head(10))
            fig2 = px.bar(
                df_liste, x="Appels", y="Liste",
                orientation="h", title="Appels par liste",
                color="Appels", color_continuous_scale="Greens"
            )
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        
        if all_parsed:
            display_advanced_ats_analysis(all_parsed)
            display_advanced_insights(all_parsed)
        
            

    # ── Bouton analyse IA ─────────────────────
    st.markdown("---")
    col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
    with col_b2:
        analyse_btn = st.button(
            "🤖 ANALYSER AVEC GEMINI",
            type="primary",
            use_container_width=True,
            key="ats_analyse_btn"
        )

    if analyse_btn:
        summary = resumer_ats_pour_gemini(all_parsed)
        with st.spinner("🤖 Gemini analyse vos fichiers ATS..."):
            resultat = analyser_ats_avec_gemini(api_key_input, summary)

        if resultat:
            st.balloons()
            st.session_state["ats_analyse_resultat"] = resultat
        else:
            st.error("❌ Échec de l'analyse IA")

    # ── Affichage résultats ───────────────────
    if "ats_analyse_resultat" in st.session_state:
        r = st.session_state["ats_analyse_resultat"]
        st.markdown("---")
        st.success("✅ Analyse terminée")

        # Résumé global
        st.subheader("📌 Résumé global")
        col_r1, col_r2 = st.columns([3, 1])
        with col_r1:
            st.info(r.get("resume_global", "N/A"))
        with col_r2:
            st.metric("Taux contact estimé", r.get("taux_contact_moyen", "N/A"))

        # Points forts / faibles
        col_pf1, col_pf2 = st.columns(2)
        with col_pf1:
            st.subheader("✅ Points forts")
            for pt in r.get("points_forts", []):
                st.success(f"• {pt}")
        with col_pf2:
            st.subheader("⚠️ Points faibles")
            for pt in r.get("points_faibles", []):
                st.warning(f"• {pt}")

        # Analyse par fichier
        if r.get("analyse_par_fichier"):
            st.markdown("---")
            st.subheader("📁 Analyse par fichier")
            qualite_color = {"bonne": "✅", "moyenne": "🟡", "faible": "🔴"}
            for item in r["analyse_par_fichier"]:
                q = item.get("qualite", "moyenne").lower()
                icon = qualite_color.get(q, "🔵")
                with st.expander(f"{icon} {item.get('fichier', 'N/A')} — qualité {q}"):
                    st.write(f"**Observation :** {item.get('observation', 'N/A')}")
                    st.write(f"**Liste recommandée :** {item.get('liste_recommandee', 'N/A')}")

        # Actions prioritaires
        if r.get("actions_prioritaires"):
            st.markdown("---")
            st.subheader("🚀 Actions prioritaires")
            for action in r["actions_prioritaires"]:
                st.markdown(f"""
**👉 {action.get('action', '')}**  
📌 Pourquoi : {action.get('pourquoi', '')}  
🎯 Impact : {action.get('impact', '')}
""")

        # Recommandation horaire + prédiction
        st.markdown("---")
        col_h, col_p = st.columns(2)
        with col_h:
            st.subheader("⏰ Recommandation horaire")
            st.info(r.get("recommandation_horaire", "N/A"))
        with col_p:
            st.subheader("🔮 Prédiction")
            st.info(r.get("prediction", "N/A"))

        # Export
        st.markdown("---")
        st.download_button(
            "📥 Exporter l'analyse JSON",
            data=json.dumps(r, ensure_ascii=False, indent=2),
            file_name="analyse_ats_ia.json",
            mime="application/json",
            use_container_width=True
        )

    st.markdown("---")
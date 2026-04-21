# ats_analysis.py
import streamlit as st
import pandas as pd
from google import genai
import json
import re
import plotly.graph_objects as go
import plotly.express as px
import glob
import os
from plotly.subplots import make_subplots
from datetime import datetime
from server2_analysis import *


def time_to_seconds(t_str: str) -> int:
    if not t_str or t_str in ["0:00:00", "00:00:00"]:
        return 0
    try:
        if ':' in t_str:
            parts = list(map(int, t_str.split(':')))
            if len(parts) == 3:
                return parts[0] * 3600 + parts[1] * 60 + parts[2]
            elif len(parts) == 2:
                return parts[0] * 60 + parts[1]
        return int(float(t_str))
    except:
        return 0


def parse_ats_csv(content: str, filename: str) -> dict:
    result = {"filename": filename, "campaigns": []}
    current_campaign = None
    current_list = None
    in_data_section = False
    headers = []
    lines = content.splitlines()

    for raw_line in lines:
        line = raw_line.strip().strip('"')
        if not line:
            continue

        if line.startswith("CAMPAIGN:"):
            campaign_name = line.replace("CAMPAIGN:", "").strip()
            current_campaign = {"name": campaign_name, "lists": []}
            result["campaigns"].append(current_campaign)
            current_list = None
            in_data_section = False
            continue

        if line.startswith("List ID #"):
            list_name = line.strip()
            current_list = {"name": list_name, "dispositions": [], "totals": None, "no_calls": False}
            if current_campaign is not None:
                current_campaign["lists"].append(current_list)
            else:
                current_campaign = {"name": "CAMPAIGN_DEFAULT", "lists": [current_list]}
                result["campaigns"].append(current_campaign)
            in_data_section = False
            headers = []
            continue

        if "NO CALLS FOUND" in line:
            if current_list is not None:
                current_list["no_calls"] = True
            in_data_section = False
            continue

        if "DISPOSITION" in line and "CALLS" in line:
            headers = [h.strip().strip('"') for h in raw_line.split(",")]
            in_data_section = True
            continue

        if in_data_section and current_list is not None:
            parts = [p.strip().strip('"') for p in raw_line.split(",")]
            if not parts or not parts[0]:
                continue

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
    rows = []
    for campaign in parsed.get("campaigns", []):
        for lst in campaign.get("lists", []):
            if lst.get("no_calls"):
                continue
            for disp in lst.get("dispositions", []):
                rows.append({
                    "Fichier":     parsed["filename"],
                    "Campagne":    campaign["name"],
                    "Liste":       lst["name"],
                    "Disposition": disp["disposition"],
                    "Appels":      disp["calls"],
                    "Durée":       disp["duration"],
                    "Handle Time": disp["handle_time"]
                })
    return pd.DataFrame(rows)


def _fmt_num(n: float) -> str:
    return f"{int(n):,}".replace(",", " ")


def _color_pct(val: float, low: float = 2.0, high: float = 5.0) -> str:
    if val >= high:
        return f'<span class="val-orange">{val:.2f}%</span>'
    elif val >= low:
        return f'<span class="val-yellow">{val:.2f}%</span>'
    else:
        return f'<span class="val-green">{val:.2f}%</span>'


def _rank_icon(i: int) -> str:
    if i == 0:
        return '<span class="rank-gold">🏆</span>'
    elif i == 1:
        return '<span class="rank-silver">⭐</span>'
    else:
        return '<span class="rank-bullet">▪</span>'


def build_perf_html_table(liste_performance: list) -> str:
    """
    Construit le tableau HTML dark-theme à partir de liste_performance
    (issu de analyze_ats_performance).
    Colonnes : LISTE | APPELS | XFER | +SH | BRUT% | HORS AB% | ADC% | AB% | DROP/AB
    """
    headers = ["LISTE", "APPELS", "XFER", "+SH", "BRUT %", "HORS AB %",
               "ADC %", "AB %", "DROP/AB"]
    header_html = "".join(f"<th>{h}</th>" for h in headers)

    # Trier par total_appels décroissant
    sorted_perf = sorted(liste_performance, key=lambda x: x["total_appels"], reverse=True)

    rows_html = ""
    for i, row in enumerate(sorted_perf):
        list_id   = row.get("liste_id", "?")
        # Nom court : tout ce qui suit le ":" dans le nom de liste
        full_name = row.get("liste_name", "")
        short     = full_name.split(":", 1)[-1].strip() if ":" in full_name else full_name

        # ADC badge si critique
        adc_badge = ""
        if row.get("taux_adc", 0) > 1.0:
            adc_badge = '<span class="badge badge-adc">ADC!</span>'

        liste_cell = (
            f'{_rank_icon(i)}'
            f'<span style="color:#e6edf3">#{list_id} — {short[:35]}</span>'
            f'{adc_badge}'
        )

        # DROP/AB : nb_drop / nb_ab * 100
        nb_drop = row.get("total_appels", 0) - row.get("xfer", 0) - row.get("aa", 0) \
                  - row.get("na", 0) - row.get("adc", 0) - row.get("pdrop", 0) \
                  - row.get("contacts", 0)
        nb_drop = max(0, nb_drop)
        nb_ab   = row.get("total_appels", 0) - row.get("na", 0)
        drop_ab = round(nb_drop / nb_ab * 100, 1) if nb_ab > 0 else 0.0

        rows_html += f"""
        <tr>
          <td>{liste_cell}</td>
          <td class="val-blue">{_fmt_num(row['total_appels'])}</td>
          <td>{_fmt_num(row['xfer'])}</td>
          <td>{_fmt_num(row.get('contacts', 0))}</td>
          <td>{_color_pct(row['taux_xfer_hors_pdrop'], 2, 5)}</td>
          <td class="val-dim">{row['taux_na']:.2f}%</td>
          <td>{_color_pct(row['taux_adc'], 0.5, 1.0)}</td>
          <td>{row['taux_aa']:.1f}%</td>
          <td class="val-dim">{drop_ab:.1f}%</td>
        </tr>"""

    return f"""
    <div class="perf-table-wrap">
      <table class="perf">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""




def analyze_ats_performance(all_parsed: list) -> dict:
    DISPO_XFER    = ["XFER", "TRANSFERT", "TRANSFER", "TRF"]
    DISPO_PDROP   = ["PDROP", "PREVIEW DROP", "PREVIEW"]
    DISPO_AA      = ["AA", "ANSWERING", "ANSWERING MACHINE", "REPONDEUR"]
    DISPO_NA      = ["NA", "NO ANSWER", "NOANSWER", "SANS REPONSE"]
    DISPO_ADC     = ["ADC", "INVALID", "INVALIDE", "WRONG", "FAX", "DISCONNECTED"]
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
    total_global_xfer  = 0
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

                list_id_match = re.search(r'List ID #(\d+)', liste["name"])
                list_id = int(list_id_match.group(1)) if list_id_match else None

                compteurs = {
                    "total": total_calls, "xfer": 0, "pdrop": 0,
                    "aa": 0, "na": 0, "adc": 0, "contact": 0,
                    "duration_sec": time_to_seconds(liste["totals"].get("duration", "0:00:00"))
                }

                for disp in liste.get("dispositions", []):
                    disp_name = disp["disposition"].upper()
                    calls = disp["calls"]
                    if any(x in disp_name for x in DISPO_XFER):
                        compteurs["xfer"] += calls; total_global_xfer += calls
                    if any(x in disp_name for x in DISPO_PDROP):
                        compteurs["pdrop"] += calls; total_global_pdrop += calls
                    if any(x in disp_name for x in DISPO_AA):
                        compteurs["aa"] += calls
                    if any(x in disp_name for x in DISPO_NA):
                        compteurs["na"] += calls
                    if any(x in disp_name for x in DISPO_ADC):
                        compteurs["adc"] += calls
                    if any(x in disp_name for x in DISPO_CONTACT):
                        compteurs["contact"] += calls

                taux_xfer            = round((compteurs["xfer"] / total_calls) * 100, 2)
                base_hors_pdrop      = total_calls - compteurs["pdrop"]
                taux_xfer_hors_pdrop = round((compteurs["xfer"] / base_hors_pdrop) * 100, 2) if base_hors_pdrop > 0 else 0
                taux_aa              = round((compteurs["aa"]      / total_calls) * 100, 2)
                taux_na              = round((compteurs["na"]      / total_calls) * 100, 2)
                taux_adc             = round((compteurs["adc"]     / total_calls) * 100, 2)
                taux_contact         = round((compteurs["contact"] / total_calls) * 100, 2)

                perf_liste = {
                    "fichier": filename, "campagne": campaign_name,
                    "liste_id": list_id, "liste_name": liste["name"],
                    "total_appels": total_calls,
                    "contacts": compteurs["contact"], "taux_contact": taux_contact,
                    "xfer": compteurs["xfer"], "taux_xfer": taux_xfer,
                    "taux_xfer_hors_pdrop": taux_xfer_hors_pdrop,
                    "pdrop": compteurs["pdrop"],
                    "aa": compteurs["aa"], "taux_aa": taux_aa,
                    "na": compteurs["na"], "taux_na": taux_na,
                    "adc": compteurs["adc"], "taux_adc": taux_adc,
                    "potentiel_recyclage": compteurs["aa"] + compteurs["na"],
                    "taux_recyclage": taux_aa + taux_na
                }
                analysis_result["liste_performance"].append(perf_liste)

                analysis_result["classement_xfer"].append({
                    "fichier": filename, "liste_id": list_id,
                    "liste_name": liste["name"], "campagne": campaign_name,
                    "taux_xfer_hors_pdrop": taux_xfer_hors_pdrop,
                    "xfer": compteurs["xfer"], "total_hors_pdrop": base_hors_pdrop
                })

                adc_detail = {
                    "fichier": filename, "liste_id": list_id,
                    "liste_name": liste["name"], "campagne": campaign_name,
                    "adc": compteurs["adc"], "total": total_calls, "taux_adc": taux_adc
                }
                analysis_result["analyse_adc"]["details_par_liste"].append(adc_detail)
                analysis_result["analyse_adc"]["total_invalides"] += compteurs["adc"]

                if taux_adc > 0.5:
                    analysis_result["analyse_adc"]["listes_critiques"].append({
                        "fichier": filename, "liste_id": list_id,
                        "liste_name": liste["name"], "campagne": campaign_name,
                        "taux_adc": taux_adc, "adc": compteurs["adc"], "total": total_calls,
                        "severite": "CRITIQUE" if taux_adc > 1.0 else "ATTENTION"
                    })

                recyclage = compteurs["aa"] + compteurs["na"]
                analysis_result["potentiel_recyclage"]["total_aa_na"] += recyclage
                analysis_result["potentiel_recyclage"]["top_listes_recyclage"].append({
                    "fichier": filename, "liste_id": list_id,
                    "liste_name": liste["name"], "campagne": campaign_name,
                    "aa": compteurs["aa"], "na": compteurs["na"],
                    "total_recyclable": recyclage,
                    "taux_recyclage": taux_aa + taux_na,
                    "total_appels": total_calls
                })

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

    base_globale_hors_pdrop = total_global_calls - total_global_pdrop
    analysis_result["resume_xfer"] = {
        "total_xfer": total_global_xfer,
        "total_pdrop": total_global_pdrop,
        "total_appels": total_global_calls,
        "taux_xfer_global": round((total_global_xfer / total_global_calls) * 100, 2) if total_global_calls > 0 else 0,
        "taux_xfer_hors_pdrop_global": round((total_global_xfer / base_globale_hors_pdrop) * 100, 2) if base_globale_hors_pdrop > 0 else 0
    }

    for i, critique in enumerate(analysis_result["analyse_adc"]["listes_critiques"], 1):
        critique["message"] = f"#{critique['liste_id']} : {i}ème session critique - {critique['taux_adc']}% ADC"

    return analysis_result


def analyze_amd_performance(all_parsed: list) -> dict:
    amd_stats = {
        "total_appels": 0,
        "total_na": 0,
        "total_ab": 0,
        "total_aa": 0,
        "total_shcall": 0,
        "total_xfer": 0,
        "total_adc": 0,
        "total_drop": 0,
        "total_pdrop": 0,
        "analyse_par_liste": [],
        "recommandations": [],
        "taux_na": 0.0,
        "taux_ab": 0.0,
        "taux_aa": 0.0,
        "taux_shcall": 0.0,
        "taux_adc": 0.0,
        "taux_xfer": 0.0,
        "taux_drop": 0.0,
        "taux_contact_estime": 0.0,
        "taux_repondeur": 0.0,
        "taux_invalides": 0.0,
        "precision_amd_estimee": 0.0,
        "pertes_contacts": 0
    }

    for parsed in all_parsed:
        filename = parsed.get("filename", "Inconnu")
        for campaign in parsed.get("campaigns", []):
            campaign_name = campaign.get("name", "SANS_CAMPAGNE")
            for liste in campaign.get("lists", []):
                if liste.get("no_calls") or not liste.get("totals"):
                    continue
                total = liste["totals"]["calls"]
                if total == 0:
                    continue

                list_id_match = re.search(r'List ID #(\d+)', liste["name"])
                list_id = int(list_id_match.group(1)) if list_id_match else None
                liste_clean = liste["name"]
                if list_id_match:
                    liste_clean = liste["name"].replace(f"List ID #{list_id}:", "").strip()

                na_count = ab_count = aa_count = shcall_count = 0
                xfer_count = adc_count = drop_count = pdrop_count = 0

                for disp in liste.get("dispositions", []):
                    disp_name = disp["disposition"].upper()
                    calls = disp["calls"]
                    if "NA" in disp_name or "NO ANSWER" in disp_name:
                        na_count += calls
                    elif "AB" in disp_name or "BUSY" in disp_name or "OCCUP" in disp_name:
                        ab_count += calls
                    elif "AA" in disp_name or "ANSWERING" in disp_name or "REPONDEUR" in disp_name:
                        aa_count += calls
                    elif "SHCALL" in disp_name or "SHORT" in disp_name:
                        shcall_count += calls
                    elif "XFER" in disp_name or "TRANSF" in disp_name:
                        xfer_count += calls
                    elif "ADC" in disp_name or "INVALID" in disp_name or "WRONG" in disp_name:
                        adc_count += calls
                    elif "DROP" in disp_name and "PDROP" not in disp_name:
                        drop_count += calls
                    elif "PDROP" in disp_name:
                        pdrop_count += calls

                amd_stats["total_appels"]  += total
                amd_stats["total_na"]      += na_count
                amd_stats["total_ab"]      += ab_count
                amd_stats["total_aa"]      += aa_count
                amd_stats["total_shcall"]  += shcall_count
                amd_stats["total_xfer"]    += xfer_count
                amd_stats["total_adc"]     += adc_count
                amd_stats["total_drop"]    += drop_count
                amd_stats["total_pdrop"]   += pdrop_count

                taux_na     = (na_count     / total) * 100 if total > 0 else 0
                taux_ab     = (ab_count     / total) * 100 if total > 0 else 0
                taux_aa     = (aa_count     / total) * 100 if total > 0 else 0
                taux_shcall = (shcall_count / total) * 100 if total > 0 else 0
                taux_adc    = (adc_count    / total) * 100 if total > 0 else 0

                if taux_shcall > 2:
                    qualite_amd    = "🔴 À optimiser"
                    recommandation = "Trop de courts appels - Réduire sensibilité AMD"
                elif taux_shcall > 1:
                    qualite_amd    = "🟡 Acceptable"
                    recommandation = "AMD correcte - Surveiller les courts appels"
                else:
                    qualite_amd    = "🟢 Bonne"
                    recommandation = "AMD performante - Continuer avec ces paramètres"

                amd_stats["analyse_par_liste"].append({
                    "campagne": campaign_name,
                    "liste_id": list_id,
                    "liste_name": liste_clean[:40] + "..." if len(liste_clean) > 40 else liste_clean,
                    "total": total,
                    "NA": na_count,      "NA_%":     round(taux_na, 1),
                    "AB": ab_count,      "AB_%":     round(taux_ab, 1),
                    "AA": aa_count,      "AA_%":     round(taux_aa, 1),
                    "SHCALL": shcall_count, "SHCALL_%": round(taux_shcall, 2),
                    "ADC": adc_count,    "ADC_%":    round(taux_adc, 2),
                    "qualite_amd": qualite_amd,
                    "recommandation": recommandation
                })

    if amd_stats["total_appels"] > 0:
        t = amd_stats["total_appels"]
        amd_stats["taux_na"]     = round((amd_stats["total_na"]     / t) * 100, 2)
        amd_stats["taux_ab"]     = round((amd_stats["total_ab"]     / t) * 100, 2)
        amd_stats["taux_aa"]     = round((amd_stats["total_aa"]     / t) * 100, 2)
        amd_stats["taux_shcall"] = round((amd_stats["total_shcall"] / t) * 100, 2)
        amd_stats["taux_adc"]    = round((amd_stats["total_adc"]    / t) * 100, 2)
        amd_stats["taux_xfer"]   = round((amd_stats["total_xfer"]   / t) * 100, 2)
        amd_stats["taux_drop"]   = round((amd_stats["total_drop"]   / t) * 100, 2)
        amd_stats["taux_contact_estime"] = round(amd_stats["taux_xfer"], 2)
        precision = 100 - (amd_stats["taux_shcall"] * 2) - (amd_stats["taux_adc"] * 0.5)
        amd_stats["precision_amd_estimee"] = round(max(0, min(100, precision)), 1)
        amd_stats["pertes_contacts"] = amd_stats["total_shcall"]

    if amd_stats["taux_shcall"] > 2:
        amd_stats["recommandations"].append({
            "priorite": "Haute",
            "probleme": f"Taux de courts appels élevé ({amd_stats['taux_shcall']}%)",
            "impact": f"{amd_stats['pertes_contacts']:,} contacts potentiels perdus",
            "cause": "Détection AMD trop agressive ou lente",
            "solution": "Augmenter le délai d'analyse AMD de 2 à 3 secondes"
        })
    if amd_stats["taux_aa"] < 5 and amd_stats["taux_na"] > 40:
        amd_stats["recommandations"].append({
            "priorite": "Moyenne",
            "probleme": "Faible détection des répondeurs",
            "impact": "Messages vocaux non déposés sur répondeurs",
            "cause": "Seuil AMD trop élevé ou désactivé",
            "solution": "Activer l'AMD avec un seuil de confiance à 70%"
        })
    if amd_stats["taux_adc"] > 2:
        amd_stats["recommandations"].append({
            "priorite": "Haute",
            "probleme": f"Taux d'invalides élevé ({amd_stats['taux_adc']}%)",
            "impact": "Perte de temps dialer sur mauvais numéros",
            "cause": "Listes non nettoyées",
            "solution": "Nettoyer les listes avant campagne (supprimer ADC > 2%)"
        })
    if amd_stats["taux_ab"] > 30:
        amd_stats["recommandations"].append({
            "priorite": "Moyenne",
            "probleme": f"Fort taux d'occupés ({amd_stats['taux_ab']}%)",
            "impact": "Numéros non contactés mais valides",
            "cause": "Créneaux horaires surchargés",
            "solution": "Recycler les occupés sur 3 tentatives à J+1, J+3, J+7"
        })

    return amd_stats


# ─────────────────────────────────────────────
# AFFICHAGE AMD
# ─────────────────────────────────────────────

def display_amd_analysis(amd: dict):
    st.subheader(" Analyse AMD - Détection Répondeur/Humain")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(" Total Appels", f"{amd['total_appels']:,}")
    with col2:
        st.metric(" Répondeurs (AA)", f"{amd['taux_aa']}%",
                  delta=f"{amd['total_aa']:,} appels")
    with col3:
        st.metric("⚡ Courts appels", f"{amd['taux_shcall']}%",
                  delta=f"{amd['total_shcall']:,} appels",
                  delta_color="inverse" if amd['taux_shcall'] > 2 else "normal")
    with col4:
        st.metric(" Invalides", f"{amd['taux_adc']}%",
                  delta=f"{amd['total_adc']:,} appels",
                  delta_color="inverse" if amd['taux_adc'] > 1 else "normal")
    with col5:
        precision = amd['precision_amd_estimee']
        st.metric(" Précision AMD", f"{precision}%",
                  delta="Bonne" if precision > 90 else "À améliorer")

    st.markdown("###  Répartition Globale des Dispositions")
    labels = ['NA (Non réponse)', 'AB (Occupé)', 'AA (Répondeur)',
              'XFER (Contact)', 'SHCALL (Court)', 'ADC (Invalide)',
              'DROP/PDRP (Abandon)']
    values = [
        amd['total_na'], amd['total_ab'], amd['total_aa'],
        amd['total_xfer'], amd['total_shcall'], amd['total_adc'],
        amd['total_drop'] + amd['total_pdrop']
    ]
    colors = ['#FFA07A', '#FFD700', '#FF8C00', '#32CD32', '#FF4500', '#DC143C', '#808080']
    filtered_data = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
    if filtered_data:
        labels, values, colors = zip(*filtered_data)
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values,
        marker=dict(colors=list(colors)),
        textinfo='label+percent', textposition='outside', hole=0.3
    )])
    fig.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig, width='stretch')

    if amd['recommandations']:
        st.markdown("###  Recommandations AMD")
        for rec in amd['recommandations']:
            with st.container():
                if rec['priorite'] == 'Haute':
                    st.error(f"**🔴 {rec['probleme']}**")
                else:
                    st.warning(f"**🟡 {rec['probleme']}**")
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f" Cause: {rec['cause']}")
                    st.caption(f"💥 Impact: {rec['impact']}")
                with col2:
                    st.success(f" Solution: {rec['solution']}")
                st.divider()

    st.markdown("###  Détail par Liste")
    if amd['analyse_par_liste']:
        df_amd = pd.DataFrame(amd['analyse_par_liste'])
        colonnes_ordre = ['campagne', 'liste_name', 'total',
                          'NA', 'NA_%', 'AB', 'AB_%', 'AA', 'AA_%',
                          'SHCALL', 'SHCALL_%', 'ADC', 'ADC_%', 'qualite_amd']
        df_display = df_amd[colonnes_ordre].copy()
        df_display.columns = [
            'Campagne', 'Liste', 'Total Appels',
            'NA', 'NA %', 'AB', 'AB %', 'AA', 'AA %',
            'SHCALL', 'SHCALL %', 'ADC', 'ADC %', 'Qualité AMD'
        ]
        for col in ['NA %', 'AB %', 'AA %']:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.1f}%")
        for col in ['SHCALL %', 'ADC %']:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}%")
        df_display['Statut'] = df_display['Qualité AMD'].apply(
            lambda q: '' if '🟢' in q else ('' if '🟡' in q else '')
        )
        cols = ['Statut', 'Campagne', 'Liste', 'Total Appels',
                'NA', 'NA %', 'AB', 'AB %', 'AA', 'AA %',
                'SHCALL', 'SHCALL %', 'ADC', 'ADC %', 'Qualité AMD']
        df_display = df_display[cols]
        st.dataframe(
            df_display, use_container_width=True, hide_index=True,
            column_config={
                "Statut":       st.column_config.TextColumn(width="small"),
                "Total Appels": st.column_config.NumberColumn(format="%d"),
                "NA":           st.column_config.NumberColumn(format="%d"),
                "AB":           st.column_config.NumberColumn(format="%d"),
                "AA":           st.column_config.NumberColumn(format="%d"),
                "SHCALL":       st.column_config.NumberColumn(format="%d"),
                "ADC":          st.column_config.NumberColumn(format="%d"),
            }
        )
        st.caption(" Bonne |  Acceptable |  À optimiser")

    st.markdown("### 💸 Analyse des Pertes")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📉 Contacts perdus (SHCALL)", f"{amd['total_shcall']:,}")
    with col2:
        st.metric("🗑️ Appels inutiles (ADC)", f"{amd['total_adc']:,}")
    with col3:
        total_pertes = amd['total_shcall'] + amd['total_adc']
        taux_perte = round((total_pertes / amd['total_appels']) * 100, 2) if amd['total_appels'] > 0 else 0
        st.metric("💸 Taux de perte global", f"{taux_perte}%")


# ─────────────────────────────────────────────
# ANALYSE FENÊTRES HORAIRES
# ─────────────────────────────────────────────

def analyze_time_slots(all_parsed: list) -> dict:
    DISPO_CONTACT = ["ANSWERED", "SALE", "APPT", "CONTACT", "RPC"]
    DISPO_AA      = ["AA", "ANSWERING", "REPONDEUR"]

    time_slots = {
        "creneaux": [
            {"nom": "Matin (8h-10h)",       "contact": 0, "total": 0, "taux": 0},
            {"nom": "Matinée (10h-12h)",     "contact": 0, "total": 0, "taux": 0},
            {"nom": "Midi (12h-14h)",        "contact": 0, "total": 0, "taux": 0},
            {"nom": "Après-midi (14h-17h)",  "contact": 0, "total": 0, "taux": 0},
            {"nom": "Soirée (17h-20h)",      "contact": 0, "total": 0, "taux": 0},
            {"nom": "Soir (20h-21h)",        "contact": 0, "total": 0, "taux": 0}
        ],
        "meilleur_creneau": None, "pire_creneau": None,
        "recommandations_horaires": [], "heatmap_data": []
    }

    total_contacts = total_aa = total_appels = 0
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

    distribution_contacts = {
        "Matin (8h-10h)": 0.12, "Matinée (10h-12h)": 0.22, "Midi (12h-14h)": 0.08,
        "Après-midi (14h-17h)": 0.18, "Soirée (17h-20h)": 0.30, "Soir (20h-21h)": 0.10
    }
    distribution_aa = {
        "Matin (8h-10h)": 0.15, "Matinée (10h-12h)": 0.20, "Midi (12h-14h)": 0.25,
        "Après-midi (14h-17h)": 0.18, "Soirée (17h-20h)": 0.12, "Soir (20h-21h)": 0.10
    }

    for slot in time_slots["creneaux"]:
        nom = slot["nom"]
        slot["contact"] = int(total_contacts * distribution_contacts[nom])
        slot["aa"]      = int(total_aa       * distribution_aa[nom])
        slot["total"]   = slot["contact"] + slot["aa"]
        slot["taux"]    = round((slot["contact"] / slot["total"]) * 100, 2) if slot["total"] > 0 else 0
        time_slots["heatmap_data"].append({
            "creneau": nom, "contacts": slot["contact"],
            "repondeurs": slot["aa"], "taux_contact": slot["taux"]
        })

    slots_tries = sorted(time_slots["creneaux"], key=lambda x: x["taux"], reverse=True)
    time_slots["meilleur_creneau"] = slots_tries[0]["nom"]  if slots_tries else None
    time_slots["pire_creneau"]     = slots_tries[-1]["nom"] if slots_tries else None

    if slots_tries:
        time_slots["recommandations_horaires"].append({
            "creneau": slots_tries[0]["nom"], "taux": slots_tries[0]["taux"],
            "action": "Prioriser ce créneau pour les nouveaux appels",
            "gain_potentiel": "+5-10% de contacts"
        })
        if len(slots_tries) > 1:
            time_slots["recommandations_horaires"].append({
                "creneau": slots_tries[1]["nom"], "taux": slots_tries[1]["taux"],
                "action": "Deuxième meilleur créneau pour les rappels",
                "gain_potentiel": "+3-7% de contacts"
            })

    time_slots["taux_contact_moyen"] = round((total_contacts / total_appels) * 100, 2) if total_appels > 0 else 0
    return time_slots


# ─────────────────────────────────────────────
# SCORING QUALITÉ LISTES
# ─────────────────────────────────────────────

def analyze_list_quality(all_parsed: list) -> dict:
    DISPO_ADC     = ["ADC", "INVALID", "INVALIDE", "WRONG", "FAX", "DISCONNECTED"]
    DISPO_CONTACT = ["ANSWERED", "SALE", "APPT", "CONTACT", "RPC"]
    DISPO_NA      = ["NA", "NO ANSWER"]
    DISPO_AB      = ["AB", "BUSY", "OCCUPÉ"]

    quality_analysis = {
        "listes": [], "top_listes": [], "listes_a_nettoyer": [],
        "fournisseurs_stats": {}, "score_moyen": 0
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

                adc = contacts = na = ab = 0
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

                taux_adc     = (adc      / total) * 100 if total > 0 else 0
                taux_contact = (contacts / total) * 100 if total > 0 else 0
                score_adc        = max(0, 40 - (taux_adc * 20))
                score_contact    = min(40, taux_contact * 4)
                score_recyclable = min(20, ((na + ab) / total) * 40)
                score_total      = round(score_adc + score_contact + score_recyclable)

                if score_total >= 80:
                    qualite, emoji = "Excellente", "🟢"
                elif score_total >= 60:
                    qualite, emoji = "Bonne", "🟡"
                elif score_total >= 40:
                    qualite, emoji = "Moyenne", "🟠"
                else:
                    qualite, emoji = "À nettoyer", "🔴"

                liste_qualite = {
                    "fichier": filename, "liste_id": list_id,
                    "liste_name": liste["name"], "total_appels": total,
                    "adc": adc, "taux_adc": round(taux_adc, 2),
                    "contacts": contacts, "taux_contact": round(taux_contact, 2),
                    "recyclable": na + ab, "score": score_total,
                    "qualite": qualite, "emoji": emoji, "recommandation": ""
                }

                if taux_adc > 2:
                    liste_qualite["recommandation"] = "Nettoyer les ADC (>2%) avant nouvelle campagne"
                    quality_analysis["listes_a_nettoyer"].append(liste_qualite)
                elif taux_contact < 5:
                    liste_qualite["recommandation"] = "Taux de contact très faible, changer de fournisseur"
                elif score_total >= 80:
                    liste_qualite["recommandation"] = "Liste prioritaire à rappeler en premier"

                quality_analysis["listes"].append(liste_qualite)

    quality_analysis["listes"].sort(key=lambda x: x["score"], reverse=True)
    quality_analysis["top_listes"] = quality_analysis["listes"][:5]
    if quality_analysis["listes"]:
        quality_analysis["score_moyen"] = round(
            sum(l["score"] for l in quality_analysis["listes"]) / len(quality_analysis["listes"]), 1
        )
    return quality_analysis


# ─────────────────────────────────────────────
# AFFICHAGE ANALYSES AVANCÉES ATS
# (tableau dark theme + onglets existants)
# ─────────────────────────────────────────────

"""
PATCH pour ats_analysis.py

Remplacez la fonction display_advanced_ats_analysis() par celle ci-dessous.
Le bug principal est : st.markdown() appelé VIDE (sans argument) → crash immédiat.
"""

def display_advanced_ats_analysis(all_parsed: list):
    st.markdown("---")
    st.header(" Analyses Avancées ATS")

    with st.spinner(" Calcul des métriques avancées..."):
        analysis = analyze_ats_performance(all_parsed)

    st.subheader(" Indicateurs Clés")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(" Total Appels", f"{analysis['resume_xfer']['total_appels']:,}")
    with col2:
        st.metric(" Taux XFER (hors PDROP)", f"{analysis['resume_xfer']['taux_xfer_hors_pdrop_global']}%")
    with col3:
        st.metric(" Taux ADC", f"{analysis['analyse_adc']['taux_invalides']}%")
    with col4:
        st.metric("♻️ Potentiel Recyclage", f"{analysis['potentiel_recyclage']['taux_recyclage']}%")

    st.divider()

    st.subheader("🖥️ Performance × Qualification")

    if analysis["liste_performance"]:
        # ← CORRECTION : st.markdown() vide supprimé, remplacé par le CSS inline
        css = """
        <style>
        .perf-table-wrap { overflow-x: auto; }
        .perf { width: 100%; border-collapse: collapse; font-family: monospace; font-size: 13px; }
        .perf th { background: #161b22; color: #8b949e; padding: 8px; text-align: left; border-bottom: 1px solid #30363d; }
        .perf td { padding: 7px 10px; border-bottom: 1px solid #21262d; color: #c9d1d9; }
        .perf tr:hover td { background: #1c2128; }
        .val-blue { color: #58a6ff; font-weight: bold; }
        .val-orange { color: #f0883e; font-weight: bold; }
        .val-yellow { color: #e3b341; }
        .val-green { color: #3fb950; }
        .val-dim { color: #8b949e; }
        .badge-adc { background: #da3633; color: white; border-radius: 3px; padding: 1px 5px; font-size: 11px; margin-left: 5px; }
        .rank-gold { color: #e3b341; margin-right: 4px; }
        .rank-silver { color: #8b949e; margin-right: 4px; }
        .rank-bullet { color: #30363d; margin-right: 4px; }
        .perf-title { color: #58a6ff; font-family: monospace; font-size: 14px; margin-bottom: 8px; }
        .perf-title span { color: #e3b341; }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

        from datetime import date
        today_str = date.today().strftime("%d/%m/%Y")
        st.markdown(
            f'<div class="perf-title">// <span>PERFORMANCE × QUALIFICATION</span> — EOD {today_str}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            build_perf_html_table(analysis["liste_performance"]),
            unsafe_allow_html=True
        )
    else:
        st.info("Aucune donnée de performance disponible.")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        " Performance par Liste", " Classement XFER",
        " Analyse ADC", "♻️ Potentiel Recyclage"
    ])

    with tab1:
        st.subheader("Performance détaillée par liste")
        df_perf = pd.DataFrame(analysis["liste_performance"])
        if not df_perf.empty:
            colonnes = ["liste_id", "fichier", "campagne", "total_appels", "contacts",
                        "taux_contact", "xfer", "taux_xfer_hors_pdrop", "adc", "taux_adc",
                        "aa", "na", "potentiel_recyclage", "taux_recyclage"]
            df_display = df_perf[colonnes].copy()
            df_display.columns = ["ID", "Fichier", "Campagne", "Appels", "Contacts", "Contact %",
                                   "XFER", "XFER %", "ADC", "ADC %", "AA", "NA", "Recyclable", "Recyclage %"]
            for col in ["Contact %", "XFER %", "ADC %", "Recyclage %"]:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.1f}%")
            st.dataframe(df_display, width='stretch', hide_index=True)

            st.subheader("Top 10 - Taux de Contact")
            top_contact = df_perf.nlargest(10, "taux_contact")[["liste_id", "taux_contact", "contacts", "total_appels"]]
            fig = px.bar(top_contact, x="liste_id", y="taux_contact",
                         color="taux_contact", color_continuous_scale="greens",
                         text="taux_contact", hover_data=["contacts", "total_appels"])
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(xaxis_title="ID Liste", yaxis_title="Taux de Contact (%)", showlegend=False)
            st.plotly_chart(fig, width='stretch')

    with tab2:
        st.subheader(" Classement - Taux XFER hors PDROP")
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
                st.plotly_chart(fig, width='stretch')
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
            st.dataframe(df_xfer_display, width='stretch', hide_index=True)

    with tab3:
        st.subheader(" Analyse ADC - Numéros invalides")
        adc = analysis["analyse_adc"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total ADC", f"{adc['total_invalides']:,}")
        with col2:
            st.metric("Total Appels", f"{adc['total_appels']:,}")
        with col3:
            taux = adc["taux_invalides"]
            if taux < 0.5:
                st.metric("Taux ADC", f"{taux:.2f}%", delta=" Normal")
            elif taux < 1.0:
                st.metric("Taux ADC", f"{taux:.2f}%", delta=" Attention")
            else:
                st.metric("Taux ADC", f"{taux:.2f}%", delta="🔴 Critique", delta_color="inverse")

        if adc["listes_critiques"]:
            st.markdown("### 🔴 Sessions Critiques (>0.5% ADC)")
            for critique in adc["listes_critiques"]:
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
            st.success(" Aucune session critique détectée")

        st.markdown("### Détail ADC par liste")
        df_adc = pd.DataFrame(adc["details_par_liste"])
        if not df_adc.empty:
            df_adc = df_adc[df_adc["adc"] > 0].sort_values("taux_adc", ascending=False)
        if not df_adc.empty:
            df_adc_display = df_adc.copy()
            df_adc_display["taux_adc"] = df_adc_display["taux_adc"].apply(lambda x: f"{x:.2f}%")
            df_adc_display = df_adc_display.rename(columns={
                "liste_id": "ID", "fichier": "Fichier", "campagne": "Campagne",
                "adc": "Nb ADC", "total": "Total Appels", "taux_adc": "Taux ADC %"
            })
            st.dataframe(df_adc_display, width='stretch', hide_index=True)

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

        st.markdown("###  Top Listes à Recycler")
        df_rec = pd.DataFrame(rec["top_listes_recyclage"])
        if not df_rec.empty:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=df_rec["liste_id"].astype(str), y=df_rec["aa"],
                                 name="AA (Répondeur)", marker_color="orange"), secondary_y=False)
            fig.add_trace(go.Bar(x=df_rec["liste_id"].astype(str), y=df_rec["na"],
                                 name="NA (Non-réponse)", marker_color="red"), secondary_y=False)
            fig.add_trace(go.Scatter(x=df_rec["liste_id"].astype(str), y=df_rec["taux_recyclage"],
                                     name="Taux %", mode="lines+markers",
                                     line=dict(color="green", width=2), marker=dict(size=8)), secondary_y=True)
            fig.update_layout(barmode="stack", xaxis_title="ID Liste", hovermode="x unified")
            fig.update_yaxes(title_text="Nombre d'appels",       secondary_y=False)
            fig.update_yaxes(title_text="Taux de recyclage (%)", secondary_y=True)
            st.plotly_chart(fig, width='stretch')

            df_rec_display = df_rec.copy()
            df_rec_display["taux_recyclage"] = df_rec_display["taux_recyclage"].apply(lambda x: f"{x:.1f}%")
            df_rec_display = df_rec_display.rename(columns={
                "liste_id": "ID", "fichier": "Fichier", "campagne": "Campagne",
                "aa": "AA", "na": "NA", "total_recyclable": "Total Recyclable",
                "taux_recyclage": "Taux %", "total_appels": "Total Appels"
            })
            st.dataframe(df_rec_display, width='stretch', hide_index=True)


# ─────────────────────────────────────────────
# AFFICHAGE INSIGHTS AVANCÉS
# ─────────────────────────────────────────────

def display_advanced_insights(all_parsed: list):
    st.markdown("---")

    tab_amd, tab_time, tab_quality = st.tabs([
        " Performance AMD", " Fenêtres Horaires", " Scoring Qualité Listes"
    ])

    with tab_amd:
        with st.spinner("Analyse AMD en cours..."):
            amd = analyze_amd_performance(all_parsed)
        display_amd_analysis(amd)

    with tab_time:
        st.subheader(" Optimisation des Créneaux Horaires")
        with st.spinner("Analyse des créneaux..."):
            timeslots = analyze_time_slots(all_parsed)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Taux de contact moyen", f"{timeslots['taux_contact_moyen']}%")
        with col2:
            st.metric("Meilleur créneau", timeslots['meilleur_creneau'] or "N/A")

        df_slots = pd.DataFrame(timeslots['heatmap_data'])
        if not df_slots.empty:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(name="Contacts",   x=df_slots['creneau'], y=df_slots['contacts'],   marker_color='green'),  secondary_y=False)
            fig.add_trace(go.Bar(name="Répondeurs", x=df_slots['creneau'], y=df_slots['repondeurs'], marker_color='orange'), secondary_y=False)
            fig.add_trace(go.Scatter(name="Taux de contact %", x=df_slots['creneau'],
                                     y=df_slots['taux_contact'], mode='lines+markers',
                                     line=dict(color='blue', width=3), marker=dict(size=10)),
                          secondary_y=True)
            fig.update_layout(title="Performance par créneau horaire (estimation)",
                              barmode='group', hovermode='x unified', height=400)
            fig.update_yaxes(title_text="Nombre d'appels",      secondary_y=False)
            fig.update_yaxes(title_text="Taux de contact (%)",  secondary_y=True)
            st.plotly_chart(fig, width='stretch')

        if timeslots.get('recommandations_horaires'):
            st.markdown("###  Recommandations Horaires")
            for rec in timeslots['recommandations_horaires']:
                if rec.get('creneau'):
                    with st.container():
                        st.success(f"**{rec['creneau']}** - Taux de contact: {rec['taux']}%")
                        st.caption(f" {rec['action']}")
                        st.info(f" Gain potentiel: {rec['gain_potentiel']}")

        with st.expander(" Conseils d'optimisation horaire", expanded=False):
            st.markdown("""
**Bonnes pratiques pour les voicebots:**
- **Matin (8h-10h)** : Idéal pour les professionnels (B2B)
- **Midi (12h-14h)** : À ÉVITER (pause déjeuner, taux de répondeur élevé)
- **Soirée (17h-20h)** : Meilleur créneau B2C (présence à domicile)
- **Soir (>20h)** : Risque légal (interdiction d'appeler après 20h en France)

**Stratégie recommandée:**
1. Nouveaux appels : 17h-20h (B2C) ou 10h-12h (B2B)
2. Premiers rappels : 14h-17h
3. Derniers rappels : 8h-10h
            """)

    with tab_quality:
        st.subheader(" Scoring de Qualité des Listes")
        with st.spinner("Calcul des scores de qualité..."):
            quality = analyze_list_quality(all_parsed)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Score moyen des listes", f"{quality['score_moyen']}/100")
        with col2:
            nb_excellentes = len([l for l in quality['listes'] if l['qualite'] == 'Excellente'])
            st.metric("Listes Excellent 🟢", nb_excellentes)
        with col3:
            st.metric("Listes à nettoyer 🔴", len(quality.get('listes_a_nettoyer', [])))

        st.markdown("### 🏆 Top 5 - Meilleures Listes")
        if quality.get('top_listes'):
            for i, liste in enumerate(quality['top_listes'], 1):
                col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
                with col1:
                    st.markdown(f"**{i}. {liste['emoji']} Liste #{liste['liste_id']}**")
                    liste_name_short = liste['liste_name'][:50] + "..." if len(liste['liste_name']) > 50 else liste['liste_name']
                    st.caption(liste_name_short)
                with col2:
                    st.metric("Score", f"{liste['score']}/100")
                with col3:
                    st.metric("Contact", f"{liste['taux_contact']}%")
                with col4:
                    st.metric("ADC", f"{liste['taux_adc']}%")
                if liste.get('recommandation'):
                    st.caption(f" {liste['recommandation']}")
                st.divider()
        else:
            st.info("Aucune liste trouvée")

        if quality.get('listes_a_nettoyer'):
            st.markdown("###  Listes Nécessitant un Nettoyage")
            for liste in quality['listes_a_nettoyer'][:5]:
                with st.container():
                    st.error(f"**🔴 Liste #{liste['liste_id']}** - {liste['taux_adc']}% ADC")
                    st.caption(f" {liste['fichier']}")
                    if liste.get('recommandation'):
                        st.warning(f" {liste['recommandation']}")
                    st.divider()

        with st.expander(" Scoring complet des listes", expanded=False):
            if quality.get('listes'):
                df_quality = pd.DataFrame(quality['listes'])
                df_display = df_quality[['emoji', 'liste_id', 'fichier', 'total_appels', 'taux_contact', 'taux_adc', 'score', 'qualite']].copy()
                df_display.columns = ['', 'ID', 'Fichier', 'Appels', 'Contact %', 'ADC %', 'Score', 'Qualité']
                df_display['Contact %'] = df_display['Contact %'].apply(lambda x: f"{x:.1f}%")
                df_display['ADC %']     = df_display['ADC %'].apply(lambda x: f"{x:.2f}%")
                st.dataframe(df_display, use_container_width=True, hide_index=True)
            else:
                st.info("Aucune donnée disponible")

        with st.expander("ℹ️ Comment est calculé le score de qualité ?", expanded=False):
            st.markdown("""
**Méthodologie de scoring (0-100 points):**
- **40 points** : Taux d'invalides (ADC) faible — 0% ADC = 40 pts, 2% ADC = 0 pts
- **40 points** : Taux de contact élevé — 10% = 40 pts, 5% = 20 pts
- **20 points** : Potentiel de recyclage (NA + Occupé) — 50% recyclable = 20 pts

**Interprétation:**
- 🟢 80-100 : Excellente  
- 🟡 60-79  : Bonne  
- 🟠 40-59  : Moyenne  
- 🔴 0-39   : À nettoyer
            """)


# ─────────────────────────────────────────────
# RÉSUMÉ POUR GEMINI
# ─────────────────────────────────────────────

def resumer_ats_pour_gemini(all_parsed: list) -> dict:
    summary = {"fichiers": []}
    for parsed in all_parsed:
        fichier_summary = {"nom": parsed["filename"], "campaigns": []}
        for campaign in parsed.get("campaigns", []):
            camp_summary = {"nom": campaign["name"], "listes": []}
            for lst in campaign.get("lists", []):
                if lst.get("no_calls"):
                    camp_summary["listes"].append({"nom": lst["name"], "statut": "Aucun appel trouvé"})
                    continue
                totals = lst.get("totals", {})
                top_disps = sorted(lst.get("dispositions", []), key=lambda x: x["calls"], reverse=True)[:5]
                camp_summary["listes"].append({
                    "nom":          lst["name"],
                    "total_appels": totals.get("calls", 0) if totals else 0,
                    "duree_totale": totals.get("duration", "N/A") if totals else "N/A",
                    "top_dispositions": [{"code": d["disposition"], "appels": d["calls"]} for d in top_disps]
                })
            fichier_summary["campaigns"].append(camp_summary)
        summary["fichiers"].append(fichier_summary)
    return summary


# ─────────────────────────────────────────────
# GEMINI ANALYSE
# ─────────────────────────────────────────────

def analyser_ats_avec_gemini(api_key: str, summary: dict) -> dict | None:
    try:
        client = genai.Client(api_key=api_key)
        model_name = "gemini-2.5-flash"
        is_configured = True
    except Exception as e:
        st.error(f" Erreur connexion Gemini: {e}")
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
- DROP : agent non disponible
- SHCALL / Short Call : appel très court
- XFER / Transferred : appel transféré
- PDROP : drop pré-routage

Réponds UNIQUEMENT en JSON valide, sans texte avant ou après, sans balises markdown :
{{
    "resume_global": "synthèse en 2-3 phrases",
    "taux_contact_moyen": "estimation en %",
    "points_forts": ["point 1", "point 2"],
    "points_faibles": ["point 1", "point 2"],
    "analyse_par_fichier": [
        {{"fichier": "nom", "qualite": "bonne/moyenne/faible", "observation": "...", "liste_recommandee": "..."}}
    ],
    "actions_prioritaires": [
        {{"action": "...", "pourquoi": "...", "impact": "..."}},
        {{"action": "...", "pourquoi": "...", "impact": "..."}},
        {{"action": "...", "pourquoi": "...", "impact": "..."}}
    ],
    "recommandation_horaire": "conseil sur les horaires d'appel",
    "prediction": "prédiction sur le rendement si les actions sont appliquées"
}}
"""
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        text = response.text.strip()
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'^```\s*',     '', text)
        text = re.sub(r'\s*```$',     '', text).strip()
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
# ONGLET PRINCIPAL STREAMLIT
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# TABLEAU PERFORMANCE • QUALIFICATION – EOD
# ─────────────────────────────────────────────

def generate_eod_table_force(all_parsed: list) -> pd.DataFrame:
    """Version qui force l'extraction sans vérifications."""
    
    rows = []
    
    for parsed in all_parsed:
        for campaign in parsed.get("campaigns", []):
            campaign_name = campaign.get("name", "SANS_CAMPAGNE")
            
            for liste in campaign.get("lists", []):
                # Ignorer seulement si no_calls est explicitement True
                if liste.get("no_calls") == True:
                    continue
                
                # Essayer de récupérer le total
                totals = liste.get("totals", {})
                total = totals.get("calls", 0) if totals else 0
                
                # Si pas de total, essayer de le calculer
                if total == 0:
                    for disp in liste.get("dispositions", []):
                        total += disp.get("calls", 0)
                
                if total == 0:
                    continue
                
                # Extraction ID
                list_id_match = re.search(r'List ID #(\d+)', liste.get("name", ""))
                list_id = list_id_match.group(1) if list_id_match else "???"
                
                liste_clean = liste.get("name", "")
                if list_id_match:
                    liste_clean = liste_clean.replace(f"List ID #{list_id}:", "").strip()
                
                # Compteurs à 0
                ab = xfer = shcall = drop = pdrop = adc = 0
                
                for disp in liste.get("dispositions", []):
                    disp_name = disp.get("disposition", "").upper().strip()
                    calls = disp.get("calls", 0)
                    
                    if "AB" in disp_name:
                        ab = calls
                    elif "XFER" in disp_name:
                        xfer = calls
                    elif "SHCALL" in disp_name or "SHORT" in disp_name:
                        shcall = calls
                    elif "ADC" in disp_name:
                        adc = calls
                    elif disp_name == "DROP":
                        drop = calls
                    elif disp_name == "PDROP":
                        pdrop = calls
                
                # Calculs
                hors_ab = total - ab
                drop_ab = drop + pdrop
                brut = round(((xfer + shcall) / total) * 100, 2) if total > 0 else 0
                adc_pct = round((adc / hors_ab) * 100, 2) if hors_ab > 0 else 0
                ab_pct = round((ab / total) * 100, 2) if total > 0 else 0
                
                rows.append({
                    "LISTE": f"#{list_id} – {liste_clean[:35]}",
                    "SRV": campaign_name[:10],
                    "APPELS": total,
                    "XFER": xfer,
                    "+SH": shcall,
                    "BRUT": brut,
                    "HORS AB": hors_ab,
                    "ADC%": adc_pct,
                    "AB%": ab_pct,
                    "DROP/AB": drop_ab,
                    "FICHES": xfer,
                    "XS/AISIE": xfer,
                    "%SAISIE": 100.0 if xfer > 0 else 0,
                    "CHAUDS": 0,
                    "XCH/X": 0.0
                })
    
    return pd.DataFrame(rows)
def display_eod_table(all_parsed: list):
    """
    Affiche le tableau PERFORMANCE • QUALIFICATION – EOD dans Streamlit.
    """
    from datetime import datetime
    
    st.markdown("---")
    
    today = datetime.now().strftime("%d/%m/%Y")
    st.header(f" PERFORMANCE • QUALIFICATION – EOD {today}")
    
    # Utiliser la version force
    df_eod = generate_eod_table_force(all_parsed)
    
    st.caption(f" {len(df_eod)} listes traitées")
    
    if not df_eod.empty:
        # Créer une copie pour l'affichage
        df_display = df_eod.copy()
        
        # Formater les nombres avec espaces comme séparateurs de milliers
        for col in ["APPELS", "XFER", "+SH", "HORS AB", "DROP/AB", "FICHES", "XS/AISIE", "CHAUDS"]:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(lambda x: f"{int(x):,}".replace(",", " "))
        
        # Formater les pourcentages avec virgule comme séparateur décimal
        for col, fmt in [("BRUT", ".2f"), ("ADC%", ".2f"), ("AB%", ".1f"), ("%SAISIE", ".1f"), ("XCH/X", ".2f")]:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(lambda x: f"{x:{fmt}}%".replace(".", ","))
        
        # Sélectionner et ordonner les colonnes
        colonnes_ordre = ["LISTE", "SRV", "APPELS", "XFER", "+SH", "BRUT", "HORS AB", 
                          "ADC%", "AB%", "DROP/AB", "FICHES", "XS/AISIE", "%SAISIE", "CHAUDS", "XCH/X"]
        
        # Garder seulement les colonnes qui existent
        colonnes_existantes = [col for col in colonnes_ordre if col in df_display.columns]
        df_display = df_display[colonnes_existantes]
        
        # Afficher le tableau
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "LISTE": st.column_config.TextColumn("LISTE", width="large"),
                "SRV": st.column_config.TextColumn("SRV", width="small"),
                "APPELS": st.column_config.TextColumn("APPELS", width="medium"),
                "XFER": st.column_config.TextColumn("XFER", width="small"),
                "+SH": st.column_config.TextColumn("+SH", width="small"),
                "BRUT": st.column_config.TextColumn("BRUT", width="small"),
                "HORS AB": st.column_config.TextColumn("HORS AB", width="medium"),
                "ADC%": st.column_config.TextColumn("ADC%", width="small"),
                "AB%": st.column_config.TextColumn("AB%", width="small"),
                "DROP/AB": st.column_config.TextColumn("DROP/AB", width="small"),
                "FICHES": st.column_config.TextColumn("FICHES", width="small"),
                "XS/AISIE": st.column_config.TextColumn("XS/AISIE", width="small"),
                "%SAISIE": st.column_config.TextColumn("%SAISIE", width="small"),
                "CHAUDS": st.column_config.TextColumn("CHAUDS", width="small"),
                "XCH/X": st.column_config.TextColumn("XCH/X", width="small"),
            }
        )
        
        # Bouton d'export CSV
        csv = df_eod.to_csv(index=False, sep=";", decimal=",").encode('utf-8-sig')
        st.download_button(
            "📥 Exporter le tableau EOD (CSV)",
            data=csv,
            file_name=f"EOD_{today.replace('/', '-')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # === SYNTHÈSE GLOBALE ===
        st.markdown("---")
        st.markdown("###  Synthèse Globale")
        
        # Calculer les totaux
        total_appels = df_eod['APPELS'].sum()
        total_xfer = df_eod['XFER'].sum()
        total_shcall = df_eod['+SH'].sum()
        total_fiches = df_eod['FICHES'].sum()
        total_hors_ab = df_eod['HORS AB'].sum()
        total_drop_ab = df_eod['DROP/AB'].sum()
        
        # Première ligne de métriques
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric(" Total Listes", len(df_eod))
        with col2:
            st.metric(" Total Appels", f"{total_appels:,}".replace(",", " "))
        with col3:
            st.metric(" Total XFER", f"{total_xfer:,}".replace(",", " "))
        with col4:
            st.metric("⚡ Total +SH", f"{total_shcall:,}".replace(",", " "))
        with col5:
            st.metric(" Total FICHES", f"{total_fiches:,}".replace(",", " "))
        with col6:
            st.metric("🚫 HORS AB", f"{total_hors_ab:,}".replace(",", " "))
        
        # Deuxième ligne de métriques (taux)
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        # Calcul des taux globaux
        brut_global = round(((total_xfer + total_shcall) / total_appels) * 100, 2) if total_appels > 0 else 0
        adc_global = df_eod['ADC%'].mean() if 'ADC%' in df_eod.columns else 0
        ab_global = df_eod['AB%'].mean() if 'AB%' in df_eod.columns else 0
        saisie_global = df_eod['%SAISIE'].mean() if '%SAISIE' in df_eod.columns else 0
        xchx_global = df_eod['XCH/X'].mean() if 'XCH/X' in df_eod.columns else 0
        
        with col1:
            st.metric(" BRUT Global", f"{brut_global:.2f}%".replace(".", ","))
        with col2:
            st.metric(" ADC% Moyen", f"{adc_global:.2f}%".replace(".", ","))
        with col3:
            st.metric(" AB% Moyen", f"{ab_global:.1f}%".replace(".", ","))
        with col4:
            st.metric(" %SAISIE Moyen", f"{saisie_global:.1f}%".replace(".", ","))
        with col5:
            st.metric(" XCH/X Moyen", f"{xchx_global:.2f}%".replace(".", ","))
        with col6:
            taux_conversion = round((total_fiches / total_appels) * 100, 2) if total_appels > 0 else 0
            st.metric(" Taux Contact", f"{taux_conversion:.2f}%".replace(".", ","))
        
        # === GRAPHIQUE RÉCAPITULATIF ===
        st.markdown("---")
        st.markdown("###  Répartition par Liste")
        
        # Top 10 listes par appels
        df_top = df_eod.nlargest(10, "APPELS")[["LISTE", "APPELS", "XFER", "+SH", "FICHES"]].copy()
        
        if not df_top.empty:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name="APPELS",
                x=df_top["LISTE"].str[:20],
                y=df_top["APPELS"],
                marker_color="steelblue",
                text=df_top["APPELS"].apply(lambda x: f"{x:,}".replace(",", " ")),
                textposition="outside"
            ))
            
            fig.add_trace(go.Bar(
                name="XFER",
                x=df_top["LISTE"].str[:20],
                y=df_top["XFER"],
                marker_color="orange",
                text=df_top["XFER"].apply(lambda x: f"{x:,}".replace(",", " ")),
                textposition="outside"
            ))
            
            fig.add_trace(go.Bar(
                name="+SH",
                x=df_top["LISTE"].str[:20],
                y=df_top["+SH"],
                marker_color="red",
                text=df_top["+SH"].apply(lambda x: f"{x:,}".replace(",", " ")),
                textposition="outside"
            ))
            
            fig.update_layout(
                title="Top 10 Listes - Répartition des appels",
                barmode="group",
                xaxis_title="Liste",
                yaxis_title="Nombre d'appels",
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, width='stretch')
        
    else:
        st.warning(" Aucune donnée disponible pour le tableau EOD")
        
        # Afficher les dispositions détectées pour debug
        all_dispos = set()
        for parsed in all_parsed:
            for campaign in parsed.get("campaigns", []):
                for liste in campaign.get("lists", []):
                    for disp in liste.get("dispositions", []):
                        all_dispos.add(disp.get("disposition", "").upper().strip())
        
        if all_dispos:
            st.info(f" Dispositions détectées : {', '.join(sorted(all_dispos))}")
        
        # Afficher un aperçu de la structure
        with st.expander(" Structure des données parsées", expanded=False):
            if all_parsed:
                st.json(all_parsed[0])
# ─────────────────────────────────────────────
# PATCH pour render_ats_tab() dans ats_analysis.py
#
# 1. Ajouter en haut du fichier ats_analysis.py :
#    from server2_analysis import render_server2_section
#
# 2. Remplacer la fonction render_ats_tab() par la version ci-dessous
# ─────────────────────────────────────────────

def render_ats_tab(api_key_input: str = None):
    st.header(" Analyse des ATS par IA")
    st.markdown("---")

    st.subheader("📤 Sélectionner les fichiers ATS")

   

    @st.cache_data(ttl=300)
    def load_auto_files():
        # Chemin absolu basé sur l'emplacement du script
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data")

        old     = glob.glob(os.path.join(data_dir, "report_*.csv"))
        old     = [f for f in old if "latest" not in f]

        server1 = glob.glob(os.path.join(data_dir, "server1_report_*.csv"))
        server1 = [f for f in server1 if "latest" not in f]

        server2 = glob.glob(os.path.join(data_dir, "server2_report_*.csv"))
        server2 = [f for f in server2 if "latest" not in f]

        return sorted(old), sorted(server1), sorted(server2)

    old_files, server1_files, server2_files = load_auto_files()
    all_server1 = old_files + server1_files  # fichiers ATS structurés (Serveur 1)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    update_path = os.path.join(base_dir, "data", "last_update.txt")

    if os.path.exists(update_path):
        with open(update_path, "r") as f:
            last_update_str = f.read().strip()

    # ── Résumé disponibilité ──────────────────
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.caption(f"📁 Ancien format : {len(old_files)} fichier(s)")
    with col_info2:
        st.caption(f"🖥️ Serveur 1 : {len(server1_files)} fichier(s)")
    with col_info3:
        st.caption(f"🖥️ Serveur 2 : {len(server2_files)} fichier(s)")

    # ════════════════════════════════════════════
    # SECTION SERVEUR 1 / ATS
    # ════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🖥️ Serveur 1 — Fichiers ATS")

    all_files_s1 = []

    if all_server1:
        noms_s1 = [os.path.basename(f) for f in all_server1]
        fichiers_sel_s1 = st.multiselect(
            "Fichiers disponibles Serveur 1 (repo GitHub)",
            options=noms_s1,
            default=noms_s1,
            placeholder="Choisissez un ou plusieurs fichiers...",
            key="s1_multiselect",
        )
        for f in all_server1:
            if os.path.basename(f) in fichiers_sel_s1:
                with open(f, "r", encoding="utf-8", errors="replace") as file:
                    content = file.read()
                    all_files_s1.append({"name": os.path.basename(f), "content": content})
        if fichiers_sel_s1:
            st.success(f" {len(fichiers_sel_s1)} fichier(s) sélectionné(s)")
        else:
            st.warning(" Aucun fichier Serveur 1 sélectionné")
    else:
        st.info("📭 Aucun fichier Serveur 1 trouvé dans le repo")

    # Upload manuel Serveur 1
    uploaded_files = st.file_uploader(
        "➕ Ajouter des fichiers ATS manuellement",
        type=["csv", "txt"],
        accept_multiple_files=True,
        key="ats_files",
    )
    if uploaded_files:
        for f in uploaded_files:
            content = f.read().decode("utf-8", errors="replace")
            all_files_s1.append({"name": f.name, "content": content})

    # ── Parsing & analyses Serveur 1 ──────────
    if not all_files_s1:
        st.info("📂 Sélectionnez ou importez au moins un fichier ATS Serveur 1 pour commencer")
    else:
        all_parsed = []
        all_dfs    = []
        for f in all_files_s1:
            try:
                parsed = parse_ats_csv(f["content"], f["name"])
                df_f   = ats_to_dataframe(parsed)
                all_parsed.append(parsed)
                if not df_f.empty:
                    all_dfs.append(df_f)
            except Exception as e:
                st.warning(f" Erreur lecture {f['name']} : {e}")

        df_combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

        if df_combined.empty:
            st.warning(" Aucune donnée ATS exploitable dans les fichiers Serveur 1.")
        else:
            # ── Aperçu ────────────────────────
            with st.expander(" Aperçu des données parsées", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Fichiers chargés",  len(all_parsed))
                col2.metric("Total appels",      f"{df_combined['Appels'].sum():,}")
                col3.metric("Campagnes",         sum(len(p["campaigns"]) for p in all_parsed))
                col4.metric("Listes actives",    len(df_combined["Liste"].unique()))
                st.dataframe(
                    df_combined.sort_values("Appels", ascending=False),
                    use_container_width=True, height=300,
                )

            # ── Visualisation rapide ──────────
            st.markdown("---")
            st.subheader(" Visualisation rapide — Serveur 1")

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric(" Total Appels", f"{df_combined['Appels'].sum():,}")
            with col_m2:
                st.metric(" Campagnes", df_combined["Campagne"].nunique())
            with col_m3:
                st.metric(" Listes", df_combined["Liste"].nunique())
            with col_m4:
                total_sec = df_combined["Durée"].apply(time_to_seconds).sum()
                h = total_sec // 3600
                m = (total_sec % 3600) // 60
                st.metric("⏱️ Durée totale", f"{h}h {m}m")

            st.markdown("---")
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                df_disp = (df_combined.groupby("Disposition")["Appels"]
                           .sum().reset_index()
                           .sort_values("Appels", ascending=False)
                           .head(10))
                fig1 = px.bar(df_disp, x="Appels", y="Disposition", orientation="h",
                              title="Top 10 Dispositions",
                              color="Appels", color_continuous_scale="Blues")
                fig1.update_layout(showlegend=False)
                st.plotly_chart(fig1, use_container_width=True)
            with col_g2:
                df_liste = (df_combined.groupby("Liste")["Appels"]
                            .sum().reset_index()
                            .sort_values("Appels", ascending=False)
                            .head(10))
                fig2 = px.bar(df_liste, x="Appels", y="Liste", orientation="h",
                              title="Top 10 Listes par appels",
                              color="Appels", color_continuous_scale="Greens")
                fig2.update_layout(showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("---")
            st.subheader(" Détail par Liste et Disposition")
            df_detail = (df_combined.groupby(["Campagne", "Liste", "Disposition"])
                         .agg(Appels=("Appels", "sum"))
                         .reset_index()
                         .sort_values(["Campagne", "Liste", "Appels"], ascending=[True, True, False]))
            df_detail["Part %"] = (
                df_detail.groupby("Liste")["Appels"]
                .transform(lambda x: (x / x.sum() * 100).round(1))
            )
            df_detail["Part %"] = df_detail["Part %"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(df_detail, use_container_width=True, hide_index=True)

            # ── EOD Table ─────────────────────
            display_eod_table(all_parsed)

           
            

    # ════════════════════════════════════════════
    # SECTION SERVEUR 2 — vicidial_log
    # ════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🖥️ Serveur 2 — Sélectionner les fichiers")

    selected_s2_paths = []
    if server2_files:
        noms_s2 = [os.path.basename(f) for f in server2_files]
        fichiers_sel_s2 = st.multiselect(
            "Fichiers disponibles Serveur 2 (repo GitHub)",
            options=noms_s2,
            default=noms_s2,
            placeholder="Choisissez un ou plusieurs fichiers...",
            key="s2_multiselect",
        )
        selected_s2_paths = [f for f in server2_files if os.path.basename(f) in fichiers_sel_s2]
        if fichiers_sel_s2:
            st.success(f" {len(fichiers_sel_s2)} fichier(s) Serveur 2 sélectionné(s)")
        else:
            st.warning(" Aucun fichier Serveur 2 sélectionné")
    else:
        st.info("📭 Aucun fichier Serveur 2 trouvé dans le repo")

    # Upload manuel Serveur 2
    uploaded_s2 = st.file_uploader(
        "➕ Ajouter des fichiers Serveur 2 manuellement",
        type=["csv"],
        accept_multiple_files=True,
        key="s2_files_upload",
    )
    if uploaded_s2:
        import tempfile
        for uf in uploaded_s2:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            tmp.write(uf.read())
            tmp.close()
            selected_s2_paths.append(tmp.name)

    # Lancer l'analyse Serveur 2
    render_server2_section(selected_s2_paths)

    st.markdown("---")
    st.header(" Analyse IA — Gemini")

    if not api_key_input:
        st.info("👈 Entrez votre clé API Gemini dans la barre latérale pour activer les recommandations IA")

    col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
    with col_b2:
        analyse_btn = st.button(
            " ANALYSER AVEC GEMINI",
            type="primary",
            use_container_width=True,
            key="ats_analyse_btn",
            disabled=not api_key_input
        )
    if analyse_btn:
        summary = resumer_ats_pour_gemini(all_parsed) if all_parsed else {"fichiers": []}
        
        if selected_s2_paths:
            dfs_s2 = []
            for path in selected_s2_paths:
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    df_raw = parse_server2_csv(content, os.path.basename(path))
                    df_norm = normalize_server2(df_raw)
                    if not df_norm.empty:
                        dfs_s2.append(df_norm)
                except:
                    pass
            if dfs_s2:
                df_s2 = pd.concat(dfs_s2, ignore_index=True)
                summary["serveur2"] = {
                    "total_appels": len(df_s2),
                    "agents": df_s2["user"].nunique() if "user" in df_s2.columns else 0,
                    "top_statuts": df_s2["status"].value_counts().head(10).to_dict() if "status" in df_s2.columns else {},
                    "campagnes": df_s2["campaign_id"].unique().tolist() if "campaign_id" in df_s2.columns else [],
                    "duree_moy_sec": int(df_s2["length_in_sec"].mean()) if "length_in_sec" in df_s2.columns else 0,
                }

        with st.spinner(" Gemini analyse vos fichiers ATS..."):
            resultat = analyser_ats_avec_gemini(api_key_input, summary)
        
        if resultat:
            st.balloons()
            st.session_state["ats_analyse_resultat"] = resultat
        else:
            st.error(" Échec de l'analyse IA")
            st.session_state["ats_analyse_resultat"] = None

    # ── Résultats Gemini ─────────────────────
    if "ats_analyse_resultat" in st.session_state and st.session_state["ats_analyse_resultat"] is not None:
        r = st.session_state["ats_analyse_resultat"]
        st.markdown("---")
        st.success(" Analyse terminée")

        st.subheader(" Résumé global")
        col_r1, col_r2 = st.columns([3, 1])
        with col_r1:
            st.info(r.get("resume_global", "N/A"))
        with col_r2:
            st.metric("Taux contact estimé", r.get("taux_contact_moyen", "N/A"))

        col_pf1, col_pf2 = st.columns(2)
        with col_pf1:
            st.subheader(" Points forts")
            for pt in r.get("points_forts", []):
                st.success(f"• {pt}")
        with col_pf2:
            st.subheader(" Points faibles")
            for pt in r.get("points_faibles", []):
                st.warning(f"• {pt}")

        if r.get("analyse_par_fichier"):
            st.markdown("---")
            st.subheader(" Analyse par fichier")
            qualite_color = {"bonne": "", "moyenne": "🟡", "faible": "🔴"}
            for item in r["analyse_par_fichier"]:
                q    = item.get("qualite", "moyenne").lower()
                icon = qualite_color.get(q, "🔵")
                with st.expander(f"{icon} {item.get('fichier', 'N/A')} — qualité {q}"):
                    st.write(f"**Observation :** {item.get('observation', 'N/A')}")
                    st.write(f"**Liste recommandée :** {item.get('liste_recommandee', 'N/A')}")

        if r.get("actions_prioritaires"):
            st.markdown("---")
            st.subheader(" Actions prioritaires")
            for action in r["actions_prioritaires"]:
                st.markdown(f"""
**👉 {action.get('action', '')}**  
 Pourquoi : {action.get('pourquoi', '')}  
 Impact : {action.get('impact', '')}
""")

        

    st.markdown("---")
import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import re


class GeminiAdvisor:
    def __init__(self, api_key=None):
        if not api_key:
            self.model = None
            self.is_configured = False
            return
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-2.5-flash")
            self.model_name = "gemini-2.5-flash"
            self.is_configured = True
            
        except Exception as e:
            st.error(f"❌ Erreur de configuration: {str(e)}")
            self.model = None
            self.is_configured = False
    
    def analyser_tous_les_volets(self, df):
        if not self.is_configured:
            return None
        
        # On calcule le contexte côté Python (pas Gemini)
        contexte = self._preparer_contexte_complet(df)
        
        # On demande à Gemini uniquement les recommandations textuelles
        prompt = self._construire_prompt(contexte)
        
        try:
            response = self.model.generate_content(prompt)
            recommandations_ia = self._parser_reponse(response.text)
            
            if recommandations_ia is None:
                return None
            
            # On fusionne les données calculées + les recommandations IA
            return {
                **contexte,                    # analyse_fournisseurs, analyse_horaire, analyse_logements
                **recommandations_ia           # resume_executif, actions_prioritaires, recommandations, prediction
            }
        except Exception as e:
            st.error(f"Erreur API Gemini: {str(e)}")
            return None
    
    def _preparer_contexte_complet(self, df):
        contexte = {
            "total_appels": len(df),
            "analyse_fournisseurs": [],
            "analyse_horaire": {},
            "analyse_logements": []
        }
        
        non_utiles = ["", "nan", "none", "non trouvé", "non trouve"]
        
        # Analyse fournisseurs
        if "list_name" in df.columns:
            for fournisseur in df["list_name"].dropna().unique()[:10]:
                df_f = df[df["list_name"] == fournisseur]
                total = len(df_f)
                taux = 0
                if "Classification" in df.columns:
                    utile_mask = ~df_f["Classification"].astype(str).str.lower().str.strip().isin(non_utiles)
                    taux = round(utile_mask.sum() / total * 100, 1) if total > 0 else 0
                
                contexte["analyse_fournisseurs"].append({
                    "nom": str(fournisseur),
                    "appels": total,
                    "taux_classification": taux
                })
        
        # Analyse horaire
        if "Timestamp" in df.columns and "Classification" in df.columns:
            df_time = df.copy()
            df_time["datetime"] = pd.to_datetime(df_time["Timestamp"], errors="coerce", dayfirst=True)
            df_time = df_time.dropna(subset=["datetime"])
            df_time["heure"] = df_time["datetime"].dt.hour
            
            utile_mask = ~df_time["Classification"].astype(str).str.lower().str.strip().isin(non_utiles)
            
            perf_heure = {}
            meilleure_heure, meilleur_taux = 0, 0
            heure_plus_appels, max_appels = 0, 0
            
            for heure in range(24):
                df_h = df_time[df_time["heure"] == heure]
                if len(df_h) > 0:
                    taux = round(utile_mask[df_h.index].sum() / len(df_h) * 100, 1)
                    perf_heure[str(heure)] = {"taux": taux, "appels": len(df_h)}
                    if taux > meilleur_taux:
                        meilleur_taux, meilleure_heure = taux, heure
                    if len(df_h) > max_appels:
                        max_appels, heure_plus_appels = len(df_h), heure
            
            contexte["analyse_horaire"] = {
                "meilleure_heure": meilleure_heure,
                "meilleur_taux": meilleur_taux,
                "heure_plus_appels": heure_plus_appels,
                "volume_max": max_appels,
                "performance_par_heure": perf_heure
            }
        
        # Analyse logements
        if "tipo_vivienda" in df.columns and "Classification" in df.columns:
            df_log = df[df["tipo_vivienda"].notna()]
            df_log = df_log[df_log["tipo_vivienda"].astype(str).str.strip() != ""]
            utile_mask = ~df_log["Classification"].astype(str).str.lower().str.strip().isin(non_utiles)
            
            for logement in df_log["tipo_vivienda"].unique()[:10]:
                df_l = df_log[df_log["tipo_vivienda"] == logement]
                total = len(df_l)
                taux = round(utile_mask[df_l.index].sum() / total * 100, 1) if total > 0 else 0
                contexte["analyse_logements"].append({
                    "type": str(logement),
                    "appels": total,
                    "taux_classification": taux
                })
        
        return contexte
    
    def _construire_prompt(self, contexte):
        prompt = f"""
Tu es un expert en centres d'appels. Analyse ces données et fournis des recommandations actionnables.

Total appels: {contexte.get('total_appels', 0)}
Fournisseurs: {json.dumps(contexte.get('analyse_fournisseurs', []), ensure_ascii=False)}
Horaires: {json.dumps(contexte.get('analyse_horaire', {}), ensure_ascii=False)}
Logements: {json.dumps(contexte.get('analyse_logements', [])[:5], ensure_ascii=False)}

Réponds UNIQUEMENT en JSON valide, sans texte avant ou après, sans balises markdown :
{{
    "resume_executif": "synthèse en 2 phrases",
    "actions_prioritaires": [
        {{"action": "...", "pourquoi": "...", "impact": "..."}},
        {{"action": "...", "pourquoi": "...", "impact": "..."}},
        {{"action": "...", "pourquoi": "...", "impact": "..."}}
    ],
    "recommandations": {{
        "horaires": "recommandation sur les horaires",
        "fournisseurs": "recommandation sur les fournisseurs",
        "logements": "recommandation sur les logements"
    }},
    "prediction": "prédiction pour le mois prochain"
}}
"""
        return prompt
    
    def _parser_reponse(self, response_text):
        try:
            text = response_text.strip()
            # Nettoyer les balises markdown
            text = re.sub(r'^```json\s*', '', text)
            text = re.sub(r'^```\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            text = text.strip()
            
            # Extraire le JSON
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            st.error("Aucun JSON trouvé dans la réponse Gemini")
            st.code(response_text[:500])  # Afficher pour debug
            return None
        except json.JSONDecodeError as e:
            st.error(f"Erreur parsing JSON: {str(e)}")
            st.code(response_text[:500])
            return None
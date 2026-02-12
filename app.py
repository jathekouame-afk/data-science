import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
import requests
from groq import Groq

# Configuration de la page
st.set_page_config(
    page_title="Assistant Décisionnel - Rentabilité Client",
    page_icon="😏",
    layout="wide"
)

# Initialisation du dossier de sortie
OUTPUT_DIR = "outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Fonctions Utiles ---
def make_json_serializable(obj):
    """Convertit un objet en quelque chose de sérialisable en JSON"""
    if isinstance(obj, (bool, np.bool_)):
        return int(obj)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, (float, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif pd.isna(obj):
        return None
    else:
        return str(obj)

def save_audit_log(step, details):
    log_file = os.path.join(OUTPUT_DIR, "audit_trail.json")
    log_data = []
    
    # Gérer les logs corrompus
    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                log_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            log_data = []  # Réinitialiser si le fichier est corrompu
    
    # Rendre les détails sérialisables
    serializable_details = make_json_serializable(details)
    
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "step": step,
        "details": serializable_details
    }
    log_data.append(log_entry)
    
    try:
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=4)
    except (IOError, TypeError) as e:
        print(f"Erreur lors de la sauvegarde du log: {e}")

# --- Barre latérale pour la navigation ---
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Étapes du projet",
    ["1. Import des données", "2. Exploration (EDA)", "3. Nettoyage", "4. Insights", "5. Prédiction", "6. Rapport Final", "7. Perspectives"]
)

def compute_data_profile(df: pd.DataFrame) -> dict:
    profile = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "duplicates": int(df.duplicated().sum()),
        "missing_cells": int(df.isnull().sum().sum()),
        "missing_pct": float((df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100) if df.shape[0] and df.shape[1] else 0.0),
        "numeric_cols": [str(c) for c in df.select_dtypes(include=[np.number]).columns.tolist()],
        "categorical_cols": [str(c) for c in df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()],
    }
    return profile

def call_ai_assistant(messages):
    try:
        client = Groq(
            api_key=st.secrets["groq"]["api_key"],
        )
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erreur de l'Assistant IA : {str(e)}"

def get_analysis_context():
    context = "Contexte de l'analyse de données : "
    
    if st.session_state.data is not None:
        df = st.session_state.data
        profile = compute_data_profile(df)
        context += f"- Données brutes : {profile['rows']} lignes, {profile['cols']} colonnes.\n"
        context += f"- Colonnes disponibles : {', '.join(df.columns.tolist())}.\n"
        context += f"- Types de données : {dict(df.dtypes.astype(str))}.\n"
        context += f"- Valeurs manquantes : {df.isnull().sum().sum()} au total.\n"
        
        # Ajouter des exemples de données pour les colonnes numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            context += f"- Colonnes numériques : {', '.join(numeric_cols)}.\n"
            for col in numeric_cols[:3]:  # Limiter aux 3 premières colonnes
                context += f"  * {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, moyenne={df[col].mean():.2f}\n"
        
        # Ajouter des exemples de données pour les colonnes catégorielles
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        if cat_cols:
            context += f"- Colonnes catégorielles : {', '.join(cat_cols)}.\n"
            for col in cat_cols[:3]:  # Limiter aux 3 premières colonnes
                unique_vals = df[col].unique()[:5]  # Premières 5 valeurs uniques
                context += f"  * {col}: valeurs uniques exemples = {', '.join(map(str, unique_vals))}\n"
    
    if st.session_state.df_clean is not None:
        df = st.session_state.df_clean
        clean_profile = compute_data_profile(df)
        context += f"- Données nettoyées : {clean_profile['rows']} lignes.\n"
    
    log_file = os.path.join(OUTPUT_DIR, "audit_trail.json")
    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                logs = json.load(f)
                context += f"- Historique des actions : {len(logs)} étapes réalisées.\n"
                for log in logs[-3:]:
                    context += f"  * {log['step']}: {log['details']}\n"
        except:
            pass
    
    return context

def render_ai_chatbot():
    # Chatbot intégré avec design moderne et épuré
    st.markdown("""
    <style>
    .chat-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border: none;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 30px 0;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }
    
    .chat-header {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 20px 24px;
        font-weight: 600;
        font-size: 18px;
        display: flex;
        align-items: center;
        gap: 12px;
        letter-spacing: 0.5px;
    }
    
    .chat-body {
        padding: 24px;
        max-height: 450px;
        overflow-y: auto;
        background: rgba(255, 255, 255, 0.9);
    }
    
    .chat-input-area {
        padding: 20px 24px;
        background: rgba(255, 255, 255, 0.95);
        border-top: 1px solid rgba(79, 172, 254, 0.2);
    }
    
    .chat-message-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 16px !important;
        margin: 8px 0 8px auto !important;
        max-width: 80%;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        font-weight: 500;
    }
    
    .chat-message-assistant {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #333;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 16px !important;
        margin: 8px auto 8px 0 !important;
        max-width: 80%;
        box-shadow: 0 4px 12px rgba(252, 182, 159, 0.3);
        font-weight: 500;
    }
    
    .chat-body::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-body::-webkit-scrollbar-track {
        background: rgba(0,0,0,0.1);
        border-radius: 3px;
    }
    
    .chat-body::-webkit-scrollbar-thumb {
        background: rgba(79, 172, 254, 0.5);
        border-radius: 3px;
    }
    
    .chat-body::-webkit-scrollbar-thumb:hover {
        background: rgba(79, 172, 254, 0.7);
    }
    
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid rgba(79, 172, 254, 0.3);
        padding: 12px 20px;
        font-size: 14px;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #4facfe;
        box-shadow: 0 0 0 3px rgba(79, 172, 254, 0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border: none;
        border-radius: 50%;
        width: 48px;
        height: 48px;
        font-size: 18px;
        color: white;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(79, 172, 254, 0.3);
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 172, 254, 0.4);
    }
    
    /* Correction pour les boutons avec use_container_width */
    .stButton > button[kind="primary"] {
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        min-height: 44px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 16px !important;
    }
    
    @media (max-width: 768px) {
        .chat-container {
            margin: 20px 0;
            border-radius: 12px;
        }
        
        .chat-header {
            padding: 16px 20px;
            font-size: 16px;
        }
        
        .chat-body {
            padding: 20px;
            max-height: 350px;
        }
        
        .chat-input-area {
            padding: 16px 20px;
        }
        
        .chat-message-user, .chat-message-assistant {
            max-width: 90%;
        }
        
        .stButton > button[kind="primary"] {
            min-height: 48px !important;
            font-size: 14px !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Conteneur principal du chatbot
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        st.markdown('<div class="chat-header">🤖 Assistant IA - Votre Expert Data Science</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="chat-body">', unsafe_allow_html=True)
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Afficher l'historique du chat avec design amélioré
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-message-user">💭 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message-assistant">🎯 {msg["content"]}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="chat-input-area">', unsafe_allow_html=True)
        
        # Bouton pour effacer le chat et input pour le message
        col_clear, col1, col2 = st.columns([1, 4, 1])
        with col_clear:
            clear_button = st.button("🗑️", key="clear_btn", help="Effacer la conversation")
        with col1:
            prompt = st.text_input("", key="chat_input", placeholder="💬 Posez votre question sur l'analyse...", label_visibility="collapsed")
        with col2:
            send_button = st.button("➤", key="send_btn", help="Envoyer votre question")
        
        # Gérer l'effacement du chat
        if clear_button:
            st.session_state.chat_history = []
            st.session_state.last_prompt = ""
            st.rerun()
        
        # Gérer l'envoi du message
        if (send_button or (prompt and st.session_state.get('last_prompt') != prompt)):
            if prompt and prompt.strip():
                st.session_state.chat_history.append({"role": "user", "content": prompt.strip()})
                st.session_state.last_prompt = prompt.strip()
                st.session_state.chat_history = st.session_state.chat_history[-10:]
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Générer une réponse si le dernier message est de l'utilisateur
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        context = get_analysis_context()
        
        # Ajout des données réelles dans le contexte pour des réponses concrètes
        detailed_context = context
        if st.session_state.data is not None:
            df = st.session_state.data
            detailed_context += f"\n\nDONNÉES RÉELLES ACTUELLES :\n"
            detailed_context += f"- DataFrame avec {len(df)} lignes et {len(df.columns)} colonnes\n"
            detailed_context += f"- Colonnes : {df.columns.tolist()}\n"
            detailed_context += f"- Aperçu des 10 premières lignes :\n{df.head(10).to_string()}\n"
            
            # Ajouter des statistiques descriptives pour les colonnes numériques
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                detailed_context += f"\n- Statistiques des colonnes numériques :\n{df[numeric_cols].describe().to_string()}\n"
            
            # Ajouter les valeurs uniques pour les colonnes catégorielles importantes
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            if cat_cols:
                detailed_context += f"\n- Valeurs uniques des colonnes catégorielles :\n"
                for col in cat_cols[:5]:  # Limiter aux 5 premières colonnes catégorielles
                    unique_vals = df[col].value_counts().head(10).to_dict()
                    detailed_context += f"  * {col}: {dict(list(unique_vals.items())[:5])}\n"
            
            # Ajouter les clients avec montants extrêmes
            if 'Montant' in df.columns:
                max_montant = df['Montant'].max()
                min_montant = df['Montant'].min()
                max_client = df[df['Montant'] == max_montant]
                min_client = df[df['Montant'] == min_montant]
                
                detailed_context += f"\n- CLIENTS AVEC MONTANTS EXTREMES :\n"
                detailed_context += f"  * Montant maximum ({max_montant}): {max_client.iloc[0].to_dict() if len(max_client) > 0 else 'Non trouvé'}\n"
                detailed_context += f"  * Montant minimum ({min_montant}): {min_client.iloc[0].to_dict() if len(min_client) > 0 else 'Non trouvé'}\n"
                
                # Ajouter les clients rentables et non rentables
                if 'Rentabilité' in df.columns:
                    rentables = df[df['Rentabilité'] == 'Rentable']
                    non_rentables = df[df['Rentabilité'] == 'Non rentable']
                    
                    if len(rentables) > 0:
                        max_rentable = rentables.loc[rentables['Montant'].idxmax()]
                        detailed_context += f"  * Client le plus rentable: {max_rentable.to_dict()}\n"
                    
                    if len(non_rentables) > 0:
                        min_non_rentable = non_rentables.loc[non_rentables['Montant'].idxmin()]
                        detailed_context += f"  * Client le moins rentable (non rentable): {min_non_rentable.to_dict()}\n"
        
        messages = [
            {"role": "system", "content": f"""Tu es un expert en Data Science. Réponds de manière CONCRÈTE et PRÉCISE en utilisant les données réelles fournies.
            
{detailed_context}

INSTRUCTIONS CRUCIALES :
1. Quand tu parles de données, utilise les noms de colonnes réels et les valeurs réelles
2. Donne des réponses spécifiques basées sur les données actuelles, pas des exemples génériques
3. Si on te demande "qui est le meilleur client", regarde les données et identifie le client spécifique avec les meilleures métriques
4. Sois concis et va droit au but avec des chiffres et faits réels
5. Utilise un ton professionnel mais accessible"""},
            *st.session_state.chat_history[-5:]
        ]
        
        with st.spinner("🤖 L'IA analyse vos données..."):
            response = call_ai_assistant(messages)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

# --- Gestion de l'état (Session State) ---
if 'data' not in st.session_state:
    st.session_state.data = None
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.sidebar.info("Projet de stage : Assistant d'aide à la décision pour l'analyse de la rentabilité.")

# --- Page 1: Import des données ---
if page == "1. Import des données":
    st.title("📥 Import des données")
    st.write("Téléchargez vos fichiers clients ou transactions (CSV ou Excel).")
    
    uploaded_file = st.file_uploader("Choisir un fichier", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.data = df
            st.success(f"Fichier chargé avec succès ! ({len(df)} lignes, {len(df.columns)} colonnes)")
            
            st.subheader("Aperçu des données")
            st.dataframe(df)
            
            save_audit_log("Import", {"filename": uploaded_file.name, "rows": len(df), "cols": len(df.columns)})
            
        except Exception as e:
            st.error(f"Erreur lors du chargement : {e}")

# --- Page 2: Exploration (EDA) ---
elif page == "2. Exploration (EDA)":
    st.title("🔍 Exploration des données (EDA)")
    
    if st.session_state.data is None:
        st.warning("Veuillez d'abord importer des données.")
    else:
        df = st.session_state.data
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Lignes", df.shape[0])
        col2.metric("Colonnes", df.shape[1])
        col3.metric("Doublons", df.duplicated().sum())
        
        st.subheader("Qualité des données")
        missing_df = pd.DataFrame({
            "Colonne": df.columns,
            "Manquants (%)": (df.isnull().sum() / len(df) * 100).round(2),
            "Type": df.dtypes.astype(str)
        })
        st.table(missing_df)
        
        st.subheader("Distributions")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("Choisir une colonne numérique", numeric_cols)
            fig = px.histogram(df, x=selected_col, title=f"Distribution de {selected_col}")
            st.plotly_chart(fig)
        else:
            st.write("Aucune colonne numérique détectée.")

# --- Page 3: Nettoyage ---
elif page == "3. Nettoyage":
    st.title("🧼 Nettoyage des données")
    
    if st.session_state.data is None:
        st.warning("Veuillez d'abord importer des données.")
    else:
        df = st.session_state.data.copy()
        
        st.subheader("1. Gestion des valeurs manquantes")
        cols_with_nan = df.columns[df.isnull().any()].tolist()
        
        if not cols_with_nan:
            st.success("Aucune valeur manquante détectée.")
        else:
            method = st.radio("Méthode de traitement", ["Conserver", "Supprimer les lignes", "Remplacer par la moyenne/mode"])
            
            if method == "Supprimer les lignes":
                df = df.dropna()
                st.info(f"Lignes restantes : {len(df)}")
            elif method == "Remplacer par la moyenne/mode":
                for col in cols_with_nan:
                    if df[col].dtype in [np.float64, np.int64]:
                        df[col] = df[col].fillna(df[col].mean())
                    else:
                        df[col] = df[col].fillna(df[col].mode()[0])
                st.info("Valeurs manquantes remplacées.")

        st.subheader("2. Gestion des doublons")
        if df.duplicated().any():
            if st.button("Supprimer les doublons", use_container_width=True):
                old_len = len(df)
                df = df.drop_duplicates()
                st.success(f"{old_len - len(df)} doublons supprimés.")
        else:
            st.info("Aucun doublon détecté.")

        st.subheader("3. Conversion des types")
        all_cols = df.columns.tolist()
        date_cols = st.multiselect("Sélectionner les colonnes à convertir en DATE", all_cols)
        
        if date_cols:
            for col in date_cols:
                try:
                    df[col] = pd.to_datetime(df[col])
                    st.success(f"Colonne {col} convertie en date.")
                except:
                    st.error(f"Impossible de convertir {col} en date.")

        if st.button("Valider le nettoyage", use_container_width=True):
            st.session_state.df_clean = df
            st.success("Données nettoyées sauvegardées pour l'analyse !")
            save_audit_log("Nettoyage", {
                "final_rows": len(df),
                "date_conversions": date_cols,
                "has_nans": df.isnull().any().any()
            })

# --- Page 4: Insights ---
elif page == "4. Insights":
    st.title("📊 Analyses & Insights")
    
    df = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.data
    
    if df is None:
        st.warning("Veuillez d'abord importer des données.")
    else:
        st.subheader("Indicateurs clés (KPI)")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            col_target = st.selectbox("Choisir l'indicateur principal (ex: Chiffre d'Affaires)", numeric_cols)
            
            c1, c2 = st.columns(2)
            c1.metric("Total", f"{df[col_target].sum():,.2f}")
            c2.metric("Moyenne par ligne", f"{df[col_target].mean():,.2f}")
            
            st.subheader("Analyse par segment")
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_cols:
                segment_col = st.selectbox("Choisir une dimension (Segment/Région)", cat_cols)
                fig_segment = px.bar(df.groupby(segment_col)[col_target].sum().reset_index(), 
                                   x=segment_col, y=col_target, title=f"{col_target} par {segment_col}")
                st.plotly_chart(fig_segment)
            
            st.subheader("Top 10")
            if cat_cols:
                top_col = st.selectbox("Top 10 par...", cat_cols, key="top_col")
                top_10 = df.groupby(top_col)[col_target].sum().sort_values(ascending=False).head(10)
                st.table(top_10)
        else:
            st.write("Aucune donnée numérique pour calculer des KPI.")

# --- Page 5: Prédiction ---
elif page == "5. Prédiction":
    st.title("🤖 Prédiction de la Rentabilité")
    
    df = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.data
    
    if df is None:
        st.warning("Veuillez d'abord nettoyer vos données.")
    else:
        st.write("Dans cette version simplifiée, nous définissons la 'Rentabilité' comme étant au-dessus de la moyenne.")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("Besoin de colonnes numériques pour la prédiction.")
        else:
            target_base = st.selectbox("Indicateur de base pour la rentabilité", numeric_cols)
            threshold = df[target_base].mean()
            df['target'] = (df[target_base] > threshold).astype(int)
            
            st.info(f"Seuil de rentabilité défini à : {threshold:.2f} (Moyenne)")
            st.write(f"Répartition : {df['target'].value_counts(normalize=True)[1]:.1%} rentables vs {df['target'].value_counts(normalize=True)[0]:.1%} non-rentables")
            
            if st.button("Lancer l'entraînement du modèle", use_container_width=True):
                from sklearn.model_selection import train_test_split
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import accuracy_score, classification_report
                
                # Features simplifiées (colonnes numériques sauf target)
                features = [c for c in numeric_cols if c != target_base]
                if not features:
                    st.error("Pas assez de variables explicatives numériques.")
                else:
                    X = df[features].fillna(0)
                    y = df['target']
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    model = RandomForestClassifier(n_estimators=100)
                    model.fit(X_train, y_train)
                    
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    
                    st.success(f"Modèle entraîné ! Précision : {acc:.2%}")
                    
                    st.subheader("Importance des variables")
                    feat_imp = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
                    st.plotly_chart(px.bar(feat_imp, x='Feature', y='Importance'))
                    
                    save_audit_log("Prediction", {"accuracy": acc, "features": features})

# --- Page 6: Rapport Final ---
elif page == "6. Rapport Final":
    st.title("📑 Rapport Final")
    
    if st.session_state.data is None:
        st.warning("Aucune donnée disponible.")
    else:
        st.write("Résumé de votre projet d'aide à la décision.")
        
        # Affichage des logs d'audit
        log_file = os.path.join(OUTPUT_DIR, "audit_trail.json")
        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    logs = json.load(f)
                st.json(logs)
            except (json.JSONDecodeError, IOError):
                st.warning("Le fichier d'audit est corrompu ou illisible.")
        else:
            st.info("Aucun log d'audit disponible.")
            
        if st.button("Générer le rapport (Simulation)", use_container_width=True):
            st.success("Rapport généré dans le dossier 'outputs'. (Fonctionnalité en cours de finalisation)")

# --- Page 7: Perspectives ---
elif page == "7. Perspectives":
    st.title("🚀 Perspectives & Améliorations")
    
    st.write("Ce que nous pourrions faire pour aller plus loin :")
    st.markdown("""
    - **Multi-utilisateurs** : Déploiement SaaS avec gestion des comptes.
    - **Connecteurs Cloud** : Intégration directe avec BigQuery, Snowflake ou AWS S3.
    - **IA Générative** : Amélioration de l'assistant pour générer des requêtes SQL complexes.
    - **Monitoring ML** : Suivi de la dérive des données (drift) et ré-entraînement automatique.
    - **Visualisations avancées** : Cartographie géographique et analyses de cohortes.
    """)

# --- Autres pages (Squelettes) ---
else:
    st.title(page)
    st.write("Cette section sera implémentée dans les prochains sprints.")
    if st.session_state.data is None:
        st.warning("Aucune donnée disponible.")

# Chatbot IA intégré à la fin de chaque page
render_ai_chatbot()

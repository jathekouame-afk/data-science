# Assistant Décisionnel - Rentabilité Client

Application Streamlit pour l'analyse de rentabilité client avec assistant IA intégré.

## Fonctionnalités

- 📥 Import des données (CSV/Excel)
- 🔍 Analyse exploratoire (EDA)
- 🧼 Nettoyage des données
- 📊 Analyses et insights
- 🤖 Prédiction de rentabilité
- 📑 Rapport final
- 💬 Assistant IA avec Groq

## Installation locale

1. Cloner le repository
```bash
git clone https://github.com/jathekouame-afk/data-science.git
cd data-science
```

2. Créer l'environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Installer les dépendances
```bash
pip install -r requirements.txt
```

4. Configurer les secrets
Créer un fichier `.streamlit/secrets.toml` :
```toml
[groq]
api_key = "votre_clé_api_groq"
```

5. Lancer l'application
```bash
streamlit run app.py
```

## Déploiement

### Streamlit Cloud (recommandé)

1. Connectez-vous sur [Streamlit Cloud](https://share.streamlit.io/)
2. Connectez votre compte GitHub
3. Sélectionnez ce repository
4. Configurez les secrets dans l'interface Streamlit Cloud
5. Déployez !

### Autres options

- **Heroku** : Ajouter un Procfile et configurer les buildpacks
- **Railway** : Connecter le repo et ajouter les variables d'environnement
- **AWS/Azure** : Utiliser Docker ou EC2

## Variables d'environnement

- `GROQ_API_KEY` : Clé API pour l'assistant IA

## Structure du projet

```
data-science/
├── app.py                 # Application principale
├── requirements.txt       # Dépendances Python
├── .streamlit/           # Configuration Streamlit
│   └── secrets.toml      # Secrets locaux (non versionné)
├── outputs/              # Logs et rapports
└── README.md            # Documentation
```

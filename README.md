# 🚰 MLOps Water Potability Prediction

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Poetry](https://img.shields.io/badge/Poetry-1.7.1-blue.svg)](https://python-poetry.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Pipeline-yellow.svg)](https://dvc.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-green.svg)](https://fastapi.tiangolo.com/)

Production-ready MLOps pipeline for water potability prediction using Machine Learning. Features automated training, model versioning, containerized deployment, and REST API serving.

## 🎯 Overview

Predict water potability based on 9 water quality parameters using an ensemble of ML models. The system automatically selects and optimizes the best model, tracks experiments with MLflow, and deploys via Docker with FastAPI.

## ✨ Key Features

- 🔄 **Automated ML Pipeline** - DVC-orchestrated training with 5 stages
- 🧪 **8 ML Models** - Auto-selection from RandomForest, GradientBoosting, HistGradientBoosting, AdaBoost, ExtraTrees, SVM, LogisticRegression, KNN
- 📊 **MLflow Tracking** - Experiment tracking, model registry, and versioning
- 🐳 **Docker & Docker Compose** - Containerized training and serving
- 🚀 **FastAPI** - Production-ready REST API with Swagger docs
- 📦 **Poetry** - Modern dependency management
- 🔁 **Reproducible** - Complete pipeline from raw data to deployed API

## 🏗️ Architecture

```
┌─────────────┐    ┌──────────────┐    ┌────────────┐
│ Raw Data    │───▶│ DVC Pipeline │───▶│ Best Model │
└─────────────┘    └──────────────┘    └────────────┘
                           │                    │
                           ▼                    ▼
                    ┌──────────────┐    ┌────────────┐
                    │ MLflow       │    │ FastAPI    │
                    │ Tracking     │    │ Service    │
                    └──────────────┘    └────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Poetry
- Docker & Docker Compose

### 1️⃣ Clone & Setup
```bash
git clone https://github.com/kabbstat/mlops_water_prediction.git
cd mlops_water_project
poetry install
```

### 2️⃣ Run Training Pipeline
```bash
docker-compose run --rm pipeline
```

### 3️⃣ Launch API
```bash
docker-compose up -d api
```

Access API documentation: `http://localhost:8000/docs`

### 4️⃣ Make Predictions
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "ph": 7.0,
    "Hardness": 200,
    "Solids": 20000,
    "Chloramines": 7.0,
    "Sulfate": 300,
    "Conductivity": 400,
    "Organic_carbon": 10,
    "Trihalomethanes": 60,
    "Turbidity": 3.5
  }'
```

## 📁 Project Structure

```
mlops_water_project/
├── api/                    # FastAPI application
│   ├── main.py            # API endpoints
│   ├── schemas.py         # Pydantic models
│   └── model_loader.py    # MLflow model loader
├── src/                   # ML pipeline source
│   ├── data_collection.py # Data loading & splitting
│   ├── data_prep.py       # Preprocessing & imputation
│   ├── exp1.py            # Model selection (8 models)
│   ├── exp2.py            # Hyperparameter tuning
│   └── model_eval.py      # Final evaluation & registration
├── data/                  # Data directories
│   ├── raw/              # Original datasets
│   └── processed/        # Processed features
├── mlruns/               # MLflow tracking data
├── Dockerfile            # Training container
├── Dockerfile.api        # API container
├── docker-compose.yml    # Service orchestration
├── dvc.yaml             # Pipeline definition
├── params.yaml          # Model configs & hyperparameters
└── pyproject.toml       # Poetry dependencies
```

## 🛠️ Technology Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.11 |
| **ML Frameworks** | Scikit-learn, Pandas, NumPy |
| **Experiment Tracking** | MLflow |
| **Pipeline** | DVC |
| **API** | FastAPI, Uvicorn |
| **Containerization** | Docker, Docker Compose |
| **Dependency Management** | Poetry |
| **Visualization** | Matplotlib, Seaborn |

## 📊 Pipeline Stages

1. **Data Collection** - Load and split dataset (80/20)
2. **Preprocessing** - Handle missing values with median imputation
3. **Model Selection** - Train 8 models with 5-fold CV, select best
4. **Hyperparameter Tuning** - GridSearchCV optimization
5. **Model Evaluation** - Final metrics, feature importance, confusion matrix

## 🎯 Model Performance

The pipeline automatically selects the best model. Current best:
- **Model**: RandomForest
- **Accuracy**: ~66%
- **F1-Score**: ~0.46
- **Tracked in MLflow**: Version-controlled and reproducible

## 🔧 Configuration

Edit `params.yaml` to:
- Add/remove models
- Modify hyperparameter grids
- Adjust cross-validation folds
- Change train/test split ratio

## 🐳 Docker Commands

```bash
# Build images
docker-compose build

# Run training pipeline
docker-compose run --rm pipeline

# Start API service
docker-compose up -d api

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

## 🚀 Installation et Configuration

### Prérequis

- Python 3.8+
- pip gestionnaire de packages
- Git

### Installation

1. **Cloner le repository**
   ```bash
   git clone https://github.com/kabbstat/mlops_water_prediction.git
   cd mlops_water_prediction
   ```

2. **Créer un environnement virtuel**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows: venv\Scripts\activate
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

4. **Installer le package du projet**
   ```bash
   pip install -e .
   ```

## 🔄 Pipeline MLOps Détaillé

### Vue d'ensemble du Pipeline

```
Collection de Données → Préparation → Expérimentation MLflow → Hyperparamètre Tuning → Évaluation Finale
```

### Étapes du Pipeline

#### 1. **Collection des Données** (`src/data_collection.py`)
- **Objectif** : Collecte et acquisition des données de qualité de l'eau
- **Impact sur l'évolution** : Établit la base de données fiable pour l'entraînement
- **Processus** :
  - Collecte des données à partir de sources multiples
  - Validation de l'intégrité des données
  - Stockage dans `data/raw/`

#### 2. **Préparation des Données** (`src/data_prep.py`)
- **Objectif** : Nettoyage, transformation et préparation des données
- **Impact sur l'évolution** : Améliore la qualité des données et la performance des modèles
- **Processus** :
  - Nettoyage des valeurs manquantes et aberrantes
  - Normalisation et standardisation
  - Ingénierie des caractéristiques
  - Division train/validation/test
  - Sauvegarde dans `data/processed/`

#### 3. **Expérimentation avec MLflow** (`src/exp1.py`)
- **Objectif** : Entraînement et comparaison de multiples modèles ML
- **Impact sur l'évolution** : Permet la sélection basée sur les données du meilleur modèle
- **Processus** :
  - Entraînement de différents algorithmes (Random Forest, SVM, etc.)
  - Suivi des métriques avec MLflow
  - Validation croisée
  - Logging des paramètres et artefacts
  - Comparaison des performances

#### 4. **Optimisation Hyperparamètres** (`src/exp2.py`)
- **Objectif** : Optimisation fine des hyperparamètres des meilleurs modèles
- **Impact sur l'évolution** : Maximise les performances du modèle final
- **Processus** :
  - Grid Search / Random Search
  - Bayesian Optimization
  - Validation croisée avec MLflow tracking
  - Sélection des meilleurs hyperparamètres

#### 5. **Évaluation du Modèle Final** (`src/model_eval.py`)
- **Objectif** : Évaluation complète du modèle final optimisé
- **Impact sur l'évolution** : Valide la robustesse et la fiabilité du modèle
- **Processus** :
  - Tests sur données de test
  - Métriques de performance détaillées
  - Analyse des erreurs
  - Génération de rapports
  - Visualisations des résultats

#### 6. **Fonctions Utilitaires** (`src/utils.py`)
- **Objectif** : Fonctions communes réutilisables
- **Impact sur l'évolution** : Assure la consistance et réutilisabilité du code
- **Contenu** :
  - Fonctions de preprocessing
  - Métriques personnalisées
  - Utilitaires de visualisation
  - Helpers pour MLflow

## 📈 Utilisation

### Démarrage Rapide

1. **MLflow UI**
   ```bash
   mlflow ui
   ```
   Naviguez vers `http://localhost:5000` pour voir le dashboard

2. **Exécution du Pipeline Complet**
   ```bash
   # Exécution compléte du pipeline
   dvc repro

   ```
   
## 📊 Données

Le projet utilise des jeux de données de qualité de l'eau contenant des paramètres tels que :

- **pH** : Niveau d'acidité/basicité
- **Dureté** : Concentration en minéraux
- **Solides** : Solides dissous totaux
- **Chloramines** : Désinfectant
- **Sulfate** : Composé chimique
- **Conductivité** : Capacité de conduction électrique
- **Carbone Organique** : Matière organique
- **Trihalométhanes** : Sous-produits de désinfection
- **Turbidité** : Clarté de l'eau
- **Potabilité** : Variable cible (0 = non potable, 1 = potable)

## 🤖 Modèles Implémentés

- **Logistic Regression** : Modèle de base
- **Random Forest** : Ensemble method
- **Support Vector Machine** : Classification non-linéaire
- **Gradient Boosting** : Algorithme de boosting
- **XGBoost** : Gradient boosting optimisé
- **Neural Networks** : Réseaux de neurones

## 📊 Suivi avec MLflow

### Fonctionnalités MLflow

- **Experiment Tracking** : Suivi de tous les runs avec paramètres et métriques
- **Model Registry** : Gestion des versions de modèles
- **Artifact Logging** : Sauvegarde des modèles, plots et importance des features
- **Metric Comparison** : Comparaison côte à côte des expériences
- **Reproductibilité** : Suivi de l'environnement et des dépendances

### Métriques Suivies

- **Accuracy** : Précision globale
- **Precision** : Précision par classe
- **Recall** : Rappel par classe
- **F1-Score** : Mesure harmonique
- **ROC AUC** : Aire sous la courbe ROC
- **Confusion Matrix** : Matrice de confusion
- **Feature Importance** : Importance des variables

## 📋 Résultats

### Dashboard MLflow

Tous les résultats sont disponibles via l'interface MLflow :
- Comparaison interactive des expériences
- Visualisation des métriques au fil du temps
- Artefacts de modèles et graphiques
- Suivi complet de la reproductibilité

### Rapports Générés

- **Rapport de Performance** : Métriques détaillées par modèle
- **Analyse des Features** : Importance et corrélations
- **Visualisations** : Courbes ROC, matrices de confusion
- **Recommandations** : Meilleur modèle et paramètres optimaux

## 📄 Licence

Ce projet est sous licence spécifiée dans le fichier LICENSE.

## 👨‍💻 Auteur

**Kabbaj Mohamed**
- GitHub: [@kabbstat](https://github.com/kabbstat)
- LinkedIn: [Mohamed Kabbaj](https://linkedin.com/in/mohamed-kabbaj)

## 🙏 Remerciements

- Structure du projet basée sur le template [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
- Communauté open-source pour les outils et bibliothèques utilisés
- MLflow pour l'excellent framework de suivi des expériences

## 📞 Support

Pour toute question ou aide avec le projet :
1. Consultez la documentation dans le répertoire `docs/`
2. Parcourez les issues existantes sur GitHub
3. Créez une nouvelle issue si votre question n'est pas résolue

---

*Ce projet démontre les meilleures pratiques MLOps pour la prédiction de la qualité de l'eau, combinant les workflows de data science avec les principes d'ingénierie pour des solutions d'apprentissage automatique reproductibles et évolutives.*

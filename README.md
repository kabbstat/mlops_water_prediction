# MLOps Water Potability Prediction

Un projet MLOps complet pour prédire la potabilité de l'eau en utilisant des techniques d'apprentissage automatique. Ce projet suit les meilleures pratiques de l'industrie avec un workflow de data science bien structuré et un pipeline d'analyse reproductible.

## 🎯 Aperçu du Projet

Ce projet vise à prédire la potabilité de l'eau (si l'eau est potable) en utilisant divers paramètres de qualité de l'eau. Le modèle aide à évaluer la qualité de l'eau et à garantir des normes d'eau potable sûres.

## 📊 Fonctionnalités

- **Pipeline de Traitement des Données** : Collecte, nettoyage et préparation automatisés des données
- **Expérimentation MLflow** : Suivi complet des expériences et gestion des modèles
- **Hyperparamètre Tuning** : Optimisation systématique des paramètres
- **Évaluation de Modèles** : Analyse complète des performances et comparaisons
- **Workflow Reproductible** : Structure standardisée suivant les meilleures pratiques

## 🛠️ Technologies Utilisées

- **Python** : Langage de programmation principal
- **Pandas & NumPy** : Manipulation et analyse des données
- **Scikit-learn** : Algorithmes d'apprentissage automatique
- **MLflow** : Suivi des expériences et gestion des modèles
- **Matplotlib/Seaborn** : Visualisation des données
- **Jupyter Notebooks** : Développement interactif et analyse

## 📁 Structure du Projet

```
mlops_water_prediction/
├── LICENSE
├── Makefile                 # Commandes d'automatisation du projet
├── README.md               # Documentation du projet
├── requirements.txt        # Dépendances Python
├── setup.py               # Configuration d'installation
├── data/
│   ├── external/           # Sources de données tierces
│   ├── interim/            # Données intermédiaires traitées
│   ├── processed/          # Jeux de données finaux pour la modélisation
│   └── raw/                # Données brutes originales
├── models/                 # Modèles entraînés et prédictions
├── notebooks/              # Notebooks Jupyter pour l'analyse
├── reports/                # Rapports d'analyse générés
│   └── figures/            # Graphiques et visualisations
├── src/                    # Code source principal
│   ├── __init__.py
│   ├── utils.py            # Fonctions utilitaires
│   ├── data_collection.py  # Collection des données
│   ├── data_prep.py        # Préparation et preprocessing
│   ├── exp1.py             # Expérimentation modèles avec MLflow
│   ├── exp2.py             # Optimisation hyperparamètres
│   └── model_eval.py       # Évaluation du modèle final
└── docs/                   # Documentation (Sphinx)
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
   # Collection des données
   python src/data_collection.py
   
   # Préparation des données
   python src/data_prep.py
   
   # Expérimentation modèles
   python src/exp1.py
   
   # Optimisation hyperparamètres
   python src/exp2.py
   
   # Évaluation finale
   python src/model_eval.py
   ```

### Commandes Makefile

```bash
make data      # Collecte et préparation des données
make train     # Entraînement des modèles
make optimize  # Optimisation hyperparamètres
make evaluate  # Évaluation finale
make clean     # Nettoyage des fichiers temporaires
```

### Notebooks Jupyter

```bash
jupyter notebook notebooks/
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

## 🔧 Développement et Tests

### Tests

```bash
# Exécution des tests
python -m pytest tests/

# Tests avec coverage
python -m pytest tests/ --cov=src
```

### Linting et Format

```bash
# Linting
flake8 src/
black src/
```

## 🤝 Contribution

1. Fork le repository
2. Créer une branche feature (`git checkout -b feature/nouvelle-fonctionnalité`)
3. Commit les changements (`git commit -m 'Ajout nouvelle fonctionnalité'`)
4. Push vers la branche (`git push origin feature/nouvelle-fonctionnalité`)
5. Créer une Pull Request

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

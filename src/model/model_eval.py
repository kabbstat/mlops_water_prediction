import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import pickle
import mlflow
import json
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from matplotlib import pyplot as plt
import seaborn as sns
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("water_potability_gb")
# Load the dataset
def load_data(file_path):
        try:
            data = pd.read_csv(file_path)
            return data
        except Exception as e:
            raise Exception(f"error loading data from {file_path}: {e}")
# load best model from exp2

def load_best_model():
    try:
        with open("best_model.txt", "r") as f:
            best_model_name = f.read().strip()
        return best_model_name
    except Exception as e:
        raise Exception(f"error loading best model: {e}")
def best_model_params():
    with open("best_model_params.json", "r") as f:
        best_model_params = json.load(f)
    return best_model_params
def main():
    best_model_name = load_best_model()
    best_model_params = best_model_params()
    data_train = load_data("./data/processed/train_processed.csv")
    data_test = load_data("./data/processed/test_processed.csv")
    X_train = data_train.drop("Potability", axis=1)
    y_train = data_train["Potability"]
    X_test = data_test.drop("Potability", axis=1)
    y_test = data_test["Potability"]
    n_estimators = best_model_params['n_estimators']
    max_depth = best_model_params['max_depth']
    model_class = best_model_name.split('.')[-1]
    model = model_class(**best_model_params)
    model.fit(X_train, y_train)
    with mlflow.start_run(run_name=f"{model_class}_hyperparameter_tuning") :
        data_view = X_train.head(5)
        data_view.to_csv("feature.csv", index=False)
        mlflow.log_artifact("feature.csv")
        data_train = mlflow.data.from_pandas(X_train)
        data_test = mlflow.data.from_pandas(X_test)
        mlflow.log_input(data_train, "X_train")
        mlflow.log_input(data_test,"X_test")
        with open("feature_names.txt", "w") as f :
            f.write("\n".join(X_train.columns))
        mlflow.log_artifact("feature_names.txt")
        for i , params in enumerate(grid_search.cv_results_['params']):
            with mlflow.start_run(run_name=f"Hyperparameter_{i+1}", nested=True):
                mlflow.log_params(params)
                mean_cv_accuracy = grid_search.cv_results_['mean_test_score'][i]
                std_cv_accuracy = grid_search.cv_results_['std_test_score'][i]
                mlflow.log_metric("mean_cv_accuracy", mean_cv_accuracy)
                mlflow.log_metric("std_cv_accuracy", std_cv_accuracy)
        best_model = grid_search.best_estimator_
        best_mean_cv_accuracy = grid_search.best_score_
        best_std_cv_accuracy = grid_search.cv_results_['std_test_score'][grid_search.best_index_]
        mlflow.log_metric("best_mean_cv_accuracy", best_mean_cv_accuracy)
        mlflow.log_metric("best_std_cv_accuracy", best_std_cv_accuracy)
        y_pred = best_model.predict(X_test)
        best_accuracy = accuracy_score(y_test, y_pred)
        best_f1 = f1_score(y_test, y_pred)
        best_recall = recall_score(y_test,y_pred)
        best_precision = precision_score(y_test, y_pred)
        signature = infer_signature(X_test,y_pred)
        mlflow.sklearn.log_model(best_model, "best_model", signature=signature,
                                                registered_model_name="water_potability_gb_model",
                                                metadata={"algorithm": "Gradient Boosting Classifier",
                                                            "dataset_size":len(data),
                                                            "features_count":X_train.shape[1],
                                                            "best_params": str(grid_search.best_params_),
                                                            "training_date": pd.Timestamp.now().isoformat()})
        client = MlflowClient()
        model_version = client.get_latest_versions(name= "water_potability_gb_model")[0]
        client.set_model_version_tag(name="water_potability_gb_model",version=model_version.version,key="data_preprocessing", value="missing_values_handled")
        client.set_model_version_tag(name="water_potability_gb_model", version=model_version.version, key="model_type", value="classification")
        client.update_model_version(name= "water_potability_gb_model", version = model_version.version, 
                                    description=f"""    Gradient Boosting Classifier pour la prédiction de potabilité de l'eau
        Détails du modèle:
        - Algorithme: Gradient Boosting Classifier
        - Meilleurs paramètres: {grid_search.best_params_}
        - Accuracy CV: {best_mean_cv_accuracy:.4f}
        - Accuracy Test: {best_accuracy:.4f}
        - Nombre de features: {X_train.shape[1]}
        - Taille dataset: {len(X_train)} échantillons
        - Date d'entraînement: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Preprocessing:
        - Imputation des valeurs manquantes par la médiane
        - Aucune normalisation appliquée """)
        client.set_registered_model_alias(name="water_potability_gb_model",alias="production", version=model_version.version)
        #mlflow.register_model(model_version.model_uri, "water_potability_gb_model")
        feature_importances = best_model.feature_importances_
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_importances, y=X_train.columns)
        plt.title('Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.savefig("feature_importances.png")
        plt.close()
        mlflow.log_artifact("feature_importances.png")
        mlflow.log_metric("best_accuracy", best_accuracy)
        mlflow.log_metric("best_f1", best_f1)
        mlflow.log_metric("best_recall", best_recall)
        mlflow.log_metric("best_precision", best_precision)
        mlflow.log_params(grid_search.best_params_)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig("confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("confusion_matrix.png")
        pickle.dump(best_model, open("model.pkl", "wb"))
        # trachking the code 
        mlflow.log_artifact(__file__)
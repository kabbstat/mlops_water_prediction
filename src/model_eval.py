import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import pickle
import mlflow
import json
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from matplotlib import pyplot as plt
from utils import get_dataset_path
import seaborn as sns
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("water_potability_gb")

# load best model from exp1
def load_best_model():
    try:
        with open("best_model.txt", "r") as f:
            best_model_name = f.read().strip()
        return best_model_name
    except Exception as e:
        raise Exception(f"error loading best model: {e}")
# Load the best model parameters
def best_model_params():
    with open("best_model_params.json", "r") as f:
        best_model_params = json.load(f)
    return best_model_params
def main():
    best_model_name = load_best_model()
    best_model_params = best_model_params()
    data_train = pd.read_csv(get_dataset_path('train.csv', 'processed'))
    data_test = pd.read_csv(get_dataset_path('test.csv', 'processed'))
    X_train, y_train = data_train.drop("Potability",1), data_train["Potability"] 
    X_test, y_test = data_test.drop("Potability",1), data_test["Potability"]
    #n_estimators = best_model_params['n_estimators']
    #max_depth = best_model_params['max_depth']
    model_map = {
        'RandomForestClassifier': RandomForestClassifier,
        'GradientBoostingClassifier': GradientBoostingClassifier,
        'HistGradientBoostingClassifier': HistGradientBoostingClassifier,
        'SVC': SVC,
        'LogisticRegression': LogisticRegression}
    model_class = model_map[best_model_name]
    model = model_class(**best_model_params)
    model.fit(X_train, y_train)
    with mlflow.start_run(run_name=f"{model_class}_evaluation"): 
        preds = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, preds),
            'f1_score': f1_score(y_test, preds),
            'recall': recall_score(y_test, preds),
            'precision': precision_score(y_test, preds)
        }
        mlflow.log_metrics(metrics)
        # signature , enregistrement du modèle
        signature = infer_signature(X_test, preds)
        registered_name = f"water_potability_{best_model_name}_model"
        mlflow.sklearn.log_model(model, "best_model", signature=signature, registered_model_name=registered_name)
        
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
        mlflow.log_params(best_model_params)
        client = MlflowClient()
        model_version = client.get_latest_versions(registered_name)[0]
        client.set_model_version_tag(name=registered_name,version=model_version.version,key="data_preprocessing", value="missing_values_handled_with_median")
        client.set_model_version_tag(name="water_potability_prediction", version=model_version.version, key="model_type", value="classification")
        client.update_model_version(name= "water_potability_prediction", version = model_version.version, 
                                    description=f"Évaluation finale du modèle {best_model_name} — accuracy={metrics['accuracy']:.4f}")
        #client.set_registered_model_alias(name="water_potability_gb_model",alias="production", version=model_version.version)
        #mlflow.register_model(model_version.model_uri, "water_potability_gb_model")
        feature_importances = model.feature_importances_
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_importances, y=X_train.columns)
        plt.title('Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.savefig("feature_importances.png")
        plt.close()
        mlflow.log_artifact("feature_importances.png")
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig("confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("confusion_matrix.png")
        with open("model_eval.txt", "w") as f:
            f.write(f"Model: {best_model_name}\n")
            f.write(f"Metrics:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
        mlflow.log_artifact(__file__)

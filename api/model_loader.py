import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

def load_model():
    mlflow.set_tracking_uri("file:./mlruns")
    client = MlflowClient()

    with open("best_model.txt", "r") as f : 
        best_model = f.read().strip()
        
    model_name = f"water_potability_{best_model}_model"
    versions = client.search_model_versions(f"name='{model_name}'")
    latest_version = max(versions, key=lambda v: int(v.version))
    model_uri = f"models:/{model_name}/{latest_version.version}"
    model = mlflow.sklearn.load_model(model_uri)
    model_info = {"name": model_name, "version": latest_version.version}
    return model, model_info
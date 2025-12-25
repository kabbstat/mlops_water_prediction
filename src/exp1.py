import pandas as pd
import mlflow
import mlflow.sklearn
import importlib
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from utils import load_params, get_dataset_path

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("exp1_median_models")

def get_model_class(model_config):
    module_name, class_name = model_config['class'].rsplit('.', 1)
    module = importlib.import_module(f"sklearn.{module_name}")
    return getattr(module, class_name)

def main():
    params = load_params()
    data = pd.read_csv(get_dataset_path('train.csv', 'processed'))
    X = data.drop('Potability', axis=1)
    y = data['Potability']
    models_cfg = params['exp1']['models']
    best_score = 0
    best_model = None
    with mlflow.start_run(run_name="BestModel_Selection"):
        for  model_name, model_config in models_cfg.items():
            with mlflow.start_run(run_name= model_name, nested= True):
                model_class = get_model_class(model_config)
                model = model_class(**model_config.get('params',{}))
                cv_results = cross_validate(model, X, y, cv=params['exp1']['cv'])
                mean_accuracy = cv_results['test_score'].mean()
                std_accuracy = cv_results['test_score'].std()
                mlflow.log_param("model_name", model_name)
                mlflow.log_params(model_config.get('params', {}))
                mlflow.log_metric("mean_accuracy", mean_accuracy)
                mlflow.log_metric("std_accuracy", std_accuracy)
                model.fit(X, y)
                mlflow.sklearn.log_model(model, model_name)
                if mean_accuracy > best_score:
                    best_score , best_model = mean_accuracy, model_name
        if best_model:
            mlflow.set_tag("best_model", best_model)
            mlflow.log_param("best_model", best_model)
            mlflow.log_param("best_score", best_score)    
        with open('best_model.txt','w')as f:
            f.write(best_model)
if __name__ == "__main__":
    main()
            
            

            
    
    
    
    
    
    


    
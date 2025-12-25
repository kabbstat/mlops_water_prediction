import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from utils import load_params, get_dataset_path
import mlflow 
import json
import importlib
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("exp2_bestmodel_Hyperparameter_tuning")
def get_model_class(model_config):
    modul_name, class_name = model_config['class'].rsplit('.', 1)
    module  = importlib.import_module(f"sklearn.{modul_name}")
    return getattr(module, class_name)
def main():
    with open("best_model.txt", "r") as f:
        best_model_name = f.read().strip()
    params = load_params()
    grid_config = params['exp2']['param_grid'][best_model_name]
    model_class = get_model_class(params['exp1']['models'][best_model_name])
    data = pd.read_csv(get_dataset_path('train.csv', 'processed'))
    X ,y = data.drop('Potability', axis=1) , data['Potability']
    class_distribution = y.value_counts(normalize=True)
    print(f"Distribution des classes:\n{class_distribution}")
    with mlflow.start_run(run_name=f"BestModel_Optimized_{best_model_name}_Hyperparameter_Tuning"):
        grid_search = GridSearchCV(model_class(), param_grid=grid_config, cv=params['exp2']['cv'], scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        best_model = grid_search.best_estimator_
        mlflow.sklearn.log_model(best_model, "best_model")
        mlflow.log_params(best_params)
        mlflow.log_metric("best_score", best_score)
        with open('best_model_params.json', 'w') as f:
            json.dump(best_params, f)
        mlflow.log_artifact('best_model_params.json')
if __name__ == "__main__":
    main()

'''
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

with mlflow.start_run(run_name="-Optimized_RF_Hyperparameter_Tuning"):
    mlflow.log_param("model_name", "RandomForestClassifier")
    mlflow.log_param("cv_strategy", "StratifiedKFold")
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid_search_model = GridSearchCV(model, param_grid=param_grid, n_jobs=-1, cv=cv_strategy, verbose= 1, scoring='accuracy')
    grid_search_model.fit(X_train, y_train)
    best_params = grid_search_model.best_params_
    best_score = grid_search_model.best_score_
    best_model = grid_search_model.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    f1= f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    mlflow.log_params(best_params)
    mlflow.log_metric("best_score", best_score)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric('f1_score', f1)
    mlflow.log_metric('roc_auc', roc_auc)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Potable', 'Potable'], yticklabels=['Not Potable', 'Potable'])  
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')
    feature_importances = best_model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.savefig('feature_importances.png')
    mlflow.log_artifact('feature_importances.png')
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score: {best_score:.4f}")
    print(f"test accuracy sur le test set: {accuracy:.4f}")
    print(f"roc_auc score: {roc_auc:.4f}")
    
 '''   
    
    



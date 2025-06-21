import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
import mlflow 
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("exp2_random_forest_Hyperparameter_tuning")
data = pd.read_csv(r"C:\Users\HP\OneDrive\Bureau\water_potability.csv")
imputer = SimpleImputer(strategy='median')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
X = data.drop('Potability', axis=1)
y = data['Potability']
class_distribution = y.value_counts(normalize=True)
print(f"Distribution des classes:\n{class_distribution}")
data_train, data_test = train_test_split(data, test_size=0.2, random_state=42, stratify=y)
X_train= data_train.drop('Potability', axis=1)
X_test = data_test.drop('Potability', axis=1)
y_train = data_train['Potability']
y_test = data_test['Potability']

#model = cross_validate(RandomForestClassifier(), X, y, cv=3, scoring='accuracy')
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 5, 10, 20, 30],
}
'''
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.5, 0.8],
    'bootstrap': [True, False]
}
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
    
    
    
    



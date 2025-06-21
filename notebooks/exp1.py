import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("exp1_median_models")

data = pd.read_csv(r"C:\Users\HP\OneDrive\Bureau\water_potability.csv")
def missing_values(data):
    for col in data.columns:
        if data[col].isnull().any():
            data[col].fillna(data[col].median(), inplace=True)
    return data
data = missing_values(data)
# cross-validation
#model = cross_validate(RandomForestClassifier(), data.drop('Potability', axis=1), data['Potability'], cv=5, scoring='accuracy')
X = data.drop('Potability', axis=1)
y = data['Potability']
#data_train , data_test = train_test_split(data, test_size=0.2, random_state=42)
#X_train = data_train.drop('Potability', axis=1)
#y_train= data_train['Potability']
#X_test = data_test.drop('Potability', axis=1)
#y_test = data_test['Potability']
models = {"RandomForestClassifier": RandomForestClassifier(),
          "GradientBoostingClassifier": GradientBoostingClassifier(),
          "HistGradientBoostingClassifier": HistGradientBoostingClassifier(),
          "SVC": SVC(),
          "LogisticRegression": LogisticRegression()}


with mlflow.start_run(run_name="RandomForestModel_median"):
    for model_name, model in models.items():
        with mlflow.start_run(run_name= model_name, nested= True):
            cmodel = cross_validate(model, X, y, cv=5)
            mean_accuracy = cmodel['test_score'].mean()
            std_accuracy = cmodel['test_score'].std()
            mlflow.log_param("model_name", model_name)
            mlflow.log_metric("mean_accuracy", mean_accuracy)
            mlflow.log_metric("std_accuracy", std_accuracy)
            
            
            #y_pred = model.predict(X_test)
            #accuracy = accuracy_score(y_test, y_pred)
            #mlflow.sklearn.log_model(model, model_name)
            #mlflow.log_metric(f"{model_name}_accuracy", accuracy)
            
    
    
    
    
    
    


    
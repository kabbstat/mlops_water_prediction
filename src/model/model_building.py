import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml

def data_loading(filepath):
    try:
        data = pd.read_csv(filepath)
        X_train = data.drop(columns=['Potability'])
        y_train = data['Potability']
        return X_train, y_train
    except Exception as e:
        raise Exception(f"error loading data from {filepath}: {e}")
#data_train = pd.read_csv('./data/processed/train_processed.csv')
#data_test = pd.read_csv('./data/processed/test_processed.csv')
def load_params(filepath):
    try:
        with open(filepath, 'r') as file:
            params = yaml.safe_load(file)
            return params['model_building']['n_estimators']
    except Exception as e:
        raise Exception(f"error loading parameters from {filepath}: {e}")
#n_estimators = yaml.safe_load(open('params.yaml'))['model_building']['n_estimators']
def train_model(X, y, n_estimators):
    try:
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(X, y )
        return clf
    except Exception as e:
        raise Exception(f"error in training: {e}") 
def save_model(model, filepath):
    try:
        with open(filepath, 'wb') as file :
            pickle.dump(model, file)
    except Exception as e:
        raise Exception(f"error in saving model to {filepath}: {e}")
def main():
    try:
      X_train, y_train = data_loading('./data/processed/train_processed.csv')
      n_estimators =load_params('params.yaml')
      model = train_model(X_train, y_train, n_estimators)
      save_model(model, 'model.pkl')
    except Exception as e:
        print(f"An error occurred: {e}")
if __name__ == "__main__":
    main()
       

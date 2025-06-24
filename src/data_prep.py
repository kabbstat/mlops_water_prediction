import numpy as np
import pandas as pd
import os
from sklearn.impute import SimpleImputer
from utils import get_dataset_path

def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        return data
    except Exception as e:
        raise Exception(f"error loading data from {filepath}: {e}")

def fill_missing_values(data):
    if 'Potability' in data.columns:
        features = data.drop('Potability', axis=1)
        target = data['Potability']
    else:
        features = data.copy()
        target = None
    missing_count = features.isnull().sum().sum()
    if missing_count > 0 :
        impute = SimpleImputer(strategy='median')
        features_imputed = pd.DataFrame(impute.fit_transform(features), columns=features.columns, index=features.index)
        print(f"Missing values found and imputed. Total missing values: {missing_count}")
    else:
        features_imputed = features
        print("No missing values found in the dataset.")
    if target is not None:
       result =  pd.concat([features_imputed, target], axis=1)
    else:
        result = features_imputed

    return result

def main(): 
    data_train_path = get_dataset_path('train.csv', 'raw')
    data_test_path = get_dataset_path('test.csv', 'raw')
    
    data_train = load_data(data_train_path)
    data_test = load_data(data_test_path)
    
    train_processed_data = fill_missing_values(data_train)
    test_processed_data = fill_missing_values(data_test)
    
    train_processed_path = get_dataset_path('train.csv', 'processed')
    test_processed_path = get_dataset_path('test.csv', 'processed')
    
    os.makedirs(os.path.dirname(train_processed_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_processed_path), exist_ok=True)

    train_processed_data.to_csv(train_processed_path, index=False)
    test_processed_data.to_csv(test_processed_path, index=False)
if __name__ == "__main__":
    main()  
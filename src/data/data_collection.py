import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
from utils import load_params, get_dataset_path



def main ():    
   params = load_params()
   data = pd.read_csv(params['data_collection']['data_path'])
   train , test = train_test_split(data, test_size=params['data_collection']['test_size'], random_state=42, stratify=data['Potability'])
   train_path = get_dataset_path('train.csv', 'raw')
   test_path = get_dataset_path('test.csv', 'raw')
   os.makedirs(os.path.dirname(train_path), exist_ok=True)
   os.makedirs(os.path.dirname(test_path), exist_ok=True)
   train.to_csv(train_path, index=False)
   test.to_csv(test_path, index=False)
if __name__ == "__main__":
    main()
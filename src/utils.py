import pandas as pd
import yaml
import os

def load_params():
    with open('params.yaml','r') as f:
        return yaml.safe_load(f)

def get_dataset_path(filename, stage):
    if stage == 'raw':
        return os.path.join('data', 'raw', filename)
    elif stage == 'processed':
        name, ext = os.path.splitext(filename)
        return os.path.join('data', 'processed',f"{name}_processed{ext}")

import pandas as pd
import yaml
import os

def load_params():
    with open('params.yaml','r') as f:
        return yaml.safe_load(f)

def get_dataset_path(filename, stage):
    name, ext = os.path.splitext(filename)
    if stage == 'raw':
        return os.path.join('data', 'raw', filename)
    elif stage == 'processed':
        return os.path.join('data', 'processed',f"{name}_processed{ext}")
    else:
        raise ValueError(f"Unknown stage: {stage}. Use 'raw' or 'processed'.")

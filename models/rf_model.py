import json
import os
from sklearn.ensemble import RandomForestClassifier

def create_model(params_path=None):
    """
    Load RF hyperparameters from JSON and return a fresh RandomForestClassifier.
    """
    if params_path is None:
        params_path = os.path.join(os.path.dirname(__file__), 'rf_params.json')
    with open(params_path) as f:
        params = json.load(f)
    return RandomForestClassifier(**params)

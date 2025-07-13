import os, sys
# make project root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import os
import joblib
import pandas as pd
from models.rf_model import create_model
from timer import start_timer 

DATA_DIR  = 'data/processed'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    """
    Trains and saves a Random Forest classifier using a preprocessed training dataset.
    Loads feature data and labels, fits the model, and persists it to disk.

    Usage:
        python train_rf.py
    """
    # Load the processed training dataset from compressed CSV
    train_path = os.path.join(DATA_DIR, 'train_data.csv.gz')
    df = pd.read_csv(train_path, compression='gzip')
    y = df['Label'].astype(int).values
    X = df.drop(columns=['Label']).values
    print(f"Loaded {len(y)} training samples")

    # Create and fit the Random Forest model
    rf = create_model()
    rf.fit(X, y)
    print("Random Forest training complete.")

    # Save the trained model to disk
    out_path = os.path.join(MODEL_DIR, 'rf_model.joblib')
    joblib.dump(rf, out_path)
    print(f"Model written to {out_path}")

if __name__ == '__main__':
    start_timer()
    print("\n")
    main()

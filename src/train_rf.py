import os, sys
# make the parent of src/ (i.e. your project root) importable
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
    # 1) Load the processed training set
    train_path = os.path.join(DATA_DIR, 'train_data.csv.gz')
    df = pd.read_csv(train_path, compression='gzip')
    y = df['Label'].astype(int).values
    X = df.drop(columns=['Label']).values
    print(f"Loaded {len(y)} training samples")

    # 2) Instantiate & fit the RF
    rf = create_model()
    rf.fit(X, y)
    print("Random Forest training complete.")

    # 3) Persist the model
    out_path = os.path.join(MODEL_DIR, 'rf_model.joblib')
    joblib.dump(rf, out_path)
    print(f"Model written to {out_path}")

if __name__ == '__main__':
    start_timer()
    print("\n")
    main()

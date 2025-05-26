import os
import sys
import numpy as np
# make project root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from timer import start_timer
import joblib
import pandas as pd
from models.svm_model import build_svm

DATA_DIR  = "data/processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_train_sample(path, sample_frac=0.05, random_state=42):
    """
    Read the gzipped CSV in chunks, sample fraction `sample_frac` from each chunk,
    and concatenate into one array.
    """
    reader = pd.read_csv(path, compression="gzip", chunksize=200_000)
    pieces = []
    for chunk in reader:
        pieces.append(chunk.sample(frac=sample_frac, random_state=random_state))
    df = pd.concat(pieces, ignore_index=True)
    y = df["Label"].astype(int).values
    X = df.drop(columns=["Label"]).values
    return X, y

def main():
    start_timer()
    train_path = os.path.join(DATA_DIR, "train_data.csv.gz")

    print(f"→ Loading & sampling training data from {train_path}")
    X_train, y_train = load_train_sample(train_path, sample_frac=0.05)
    print(f"   Loaded {X_train.shape[0]:,} samples with {X_train.shape[1]} features")

    print("→ Training approximate RBF-SVM (Nystroem → LinearSVC)…")
    svm = build_svm(
        kernel="rbf",
        gamma=0.1,
        n_components=300,
        C=1.0,
        max_iter=10000,
        random_state=42
    )
    svm.fit(X_train, y_train)
    print("✔ Training complete.")

    out_path = os.path.join(MODEL_DIR, "svm_pipeline.joblib")
    joblib.dump(svm, out_path)
    print(f"✔ Saved pipeline to {out_path}")

if __name__ == '__main__':
    print()  
    main()

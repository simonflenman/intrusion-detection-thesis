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
    Loads and subsamples a training dataset from a compressed CSV file.
    Streams the file in chunks, samples a fraction of rows from each chunk, and combines them.

    Args:
        path (str): Path to the compressed CSV file.
        sample_frac (float): Fraction of rows to sample from each chunk.
        random_state (int): Random seed for reproducibility.

    Returns:
        X (ndarray): Feature matrix.
        y (ndarray): Label array.
    """
    reader = pd.read_csv(path, compression="gzip", chunksize=200_000)
    pieces = []
    for chunk in reader:
        # Sample a fraction from each chunk to avoid full memory load
        pieces.append(chunk.sample(frac=sample_frac, random_state=random_state))
    df = pd.concat(pieces, ignore_index=True)
    y = df["Label"].astype(int).values
    X = df.drop(columns=["Label"]).values
    return X, y

def main():
    """
    Builds, trains, and saves an approximate RBF SVM model using a sampled training dataset.
    Uses the Nystroem kernel approximation combined with a linear SVM to approximate RBF behavior.
    Saves the final pipeline to disk.

    Usage:
        python train_svm.py
    """
    start_timer()

    # Define path to training data
    train_path = os.path.join(DATA_DIR, "train_data.csv.gz")

    print(f"→ Loading & sampling training data from {train_path}")
    X_train, y_train = load_train_sample(train_path, sample_frac=0.05)
    print(f"   Loaded {X_train.shape[0]:,} samples with {X_train.shape[1]} features")

    # Build and train approximate RBF-SVM model (Nystroem + LinearSVC)
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

    # Save the trained SVM pipeline to disk
    out_path = os.path.join(MODEL_DIR, "svm_pipeline.joblib")
    joblib.dump(svm, out_path)
    print(f"✔ Saved pipeline to {out_path}")

if __name__ == '__main__':
    print()
    main()

import os
import sys
import json
import pandas as pd
import joblib
from sklearn.utils import check_array
from timer import start_timer

# make project root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.hdbscan_model import build_hdbscan

DATA_DIR  = 'data/processed'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def load_unsup_sample(path, sample_frac=0.1, random_state=42, chunksize=200_000):
    """
    Loads a subsampled unsupervised training dataset from a compressed CSV file in chunks.
    Samples a specified fraction from each chunk and combines them into a single array.

    Args:
        path (str): Path to the compressed CSV file.
        sample_frac (float): Fraction of rows to sample from each chunk.
        random_state (int): Random seed for reproducibility.
        chunksize (int): Number of rows per chunk.

    Returns:
        ndarray: Combined sampled feature matrix.
    """
    reader = pd.read_csv(path, compression='gzip', low_memory=False, chunksize=chunksize)
    pieces = []
    for chunk in reader:
        # Sample a fraction from each chunk to avoid loading the entire file at once
        pieces.append(chunk.sample(frac=sample_frac, random_state=random_state))
    df = pd.concat(pieces, ignore_index=True)
    return df.values

def main():
    """
    Builds, trains, and saves an HDBSCAN clusterer using a subsampled unsupervised training dataset.
    Saves hyperparameters and the fitted model for later evaluation and inference.

    Usage:
        python train_hdbscan.py
    """
    start_timer()

    # Define hyperparameters for the HDBSCAN model
    params = {
        'min_cluster_size': 50,
        'min_samples':      50,
        'metric':           'euclidean'
    }

    # Save hyperparameters to a JSON file
    with open(os.path.join(MODEL_DIR, 'hdbscan_params.json'), 'w') as f:
        json.dump(params, f, indent=2)

    # Load and subsample unsupervised training data
    train_path = os.path.join(DATA_DIR, 'unsupervised_train_data.csv.gz')
    print(f"→ Sampling 10% of benign data from {train_path}")
    X_train = load_unsup_sample(train_path, sample_frac=0.1)
    print(f"   Sample shape: {X_train.shape[0]:,} × {X_train.shape[1]}")

    # Check data integrity to ensure finite values
    X_train = check_array(X_train, ensure_all_finite=True)

    # Build and fit the HDBSCAN clusterer
    clusterer = build_hdbscan(**params)
    print("→ Fitting HDBSCAN (this may still take a few minutes)…")
    clusterer.fit(X_train)

    # Report the number of clusters and detected outliers
    n_clusters = clusterer.labels_.max() + 1
    n_outliers = (clusterer.labels_ == -1).sum()
    print(f"✔ Done: {n_clusters} clusters, {n_outliers:,} outliers")

    # Save the trained clusterer to disk
    out_path = os.path.join(MODEL_DIR, 'hdbscan_clusterer.joblib')
    joblib.dump(clusterer, out_path)
    print(f"✔ Saved clusterer to {out_path}")

if __name__ == '__main__':
    main()

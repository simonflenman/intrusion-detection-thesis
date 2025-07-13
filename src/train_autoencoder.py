import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import numpy as np
import pandas as pd
import tensorflow as tf
from models.autoencoder_model import build_autoencoder
from timer import start_timer 

DATA_DIR  = 'data/processed'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def load_unsup_train(path):
    """
    Loads an unsupervised training dataset from a compressed CSV file.
    Returns the feature matrix as a NumPy array.

    Args:
        path (str): Path to the compressed CSV file.

    Returns:
        ndarray: Feature matrix.
    """
    df = pd.read_csv(path, compression='gzip')
    return df.values

def main():
    """
    Builds, trains, and saves an autoencoder model using an unsupervised training dataset.
    Computes a reconstruction error threshold for anomaly detection and saves it to disk.
    Also saves the trained model in native .keras format.

    Usage:
        python train_autoencoder.py
    """
    # Load unsupervised training data
    X_train = load_unsup_train(os.path.join(DATA_DIR, 'unsupervised_train_data.csv.gz'))
    input_dim = X_train.shape[1]

    # Build and compile the autoencoder
    autoenc = build_autoencoder(input_dim=input_dim, bottleneck_dim=16)
    autoenc.compile(optimizer='adam', loss='mse')

    # Define callbacks: early stopping and model checkpoint
    ckpt_path = os.path.join(MODEL_DIR, 'autoencoder.keras')
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            save_best_only=True,
            monitor='val_loss'
        )
    ]

    # Train the autoencoder
    autoenc.fit(
        X_train, X_train,
        epochs=50,
        batch_size=256,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=2
    )

    # Compute per-sample reconstruction error (MSE) in batches
    mse_batches = []
    for i in range(0, X_train.shape[0], 256):
        batch = X_train[i:i + 256]
        rec = autoenc.predict(batch, batch_size=256, verbose=0)
        mse_batches.append(np.mean((batch - rec) ** 2, axis=1))
    mse = np.concatenate(mse_batches)

    # Set threshold as the 99th percentile of reconstruction errors
    threshold = np.percentile(mse, 99)

    # Save threshold to file
    with open(os.path.join(MODEL_DIR, 'threshold.txt'), 'w') as f:
        f.write(f"{threshold:.6f}\n")

    print(f"\n✔ Training complete. Threshold set at {threshold:.6f}")
    print(f"✔ Model saved to {ckpt_path}")

if __name__ == '__main__':
    start_timer()
    print("\n")
    main()

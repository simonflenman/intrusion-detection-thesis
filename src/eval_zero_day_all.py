import os
import time
import joblib
import numpy as np
import tensorflow as tf
import hdbscan

def eval_rf(X, y, model_dir):
    """
    Evaluates a trained Random Forest model on provided data.
    Loads the model, predicts labels, and reports detection count, recall, and execution time.

    Args:
        X (ndarray): Feature matrix.
        y (ndarray): True label array.
        model_dir (str): Directory containing the saved model.
    """
    start = time.time()
    rf = joblib.load(os.path.join(model_dir, 'rf_model.joblib'))
    y_pred = rf.predict(X)
    detected = np.sum(y_pred == 1)
    total = len(y)
    print("--- Random Forest ---")
    print(f"Samples: {total}")
    print(f"Detected as attack: {detected}/{total}   (Recall: {detected / total:.4f})")
    print(f"Time: {time.time() - start:.2f}s\n")

def eval_svm(X, y, model_dir):
    """
    Evaluates a trained SVM (Nystroem + LinearSVC pipeline) on provided data.
    Loads the model, predicts labels, and reports detection count, recall, and execution time.

    Args:
        X (ndarray): Feature matrix.
        y (ndarray): True label array.
        model_dir (str): Directory containing the saved model.
    """
    start = time.time()
    svm = joblib.load(os.path.join(model_dir, 'svm_pipeline.joblib'))
    y_pred = svm.predict(X)
    detected = np.sum(y_pred == 1)
    total = len(y)
    print("--- SVM (Nystroem + LinearSVC) ---")
    print(f"Samples: {total}")
    print(f"Detected as attack: {detected}/{total}   (Recall: {detected / total:.4f})")
    print(f"Time: {time.time() - start:.2f}s\n")

def eval_autoencoder(X, y, model_dir):
    """
    Evaluates a trained autoencoder on provided data.
    Loads the model and threshold, computes reconstruction errors, classifies samples,
    and reports detection count, recall, and execution time.

    Args:
        X (ndarray): Feature matrix.
        y (ndarray): True label array.
        model_dir (str): Directory containing the saved model and threshold file.
    """
    start = time.time()
    keras_path = os.path.join(model_dir, 'autoencoder.keras')
    h5_path    = os.path.join(model_dir, 'autoencoder.h5')
    model_path = keras_path if os.path.exists(keras_path) else h5_path
    autoenc = tf.keras.models.load_model(model_path, compile=False)
    thresh = float(open(os.path.join(model_dir, 'threshold.txt')).read().strip())

    recon = autoenc.predict(X, batch_size=256, verbose=0)
    mse = np.mean((X - recon) ** 2, axis=1)
    y_pred = (mse > thresh).astype(int)
    detected = np.sum(y_pred == 1)
    total = len(y)

    print("--- Autoencoder ---")
    print(f"Samples: {total}")
    print(f"Detected as attack: {detected}/{total}   (Recall: {detected / total:.4f})")
    print(f"Time: {time.time() - start:.2f}s\n")

def eval_hdbscan(X, y, model_dir):
    """
    Evaluates a pre-trained HDBSCAN clusterer on provided data.
    Reclusters samples, flags outliers as attacks, and reports detection count, recall, and execution time.

    Args:
        X (ndarray): Feature matrix.
        y (ndarray): True label array.
        model_dir (str): Directory containing the saved HDBSCAN clusterer.
    """
    start = time.time()
    model_path = os.path.join(model_dir, 'hdbscan_clusterer.joblib')
    clusterer = joblib.load(model_path)

    # Reclustering directly on the test data
    labels = clusterer.fit_predict(X)

    # Label -1 is considered an outlier
    y_pred = (labels == -1).astype(int)
    detected = np.sum(y_pred == 1)
    total = len(y)

    print("--- HDBSCAN on zero day data ---")
    print(f"Samples: {total}")
    print(f"Detected as outlier: {detected}/{total}   (Recall: {detected / total:.4f})")
    print(f"Time elapsed: {time.time() - start:.4f}s\n")

def main():
    """
    Loads zero-day test data and evaluates all models (Random Forest, SVM, Autoencoder, HDBSCAN).
    Reports detection performance metrics and execution times for each model.
    """
    model_dir = 'models'

    # Load zero-day test data
    import pandas as pd
    df = pd.read_csv('data/processed/zero_day_data.csv.gz', compression='gzip')
    X = df.drop(columns=['Label']).values
    y = df['Label'].astype(int).values

    eval_rf(X, y, model_dir)
    eval_svm(X, y, model_dir)
    eval_autoencoder(X, y, model_dir)
    eval_hdbscan(X, y, model_dir)

if __name__ == '__main__':
    main()

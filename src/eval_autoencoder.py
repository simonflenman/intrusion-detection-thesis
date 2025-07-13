import os, sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from timer import start_timer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
DATA_DIR   = os.path.join(os.path.dirname(__file__),'..','data','processed')
MODEL_DIR  = os.path.join(os.path.dirname(__file__),'..','models')
CHUNK_SIZE = 200_000
AUC_FRAC   = 0.05

def main():
    """
    Evaluates a trained autoencoder model on an unsupervised test dataset.
    Loads the model and threshold, iterates through data in chunks to compute reconstruction
    error, classifies samples as normal or anomaly, and calculates performance metrics
    including confusion matrix, accuracy, precision, recall, F1-score, and approximate AUC.
    Also plots and saves the ROC curve.

    Usage:
        python eval_autoencoder.py
    """
    # Start timer for measuring execution time
    start_timer()

    # Load trained autoencoder model (without compilation)
    keras_p = os.path.join(MODEL_DIR, 'autoencoder.keras')
    model = tf.keras.models.load_model(keras_p, compile=False)

    # Load threshold value for anomaly classification
    threshold = float(open(os.path.join(MODEL_DIR, 'threshold.txt')).read().strip())

    # Read unsupervised test dataset in chunks to avoid memory issues
    reader = pd.read_csv(
        os.path.join(DATA_DIR, 'unsupervised_test_data.csv.gz'),
        compression='gzip', chunksize=CHUNK_SIZE
    )

    # Initialize counters for confusion matrix
    tn = fp = fn = tp = total = 0

    # Set random state for consistent approximate AUC sampling
    rng = np.random.RandomState(42)
    auc_trues, auc_scores = [], []

    # Iterate through each data chunk
    for i, chunk in enumerate(reader, start=1):
        # Separate features and true labels
        X = chunk.drop(columns=['Label']).values
        y = chunk['Label'].astype(int).values

        # Predict reconstruction from autoencoder
        recon = model.predict(X, batch_size=256, verbose=0)

        # Compute mean squared error per sample
        mse = np.mean((X - recon)**2, axis=1)

        # Classify samples as anomaly if MSE > threshold
        y_pred = (mse > threshold).astype(int)

        # Update confusion matrix counters
        tn += np.sum((y == 0) & (y_pred == 0))
        fp += np.sum((y == 0) & (y_pred == 1))
        fn += np.sum((y == 1) & (y_pred == 0))
        tp += np.sum((y == 1) & (y_pred == 1))
        total += len(y)

        # Randomly sample a small fraction for approximate AUC calculation
        mask = rng.rand(len(y)) < AUC_FRAC
        if mask.any():
            auc_trues.append(y[mask])
            auc_scores.append(mse[mask])

        # Log progress every 5 chunks
        if i % 5 == 0:
            print(f"  → processed {total:,} rows…")

    # Compute final metrics
    acc  = (tp + tn) / total
    prec = tp / (tp + fp) if tp + fp else 0
    rec  = tp / (tp + fn) if tp + fn else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(f"         Pred=0     Pred=1")
    print(f" True=0  {tn:10,}  {fp:10,}")
    print(f" True=1  {fn:10,}  {tp:10,}\n")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    # Calculate and plot approximate ROC AUC if samples were collected
    if auc_trues:
        y_true_all  = np.concatenate(auc_trues)
        y_score_all = np.concatenate(auc_scores)
        auc = roc_auc_score(y_true_all, y_score_all)
        print(f"\nApprox ROC AUC (on {len(y_true_all):,} samples): {auc:.4f}")

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true_all, y_score_all)

        # Plot and save ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label=f'AE ROC (AUC={auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('Autoencoder ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig('autoencoder_roc_curve.png', dpi=150)
        plt.close()

if __name__ == '__main__':
    main()

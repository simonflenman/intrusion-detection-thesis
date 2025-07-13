import os, sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from timer import start_timer

# make project root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

DATA_DIR       = os.path.join(os.path.dirname(__file__),'..','data','processed')
MODEL_PATH     = os.path.join(os.path.dirname(__file__),'..','models','svm_pipeline.joblib')
CHUNK_SIZE     = 200_000
AUC_SAMPLE_FRAC= 0.10

def main():
    """
    Evaluates a trained Support Vector Machine (SVM) pipeline on a preprocessed test dataset.
    Loads the model, reads test data in chunks, and computes predictions to update a confusion matrix.
    Calculates performance metrics including accuracy, precision, recall, F1-score, and approximate ROC AUC.
    Also plots and saves the ROC curve.

    Usage:
        python eval_svm.py
    """
    # Start timer to measure runtime
    start_timer()

    # Load the trained SVM pipeline from disk
    svm = joblib.load(MODEL_PATH)
    print(f"Loaded SVM pipeline from {MODEL_PATH}\n")

    # Initialize reader to load test dataset in chunks
    reader = pd.read_csv(
        os.path.join(DATA_DIR, 'test_data.csv.gz'),
        compression='gzip', chunksize=CHUNK_SIZE
    )

    # Initialize confusion matrix counters
    cm = np.zeros((2, 2), dtype=np.int64)

    # Random state for reproducible sampling when computing approximate AUC
    rng = np.random.RandomState(42)
    auc_trues, auc_scores = [], []
    total = 0

    # Iterate over data chunks
    for i, chunk in enumerate(reader, start=1):
        y_true = chunk['Label'].astype(int).values
        X = chunk.drop(columns=['Label']).values

        # Predict labels for the current chunk
        y_pred = svm.predict(X)

        # Update confusion matrix
        cm[0, 0] += np.sum((y_true == 0) & (y_pred == 0))
        cm[0, 1] += np.sum((y_true == 0) & (y_pred == 1))
        cm[1, 0] += np.sum((y_true == 1) & (y_pred == 0))
        cm[1, 1] += np.sum((y_true == 1) & (y_pred == 1))

        # Sample data for approximate AUC calculation
        mask = rng.rand(len(y_true)) < AUC_SAMPLE_FRAC
        if mask.any():
            try:
                # Prefer decision_function if available
                scores = svm.decision_function(X[mask])
            except AttributeError:
                # Fallback to probability if decision_function is unavailable
                scores = svm.predict_proba(X[mask])[:, 1]
            auc_trues.append(y_true[mask])
            auc_scores.append(scores)

        total += len(y_true)

        # Print progress periodically
        if i % 5 == 0:
            print(f"  → processed {total:,} rows…")

    # Extract confusion matrix elements
    tn, fp, fn, tp = cm.ravel()

    # Calculate performance metrics
    acc  = (tp + tn) / total
    prec = tp / (tp + fp) if tp + fp else 0
    rec  = tp / (tp + fn) if tp + fn else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    # Print confusion matrix and metrics
    print("\nConfusion matrix:")
    print(f"          Pred=0     Pred=1")
    print(f" True=0  {tn:10,}  {fp:10,}")
    print(f" True=1  {fn:10,}  {tp:10,}\n")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    # Compute and plot approximate ROC AUC if samples are available
    if auc_trues:
        y_true_all  = np.concatenate(auc_trues)
        y_score_all = np.concatenate(auc_scores)
        auc = roc_auc_score(y_true_all, y_score_all)
        print(f"\nApprox. ROC AUC (on {len(y_true_all):,} sampled rows): {auc:.4f}")

        fpr, tpr, _ = roc_curve(y_true_all, y_score_all)

        plt.figure()
        plt.plot(fpr, tpr, label=f'SVM ROC (AUC={auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('SVM ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig('svm_roc_curve.png', dpi=150)
        plt.close()

if __name__ == '__main__':
    main()

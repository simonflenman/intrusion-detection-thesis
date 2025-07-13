import os, sys
# make project root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import os
import joblib
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt

DATA_DIR  = 'data/processed'
MODEL_DIR = 'models'

def load_test(path):
    """
    Loads a test dataset from a compressed CSV file.
    Returns feature matrix X and label array y.

    Args:
        path (str): Path to the compressed CSV file.

    Returns:
        X (ndarray): Feature matrix.
        y (ndarray): Label array.
    """
    df = pd.read_csv(path, compression='gzip')
    y = df['Label'].astype(int).values
    X = df.drop(columns=['Label']).values
    return X, y

def main():
    """
    Evaluates a trained Random Forest classifier on a preprocessed test dataset.
    Loads test data and a saved model, predicts labels, and reports metrics
    including confusion matrix, classification report, and ROC AUC score.
    Also plots and saves the ROC curve to a file.

    Usage:
        python eval_rf.py
    """
    # Load the test dataset (features and labels)
    test_path = os.path.join(DATA_DIR, 'test_data.csv.gz')
    X_test, y_test = load_test(test_path)
    print(f"Loaded {len(y_test)} test samples")

    # Load the trained Random Forest model from disk
    model_path = os.path.join(MODEL_DIR, 'rf_model.joblib')
    rf = joblib.load(model_path)
    print(f"Loaded RF model from {model_path}\n")

    # Predict labels on the test set
    y_pred = rf.predict(X_test)

    # Display the confusion matrix
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred), "\n")

    # Display detailed classification metrics
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Calculate predicted probabilities for ROC AUC computation
    scores = rf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, scores)
    print(f"\nROC AUC: {auc:.4f}")

    # Compute ROC curve points
    fpr, tpr, _ = roc_curve(y_test, scores)

    # Plot ROC curve and save it to file
    plt.figure()
    plt.plot(fpr, tpr, label=f"RF (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], '--', color='gray', label="random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Random Forest ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    out_fig = 'rf_roc_curve.png'
    plt.savefig(out_fig)
    print(f"ROC curve saved to {out_fig}")

if __name__ == '__main__':
    main()

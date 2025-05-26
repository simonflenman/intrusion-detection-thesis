import os, sys
# make the parent of src/ (i.e. your project root) importable
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
    df = pd.read_csv(path, compression='gzip')
    y = df['Label'].astype(int).values
    X = df.drop(columns=['Label']).values
    return X, y

def main():
    # 1) Load test data
    test_path = os.path.join(DATA_DIR, 'test_data.csv.gz')
    X_test, y_test = load_test(test_path)
    print(f"Loaded {len(y_test)} test samples")

    # 2) Load trained RF
    model_path = os.path.join(MODEL_DIR, 'rf_model.joblib')
    rf = joblib.load(model_path)
    print(f"Loaded RF model from {model_path}\n")

    # 3) Predict & report
    y_pred = rf.predict(X_test)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred), "\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))

    # 4) ROC & AUC
    #    RF supports predict_proba
    scores = rf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, scores)
    print(f"\nROC AUC: {auc:.4f}")

    fpr, tpr, _ = roc_curve(y_test, scores)

    # 5) Plot & save (headless-friendly)
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

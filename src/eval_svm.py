import os
import sys
import joblib
import pandas as pd
import numpy as np
from timer import start_timer
from sklearn.metrics import roc_auc_score

# make the project root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA_DIR   = os.path.join(os.path.dirname(__file__),"..","data","processed")
MODEL_PATH = os.path.join(os.path.dirname(__file__),"..","models","svm_pipeline.joblib")

# how big each chunk to read (tweak to your RAM)
CHUNK_SIZE = 200_000

# fraction of test set to sample for AUC
AUC_SAMPLE_FRAC = 0.10

def main():
    start_timer()

    # 1) load model
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: no model at {MODEL_PATH}. Run train_svm.py first.")
        sys.exit(1)
    svm = joblib.load(MODEL_PATH)
    print(f"Loaded SVM pipeline from {MODEL_PATH}\n")

    # 2) stream through test set
    test_file = os.path.join(DATA_DIR, "test_data.csv.gz")
    if not os.path.exists(test_file):
        print(f"ERROR: no test data at {test_file}")
        sys.exit(1)

    print(f"Evaluating on {test_file} in chunks of {CHUNK_SIZE:,} rows…\n")

    # running counts for confusion matrix: [ [TN, FP], [FN, TP] ]
    cm = np.zeros((2,2), dtype=np.int64)

    # for AUC: sample a subset
    rng = np.random.RandomState(42)
    auc_trues = []
    auc_scores = []

    reader = pd.read_csv(test_file, compression="gzip", chunksize=CHUNK_SIZE)
    total_rows = 0
    for i, chunk in enumerate(reader, start=1):
        y_true = chunk["Label"].astype(int).values
        X = chunk.drop(columns=["Label"]).values
        y_pred = svm.predict(X)

        # accumulate confusion
        # TN
        cm[0,0] += np.sum((y_true==0)&(y_pred==0))
        # FP
        cm[0,1] += np.sum((y_true==0)&(y_pred==1))
        # FN
        cm[1,0] += np.sum((y_true==1)&(y_pred==0))
        # TP
        cm[1,1] += np.sum((y_true==1)&(y_pred==1))

        # AUC sampling
        mask = rng.rand(len(y_true)) < AUC_SAMPLE_FRAC
        if mask.any():
            try:
                scores = svm.decision_function(X[mask])
            except AttributeError:
                scores = svm.predict_proba(X[mask])[:,1]
            auc_trues.append(y_true[mask])
            auc_scores.append(scores)

        total_rows += len(y_true)
        if i % 5 == 0:
            print(f"  → processed {total_rows:,} rows…")

    print("\n✔ Done streaming predictions.")
    print(f"Total test samples: {total_rows:,}\n")

    # 3) report confusion matrix
    tn, fp, fn, tp = cm.ravel()
    print("Confusion matrix:")
    print(f"          Pred=0     Pred=1")
    print(f" True=0  {tn:10,}  {fp:10,}")
    print(f" True=1  {fn:10,}  {tp:10,}")
    print()

    # 4) derived metrics
    precision = tp / (tp + fp) if tp+fp else 0.0
    recall    = tp / (tp + fn) if tp+fn else 0.0
    f1        = 2 * precision * recall / (precision + recall) if precision+recall else 0.0
    acc       = (tp + tn) / total_rows

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    # 5) ROC AUC on sampled subset
    if auc_trues:
        y_true_all = np.concatenate(auc_trues)
        y_score_all= np.concatenate(auc_scores)
        auc = roc_auc_score(y_true_all, y_score_all)
        print(f"\nApprox. ROC AUC (on {len(y_true_all):,} sampled rows): {auc:.4f}")

if __name__ == "__main__":
    main()

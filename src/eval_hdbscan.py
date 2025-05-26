# src/eval_hdbscan.py

import os
import sys
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from timer import start_timer

# make project root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hdbscan

DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODEL_PATH  = os.path.join(os.path.dirname(__file__), '..', 'models', 'hdbscan_clusterer.joblib')

# read 200 k rows at a time, sample 10 % for evaluation
CHUNK_SIZE  = 200_000
SAMPLE_FRAC = 0.10

def main():
    start_timer()
    print("→ Loading base HDBSCAN clusterer…")
    base = joblib.load(MODEL_PATH)

    # pull out the training hyper-parameters
    params = dict(
        min_cluster_size = base.min_cluster_size,
        min_samples      = base.min_samples,
        metric           = base.metric,
        cluster_selection_method = base.cluster_selection_method
    )

    reader = pd.read_csv(
        os.path.join(DATA_DIR, 'unsupervised_test_data.csv.gz'),
        compression = 'gzip',
        chunksize    = CHUNK_SIZE,
        low_memory   = False
    )

    # confusion-matrix counters, but only on the **sampled** points
    tn = fp = fn = tp = total = 0

    rng = np.random.RandomState(42)
    auc_trues, auc_scores = [], []

    for i, chunk in enumerate(reader, start=1):
        y = chunk['Label'].astype(int).values
        X = chunk.drop(columns=['Label']).values

        # sample 10 % of this chunk
        mask = rng.rand(len(y)) < SAMPLE_FRAC
        if not mask.any():
            continue

        Xs = X[mask]
        ys = y[mask]
        total += len(ys)

        # recluster **only** on the sampled subset
        clusterer = hdbscan.HDBSCAN(**params)
        labels = clusterer.fit_predict(Xs)
        # use outlier_scores_ as your continuous anomaly strength
        strengths = clusterer.outlier_scores_

        # binary predictions
        y_pred = (labels == -1).astype(int)

        # accumulate confusion
        tn += np.sum((ys == 0) & (y_pred == 0))
        fp += np.sum((ys == 0) & (y_pred == 1))
        fn += np.sum((ys == 1) & (y_pred == 0))
        tp += np.sum((ys == 1) & (y_pred == 1))

        # store for ROC AUC
        auc_trues .append(ys)
        auc_scores.append(strengths)

        if i % 5 == 0:
            processed = i * CHUNK_SIZE
            print(f"  → read ~{processed:,} rows, sampled {total:,} so far…")

    # final metrics
    print("\n✔ Done streaming HDBSCAN eval on 10 % sample")
    print(f"Sample size: {total:,}\n")

    # confusion matrix
    print("Confusion Matrix (on sampled points):")
    print(f"         Pred=0     Pred=1")
    print(f" True=0  {tn:10,}  {fp:10}")
    print(f" True=1  {fn:10,}  {tp:10}\n")

    precision = tp/(tp+fp) if tp+fp else 0.0
    recall    = tp/(tp+fn) if tp+fn else 0.0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
    acc       = (tp+tn)/total

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    # === ROC AUC on sampled points (drop NaNs) ===
    y_true_all  = np.concatenate(auc_trues)
    y_score_all = np.concatenate(auc_scores)
    mask = ~np.isnan(y_score_all)
    y_true_all  = y_true_all[mask]
    y_score_all = y_score_all[mask]

    if len(y_true_all):
        auc = roc_auc_score(y_true_all, y_score_all)
        fpr, tpr, _ = roc_curve(y_true_all, y_score_all)

        print(f"\nApprox ROC AUC (on {len(y_true_all):,} samples): {auc:.4f}")

        # plot + save
        plt.figure()
        plt.plot(fpr, tpr, label=f'HDBSCAN ROC (AUC={auc:.4f})')
        plt.plot([0,1], [0,1], 'k--', linewidth=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('HDBSCAN ROC Curve')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig('hdbscan_roc_curve.png', dpi=150)
        plt.close()
    else:
        print("\nNo valid scores to compute ROC AUC (all NaN).")

if __name__ == '__main__':
    main()

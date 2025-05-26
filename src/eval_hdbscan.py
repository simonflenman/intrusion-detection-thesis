import os, sys
import pandas as pd
import joblib
import numpy as np
from timer import start_timer

# make project root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import hdbscan

DATA_DIR   = os.path.join(os.path.dirname(__file__),'..','data','processed')
MODEL_PATH = os.path.join(os.path.dirname(__file__),'..','models','hdbscan_clusterer.joblib')
CHUNK_SIZE = 200_000

def main():
    start_timer()

    print("→ Loading HDBSCAN clusterer…")
    clusterer = joblib.load(MODEL_PATH)

    reader = pd.read_csv(
        os.path.join(DATA_DIR,'unsupervised_test_data.csv.gz'),
        compression='gzip', chunksize=CHUNK_SIZE, low_memory=False
    )

    tn = fp = fn = tp = total = 0

    for i, chunk in enumerate(reader, start=1):
        X = chunk.drop(columns=['Label']).values
        y = chunk['Label'].astype(int).values

        # if prediction_data_ exists we can do approximate_predict
        if getattr(clusterer, 'prediction_data_', None) is not None:
            labels, _ = hdbscan.approximate_predict(clusterer, X)
        else:
            # fallback: recluster this CHUNK from scratch
            labels = clusterer.fit_predict(X)

        y_pred = (labels == -1).astype(int)

        tn += np.sum((y==0)&(y_pred==0))
        fp += np.sum((y==0)&(y_pred==1))
        fn += np.sum((y==1)&(y_pred==0))
        tp += np.sum((y==1)&(y_pred==1))
        total += len(y)

        if i % 5 == 0:
            print(f"  → processed {total:,} rows…")

    print("\n✔ Done streaming HDBSCAN eval")
    print(f"Total samples: {total:,}\n")

    acc  = (tp+tn)/total
    prec = tp/(tp+fp) if tp+fp else 0.0
    rec  = tp/(tp+fn) if tp+fn else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0

    print("Confusion Matrix:")
    print(f"         Pred=0     Pred=1")
    print(f" True=0  {tn:10,}  {fp:10,}")
    print(f" True=1  {fn:10,}  {tp:10,}\n")

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

if __name__ == '__main__':
    main()

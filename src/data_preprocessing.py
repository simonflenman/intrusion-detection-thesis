import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
RAW_DIR = 'data/CIC-IDS2018'
PROCESSED_DIR = 'data/processed'
CHUNK_SIZE = 10000

# Ensure output directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

def clean_chunk(chunk):
    """
    Drop IP/port/timestamp columns, ensure labels present and valid,
    coerce features to numeric, and drop any rows with NaNs in features.
    """
    drop_cols = [
        'Flow ID', 'Source IP', 'Source Port',
        'Destination IP', 'Destination Port', 'Timestamp'
    ]
    chunk = chunk.drop(columns=drop_cols, errors='ignore')
    if 'Label' not in chunk.columns:
        return pd.DataFrame()
    chunk = chunk.dropna(subset=['Label'])
    chunk['Label'] = chunk['Label'].astype(str).str.strip()
    valid = chunk['Label'].ne('') & ~chunk['Label'].str.lower().eq('nan')
    chunk = chunk.loc[valid]
    labels = chunk['Label'].astype(str)
    feats = chunk.drop(columns=['Label']).apply(pd.to_numeric, errors='coerce')
    mask = feats.notna().all(axis=1)
    df = feats.loc[mask].copy()
    df['Label'] = labels.loc[mask].values
    return df


def first_pass(zero_day=None):
    if zero_day is None:
        counts = {}
        for fname in sorted(os.listdir(RAW_DIR)):
            if not fname.endswith('.csv'): continue
            path = os.path.join(RAW_DIR, fname)
            for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):
                cl = clean_chunk(chunk)
                if cl.empty: continue
                for lbl, cnt in cl['Label'].value_counts().items():
                    counts[lbl] = counts.get(lbl, 0) + cnt
        zero_day = min(counts, key=counts.get)
        print(f"Identified zero-day attack: {zero_day}")
    scaler = StandardScaler()
    print("Fitting scaler on supervised training data...")
    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.endswith('.csv'): continue
        path = os.path.join(RAW_DIR, fname)
        for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):
            cl = clean_chunk(chunk)
            if cl.empty: continue
            sub = cl.loc[cl['Label'] != zero_day]
            if sub.empty: continue
            X = sub.drop(columns=['Label']).apply(pd.to_numeric, errors='coerce')
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            X.dropna(inplace=True)
            if not X.empty:
                scaler.partial_fit(X)
    print("Scaler fitted.\n")
    return zero_day, scaler


def write_split(mode, zero_day, scaler):
    fn_map = {
        'train':       'train_data.csv.gz',
        'test':        'test_data.csv.gz',
        'unsup_train': 'unsupervised_train_data.csv.gz',
        'unsup_test':  'unsupervised_test_data.csv.gz',
        'zero_day':    'zero_day_data.csv.gz'
    }
    out_fn = fn_map[mode]
    out_path = os.path.join(PROCESSED_DIR, out_fn)
    if os.path.exists(out_path): os.remove(out_path)

    print(f"Writing '{mode}' split to {out_fn}...")
    header = False
    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.endswith('.csv'): continue
        path = os.path.join(RAW_DIR, fname)
        for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):
            cl = clean_chunk(chunk)
            if cl.empty: continue

            orig = cl['Label'].astype(str).str.strip()
            X = cl.drop(columns=['Label']).apply(pd.to_numeric, errors='coerce')
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            X.fillna(0, inplace=True)
            Xs = scaler.transform(X)
            df = pd.DataFrame(Xs, columns=X.columns)
            df['Label'] = orig.values

            if mode == 'train':
                mask = df['Label'] != zero_day
                df2 = df.loc[mask].copy()
                df2['Label'] = (df2['Label'] != 'BENIGN').astype(int)
            elif mode == 'test':
                df2 = df.copy()
                df2['Label'] = (df2['Label'] != 'BENIGN').astype(int)
            elif mode == 'unsup_train':
                mask = df['Label'] == 'BENIGN'
                df2 = df.loc[mask].drop(columns=['Label'])
            elif mode == 'unsup_test':
                df2 = df.copy()
                df2['Label'] = (df2['Label'] != 'BENIGN').astype(int)
            elif mode == 'zero_day':
                mask = df['Label'] == zero_day
                df2 = df.loc[mask].copy()
                df2['Label'] = 1
            else:
                raise ValueError(f"Unknown mode: {mode}")

            df2.to_csv(
                out_path,
                index=False,
                header=not header,
                mode='a',
                compression='gzip'
            )
            header = True
    print(f"✓ '{mode}' split complete.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        choices=['train','test','unsup_train','unsup_test','zero_day'],
        required=True,
        help='Which split to generate'
    )
    args = parser.parse_args()

    zero_day, scaler = first_pass()
    write_split(args.mode, zero_day, scaler)

if __name__ == '__main__':
    main()

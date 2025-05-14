import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from math import sqrt

# --- Configuration ---
RAW_DIR = 'data/CIC-IDS2018'
PROCESSED_DIR = 'data/processed'
CHUNK_SIZE = 10000

# Ensure output directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Columns to drop (both full and abbreviated variants)
DROP_COLS = [
    'Flow ID',
    'Source IP', 'Destination IP', 'Src IP', 'Dst IP',
    'Source Port', 'Destination Port', 'Src Port', 'Dst Port',
    'Timestamp'
]

VALID_LABEL_FILTERS = ['nan', 'label']  # lowercase


def clean_chunk(chunk):
    """
    Drop irrelevant columns and ensure labels are present & valid.
    Does *not* drop feature-NaN rows (they will be filled downstream).
    """
    # 1) Drop IP/port/timestamp cols
    chunk = chunk.drop(columns=DROP_COLS, errors='ignore')

    # 2) Require a Label col
    if 'Label' not in chunk.columns:
        return pd.DataFrame()

    # 3) Drop rows with missing Label
    chunk = chunk.dropna(subset=['Label'])

    # 4) Normalize & strip label strings
    chunk['Label'] = chunk['Label'].astype(str).str.strip()

    # 5) Remove empty or invalid labels ("nan", header strings like "label")
    lbl_lower = chunk['Label'].str.lower()
    valid = chunk['Label'].ne('') & ~lbl_lower.isin(VALID_LABEL_FILTERS)
    chunk = chunk.loc[valid]

    return chunk


def first_pass(zero_day=None):
    # Identify least frequent valid Label
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
        # Drop any accidental 'Label' key
        counts.pop('Label', None)
        zero_day = min(counts, key=counts.get)
        print(f"Identified zero-day attack: {zero_day}")

    # Fit scaler on supervised (exclude zero-day)
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
    out_path = os.path.join(PROCESSED_DIR, fn_map[mode])
    if os.path.exists(out_path): os.remove(out_path)

    print(f"Writing '{mode}' split to {fn_map[mode]}...")
    header = False
    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.endswith('.csv'): continue
        path = os.path.join(RAW_DIR, fname)
        for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):
            cl = clean_chunk(chunk)
            if cl.empty: continue
            orig = cl['Label'].astype(str).str.strip()

            # Features to numeric, fill for scaling
            X = cl.drop(columns=['Label']).apply(pd.to_numeric, errors='coerce')
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            X.fillna(0, inplace=True)
            Xs = scaler.transform(X)
            df = pd.DataFrame(Xs, columns=X.columns)
            df['Label'] = orig.values

            # Mode-specific splits
            if mode == 'train':
                mask = df['Label'] != zero_day
                df2 = df.loc[mask].copy()
                df2['Label'] = (df2['Label'].str.lower() != 'benign').astype(int)
            elif mode == 'test':
                df2 = df.copy()
                df2['Label'] = (df2['Label'].str.lower() != 'benign').astype(int)
            elif mode == 'unsup_train':
                mask = df['Label'].str.lower() == 'benign'
                df2 = df.loc[mask].drop(columns=['Label'])
            elif mode == 'unsup_test':
                df2 = df.copy()
                df2['Label'] = (df2['Label'].str.lower() != 'benign').astype(int)
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
        '--mode', choices=['train','test','unsup_train','unsup_test','zero_day'],
        required=True, help='Which split to generate'
    )
    args = parser.parse_args()

    zero_day, scaler = first_pass()
    write_split(args.mode, zero_day, scaler)

if __name__ == '__main__':
    main()

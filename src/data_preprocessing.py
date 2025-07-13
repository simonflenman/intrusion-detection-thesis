import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
RAW_DIR       = 'data/CIC-IDS2018'
PROCESSED_DIR = 'data/processed'
CHUNK_SIZE    = 10000

# ensure output directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

# columns we always drop
DROP_COLS = [
    'Flow ID',
    'Source IP', 'Destination IP',
    'Src IP',    'Dst IP',
    'Source Port','Destination Port',
    'Src Port',  'Dst Port',
    'Timestamp'
]

# labels we consider invalid (when lowercased)
INVALID_LABELS = {'nan', 'label'}


def clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    1) Drop IP/port/timestamp cols
    2) Require & clean 'Label'
    3) Filter out empty/invalid labels
    """
    chunk = chunk.drop(columns=DROP_COLS, errors='ignore')
    if 'Label' not in chunk.columns:
        return pd.DataFrame()
    chunk = chunk.dropna(subset=['Label'])
    chunk['Label'] = chunk['Label'].astype(str).str.strip()
    # filter out empty or header text
    lower = chunk['Label'].str.lower()
    valid = (chunk['Label'] != '') & (~lower.isin(INVALID_LABELS))
    return chunk.loc[valid]


def first_pass(zero_day=None):
    """
    If zero_day is not provided, scan every CSV to count labels
    and pick the least frequent (minus the spurious 'Label').
    Then fit a StandardScaler on *all* non‐zero‐day rows.
    """
    if zero_day is None:
        counts = {}
        for fname in sorted(os.listdir(RAW_DIR)):
            if not fname.endswith('.csv'):
                continue
            path = os.path.join(RAW_DIR, fname)
            for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):
                cl = clean_chunk(chunk)
                if cl.empty:
                    continue
                for lbl, cnt in cl['Label'].value_counts().items():
                    counts[lbl] = counts.get(lbl, 0) + cnt
        counts.pop('Label', None)
        zero_day = min(counts, key=counts.get)
        print(f"Identified zero-day attack: {zero_day!r}")

    scaler = StandardScaler()
    print("Fitting scaler on supervised training data…")
    zd_lower = zero_day.lower()

    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.endswith('.csv'):
            continue
        path = os.path.join(RAW_DIR, fname)
        for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):
            cl = clean_chunk(chunk)
            if cl.empty:
                continue
            # exclude zero-day flows
            mask = cl['Label'].str.lower() != zd_lower
            sub = cl.loc[mask]
            if sub.empty:
                continue
            X = sub.drop(columns=['Label']).apply(pd.to_numeric, errors='coerce')
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            X = X.dropna()
            if not X.empty:
                scaler.partial_fit(X)

    print("Scaler fitted.\n")
    return zero_day, scaler


def write_split(mode, zero_day, scaler):
    """
    Generate one of: train, test, unsup_train, unsup_test, zero_day
    and write it gzipped to data/processed.
    """
    fn_map = {
        'train':       'train_data.csv.gz',
        'test':        'test_data.csv.gz',
        'unsup_train': 'unsupervised_train_data.csv.gz',
        'unsup_test':  'unsupervised_test_data.csv.gz',
        'zero_day':    'zero_day_data.csv.gz'
    }
    out_path = os.path.join(PROCESSED_DIR, fn_map[mode])
    if os.path.exists(out_path):
        os.remove(out_path)

    print(f"Writing '{mode}' split to {fn_map[mode]}…")
    header = False
    zd_lower = zero_day.lower()

    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.endswith('.csv'):
            continue
        path = os.path.join(RAW_DIR, fname)
        for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):
            cl = clean_chunk(chunk)
            if cl.empty:
                continue

            orig = cl['Label'].astype(str).str.strip()
            lower = orig.str.lower()

            # prepare feature matrix
            X = cl.drop(columns=['Label']).apply(pd.to_numeric, errors='coerce')
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            X.fillna(0, inplace=True)
            Xs = scaler.transform(X)

            df = pd.DataFrame(Xs, columns=X.columns)
            df['Label_raw']   = orig.values
            df['Label_lower'] = lower.values  # for easy boolean ops

            # build the proper split
            if mode == 'train':
                # keep everything except zero-day
                mask = df['Label_lower'] != zd_lower
                df2 = df.loc[mask].copy()
                # map benign→0, others→1
                df2['Label'] = (df2['Label_lower'] != 'benign').astype(int)

            elif mode == 'test':
                df2 = df.copy()
                df2['Label'] = (df2['Label_lower'] != 'benign').astype(int)

            elif mode == 'unsup_train':
                mask = df['Label_lower'] == 'benign'
                df2 = df.loc[mask].drop(columns=['Label_raw','Label_lower'])

            elif mode == 'unsup_test':
                df2 = df.copy()
                df2['Label'] = (df2['Label_lower'] != 'benign').astype(int)

            elif mode == 'zero_day':
                mask = df['Label_lower'] == zd_lower
                df2 = df.loc[mask].copy()
                df2['Label'] = 1

            else:
                raise ValueError(f"Unknown mode: {mode}")

            # drop our helper cols
            df2 = df2.drop(columns=['Label_raw','Label_lower'], errors='ignore')

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
    """
    Generates preprocessed data splits for different experimental modes.
    Calls first_pass() to prepare the initial dataset and scaler, then writes 
    the requested split to disk based on the provided mode.

    Usage:
        python generate_split.py --mode [train|test|unsup_train|unsup_test|zero_day]
    """
    # Set up argument parser to select which data split to generate
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        choices=['train', 'test', 'unsup_train', 'unsup_test', 'zero_day'],
        required=True,
        help='Which split to generate'
    )
    args = parser.parse_args()

    # Run initial data preparation to obtain zero-day subset and fitted scaler
    zero_day, scaler = first_pass()

    # Write out the chosen split using the generated zero-day data and scaler
    write_split(args.mode, zero_day, scaler)


if __name__ == '__main__':
    main()

import os
import argparse
import pandas as pd
import numpy as np
from math import sqrt

# Validator for processed data splits**

def compute_stats(reader, mode):
    """
    Compute statistics and validate a split file.
    Only numeric feature columns are considered for NaN/Inf checks.
    Skip scaling checks for zero_day mode.
    """
    label_values = set()
    nan_count = inf_count = 0
    nrows = 0

    # For accumulation
    sum_dict = {}
    sumsq_dict = {}

    for chunk in reader:
        # Count rows
        n = len(chunk)
        nrows += n

        # Label collection if applicable
        if 'Label' in chunk.columns and mode in ('train','test','unsup_test','zero_day'):
            # drop missing labels
            labels = chunk['Label'].dropna()
            label_values.update(labels.unique().tolist())

        # Only numeric features for NaN/Inf and scaling stats
        feat_cols = chunk.select_dtypes(include=[np.number]).columns.tolist()
        if feat_cols:
            arr = chunk[feat_cols].to_numpy(dtype=float)
            nan_count += np.isnan(arr).sum()
            inf_count += np.isinf(arr).sum()

            # accumulate sums for mean/std
            for col in feat_cols:
                col_data = arr[:, feat_cols.index(col)]
                sum_dict[col] = sum_dict.get(col, 0.0) + np.nansum(col_data)
                sumsq_dict[col] = sumsq_dict.get(col, 0.0) + np.nansum(col_data**2)

    # If zero_day mode, skip mean/std (distribution will differ)
    if mode == 'zero_day':
        return {
            'nrows': nrows,
            'labels': sorted(label_values),
            'nan_count': int(nan_count),
            'inf_count': int(inf_count),
            'mean': {},
            'std': {}
        }

    # compute per-feature mean and std
    mean_dict = {col: sum_val / nrows for col, sum_val in sum_dict.items()}
    std_dict = {}
    for col in sum_dict:
        variance = sumsq_dict[col] / nrows - mean_dict[col]**2
        std_dict[col] = sqrt(variance) if variance > 0 else 0.0

    return {
        'nrows': nrows,
        'labels': sorted(label_values),
        'nan_count': int(nan_count),
        'inf_count': int(inf_count),
        'mean': mean_dict,
        'std': std_dict
    }

def main():
    """
    Validates the structure and integrity of preprocessed data splits used for training/testing.
    It performs checks for file existence, missing or infinite values, label consistency, and 
    standardization (mean/std deviation) for different dataset modes including supervised, 
    unsupervised, and zero-day settings.
    
    Usage:
        python validate.py --mode [train|test|unsup_train|unsup_test|zero_day]
    """
    # Set up argument parser for selecting which split to validate
    parser = argparse.ArgumentParser(description="Validate processed data splits.")
    parser.add_argument(
        '--mode', choices=['train', 'test', 'unsup_train', 'unsup_test', 'zero_day'],
        required=True, help='Which split to validate'
    )
    args = parser.parse_args()

    # Map each mode to its corresponding processed data file
    path_map = {
        'train':       'data/processed/train_data.csv.gz',
        'test':        'data/processed/test_data.csv.gz',
        'unsup_train': 'data/processed/unsupervised_train_data.csv.gz',
        'unsup_test':  'data/processed/unsupervised_test_data.csv.gz',
        'zero_day':    'data/processed/zero_day_data.csv.gz'
    }

    # Retrieve the correct file path based on the selected mode
    path = path_map[args.mode]

    # Check that the file exists before proceeding
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Please generate it first.")

    print(f"Validating '{args.mode}' split at {path}")

    # Read the CSV in chunks to avoid memory issues
    reader = pd.read_csv(path, chunksize=100000, compression='gzip', low_memory=False)

    # Compute validation statistics on the dataset
    stats = compute_stats(reader, args.mode)

    # Display general statistics
    print(f"\nRows: {stats['nrows']}")
    print(f"NaN count (features): {stats['nan_count']}")
    print(f"Inf count (features): {stats['inf_count']}")

    # For unsupervised training data, there should be no label column
    if args.mode == 'unsup_train':
        print("Expect no 'Label' column; feature-only dataset.")
    else:
        print(f"Label values found: {stats['labels']}")

    # For supervised and unsupervised test sets, check that labels are binary
    if args.mode in ('train', 'test', 'unsup_test'):
        if set(stats['labels']) <= {0, 1}:
            print("✔ Labels are binary {0,1}")
        else:
            print("✗ Labels not binary!")

    # For zero-day datasets, labels must be only '1'
    if args.mode == 'zero_day':
        if stats['labels'] == [1]:
            print("✔ Zero-day data labeled '1' only")
        else:
            print("✗ Zero-day labels incorrect!")

    # Check if dataset is properly standardized (mean ~0, std ~1), if applicable
    if args.mode != 'zero_day' and stats['nrows'] > 0 and stats['mean']:
        means = np.array(list(stats['mean'].values()))
        stds  = np.array(list(stats['std'].values()))

        # Calculate average absolute deviation from expected standardized values
        mean_dev = np.abs(means).mean()
        std_dev  = np.abs(stds - 1).mean()

        print(f"\nAverage absolute mean deviation from 0: {mean_dev:.4f}")
        print(f"Average absolute std deviation from 1: {std_dev:.4f}")

    print("\nValidation complete.")

if __name__ == '__main__':
    main()

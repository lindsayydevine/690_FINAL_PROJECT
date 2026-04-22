#!/usr/bin/env python3
"""
preprocess_data.py — Convert raw accelerometer data to BioPM-ready HDF5 files.

This script implements the full preprocessing pipeline modeled after the
mHealth preprocessing used in the original BioPM research.

=== WHAT YOU NEED TO CHANGE ===

Look for lines marked with "# STUDENT:" — those are the places where you
need to adapt for your own dataset.  Everything else should work as-is
for typical 3-axis accelerometer data.

=== PIPELINE OVERVIEW ===

  1. Load raw data (CSV, etc.) → 3-axis acceleration + labels
  2. Convert units to g (divide by 9.81 if in m/s²)
  3. Resample to 30 Hz
  4. Bandpass filter → body-acceleration signal
  5. Lowpass filter  → gravity signal
  6. Sliding window (e.g. 10 s with 5 s overlap)
  7. Zero-crossing movement element extraction per window
  8. Save as HDF5 per subject

=== OUTPUT FORMAT ===

Per subject, produces two HDF5 files in the output directory:
  Data_AccLabel_{subject_id}.h5
    - window_acc_raw:         (W, T, 3)    raw acceleration per window
    - window_acc_filt_gravity: (W, T, 6)   filtered acc + gravity concat
    - window_label:           (W,)         integer labels

  Data_MeLabel_{subject_id}.h5
    - window_acc_raw:   (W, T, 3)            raw acceleration
    - x_acc_filt:       (W, pad_size, 37+)   movement elements + metadata
    - x_gravity:        (W, T, 3)            gravity signal
    - window_label:     (W,)                 integer labels

The Data_MeLabel_*.h5 files are what extract_features.py expects.

=== USAGE ===
python scripts/preprocess_data.py --raw_data_dir ./data --output_dir ./preprocessed_data --window_sec 10 --slide_sec 5 --ori_fs 100
"""

import os
import sys
import argparse
import statistics
import numpy as np
import pandas as pd
import h5py
import re
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.preprocessing import (
    resample_to_target_fs, bandpass_filter, lowpass_filter,
    detect_zero_crossings, assign_zero_crossings,
)

# ===================================================================
# Configuration defaults (match the original BioPM mHealth pipeline)
# ===================================================================
DEFAULT_CONFIG = {
    'HighF1': 12,        # bandpass upper cutoff (Hz)
    'LowF1': 0.5,       # bandpass lower cutoff / lowpass cutoff (Hz)
    'Order1': 6,         # filter order
    'ori_FS': 100,        # STUDENT: original sample rate of your data
    'target_FS': 30,     # target sample rate after resampling
    'WS': 10,            # window size in seconds
    'SlideSize': 5,      # slide / hop size in seconds
    'normalize_size_target': 32,  # ME normalisation length
    'normalize_size_assign': 32,
}


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess raw data for BioPM")
    p.add_argument("--raw_data_dir", type=str, required=True,
                   help="Directory containing raw data files")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to save preprocessed HDF5 files")
    p.add_argument("--window_sec", type=int, default=10,
                   help="Window size in seconds (default: 10)")
    p.add_argument("--slide_sec", type=int, default=5,
                   help="Slide/overlap in seconds (default: 5)")
    p.add_argument("--ori_fs", type=int, default=100,
                   help="Original sample rate of your data (default: 100)")
    p.add_argument("--target_fs", type=int, default=30,
                   help="Target sample rate after resampling (default: 30)")
    return p.parse_args()


# ===================================================================
# STUDENT: Implement this function for your own dataset
# ===================================================================
def load_raw_data(file_path, config):
    """
    Load a single raw data file and return (acc_raw, labels, timestamps).

    STUDENT: Replace this function body with your own data-loading logic.

    The example below shows the mHealth format as reference.  Your data
    may be CSV, Parquet, NumPy, MAT, or any other format.

    Must return:
        acc_raw:    (N, 3) numpy array — 3-axis acceleration in g units
                    If your data is in m/s², divide by 9.80665
        labels:     (N,) numpy array — integer activity labels per sample
        time_array: (N,) numpy array — timestamps in seconds

    If your data does not have per-sample timestamps, create synthetic
    ones: time_array = np.arange(N) / original_sample_rate
    """
    df = pd.read_csv(file_path, sep=',', header=0, low_memory=False)
    acc_raw = df.iloc[:, [1, 2, 3]].values
    
    # Get the lookup table from config
    lookup = config.get('lookup', {})

    # Match the annotation column to the dictionary
    # Default to 0 if the string isn't in your dictionary
    labels = df.iloc[:, 4].apply(lambda x: lookup.get(x, 0)).values

    # Clean invalid values
    acc_df = pd.DataFrame(acc_raw, columns=['x', 'y', 'z'])
    acc_raw = acc_df.apply(pd.to_numeric, errors='coerce').ffill().bfill().values

    # Synthetic timestamps at original sample rate
    time_array = np.arange(len(acc_raw)) / config['ori_FS']

    return acc_raw, labels, time_array


# ===================================================================
# STUDENT: Implement this function for your label mapping
# ===================================================================
# def remap_labels(labels_raw, skip_labels=None):
#     """
#     Map raw integer labels to a contiguous 0..K-1 range and optionally
#     skip certain labels (e.g. 'null' class).

#     STUDENT: adapt this for your dataset's label scheme.

#     Args:
#         labels_raw:   array of raw integer labels
#         skip_labels:  set of label values to exclude

#     Returns:
#         labels_mapped: array of remapped labels (or original if no mapping)
#     """


#     # Build mapping from old labels to contiguous new labels
#     unique_old = sorted(set(int(l) for l in np.unique(labels_raw)) - skip_labels)
#     old_to_new = {old: new for new, old in enumerate(unique_old)}

#     # print(f"--- Activity Map Created ---")
#     # for old, new in old_to_new.items():
#     #     print(f" Original Code {old} -> Model Label {new}")

#     return old_to_new, skip_labels


def get_global_label_map(lookup_dict, skip_labels={0}):
    """
    Creates a single, universal mapping for all possible CPA codes 
    found in your annotation-label-dictionary.csv.
    """
    # Get all unique CPA codes from your lookup table
    all_cpa_codes = sorted(set(lookup_dict.values()) - skip_labels)
    
    # Create the universal mapping: CPA_CODE -> Contiguous_Index
    global_to_new = {cpa: idx for idx, cpa in enumerate(all_cpa_codes)}
    
    return global_to_new

# ===================================================================
# Main preprocessing loop
# ===================================================================
def preprocess_one_subject(file_path, subject_id, config, output_dir):
    """Run full preprocessing for one subject file."""
    acc_raw, labels, time_array = load_raw_data(file_path, config)

    # Resample to target FS
    acc_resampled, time_resampled, labels_resampled = resample_to_target_fs(
        time_array, acc_raw, labels, config['target_FS'])

    # Bandpass filter (body acceleration)
    acc_filt = bandpass_filter(
        acc_resampled, config['LowF1'], config['HighF1'], config['target_FS'],
        order=config['Order1'])

    # Lowpass filter (gravity component)
    acc_gravity = lowpass_filter(
        acc_resampled, config['LowF1'], config['target_FS'],
        order=config['Order1'])

    # Window parameters
    ws = int(config['WS'] * config['target_FS'])  # window in samples
    step = int(config['SlideSize'] * config['target_FS'])

    label_map = config['global_label_map']
    skip_labels = {0}

    # Accumulate windows
    win_acc_raw, win_acc_filt_grav, win_labels = [], [], []
    win_x_acc_filt, win_x_gravity = [], []

    unique_labels = np.unique(labels_resampled)
    print(f"--- Subject {subject_id}: Found codes {unique_labels} ---")

    start = 0
    while start + ws < acc_filt.shape[0]:
        window_labels = labels_resampled[start:start + ws]
        try:
            mode_label = statistics.mode(window_labels.astype(int))
        except Exception:
            start += step
            continue

        # mode_label here is the CPA code (e.g., 7030)
        if mode_label in skip_labels:
            start += step
            continue

        w_raw = acc_resampled[start:start + ws, :]
        w_filt = acc_filt[start:start + ws, :]
        w_grav = acc_gravity[start:start + ws, :]
        w_time = time_resampled[start:start + ws]

        if np.mean(np.abs(w_filt)) < 0.01: 
            start += step
            continue

        # Zero-crossing movement-element extraction
        try:
            (_, _, me_list, me_norm, me_info, _, _,
             pos_info, zc_list, zc_time_list) = detect_zero_crossings(
                w_filt, w_time, config)

            (_, _, _, grav_norm, grav_info,
             _, _, _) = assign_zero_crossings(
                w_grav, w_time, zc_list, zc_time_list, config)
        except Exception as e:
            start += step
            continue

        if len(me_list) == 0:
            start += step
            continue

        # Build x_acc_filt: [normalised_ME(32) | pos(1) | axis,len,min,max,dirct(5)]
        x_acc = np.concatenate([
            me_norm,
            pos_info.reshape(-1, 1),
            me_info[['axis', 'len', 'min', 'max', 'dirct']].values,
        ], axis=1)

        # Pad to pad_size
        if x_acc.shape[0] < config['pad_size']:
            pad = np.full((config['pad_size'] - x_acc.shape[0], x_acc.shape[1]),
                          np.nan)
            x_acc = np.vstack([x_acc, pad])
        else:
            x_acc = x_acc[:config['pad_size']]

        x_gravity = w_grav.astype(np.float32)

        mapped_label = label_map[mode_label] # Ex: Now 7030 is ALWAYS 5 (across all participants)

        win_acc_raw.append(w_raw.astype(np.float32))
        win_acc_filt_grav.append(
            np.concatenate([w_filt, w_grav], axis=1).astype(np.float32))
        win_labels.append(float(mapped_label))
        win_x_acc_filt.append(x_acc.astype(np.float32))
        win_x_gravity.append(x_gravity.astype(np.float32))

        start += step

    if len(win_labels) == 0:
        print(f"  WARNING: no valid windows for subject {subject_id}")
        return

    # Stack arrays
    win_acc_raw = np.array(win_acc_raw, dtype=np.float32)
    win_acc_filt_grav = np.array(win_acc_filt_grav, dtype=np.float32)
    win_x_acc_filt = np.array(win_x_acc_filt, dtype=np.float32)
    win_x_gravity = np.array(win_x_gravity, dtype=np.float32)
    win_labels = np.array(win_labels, dtype=np.float32)

    # Save HDF5 files
    os.makedirs(output_dir, exist_ok=True)

    h5_acc = os.path.join(output_dir, f"Data_AccLabel_{subject_id}.h5")
    with h5py.File(h5_acc, "w") as f:
        f.create_dataset("window_acc_raw", data=win_acc_raw)
        f.create_dataset("window_acc_filt_gravity", data=win_acc_filt_grav)
        f.create_dataset("window_label", data=win_labels)

    h5_me = os.path.join(output_dir, f"Data_MeLabel_{subject_id}.h5")
    with h5py.File(h5_me, "w") as f:
        f.create_dataset("window_acc_raw", data=win_acc_raw)
        f.create_dataset("x_acc_filt", data=win_x_acc_filt)
        f.create_dataset("x_gravity", data=win_x_gravity)
        f.create_dataset("window_label", data=win_labels)

    print(f"  Saved {len(win_labels)} windows for subject {subject_id}")

def process_file_wrapper(fname, config, output_dir, raw_dir):
    """A helper function to handle the processing of a single file."""
    subject_id = fname.split(".")[0]
    filepath = os.path.join(raw_dir, fname)
    
    # We check for the file here too, to be extra safe with parallel runs
    out_check = os.path.join(output_dir, f"Data_MeLabel_{subject_id}.h5")
    if os.path.exists(out_check):
        return f"Skipped {subject_id}"

    try:
        preprocess_one_subject(filepath, subject_id, config, output_dir)
        return f"Finished {subject_id}"
    except Exception as e:
        return f"Error {subject_id}: {e}"

def main():
    args = parse_args()
    config = DEFAULT_CONFIG.copy()
    config['WS'] = args.window_sec
    config['SlideSize'] = args.slide_sec
    config['ori_FS'] = args.ori_fs
    config['target_FS'] = args.target_fs
    config['pad_size'] = int(config['WS'] * 192 / 10)

    print("=" * 60)
    print("BioPM Data Preprocessing")
    print("=" * 60)
    print(f"  Raw data dir:  {args.raw_data_dir}")
    print(f"  Output dir:    {args.output_dir}")
    print(f"  Window:        {config['WS']}s, slide {config['SlideSize']}s")
    print(f"  Resample:      {config['ori_FS']} → {config['target_FS']} Hz")
    print(f"  Pad size:      {config['pad_size']} patches/window")
    print()

    # Load the annotation dictionary
    dict_df = pd.read_csv('./data/annotation-label-dictionary.csv', usecols=['annotation'])

    # 1. Generate the CPA lookup from your CSV
    annotation_lookup = {}
    for anno in dict_df['annotation'].unique():
        match = re.search(r'(\d{4,5})', str(anno))
        if match:
            annotation_lookup[anno] = int(match.group(1))
    
    # 2. GENERATE GLOBAL MAP HERE
    # This ensures 7030 is ALWAYS index X, regardless of the subject
    global_label_map = get_global_label_map(annotation_lookup)
    config['global_label_map'] = global_label_map
    config['lookup'] = annotation_lookup # Raw string -> CPA code

    raw_dir = args.raw_data_dir
    files = sorted([
        f for f in os.listdir(raw_dir) # do not get metadata csv or annotation-label-dictionary csv
        if f.startswith('P') and f.endswith('.csv') # only care about P001 to P151 csvs
    ])

    if not files:
        print(f"ERROR: No data files found in {raw_dir}")
        sys.exit(1)
    
    print(f"Parallel processing started on {os.cpu_count()} cores...")

    worker = partial(process_file_wrapper, config=config, 
                    output_dir=args.output_dir, raw_dir=args.raw_data_dir)
    
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(worker, files, chunksize=2), 
                           total=len(files), 
                           desc="Overall Progress"))

    print("\nPreprocessing complete!")
    print(f"Output saved to: {args.output_dir}")

if __name__ == "__main__":
    main()

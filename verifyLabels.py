import pandas as pd
import numpy as np
import h5py
import re
import statistics

# 1. Reconstruct the Global Map (exactly as your preprocess script did)
dict_df = pd.read_csv('./data/annotation-label-dictionary.csv', usecols=['annotation'])
annotation_lookup = {}
for anno in dict_df['annotation'].unique():
    match = re.search(r'(\d{4,5})', str(anno))
    if match:
        annotation_lookup[anno] = int(match.group(1))

all_cpa_codes = sorted(set(annotation_lookup.values()) - {0})
global_label_map = {cpa: idx for idx, cpa in enumerate(all_cpa_codes)}

# 2. Load Raw CSV for P034
df_raw = pd.read_csv('./data/P034.csv')
raw_annotations = df_raw.iloc[:, 4] # Assuming column 4 is the annotation string

# 3. Load HDF5 Labels for P034
with h5py.File('./preprocessed_data/Data_MeLabel_P034.h5', 'r') as f:
    h5_labels = f['window_label'][:]

# 4. Verification Logic
# Your script used 10s windows (300 samples at 30Hz target, or 1000 samples at 100Hz original)
# Let's check the first valid window from the CSV
window_size_raw = 1000 # 10 seconds * 100 Hz
first_window_raw = raw_annotations[0:window_size_raw]

# Get the CPA code for the mode of the first window
mode_anno_string = statistics.mode(first_window_raw)
cpa_code = annotation_lookup.get(mode_anno_string)
expected_index = global_label_map.get(cpa_code)

print(f"--- Verification for P034 ---")
print(f"Raw String Mode (Window 1): {mode_anno_string}")
print(f"Mapped CPA Code:           {cpa_code}")
print(f"Expected Model Index:      {expected_index}")
print(f"Actual HDF5 Index (Win 1): {h5_labels[0]}")

if expected_index == h5_labels[0]:
    print("\nSUCCESS: The labels align perfectly!")
else:
    print("\nWARNING: Mismatch detected. Check window slicing or sample rate alignment.")
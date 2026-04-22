import pandas as pd
import re

# Load the same file you used in your script
dict_df = pd.read_csv('./data/annotation-label-dictionary.csv', usecols=['annotation'])

# Recreate the CPA lookup
annotation_lookup = {}
for anno in dict_df['annotation'].unique():
    match = re.search(r'(\d{4,5})', str(anno))
    if match:
        annotation_lookup[anno] = int(match.group(1))

# Recreate the global map
all_cpa_codes = sorted(set(annotation_lookup.values()) - {0})
global_label_map = {cpa: idx for idx, cpa in enumerate(all_cpa_codes)}

print("--- Universal Dictionary (CPA Code : Model Index) ---")
for cpa, idx in global_label_map.items():
    print(f"CPA {cpa} -> Index {idx}")
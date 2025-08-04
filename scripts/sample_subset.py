import gcsfs
import random
import json
import sys

# Configs
SPLIT = str(sys.argv[1])  # e.g., "training" or "validation"
SUBSET_SIZE_THOUSANDS = int(sys.argv[2])  # e.g., 50 -> 50k
SUBSET_SIZE = SUBSET_SIZE_THOUSANDS * 1000
GCS_PREFIX = f"waymo_camera_data_01082025/waymo_processed_samples/{SPLIT}"
SEED = 42 
OUTPUT_JSON = f"./data/{SPLIT}_subset_{SUBSET_SIZE_THOUSANDS}k.json"

# Init GCS
fs = gcsfs.GCSFileSystem(token='google_default')

print(f"Listing .pt files in {SPLIT} bucket...")
all_files = fs.ls(GCS_PREFIX)
pt_files = [f.split("/")[-1] for f in all_files if f.endswith(".pt")]

print(f"Found {len(pt_files)} .pt files. Sampling {SUBSET_SIZE} files...")
random.seed(SEED)
subset = random.sample(pt_files, SUBSET_SIZE)

# Save list for use in training/validation
with open(OUTPUT_JSON, 'w') as f:
    json.dump(subset, f)

print(f"Saved sampled file list to {OUTPUT_JSON}")

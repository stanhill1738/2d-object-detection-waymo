import os
import random
import argparse
import gcsfs
import shutil
from tqdm import tqdm

def download_subset(split, num_samples, local_dir, seed=42):
    # GCS prefix
    gcs_prefix = f"waymo_camera_data_01082025/waymo_processed_samples/{split}"

    # Init GCS filesystem
    fs = gcsfs.GCSFileSystem(token='google_default')

    print(f"Listing files from: gs://{gcs_prefix}")
    all_files = [f for f in fs.ls(gcs_prefix) if f.endswith(".pt")]
    print(f"Total files found: {len(all_files)}")

    if num_samples > len(all_files):
        raise ValueError("Requested more samples than available in GCS.")

    random.seed(seed)
    selected_files = random.sample(all_files, num_samples)

    # Create local output dir
    os.makedirs(local_dir, exist_ok=True)

    print(f"Downloading {num_samples} files to: {local_dir}")
    for gcs_file in tqdm(selected_files):
        filename = os.path.basename(gcs_file)
        local_path = os.path.join(local_dir, filename)

        with fs.open(gcs_file, 'rb') as src, open(local_path, 'wb') as dst:
            shutil.copyfileobj(src, dst)

    print("✅ Download complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Waymo subset from GCS")
    parser.add_argument("--split", type=str, required=True, help="Split: training, validation, test")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of .pt files to download")
    parser.add_argument("--output_dir", type=str, default="/tmp/waymo_data", help="Local output directory")
    args = parser.parse_args()

    download_subset(args.split, args.num_samples, args.output_dir)

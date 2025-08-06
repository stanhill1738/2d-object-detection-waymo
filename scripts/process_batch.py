import gcsfs
import pyarrow.parquet as pq
import pandas as pd
import torch
import os
import io
import numpy as np
from PIL import Image
import sys
import logging
import gc
import random

# ========== CONFIG ==========
SOURCE_BUCKET = "waymo_open_dataset_v_2_0_1"
SPLIT = "training"
NUM_BATCHES = 20  # total number of parallel jobs
OUTPUT_ROOT = "/tmp/waymo_data"
TARGET_PT_FILES = 18000
TARGET_PT_FILES_PER_BATCH = TARGET_PT_FILES // NUM_BATCHES
# ============================

# Accept batch number
BATCH_ID = int(sys.argv[1])

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Batch %(batch)d] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.LoggerAdapter(logging.getLogger(), {"batch": BATCH_ID})

# GCS Filesystem (read-only)
fs = gcsfs.GCSFileSystem(token="google_default")

# Shared counter
global_pt_counter = 0


def list_parquet_files(prefix):
    files = fs.ls(prefix)
    return sorted([os.path.basename(f) for f in files if f.endswith(".parquet")])


def process_file(filename, max_to_save):
    global global_pt_counter

    if global_pt_counter >= max_to_save:
        return 0  # Already hit the cap

    GCS_PREFIX = f"{SOURCE_BUCKET}/{SPLIT}"
    image_path = f"gs://{GCS_PREFIX}/camera_image/{filename}"
    box_path = f"gs://{GCS_PREFIX}/camera_box/{filename}"
    stats_path = f"gs://{GCS_PREFIX}/stats/{filename}"
    output_dir = os.path.join(OUTPUT_ROOT, SPLIT)
    os.makedirs(output_dir, exist_ok=True)

    if not fs.exists(box_path[5:]) or not fs.exists(stats_path[5:]):
        logger.warning(f"Missing box or stats for {filename}, skipping.")
        return 0

    try:
        with fs.open(image_path, 'rb') as f_img:
            df_img = pq.read_table(f_img).to_pandas()
        with fs.open(box_path, 'rb') as f_box:
            df_box = pq.read_table(f_box).to_pandas()
        with fs.open(stats_path, 'rb') as f_stats:
            df_stats = pq.read_table(f_stats).to_pandas()
    except Exception as e:
        logger.error(f"Failed to read {filename}: {e}")
        return 0

    df_stats = df_stats[['key.frame_timestamp_micros', '[StatsComponent].location',
                         '[StatsComponent].time_of_day', '[StatsComponent].weather']]
    pairs = df_img[['key.frame_timestamp_micros', 'key.camera_name']].drop_duplicates()

    processed = 0

    for _, row in pairs.iterrows():
        if global_pt_counter >= max_to_save:
            break  # Exit inner loop too

        timestamp = row['key.frame_timestamp_micros']
        camera_id = row['key.camera_name']
        output_path = os.path.join(output_dir, f"frame_{timestamp}_{camera_id}.pt")

        if os.path.exists(output_path):
            continue

        try:
            img_row = df_img[
                (df_img['key.frame_timestamp_micros'] == timestamp) &
                (df_img['key.camera_name'] == camera_id)
            ].iloc[0]

            image = Image.open(io.BytesIO(img_row['[CameraImageComponent].image'])).convert("RGB")
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

            boxes_df = df_box[
                (df_box['key.frame_timestamp_micros'] == timestamp) &
                (df_box['key.camera_name'] == camera_id)
            ]

            box_list, label_list = [], []
            for _, box in boxes_df.iterrows():
                cx = box['[CameraBoxComponent].box.center.x']
                cy = box['[CameraBoxComponent].box.center.y']
                w = box['[CameraBoxComponent].box.size.x']
                h = box['[CameraBoxComponent].box.size.y']
                x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
                box_list.append([x1, y1, x2, y2])
                label_list.append(box['[CameraBoxComponent].type'])

            stats_row = df_stats[df_stats['key.frame_timestamp_micros'] == timestamp]
            if stats_row.empty:
                continue

            meta = {
                "timestamp": int(timestamp),
                "camera_name": int(camera_id),
                "location": stats_row['[StatsComponent].location'].values[0],
                "time_of_day": stats_row['[StatsComponent].time_of_day'].values[0],
                "weather": stats_row['[StatsComponent].weather'].values[0],
                "split": SPLIT,
                "source_file": filename
            }

            sample = {
                "image": image_tensor,
                "boxes": torch.tensor(box_list, dtype=torch.float32),
                "labels": torch.tensor(label_list, dtype=torch.int64),
                "meta": meta
            }

            torch.save(sample, output_path)
            processed += 1
            global_pt_counter += 1

            # Free memory
            del image_tensor, sample, image, img_row, boxes_df, box_list, label_list, stats_row
            gc.collect()

        except Exception as e:
            logger.error(f"Error processing frame {timestamp} cam {camera_id}: {e}")

    del df_img, df_box, df_stats
    gc.collect()

    logger.info(f"Finished {filename}, saved {processed} samples.")
    return processed


if __name__ == "__main__":
    all_files = list_parquet_files(f"{SOURCE_BUCKET}/{SPLIT}/camera_image")

    random.seed(42)
    random.shuffle(all_files)

    # Split by batch
    subset_files = [f for i, f in enumerate(all_files) if i % NUM_BATCHES == BATCH_ID]

    logger.info(f"Batch {BATCH_ID} processing up to {TARGET_PT_FILES_PER_BATCH} total .pt files.")

    for fname in subset_files:
        if global_pt_counter >= TARGET_PT_FILES_PER_BATCH:
            break
        process_file(fname, max_to_save=TARGET_PT_FILES_PER_BATCH)

    logger.info(f"Batch {BATCH_ID} complete. {global_pt_counter} total .pt files saved.")

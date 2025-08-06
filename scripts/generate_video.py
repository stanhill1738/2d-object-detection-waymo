import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gcsfs
import pyarrow.parquet as pq
import pandas as pd
import torch
import io
import numpy as np
from PIL import Image
import logging
import gc
import shutil
import cv2
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from models.model import get_model
from data.waymo_unlabeled_dataset import WaymoUnlabeledDataset
from utils.label_map_utils import load_label_map

# ========== CONFIG ==========
SOURCE_BUCKET = "waymo_open_dataset_v_2_0_1"
SPLIT = "training"
OUTPUT_DIR = "/tmp/video_generation"
VIDEO_NAME = "predictions.mp4"
MODEL_PATH = "../best_model.pt" # Change to actual model path in .pt format
LABEL_MAP_PATH = "label_map.py"
CONFIDENCE_THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ============================

# ===== Step 1: Setup =====
if len(sys.argv) != 2:
    print("Usage: python script.py <parquet_filename>")
    sys.exit(1)

FILENAME = sys.argv[1]

# Clean output dir
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("video_generation")

# GCS Filesystem
fs = gcsfs.GCSFileSystem(token="google_default")

# ===== Step 2: Download and Convert Parquet to .pt =====
def process_file(filename):
    GCS_PREFIX = f"{SOURCE_BUCKET}/{SPLIT}"
    image_path = f"gs://{GCS_PREFIX}/camera_image/{filename}"
    box_path = f"gs://{GCS_PREFIX}/camera_box/{filename}"

    if not fs.exists(box_path[5:]):
        logger.warning(f"Missing box file for {filename}, skipping.")
        return

    try:
        with fs.open(image_path, 'rb') as f_img:
            df_img = pq.read_table(f_img).to_pandas()
        with fs.open(box_path, 'rb') as f_box:
            df_box = pq.read_table(f_box).to_pandas()
    except Exception as e:
        logger.error(f"Failed to read {filename}: {e}")
        return

    pairs = df_img[['key.frame_timestamp_micros', 'key.camera_name']].drop_duplicates()
    saved = 0

    for _, row in pairs.iterrows():
        timestamp = row['key.frame_timestamp_micros']
        camera_id = row['key.camera_name']
        output_path = os.path.join(OUTPUT_DIR, f"frame_{timestamp}_{camera_id}.pt")

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

            sample = {
                "image": image_tensor,
                "boxes": torch.tensor(box_list, dtype=torch.float32),
                "labels": torch.tensor(label_list, dtype=torch.int64)
            }

            torch.save(sample, output_path)
            saved += 1

            # Clean up
            del image_tensor, sample, image, img_row, boxes_df, box_list, label_list
            gc.collect()

        except Exception as e:
            logger.error(f"Error processing frame {timestamp} cam {camera_id}: {e}")

    del df_img, df_box
    gc.collect()
    logger.info(f"Saved {saved} .pt files to {OUTPUT_DIR}")

print(f"Processing parquet: {FILENAME}")
process_file(FILENAME)

# ===== Step 3: Inference and Video Generation =====
# Load model
model = get_model(num_classes=4)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state'])
model.to(DEVICE)
model.eval()

# Load label map
label_map = load_label_map(LABEL_MAP_PATH)
id_to_name = {v: k for k, v in label_map.items()}

# Load data
file_list = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".pt")])
dataset = WaymoUnlabeledDataset(OUTPUT_DIR, file_list)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Font
font = cv2.FONT_HERSHEY_SIMPLEX

def draw_predictions(img_tensor, boxes, labels, scores):
    img = F.to_pil_image(img_tensor.cpu().squeeze(0))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    for box, label, score in zip(boxes, labels, scores):
        if score < CONFIDENCE_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, box)
        class_name = id_to_name.get(label.item(), f"Class {label.item()}")
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{class_name} {score:.2f}", (x1, y1 - 10),
                    font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img

# Save frames with predictions
print("🔍 Running inference on frames...")
frame_paths = []
for idx, (img, fname) in enumerate(loader):
    img = img.to(DEVICE)
    with torch.no_grad():
        outputs = model(img)

    output = outputs[0]
    boxes = output["boxes"].cpu()
    labels = output["labels"].cpu()
    scores = output["scores"].cpu()

    frame = draw_predictions(img[0], boxes, labels, scores)
    frame_path = os.path.join(OUTPUT_DIR, f"pred_{idx:04d}.jpg")
    cv2.imwrite(frame_path, frame)
    frame_paths.append(frame_path)

# Create video
print("🎞️ Generating video...")
if frame_paths:
    first_frame = cv2.imread(frame_paths[0])
    height, width, _ = first_frame.shape
    video_writer = cv2.VideoWriter(VIDEO_NAME, cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    video_writer.release()
    print(f"✅ Video saved: {VIDEO_NAME}")
else:
    print("❌ No frames to generate video.")

# Clean up
shutil.rmtree(OUTPUT_DIR)
print("🧹 Cleaned up temporary files.")

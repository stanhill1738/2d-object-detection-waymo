import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
import cv2
import os
import matplotlib.pyplot as plt
from models.model import get_model
from data.waymo_unlabeled_dataset import WaymoUnlabeledDataset
from utils.label_map_utils import load_label_map
import numpy as np

# ---------- Config ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "../model_pt/best_model.pt"  # Path to trained model (too big for repo)
INPUT_DIR = "/tmp/video"  # Download videos to here, making sure they are from same camera.
OUTPUT_DIR = "video_frames"
VIDEO_NAME = "predictions.mp4"
LABEL_MAP_PATH = "label_map.py"
CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = (1280, 720)  # Resize if needed
# ----------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
file_list = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".pt")])
dataset = WaymoUnlabeledDataset(INPUT_DIR, file_list)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

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

# Generate and save frames
print("🔍 Running inference on frames...")
for idx, (img, fname) in enumerate(loader):
    img = img.to(DEVICE)
    with torch.no_grad():
        outputs = model(img)

    output = outputs[0]
    boxes = output["boxes"].cpu()
    labels = output["labels"].cpu()
    scores = output["scores"].cpu()

    frame = draw_predictions(img[0], boxes, labels, scores)
    frame_path = os.path.join(OUTPUT_DIR, f"frame_{idx:04d}.jpg")
    cv2.imwrite(frame_path, frame)

# Generate video
print("🎞️ Generating video...")
frame_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".jpg")])
first_frame = cv2.imread(os.path.join(OUTPUT_DIR, frame_files[0]))
height, width, _ = first_frame.shape
video_writer = cv2.VideoWriter(VIDEO_NAME, cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))

for frame_file in frame_files:
    frame = cv2.imread(os.path.join(OUTPUT_DIR, frame_file))
    video_writer.write(frame)

video_writer.release()
print(f"✅ Video saved: {VIDEO_NAME}")

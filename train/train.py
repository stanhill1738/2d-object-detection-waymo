import torch
from torch.utils.data import DataLoader
from data.waymo_dataset import WaymoDataset
from utils.gcs_utils import save_checkpoint, list_gcs_files
from models.model import get_model
import torchvision.transforms as T
import time
from utils.label_map_utils import load_label_map
import json

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()

    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

    return total_loss / len(dataloader)

def run_training(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=4)  # Assuming 3 foreground classes + background
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"]
    )

    # Load file list from subset JSON
    with open(config["train_subset_json"]) as f:
        train_files = json.load(f)
    with open(config["val_subset_json"]) as f:
        val_files = json.load(f)

    # Load label map
    label_map = load_label_map(config["label_map_path"])

    train_dataset = WaymoDataset(config["train_prefix"], train_files, label_map=label_map)
    val_dataset = WaymoDataset(config["val_prefix"], val_files, label_map=label_map)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=8,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=4,
        pin_memory=True
    )

    best_val_loss = float("inf")

    for epoch in range(config["epochs"]):
        print(f"Starting Epoch: {epoch}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, config["checkpoint_path"])

        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    return best_val_loss
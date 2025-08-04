import torch
from torch.utils.data import DataLoader
from data.waymo_dataset import WaymoDataset
from utils.gcs_utils import save_checkpoint
from models.model import get_model
from utils.label_map_utils import load_label_map
import os
from datetime import datetime

def list_local_files(directory):
    """List all .pt files in a local directory."""
    return sorted([f for f in os.listdir(directory) if f.endswith('.pt')])

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, (images, targets) in enumerate(dataloader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()

        now = datetime.now()
        print(f"Time: {now.strftime('%H:%M:%S')} - [Train] Batch {batch_idx + 1}/{len(dataloader)} - Loss: {losses.item():.4f}")

    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_output = model(images, targets)

            if isinstance(loss_output, dict):
                losses = sum(loss for loss in loss_output.values())
            elif isinstance(loss_output, list):
                losses = sum(loss_output)
            else:  # assume it's a single tensor
                losses = loss_output

            total_loss += losses.item()

    return total_loss / len(dataloader)


def run_training(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=4)  
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"]
    )

    # List local .pt files
    train_files = list_local_files(config["train_prefix"])
    val_files = list_local_files(config["val_prefix"])

    # ⏱ Limit file count for faster training runs
    num_train_files = config["num_train_files"]
    num_val_files = config["num_val_files"]
    train_files = train_files[:num_train_files]
    val_files = val_files[:num_val_files]

    # Load label map
    label_map = load_label_map(config["label_map_path"])

    # Load datasets
    train_dataset = WaymoDataset(config["train_prefix"], train_files, label_map=label_map)
    val_dataset = WaymoDataset(config["val_prefix"], val_files, label_map=label_map)

    # Loaders
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
        print(f"Starting Epoch: {epoch + 1}/{config['epochs']}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, config["checkpoint_path"])

        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    return best_val_loss

import torch
from torch.utils.data import DataLoader
from data.waymo_dataset import WaymoDataset
from utils.gcs_utils import save_checkpoint, list_gcs_files
from models.model import get_model
import torchvision.transforms as T
import time

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

    model = get_model(num_classes=11)  # Waymo has 10 classes + background
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"]
    )

    # Load data
    train_files = list_gcs_files(config["train_prefix"])
    val_files = list_gcs_files(config["val_prefix"])

    train_dataset = WaymoDataset(config["train_prefix"], train_files)
    val_dataset = WaymoDataset(config["val_prefix"], val_files)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], collate_fn=lambda x: tuple(zip(*x)))

    best_val_loss = float("inf")

    for epoch in range(config["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, config["checkpoint_path"])

        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    return best_val_loss
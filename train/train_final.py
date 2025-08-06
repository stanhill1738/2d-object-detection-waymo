import torch
from torch.utils.data import DataLoader
from data.waymo_dataset import WaymoDataset
from utils.gcs_utils import save_checkpoint
from models.model import get_model
from utils.label_map_utils import load_label_map
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import os
from datetime import datetime
import matplotlib.pyplot as plt


def list_local_files(directory):
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
    metric = MeanAveragePrecision(iou_thresholds=[0.5])
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]

            outputs = model(images)
            outputs_cpu = [{k: v.cpu() for k, v in o.items()} for o in outputs]

            metric.update(outputs_cpu, targets_cpu)

    score = metric.compute()
    map_50 = score['map_50'].item()
    print(f"[Validation] mAP@0.5: {map_50:.4f}")
    return map_50


def run_training(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=4)  # 3 classes + background
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"]
    )

    train_files = list_local_files(config["train_prefix"])
    val_files = list_local_files(config["val_prefix"])
    train_files = train_files[:config["num_train_files"]]
    val_files = val_files[:config["num_val_files"]]

    label_map = load_label_map(config["label_map_path"])
    train_dataset = WaymoDataset(config["train_prefix"], train_files, label_map=label_map)
    val_dataset = WaymoDataset(config["val_prefix"], val_files, label_map=label_map)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                              collate_fn=lambda x: tuple(zip(*x)), num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False,
                            collate_fn=lambda x: tuple(zip(*x)), num_workers=4, pin_memory=True)

    best_map = 0.0
    best_epoch = 0
    patience = config.get("early_stopping_patience", 5)
    epochs_no_improve = 0

    for epoch in range(config["epochs"]):
        print(f"Starting Epoch: {epoch + 1}/{config['epochs']}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_map = validate(model, val_loader, device)

        if val_map > best_map:
            best_map = val_map
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, config["checkpoint_path"])
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val mAP@0.5 = {val_map:.4f}")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    print(f"Best model at epoch {best_epoch + 1} with mAP@0.5 = {best_map:.4f}")
    return best_map


def test_model(checkpoint_path, test_prefix, label_map_path, num_test_files=1000, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading best model from checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = get_model(num_classes=4)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    test_files = list_local_files(test_prefix)
    test_files = test_files[:num_test_files]

    label_map = load_label_map(label_map_path)
    test_dataset = WaymoDataset(test_prefix, test_files, label_map=label_map)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=lambda x: tuple(zip(*x)), num_workers=4, pin_memory=True)

    print("Runing Testing:")
    metric = MeanAveragePrecision(class_metrics=True)
    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]
            outputs = model(images)
            outputs_cpu = [{k: v.cpu() for k, v in o.items()} for o in outputs]
            metric.update(outputs_cpu, targets_cpu)

    results = metric.compute()
    print("Final Test Set Evaluation:")
    for k, v in results.items():
        if isinstance(v, torch.Tensor):
            if v.ndim == 0:
                print(f"{k}: {v.item():.4f}")
            else:
                print(f"{k}: {v}")
        else:
            print(f"{k}: {v}")


    # Plot per-class AP
    if "classes" in results and "map_per_class" in results:
        class_ids = list(range(len(results["map_per_class"])))
        ap_values = results["map_per_class"].tolist()

        plt.figure(figsize=(8, 5))
        plt.bar(class_ids, ap_values)
        plt.xlabel("Class ID")
        plt.ylabel("AP")
        plt.title("Per-Class Average Precision (AP)")
        plt.savefig("per_class_ap.png")
        print("Saved per-class AP plot to per_class_ap.png")

    return results

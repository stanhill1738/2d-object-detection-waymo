import yaml
from train.train_final import run_training, test_model

with open("final_config.yaml") as f:
    base = yaml.safe_load(f)

config = {
    "lr": base["optimizer"]["lr"],
    "momentum": base["optimizer"]["momentum"],
    "weight_decay": base["optimizer"]["weight_decay"],
    "batch_size": base["training"]["batch_size"],
    "epochs": base["training"]["num_epochs"],
    "checkpoint_path": f"{base['training']['checkpoint_dir']}/best_model.pt",
    "label_map_path": base["data"]["label_map"],
    "train_prefix": base["data"]["train_prefix"],
    "val_prefix": base["data"]["val_prefix"],
    "test_prefix": base["data"]["test_prefix"],
    "num_train_files": base["files"]["num_train_files"],
    "num_val_files": base["files"]["num_val_files"],
    "num_test_files": base["files"]["num_test_files"],
    "early_stopping_patience": base["training"]["early_stopping_patience"],
}

# Train
run_training(config)

# Test
test_model(
    checkpoint_path=config["checkpoint_path"],
    test_prefix=config["test_prefix"],
    label_map_path=config["label_map_path"],
    num_test_files=config["num_test_files"],
    batch_size=config["batch_size"]
)

import optuna
from train.train import run_training

def objective(trial):
    config = {
        "lr": trial.suggest_loguniform("lr", 1e-5, 1e-2),
        "momentum": trial.suggest_float("momentum", 0.7, 0.99),
        "weight_decay": trial.suggest_loguniform("weight_decay", 1e-6, 1e-3),
        "batch_size": trial.suggest_categorical("batch_size", [2, 4, 8]),
        "epochs": 3,  # keep small for tuning
        "train_prefix": "waymo_camera_data_01082025_2/waymo_processed_samples/training",
        "val_prefix": "waymo_camera_data_01082025_2/waymo_processed_samples/validation",
        "checkpoint_path": f"waymo_camera_data_01082025_2/fasterrcnn_checkpoints/best_model.pt"
    }

    return run_training(config)

from train.train import run_training

def objective(trial, base_config):
    config = {
        "lr": trial.suggest_loguniform("lr", 1e-5, 1e-2),
        "momentum": trial.suggest_float("momentum", 0.7, 0.99),
        "weight_decay": trial.suggest_loguniform("weight_decay", 1e-6, 1e-3),
        "batch_size": trial.suggest_categorical("batch_size", [2, 4, 8]),
        "epochs": base_config["training"]["num_epochs"],
        "train_prefix": base_config["data"]["train_prefix"],
        "val_prefix": base_config["data"]["val_prefix"],
        "checkpoint_path": base_config["training"]["checkpoint_dir"] + "/best_model.pt",
        "label_map_path": base_config["data"]["label_map"],
        "num_train_files": base_config["training"]["num_train_files"],
        "num_val_files": base_config["training"]["num_val_files"]
    }

    return run_training(config)

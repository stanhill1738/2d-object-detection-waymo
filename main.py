import optuna
import yaml
from train.objective import objective

with open("configs/config.yaml", "r") as f:
    base_config = yaml.safe_load(f)

study = optuna.create_study(direction="minimize")
study.optimize(lambda trial: objective(trial, base_config), n_trials=base_config['optuna']['n_trials'], timeout=base_config['optuna']['timeout'])

print("Best trial:")
print(study.best_trial)

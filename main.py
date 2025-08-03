import optuna
from train.objective import objective

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

print("Best trial:")
print(study.best_trial)
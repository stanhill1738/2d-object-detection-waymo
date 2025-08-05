# Hyperparameter Tuning Instructions

Hyperparameter tuning was completed using Optuna.
The config that drives the overall training for these experiements are found in [config.yaml](../configs/config.yaml).

## Prerequisites
- Python 3.10

# Instructions
1. Create a venv:
    `python3 -m venv venv`
    `source venv/bin/activate`
2. `pip install -r requirements.txt`
3. Edit the contents of [config.yaml](../configs/config.yaml) to suite your experimentation.
4. `python main.py` (or use `nohup` to run in the background)

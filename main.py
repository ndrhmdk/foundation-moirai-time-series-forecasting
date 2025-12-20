import os
import sys
import warnings
warnings.filterwarnings('ignore')

import yaml
import torch
import pandas as pd

# NOTE. PATH SETUP
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

BASE_DIR = os.path.basename(PROJECT_DIR)
print(f"Project path (Relative): {BASE_DIR}")
# Expected output: Project path (Relative): foundation-moirai-time-series-forecasting

OUTPUTS_DIR = os.path.join(PROJECT_DIR, 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)
print(f"Detected output folder: {os.path.basename(OUTPUTS_DIR)}")
# Expected output: Detected output folder: outputs

# NOTE. PROJECT MODULES
from src.dataloader import load_data, get_splits
from src.baselines import BaselineModels
from src.moirai_wrapper import MoiraiPredictor
from src.evaluation import rolling_cv, statistical_test
from src.plotting import (
    plot_backtesting_perf,
    plot_error_by_horizon,
    plot_calibration,
    plot_forecast_overlay)

# NOTE. DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device using: {DEVICE}")

# NOTE. LOAD CONFIG
CONFIG_PATH = os.path.join(PROJECT_DIR, 'configs', 'config.yaml')
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
    
HORIZON = config["experiment"]["horizon"]
CONTEXT_LENGTH = config["experiment"]["context_length"]
MOIRAI_SMALL = config["models"]["moirai_small"]
MOIRAI_BASE = config["models"]["moirai_base"]

# NOTE. CORE PIPELINE FUNCTION
def run_experiment(dataset_name: str, config: dict):
    print(f"\n{'='*60}")
    print(f"Running experiment for dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # Load data
    df = load_data(dataset_name, config)
    train, valid, test = get_splits(df, config)
    
    seasonality = config['datasets'][dataset_name]['seasonality']
    freq = config['datasets'][dataset_name].get('frequency', 'D')
    
    # Models
    baselines = BaselineModels(
        seasonal_period=seasonality,
        horizon=HORIZON)
    
    moirai_models = {
        "Moirai-Small": MoiraiPredictor(
            model_name=MOIRAI_SMALL,
            context_length=CONTEXT_LENGTH,
            horizon=HORIZON,
            device=DEVICE,
            freq=freq
        ),
        "Moirai-Base": MoiraiPredictor(
            model_name=MOIRAI_BASE,
            context_length=CONTEXT_LENGTH,
            horizon=HORIZON,
            device=DEVICE,
            freq=freq
        )
    }
    
    # Backtesting
    valid_results, preds = rolling_cv(
        train_series=train['values'],
        valid_series=valid['values'],
        baselines=baselines,
        moirai_models=moirai_models,
        config=config
    )
    
    # Statistics
    print(f"\n[{dataset_name}] Results Summary (MAE):")
    print(valid_results.groupby("model")["MAE"].mean())

    best_bl, best_moirai, p_val = statistical_test(valid_results)
    print(f"Wilcoxon: {best_moirai} vs {best_bl}")
    print(f"- p-value = {p_val:.4f}")
    
    # Plots
    plot_backtesting_perf(valid_results, dataset_name)
    plot_error_by_horizon(preds, dataset_name)
    plot_calibration(preds, dataset_name)
    
    # Final inference
    full_history = pd.concat([train['values'], valid['values']])
    final_preds = {}
    
    for name, model in moirai_models.items():
        context = full_history.iloc[-CONTEXT_LENGTH:]
        final_preds[name] = model.predict(context)
        
    plot_forecast_overlay(
        full_history,
        test['values'],
        final_preds,
        dataset_name
    )
    
if __name__ == "__main__":
    for dataset_name in config['datasets'].keys():
        run_experiment(dataset_name, config)
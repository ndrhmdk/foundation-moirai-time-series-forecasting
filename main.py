import argparse
import os
import sys
import yaml
import warnings
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())

from src.dataloader import load_data, get_splits
from src.baselines import BaselineModels
from src.moirai_wrapper import MoiraiPredictor
from src.evaluation import rolling_cv, statistical_test, cal_metrics, pinball_loss
from src.plotting import (
    plot_backtesting_perf, 
    plot_error_by_horizon, 
    plot_calibration, 
    plot_forecast_overlay
)

def run_experiment(dataset_key):
    print(f"\n{'='*40}")
    print(f"STARTING EXPERIMENT: {dataset_key}")
    print(f"{'='*40}")

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    # Directories
    OUTPUTS_DIR = "outputs"
    CONFIGS_PATH = "configs/config.yaml"
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Load Config
    if not os.path.exists(CONFIGS_PATH):
        raise FileNotFoundError(f"Config file not found at {CONFIGS_PATH}")
        
    with open(CONFIGS_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Validate Dataset Key
    if dataset_key not in config['datasets']:
        raise ValueError(f"Dataset '{dataset_key}' not found in config.yaml. Available: {list(config['datasets'].keys())}")

    # Extract Parameters
    model_cfg = config['models']
    exp_cfg = config['experiment']
    data_cfg = config['datasets'][dataset_key]

    MOIRAI_SMALL = model_cfg['moirai_small']
    MOIRAI_BASE = model_cfg['moirai_base']
    HORIZON = exp_cfg['horizon']
    CTX_ZS = exp_cfg['context_length']['zero_shot']
    CTX_FS = exp_cfg['context_length']['few_shot']
    SEASONALITY = data_cfg['seasonality']
    FREQ = data_cfg.get('frequency', 'D')

    print(f"Parameters:")
    print(f"- Horizon: {HORIZON}")
    print(f"- Seasonality: {SEASONALITY}")
    print(f"- Zero-Shot Context: {CTX_ZS}")
    print(f"- Few-Shot Context: {CTX_FS}")

    # ==========================================
    # 2. Data Loading & EDA
    # ==========================================
    print("\n[1/5] Loading Data...")
    df = load_data(dataset_key, config)
    print(f"Data Shape: {df.shape}")
    
    train, valid, test = get_splits(df, config)
    
    # STL Decomposition Check
    print("Performing STL Decomposition...")
    stl = STL(train['values'], period=SEASONALITY)
    res = stl.fit()
    fig = res.plot()
    plt.suptitle(f"STL Decomposition - {dataset_key}")
    plt.tight_layout()
    stl_path = f"{OUTPUTS_DIR}/00_{dataset_key.lower()}_stl.png"
    plt.savefig(stl_path)
    print(f"STL Plot saved to: {stl_path}")
    plt.close() # Close to prevent display issues in non-interactive environments

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("\n[2/5] Initializing Models...")
    
    baselines = BaselineModels(
        seasonal_period=SEASONALITY,
        horizon=HORIZON
    )

    moirai_models = {
        'Moirai-Small (ZS)': MoiraiPredictor(
            model_name=MOIRAI_SMALL,
            context_length=CTX_ZS,
            horizon=HORIZON,
            device=DEVICE,
            freq=FREQ),
        
        'Moirai-Base (ZS)': MoiraiPredictor(
            model_name=MOIRAI_BASE,
            context_length=CTX_ZS,
            horizon=HORIZON,
            device=DEVICE,
            freq=FREQ),
            
        'Moirai-Small (FewS)': MoiraiPredictor(
            model_name=MOIRAI_SMALL,
            context_length=CTX_FS, 
            horizon=HORIZON,
            device=DEVICE,
            freq=FREQ),
        
        'Moirai-Base (FewS)': MoiraiPredictor(
            model_name=MOIRAI_BASE,
            context_length=CTX_FS,
            horizon=HORIZON,
            device=DEVICE,
            freq=FREQ)
    }
    print(f"Initialized {len(moirai_models)} Moirai variants + Baselines.")

    # ==========================================
    # 4. Backtesting (Validation)
    # ==========================================
    print("\n[3/5] Running Rolling Cross-Validation (Backtesting)...")
    valid_results, preds = rolling_cv(
        dataset_name=dataset_key,
        train_series=train['values'],
        valid_series=valid['values'],
        baselines=baselines,
        moirai_models=moirai_models,
        config=config
    )

    # --- Validation Analysis ---
    print("\n--- Validation Leaderboard (MAE) ---")
    print(valid_results.groupby('model')['MAE'].mean().sort_values())

    # Statistical Test
    best_bl, best_moirai, p_val = statistical_test(valid_results)
    print(f"\nWilcoxon Test ({best_bl} vs {best_moirai}): p-value = {p_val:.4f}")

    # Plotting
    print("Generating Validation Plots...")
    plot_backtesting_perf(valid_results, dataset_key)
    plot_error_by_horizon(preds, dataset_key)
    plot_calibration(preds, dataset_key)

    # ==========================================
    # 5. Final Inference (Test Set)
    # ==========================================
    print("\n[4/5] Running Final Inference on Test Set...")
    full_history = pd.concat([train['values'], valid['values']])
    y_true = test['values'].values
    y_train_hist = full_history.values
    
    final_preds = {}
    
    # Predict Moirai Models
    for name, model in moirai_models.items():
        print(f"Predicting {name}...")
        final_preds[name] = model.predict(full_history)

    # Generate Overlay Plot
    plot_forecast_overlay(full_history, test['values'], final_preds, dataset_key)

    # Predict Baselines (Test Set)
    baseline_test_preds = {
        'S-Naive': baselines.seasonal_naive(full_history),
        'ETS': baselines.ets(full_history)
    }

    # ==========================================
    # 6. Final Evaluation
    # ==========================================
    print("\n[5/5] Calculating Final Metrics...")
    
    # Combine predictions
    all_model_preds = final_preds.copy()
    all_model_preds.update(baseline_test_preds)

    stats_list = []
    for model_name, prediction in all_model_preds.items():
        # Handle dict vs array
        if isinstance(prediction, dict):
            y_pred = prediction['median']
        else:
            y_pred = prediction
        
        # Calculate Metrics
        metrics = cal_metrics(y_true, y_pred, y_train_hist, SEASONALITY)
        metrics['Pinball_0.5'] = pinball_loss(y_true, y_pred, 0.5)
        metrics['Model'] = model_name
        stats_list.append(metrics)

    # Display Final Table
    df_stats = pd.DataFrame(stats_list).set_index('Model')
    print(f"\n[{dataset_key}] FINAL TEST SET LEADERBOARD:")
    print(df_stats.sort_values(by='MAE').to_markdown()) # Requires tabulate, else just print(df_stats)
    
    # Save metrics to CSV
    metrics_path = f"{OUTPUTS_DIR}/05_{dataset_key.lower()}_metrics.csv"
    df_stats.to_csv(metrics_path)
    print(f"\nMetrics saved to {metrics_path}")
    print(f"{'='*40}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*40}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Moirai Forecasting Experiment")
    
    try:
        with open("configs/config.yaml", "r") as f:
            pre_config = yaml.safe_load(f)
            available_datasets = list(pre_config['datasets'].keys())
    except:
        available_datasets = ["SPY", "AQI_LA"]

    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        choices=available_datasets,
        help=f"Name of the dataset to run. Options: {available_datasets}"
    )

    args = parser.parse_args()
    
    try:
        run_experiment(args.dataset)
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
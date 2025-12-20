import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_backtesting_perf(results_df, dataset_name):
    PATH = f"{PROJECT_ROOT}/outputs/01_{dataset_name.lower()}_backtesting.png"
    SAVE_PATH = os.path.relpath(PATH, PROJECT_ROOT)
    print(f"Plot saved: {SAVE_PATH}")
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x='fold', y='MAE', hue='model', marker='o')
    plt.title(f"[{dataset_name}] Backtesting Performance (MAE)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig(PATH)
    plt.show()
    plt.close()
    
def plot_error_by_horizon(preds_storage, dataset_name): 
    PATH = f"{PROJECT_ROOT}/outputs/02_{dataset_name.lower()}_horizon_error.png"
    SAVE_PATH = os.path.relpath(PATH, PROJECT_ROOT)
    print(f"Plot saved: {SAVE_PATH}")
    
    horizon_errors = []
    for item in preds_storage:
        if 'pred' in item:
            residuals = np.abs(item['truth'] - item['pred'])
            for h, err in enumerate(residuals):
                horizon_errors.append({
                    'model': item['model'],
                    'horizon': h + 1,
                    'AbsError': err})
    if not horizon_errors:
        return
    
    df_err = pd.DataFrame(horizon_errors)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_err, x='horizon', y='AbsError', hue='model')
    plt.title(f"[{dataset_name}] MAE by Forecast Horizon")
    plt.savefig(PATH)
    plt.show()
    plt.close()
    
def plot_calibration(preds_storage, dataset_name):
    """
    Interval converage.
    """
    PATH = f"{PROJECT_ROOT}/outputs/03_{dataset_name.lower()}_calibration.png"
    SAVE_PATH = os.path.relpath(PATH, PROJECT_ROOT)
    print(f"Plot saved: {SAVE_PATH}")
    
    records = []
    for item in preds_storage:
        if 'q10' in item:
            truth = item['truth']
            low = item['q10']
            high = item['q90']
            
            inside = ((truth >= low) & (truth <= high)).astype(int)
            records.append({
                'model': item['model'],
                'expected': 0.8,
                'empirical': np.mean(inside)})
    if not records:
        return
    df_cal = pd.DataFrame(records)
    plt.figure(figsize=(6, 6))
    sns.barplot(data=df_cal, x='model', y='empirical')
    plt.axhline(0.8, color='red', linestyle='--', label='Expected (0.8)')
    plt.ylim(0, 1.0)
    plt.title(f"[{dataset_name}] Interval Coverage (Nomial 80%)")
    plt.legend()
    plt.savefig(PATH)
    plt.show()

def plot_forecast_overlay(train_series, test_series, prediction_dict, dataset_name):
    PATH = f"{PROJECT_ROOT}/outputs/04_{dataset_name.lower()}_overlay_prediction.png"
    SAVE_PATH = os.path.relpath(PATH, PROJECT_ROOT)
    print(f"Plot saved: {SAVE_PATH}")
    models_to_plot = ['Moirai-Small', 'Moirai-Base']
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"[{dataset_name}] Forecast Visualization", fontsize=16, y=0.95, fontweight='bold')
    
    history_len = 100
    history = train_series.iloc[-history_len:] if len(train_series) > history_len else train_series

    for ax, model_key in zip(axes, models_to_plot):
        forecast_len = len(test_series)
        has_prediction = model_key in prediction_dict
        
        if has_prediction:
            forecast_len = len(prediction_dict[model_key]['median'])

        ax.plot(history.index, history.values, color='black', linewidth=1.5, label='Target (History)')
        actual_subset = test_series.iloc[:forecast_len]
        ax.plot(actual_subset.index, actual_subset.values, color='black', linewidth=1.5, linestyle=':', label='Target (Actual)')
        
        # Plot Prediction
        if has_prediction:
            m = prediction_dict[model_key]
            dates = test_series.index[:forecast_len]
            
            # Median forecast
            ax.plot(dates, m['median'], color='#2b6ce6', linewidth=2, label=f'{model_key} (Median)')
            
            # Inner Band (50% CI)
            ax.fill_between(
                dates, m['q25'], m['q75'],
                color='#2b6ce6', alpha=0.4, 
                label='50% CI')
            
            # Outer Band (90% CI)
            ax.fill_between(
                dates, m['q05'], m['q95'], 
                color='#2b6ce6', alpha=0.15, 
                label='90% CI')
            
            ax.set_title(f"{model_key} Prediction", loc='left', fontsize=12, fontweight='bold')
        else:
            ax.set_title(f"{model_key} (Not Found)", loc='left', fontsize=12, color='red')

        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize='small')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(PATH, dpi=300)
    plt.show()
    plt.close()
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from scipy.stats import wilcoxon
from tqdm import tqdm

def cal_metrics(y_true, y_pred, y_train_hist, m):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    
    # MASE
    n = len(y_train_hist)
    if n > m:
        denom = np.mean(np.abs(y_train_hist[m:] - y_train_hist[:-m]))
        mase = mae / denom if denom != 0 else np.nan
    else:
        mase = np.nan
    
    return {
        'MAE': mae, 
        'RMSE': rmse, 
        'MASE': mase}

def pinball_loss(y_true, q_pred, tau):
    """Pinball loss for quantile tau."""
    y_true = np.asarray(y_true)
    q_pred = np.asarray(q_pred)
    pinball = np.mean(
        np.maximum(
            tau * (y_true - q_pred),
            (tau - 1) * (y_true - q_pred)
        )
    )
    return pinball

def rolling_cv(dataset_name, train_series, valid_series, baselines, moirai_models, config):
    experiment = config['experiment']
    datasets_cfg = config['datasets']
    horizon = experiment['horizon']
    n_splits = experiment['n_splits']
    
    # Seasonal period
    if dataset_name not in datasets_cfg:
        raise ValueError(f"Dataset '{dataset_name}' not found in config['datasets'].")
    
    m = datasets_cfg[dataset_name].get('seasonality')
    if m is None:
        raise ValueError("Seasonality `m` must be provided in config ['experiment'].")
    
    full_cv_data = pd.concat([train_series, valid_series])
    total_len = len(full_cv_data)
    
    results = []
    preds_storage = []
    
    start_indices = [
        total_len - (i * horizon) - horizon 
        for i in range(n_splits, 0, -1)]
    
    for fold, start_idx in enumerate(start_indices):
        train = full_cv_data.iloc[:start_idx]
        test = full_cv_data.iloc[start_idx : start_idx + horizon]
        
        if len(test) < horizon:
            continue
        
        print(f"[FOLD {fold + 1}]: Predict {test.index[0].date()} to {test.index[-1].date()}")
        
        # 1. Baselines
        fold_preds = {}
        fold_preds['S-Naive'] = baselines.seasonal_naive(train)
        fold_preds['ETS'] = baselines.ets(train)
        
        baseline_preds = {
            'S-Naive': baselines.seasonal_naive(train),
            'ETS': baselines.ets(train)
        }
        
        for model_name, y_hat in baseline_preds.items():
            metrics = cal_metrics(test.values, y_hat, train.values, m)
            metrics.update({
                'model': model_name,
                'fold': fold
            })
            results.append(metrics)
            
            preds_storage.append({
                'model': model_name,
                'fold': fold,
                'truth': test.values,
                'pred': y_hat
            })
        
        # 2. Moirai
        for model_name, model in moirai_models.items():
            out = model.predict(train)
            y_med = out['median']
            
            metrics = cal_metrics(test.values, y_med, train.values, m)
            metrics.update({
                'model': model_name,
                'fold': fold,
                'pinball_0.1': pinball_loss(test.values, out['q10'], 0.10),
                'pinball_0.5': pinball_loss(test.values, y_med, 0.59),
                'pinball_0.9': pinball_loss(test.values, out['q90'], 0.90)
            })
            results.append(metrics)
            
            preds_storage.append({
                'model': model_name,
                'fold': fold,
                'truth': test.values,
                'pred': y_med,
                'q05': out['q05'],
                'q25': out['q25'],
                'q50': y_med,
                'q75': out['q75'],
                'q95': out['q95'],
                'q10': out['q10'],
                'q90': out['q90']
            })
            
    return pd.DataFrame(results), preds_storage
    
def statistical_test(results_df):
    agg = results_df.groupby('model')['MAE'].mean().sort_values()
    try:
        best_bl = [m for m in agg.index if 'Moirai' not in m][0]
        best_moirai = [m for m in agg.index if 'Moirai' in m][0]
        
        err_bl = results_df[results_df['model'] == best_bl]['MAE'].values
        err_moirai = results_df[results_df['model'] == best_moirai]['MAE'].values
        
        _, p = wilcoxon(err_bl, err_moirai)
        return best_bl, best_moirai, p
    except:
        return "None", "None", 1.0
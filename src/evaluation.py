import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from scipy.stats import wilcoxon
from tqdm import tqdm

def metrics(y_true, y_pred, y_train_hist):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    
    # MASE
    n = len(y_train_hist)
    d = np.abs(np.diff(y_train_hist)).sum() / (n - 1)
    mase = mae / d if d != 0 else np.nan
    
    return {'MAE': mae, 'RMSE': rmse, 'MASE': mase}

def rolling_cv(train_series, valid_series, baselines, moirai_models, config):
    experiment = config['experiment']
    
    horizon = experiment['horizon']
    n_splits = experiment['n_splits']
    
    full_cv_data = pd.concat([train_series, valid_series])
    total_len = len(full_cv_data)
    
    results = []
    preds_storage = []
    
    start_indices = [total_len - (i * horizon) - horizon for i in range(n_splits, 0, -1)]
    
    for i, start_idx in enumerate(start_indices):
        train = full_cv_data.iloc[:start_idx]
        test = full_cv_data.iloc[start_idx : start_idx + horizon]
        
        if len(test) < horizon:
            continue
        
        print(f"[FOLD {i + 1}]: Predict {test.index[0].date()} to {test.index[-1].date()}")
        
        # 1. Baselines
        fold_preds = {}
        fold_preds['S-Naive'] = baselines.seasonal_naive(train)
        fold_preds['ETS'] = baselines.ets(train)
        
        # 2. Moirai
        for name, model in moirai_models.items():
            out = model.predict(train)
            fold_preds[name] = out['median']
            
            # Store probabilistic info
            preds_storage.append({
                'model': name, 'fold': i, 'truth': test.values,
                'q05': out['q05'], 'q95': out['q95'],
                'q25': out['q25'], 'q75': out['q75'],
                'q10': out['q10'], 'q90': out['q90']
            })
            
        # 3. Metrics
        for model_name, y_hat in fold_preds.items():
            metrics = metrics(test.values, y_hat, train.values)
            metrics['models'] = model_name
            metrics['fold'] = i
            results.append(metrics)
            
            preds_storage.append({
                'model': model_name, 'fold': i, 'truth': test.values, 'pred': y_hat})
            
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
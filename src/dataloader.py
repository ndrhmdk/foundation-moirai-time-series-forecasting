import yfinance as yf
import pandas as pd
import requests 
import time

def load_yahoo(name, config):
    """
    Fetch data from Yahoo Finance.
    """
    cfg = config['datasets'][name]
    print(f"Downloading {cfg['ticker']} from Yahoo Finance...")
    
    df = yf.download(
        cfg['ticker'],
        start=cfg['start_date'],
        end=cfg['end_date'],
        interval='1d',
        progress=False,
        auto_adjust=True
    )
    
    # MultiIndex Handling
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs('Close', axis=1, level=0) if 'Close' in df.columns.levels[0] else df['Close']
        except:
            df = df.iloc[:, 0]
    elif 'Close' in df.columns:
        df = df[['Close']]
        
    df.columns = ['values']
    df = df.ffill().bfill()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    
    df.index.name = 'date'
    df = df.asfreq('B')
    df['values'] = df['values'].ffill()
    
    return df

def load_aqi(name, config):
    """
    Fetch AQI from Open-Meteo.
    """
    cfg = config['datasets'][name]
    
    print(f"Downloading AQI for lat={cfg['lat']}, lon={cfg['lon']}...")
    
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": cfg['lat'],
        "longitude": cfg['lon'],
        "start_date": cfg['start_date'],
        "end_date": cfg['end_date'],
        "hourly": "us_aqi",  # US Air Quality Index
        "timezone": "auto"
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"API Failed: {response.text}")
        
    data = response.json()
    
    df = pd.DataFrame({
        'date': pd.to_datetime(data['hourly']['time']),
        'values': data['hourly']['us_aqi']
    })
    
    df.set_index('date', inplace=True)
    
    df = df.resample('D').max()
    df['values'] = df['values'].interpolate(method='linear')
    return df


def load_data(dataset_name, config):
    source_type = config['datasets'][dataset_name]['type']
    
    if source_type == 'yahoo':
        return load_yahoo(dataset_name, config)
    elif source_type == 'aqi':
        return load_aqi(dataset_name, config)
    else:
        raise ValueError(f"Unknown type: {source_type}")
    
def get_splits(df, config):
    """Splits data preventing leakage"""
    experiment = config['experiment']
    horizon = experiment['horizon']
    n_splits = experiment['n_splits']
    
    # Calculating sizes
    test_size = experiment['test_size']
    valid_size = n_splits * horizon
    
    # Splits
    test = df.iloc[-test_size:].copy()
    valid = df.iloc[len(df) - test_size - valid_size : -test_size].copy()
    train = df.iloc[: len(df) - test_size - valid_size].copy()
    
    print(f"Splits created:\n" +
          f"- train: {len(train)}\n" + 
          f"- valid: {len(valid)}\n" +
          f"-  test: {len(test)}\n")
    
    return train, valid, test
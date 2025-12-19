import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

class BaselineModels:
    def __init__(self, seasonal_period=5, horizon=14):
        self.m = seasonal_period
        self.h = horizon
        
    def seasonal_naive(self, train_series):
        last_window = train_series.iloc[-self.m:].values
        forecast = np.tile(last_window, int(np.ceil(self.h/self.m)))[:self.h]
        return forecast
    
    def ets(self, train_series):
        try:
            model = ExponentialSmoothing(
                train_series,
                seasonal_periods=self.m,
                trend='add',
                seasonal='add',
                initialization_method='estimated').fit()
            return model.forecast(self.h).values
        except:
            return self.seasonal_naive(train_series)
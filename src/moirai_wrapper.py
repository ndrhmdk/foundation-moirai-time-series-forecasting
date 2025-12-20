import numpy as np
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

class MoiraiPredictor:
    def __init__(self, model_name, context_length, horizon, device, freq="D"):
        # Load Pre-trained Module
        module = MoiraiModule.from_pretrained(model_name)
        
        # Initialize Forecaster Wrapper
        self.model = MoiraiForecast(
            module=module,
            patch_size='auto',
            context_length=context_length,
            prediction_length=horizon,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0
        )
        
        # Predictor
        self.predictor = self.model.create_predictor(batch_size=8, device=str(device))
        self.freq = freq
        
    def predict(self, series_pd):
        ds = PandasDataset(
            {'target': series_pd},
            target='target',
            freq=self.freq
        )
        
        forecast_it = self.predictor.predict(ds)
        forecast = list(forecast_it)[0]
        
        return {
            'mean': forecast.mean,
            'median': forecast.quantile(0.5),
            'q05': forecast.quantile(0.05),
            'q95': forecast.quantile(0.95),
            'q25': forecast.quantile(0.25),
            'q75': forecast.quantile(0.75),
            'q10': forecast.quantile(0.1),
            'q90': forecast.quantile(0.9),
            'samples': forecast.samples
        }
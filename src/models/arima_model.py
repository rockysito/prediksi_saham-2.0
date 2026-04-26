import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

class StockARIMAModel:
    def __init__(self, order=(5,1,0)):
        self.order = order
        self.model_fit = None

    def train(self, data: pd.Series):
        """
        Train the ARIMA model using historical stock data.
        Extracted logic from 'ARIMA - Apple.ipynb'
        """
        print(f"Training ARIMA model with order {self.order}...")
        model = ARIMA(data, order=self.order)
        self.model_fit = model.fit()
        return self.model_fit

    def predict(self, steps=1) -> np.ndarray:
        """
        Forecast future prices.
        """
        if self.model_fit is None:
            raise ValueError("Model is not trained yet.")
        forecast = self.model_fit.forecast(steps=steps)
        return forecast.values if hasattr(forecast, 'values') else forecast

    def evaluate(self, test_data: pd.Series):
        predictions = self.predict(steps=len(test_data))
        rmse = np.sqrt(mean_squared_error(test_data, predictions))
        print(f"Model RMSE: {rmse}")
        return rmse

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

class StockLSTMModel:
    def __init__(self, look_back=60):
        self.look_back = look_back
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def _build_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def prepare_data(self, data: np.ndarray):
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        X, y = [], []
        for i in range(self.look_back, len(scaled_data)):
            X.append(scaled_data[i-self.look_back:i, 0])
            y.append(scaled_data[i, 0])
        return np.array(X), np.array(y)

    def train(self, data: pd.Series, epochs=50, batch_size=32):
        """
        Train LSTM model using historical stock data.
        Extracted logic from 'LSTM - Apple.ipynb'
        """
        print("Preparing data and training LSTM model...")
        X, y = self.prepare_data(data.values)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        self.model = self._build_model((X.shape[1], 1))
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, recent_data: np.ndarray) -> float:
        """
        Predict next step.
        """
        scaled = self.scaler.transform(recent_data.reshape(-1, 1))
        X = np.reshape(scaled, (1, self.look_back, 1))
        pred_scaled = self.model.predict(X)
        pred = self.scaler.inverse_transform(pred_scaled)
        return pred[0][0]

    def save(self, model_path, scaler_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        if self.model:
            self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)

    def load(self, model_path, scaler_path):
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)

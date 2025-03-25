import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Загружаем данные
dataset = pd.read_csv("LST_final_TRUE.csv")

features = ["H", "TWI", "Aspect", "Hillshade", "Roughness", "Slope",
            "Temperature_merra_1000hpa", "Time", "DayOfYear", "X", "Y"]
target = "T_rp5"

X = dataset[features].values
y = dataset[target].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# Функция для тестирования разных гиперпараметров LSTM
def test_lstm_models():
    epochs = [50, 100, 200, 500]
    lstm_units = [64, 128, 256]
    activations = ["relu", "tanh", "LeakyReLU"]
    optimizers = ["adam", "rmsprop", "sgd"]
    batch_sizes = [16, 32, 64]
    dropout_rates = [0.2, 0.3, 0.5]

    for epoch in epochs:
        for units in lstm_units:
            for activation in activations:
                for optimizer in optimizers:
                    for batch_size in batch_sizes:
                        for dropout_rate in dropout_rates:
                            # Строим модель
                            model = keras.Sequential()
                            model.add(layers.LSTM(units, activation=activation, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
                            model.add(layers.LSTM(units, activation=activation))
                            model.add(layers.Dropout(dropout_rate))
                            model.add(layers.Dense(1))

                            model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
                            history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size,
                                                validation_data=(X_test, y_test), verbose=0)

                            # Предсказания и метрики
                            predictions = model.predict(X_test)
                            mse = mean_squared_error(y_test, predictions)
                            r2 = r2_score(y_test, predictions)
                            mae = history.history['val_mae'][-1]

                            print(f"LSTM - Epochs: {epoch}, Units: {units}, Activation: {activation}, "
                                  f"Optimizer: {optimizer}, Batch Size: {batch_size}, Dropout: {dropout_rate} => "
                                  f"MSE: {mse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")

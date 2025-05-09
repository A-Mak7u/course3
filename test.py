import os
import time
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------
# CONFIG
# ------------------------

# Использование mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# ------------------------
# LOAD DATA
# ------------------------

# Замените путь на ваш файл
dataset = pd.read_csv("LST_final_TRUE.csv")
features = ["H", "TWI", "Aspect", "Hillshade", "Roughness", "Slope", "Temperature_merra_1000hpa", "Time", "DayOfYear", "X", "Y"]
target = "T_rp5"

X = dataset[features].values
y = dataset[target].values

# Масштабируем данные
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ------------------------
# RANDOM HYPERPARAM SEARCH CONFIG
# ------------------------

# Уменьшенные опции гиперпараметров для быстрого теста
epochs_options = [50]  # Оставляем одну эпоху для быстроты
batch_size_options = [32]  # Размер батча 32
activation_options = ['relu']  # Оставляем одну функцию активации
neurons_options = [(64, 32)]  # Оставляем одну конфигурацию нейронов
dropout_options = [0.0]  # Без dropout для простоты
learning_rate_options = [0.001]  # Оставляем одну скорость обучения

all_combinations = list(itertools.product(
    epochs_options,
    batch_size_options,
    activation_options,
    neurons_options,
    dropout_options,
    learning_rate_options
))

# ------------------------
# LSTM MODEL DEFINITION
# ------------------------

def run_model(epochs, batch_size, activation, neurons, dropout_rate, learning_rate):
    # Создание модели
    model = models.Sequential()
    model.add(layers.LSTM(128, input_shape=(X_train.shape[1], 1), return_sequences=True))
    model.add(layers.LSTM(64, return_sequences=False))

    for units in neurons:
        model.add(layers.Dense(units, activation=activation))
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1))

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    # Обучение модели
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return history

# ------------------------
# FUNCTION TO TIME PERFORMANCE
# ------------------------

def test_parallelization(processes):
    start_time = time.time()

    # Тестируем выполнение с разными гиперпараметрами
    for params in all_combinations:
        e, b, a, n, d, lr = params
        run_model(e, b, a, n, d, lr)  # Не выводим метрики, просто тестируем время

    duration = time.time() - start_time
    print(f"\nИспользование {processes} процессов заняло {duration:.2f} секунд.")

# ------------------------
# TESTING WITH VARIOUS PARALLELIZATION CONFIGURATIONS
# ------------------------

if __name__ == "__main__":
    # Тестируем с различным количеством процессов
    for processes in [1, 2, 4]:  # Меньше процессов для ускоренного теста
        test_parallelization(processes)

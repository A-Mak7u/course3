import pandas as pd
import numpy as np
import os
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tqdm import tqdm
import itertools

# Оптимизация вычислений
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    try:
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except RuntimeError as e:
        print(e)

# === Загрузка и подготовка данных ===
df = pd.read_csv("LST_final_TRUE.csv")
target_col = "T_rp5"

# Фильтрация выбросов
q_low = df[target_col].quantile(0.07)
q_high = df[target_col].quantile(0.93)
df = df[(df[target_col] >= q_low) & (df[target_col] <= q_high)]

# Признаки и целевая переменная
X = df.drop(columns=["T_rp5"])
y = df["T_rp5"]

# Масштабирование
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Формирование временных окон (3 шага в прошлое)
def create_sequences(X, y, time_steps=3):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_scaled)
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, shuffle=False)

# === Гиперпараметры ===
param_grid = {
    'units': [64, 128],
    'dropout': [0.1, 0.2],
    'lr': [0.001, 0.0005],
    'batch_size': [32, 64],
    'epochs': [50, 100]
}

# === Функция создания модели ===
def build_model(input_shape, units, dropout, lr):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(units, return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(units // 2),
        layers.Dropout(dropout),
        layers.Dense(1)
    ])
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    return model

# === Поиск по сетке ===
results = []
progress = list(itertools.product(*param_grid.values()))
total = len(progress)

for units, dropout, lr, batch_size, epochs in tqdm(progress, total=total, desc="Grid Search"):
    model = build_model(X_train.shape[1:], units, dropout, lr)
    es = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )

    y_pred = model.predict(X_test)
    y_pred_inv = scaler_y.inverse_transform(y_pred)
    y_test_inv = scaler_y.inverse_transform(y_test)

    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
    r2 = r2_score(y_test_inv, y_pred_inv)

    results.append({
        'units': units,
        'dropout': dropout,
        'lr': lr,
        'batch_size': batch_size,
        'epochs': epochs,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    })

# === Сохранение результатов ===
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="R2", ascending=False)
today_str = date.today().isoformat()
results_df.to_csv(f"lstm_grid_results_{today_str}.csv", index=False)

print(f"\nГотово. Лучшие результаты сохранены в lstm_grid_results_{today_str}.csv")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

# Проверяем доступные устройства (GPU)
print("Доступные устройства:", tf.config.list_physical_devices())

# Загружаем данные
dataset = pd.read_csv("LST_final_TRUE.csv")

# Увеличиваем выборку до 50% для теста
dataset = dataset.sample(frac=0.5, random_state=42)

features = ["H", "TWI", "Aspect", "Hillshade", "Roughness", "Slope",
            "Temperature_merra_1000hpa", "Time", "DayOfYear", "X", "Y"]
target = "T_rp5"

X = dataset[features].values
y = dataset[target].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Опции гиперпараметров для более долгого тренинга
epochs_options = [50]  # Увеличим до 50 эпох для большей нагрузки
batch_size_options = [64]  # Увеличиваем размер батча
neurons_options = [(256, 128, 64)]  # Увеличиваем количество нейронов в слоях
activation_options = ['relu']
regularization_options = [None]
optimizer_options = ['adam']
learning_rate_options = [0.001]


# Функция для обучения модели
def run_model(epochs, batch_size, neurons, activation, regularization, optimizer, learning_rate, device='/CPU:0'):
    model = keras.Sequential()

    model.add(layers.Dense(neurons[0], activation=activation, input_shape=(X_train.shape[1],)))
    if regularization == 'dropout':
        model.add(layers.Dropout(0.3))

    model.add(layers.Dense(neurons[1], activation=activation))
    if regularization == 'dropout':
        model.add(layers.Dropout(0.3))

    model.add(layers.Dense(neurons[2], activation=activation))
    if regularization == 'dropout':
        model.add(layers.Dropout(0.3))

    model.add(layers.Dense(1))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])

    # Колбэк для логирования раз в 10 эпох
    class EpochLogger(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 10 == 0:  # Лог каждые 10 эпох
                print(f"Эпоха {epoch + 1}: val_loss={logs['val_loss']:.4f}, val_mae={logs['val_mae']:.4f}")

    with tf.device(device):  # Явное указание устройства для вычислений
        start_time = time.time()
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),
                            verbose=0, callbacks=[EpochLogger()])
        elapsed_time = time.time() - start_time

        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

    return mse, r2, history, elapsed_time


# Запуск на CPU
print("\nЗапуск на CPU:")
mse_cpu, r2_cpu, history_cpu, elapsed_time_cpu = run_model(epochs=50, batch_size=64, neurons=(256, 128, 64),
                                                           activation='relu', regularization=None, optimizer='adam',
                                                           learning_rate=0.001, device='/CPU:0')

print(f"CPU время: {elapsed_time_cpu:.2f} секунд, MSE={mse_cpu:.4f}, R²={r2_cpu:.4f}")

# Запуск на GPU, если доступен
if tf.config.list_physical_devices('GPU'):
    print("\nЗапуск на GPU:")
    mse_gpu, r2_gpu, history_gpu, elapsed_time_gpu = run_model(epochs=50, batch_size=64, neurons=(256, 128, 64),
                                                               activation='relu', regularization=None, optimizer='adam',
                                                               learning_rate=0.001, device='/GPU:0')

    print(f"GPU время: {elapsed_time_gpu:.2f} секунд, MSE={mse_gpu:.4f}, R²={r2_gpu:.4f}")
    speedup = elapsed_time_cpu / elapsed_time_gpu
    print(f"Ускорение на GPU: {speedup:.2f}x")
else:
    print("GPU не обнаружен!")

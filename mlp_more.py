import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

# Загружаем данные
dataset = pd.read_csv("LST_final_TRUE.csv")

features = ["H", "TWI", "Aspect", "Hillshade", "Roughness", "Slope",
            "Temperature_merra_1000hpa", "Time", "DayOfYear", "X", "Y"]
target = "T_rp5"

X = dataset[features].values
y = dataset[target].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Список параметров для тестирования
epochs_options = [100, 200, 300]
batch_size_options = [32, 64, 128]
neurons_options = [(128, 64, 32), (256, 128, 64)]
activation_options = ['relu', 'leakyrelu', 'elu']
regularization_options = [None, 'dropout', 'l2']
optimizer_options = ['adam', 'nadam']
learning_rate_options = [0.001, 0.0001]


# Функция для обучения модели и получения метрик
def run_model(epochs, batch_size, neurons, activation, regularization, optimizer, learning_rate):
    print(
        f"Тестирование: epochs={epochs}, batch_size={batch_size}, neurons={neurons}, activation={activation}, regularization={regularization}, optimizer={optimizer}, learning_rate={learning_rate}")

    # Модель
    model = keras.Sequential()

    # Скрытые слои
    model.add(layers.Dense(neurons[0], activation=activation, input_shape=(X_train.shape[1],)))
    if regularization == 'dropout':
        model.add(layers.Dropout(0.2))

    model.add(layers.Dense(neurons[1], activation=activation))
    if regularization == 'dropout':
        model.add(layers.Dropout(0.2))

    model.add(layers.Dense(neurons[2], activation=activation))
    if regularization == 'dropout':
        model.add(layers.Dropout(0.2))

    # Выходной слой
    model.add(layers.Dense(1))

    # Регуляризация L2
    if regularization == 'l2':
        for layer in model.layers:
            if isinstance(layer, layers.Dense):
                layer.kernel_regularizer = regularizers.l2(0.01)

    # Оптимизатор
    if optimizer == 'adam':
        optimizer_instance = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'nadam':
        optimizer_instance = keras.optimizers.Nadam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer_instance, loss="mse", metrics=["mae"])

    # Обучение модели
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),
                        verbose=0)

    # Предсказания
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return mse, r2, history


# Тестирование всех параметров
results = []

for epochs in epochs_options:
    for batch_size in batch_size_options:
        for neurons in neurons_options:
            for activation in activation_options:
                for regularization in regularization_options:
                    for optimizer in optimizer_options:
                        for learning_rate in learning_rate_options:
                            mse, r2, history = run_model(epochs, batch_size, neurons, activation, regularization,
                                                         optimizer, learning_rate)
                            results.append((epochs, batch_size, neurons, activation, regularization, optimizer,
                                            learning_rate, mse, r2))

# Сортировка результатов по MSE (по возрастанию)
results_sorted = sorted(results, key=lambda x: x[7])

# Вывод лучших результатов
print("\nЛучшие результаты:")
for result in results_sorted[:5]:
    print(
        f"Epochs: {result[0]}, Batch Size: {result[1]}, Neurons: {result[2]}, Activation: {result[3]}, Regularization: {result[4]}, Optimizer: {result[5]}, Learning Rate: {result[6]}, MSE: {result[7]:.4f}, R²: {result[8]:.4f}")

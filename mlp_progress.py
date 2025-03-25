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
from tqdm import tqdm  # Импортируем tqdm для прогресс-бара

# Загружаем данные
dataset = pd.read_csv("LST_final_TRUE.csv")

features = ["H", "TWI", "Aspect", "Hillshade", "Roughness", "Slope",
            "Temperature_merra_1000hpa", "Time", "DayOfYear", "X", "Y"]
target = "T_rp5"


y = dataset[target].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

epochs_options = [100, 200, 300]
batch_size_options = [32, 64, 128]
neurons_options = [(128, 64, 32)]
activation_options = ['relu', 'elu']
regularization_options = [None, 'dropout']
optimizer_options = ['adam']
learning_rate_options = [0.001]


def run_model(epochs, batch_size, neurons, activation, regularization, optimizer, learning_rate):
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

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),
                        verbose=0)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return mse, r2, history


results = []

combinations = (
        len(epochs_options) *
        len(batch_size_options) *
        len(neurons_options) *
        len(activation_options) *
        len(regularization_options) *
        len(optimizer_options) *
        len(learning_rate_options)
)

with tqdm(total=combinations) as pbar:
    for epochs in epochs_options:
        for batch_size in batch_size_options:
            for neurons in neurons_options:
                for activation in activation_options:
                    for regularization in regularization_options:
                        for optimizer in optimizer_options:
                            for learning_rate in learning_rate_options:
                                print(f"\nЗапуск комбинации: Эпохи={epochs}, Пакет={batch_size}, Нейроны={neurons}, "
                                      f"Активация={activation}, Регуляризация={regularization}, Оптимизатор={optimizer}, "
                                      f"Скорость обучения={learning_rate}")

                                mse, r2, history = run_model(epochs, batch_size, neurons, activation, regularization,
                                                             optimizer, learning_rate)

                                results.append((epochs, batch_size, neurons, activation, regularization, optimizer,
                                                learning_rate, mse, r2))

                                pbar.update(1)


df_results = pd.DataFrame(results,
                          columns=["Epochs", "Batch Size", "Neurons", "Activation", "Regularization", "Optimizer",
                                   "Learning Rate", "MSE", "R²"])

df_results_sorted = df_results.sort_values(by="MSE", ascending=True)

print("\nЛучшие результаты:")
print(df_results_sorted.head())

df_results_sorted.to_csv("model_results.csv", index=False)

# Визуализация: MSE vs Epochs и Batch Size
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_results_sorted['Epochs'], y=df_results_sorted['MSE'], hue=df_results_sorted['Batch Size'],
                palette='coolwarm', size=df_results_sorted['Learning Rate'], sizes=(20, 200))
plt.title("MSE vs Epochs, Batch Size and Learning Rate")
plt.xlabel("Количество эпох")
plt.ylabel("MSE")
plt.show()

# Визуализация: R² vs Epochs и Batch Size
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_results_sorted['Epochs'], y=df_results_sorted['R²'], hue=df_results_sorted['Batch Size'],
                palette='coolwarm', size=df_results_sorted['Learning Rate'], sizes=(20, 200))
plt.title("R² vs Epochs, Batch Size and Learning Rate")
plt.xlabel("Количество эпох")
plt.ylabel("R²")
plt.show()

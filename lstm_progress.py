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

X = dataset[features].values
y = dataset[target].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Для LSTM нам нужно будет reshape X для подачи на вход сети (формат [samples, timesteps, features])
# В данном случае для простоты мы считаем, что каждый "sample" имеет 1 timestep (будем использовать 1D LSTM).
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Определяем параметры модели
epochs_options = [100, 200, 300]  # Увеличено количество эпох
batch_size_options = [32, 64, 128]  # Добавлен размер пакета 128
neurons_options = [64, 128, 256]  # Число нейронов для LSTM
activation_options = ['relu', 'elu']  # Добавлена активация tanh
regularization_options = [None, 'dropout']  # Регуляризация оставлена как есть, можно добавить dropout 0.3
optimizer_options = ['adam']
learning_rate_options = [0.001]

# Функция для обучения модели и получения метрик
def run_lstm_model(epochs, batch_size, neurons, activation, regularization, optimizer, learning_rate):
    model = keras.Sequential()

    model.add(layers.LSTM(neurons, activation=activation, input_shape=(X_train.shape[1], X_train.shape[2])))
    if regularization == 'dropout':
        model.add(layers.Dropout(0.3))  # Увеличен dropout до 0.3

    model.add(layers.Dense(1))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),
                        verbose=0)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return mse, r2, history

# Массив для записи результатов
results_lstm = []

# Создание прогресс-бара
combinations = (
        len(epochs_options) *
        len(batch_size_options) *
        len(neurons_options) *
        len(activation_options) *
        len(regularization_options) *
        len(optimizer_options) *
        len(learning_rate_options)
)

# Тестирование параметров LSTM с прогресс-баром
with tqdm(total=combinations) as pbar:
    for epochs in epochs_options:
        for batch_size in batch_size_options:
            for neurons in neurons_options:
                for activation in activation_options:
                    for regularization in regularization_options:
                        for optimizer in optimizer_options:
                            for learning_rate in learning_rate_options:
                                # Печатаем информацию о текущей комбинации
                                print(f"\nЗапуск комбинации: Эпохи={epochs}, Пакет={batch_size}, Нейроны={neurons}, "
                                      f"Активация={activation}, Регуляризация={regularization}, Оптимизатор={optimizer}, "
                                      f"Скорость обучения={learning_rate}")

                                # Запуск модели
                                mse, r2, history = run_lstm_model(epochs, batch_size, neurons, activation, regularization,
                                                                   optimizer, learning_rate)

                                # Записываем результаты
                                results_lstm.append((epochs, batch_size, neurons, activation, regularization, optimizer,
                                                     learning_rate, mse, r2))

                                # Обновляем прогресс-бар
                                pbar.update(1)

# Создаем DataFrame для результатов LSTM
df_results_lstm = pd.DataFrame(results_lstm,
                                columns=["Epochs", "Batch Size", "Neurons", "Activation", "Regularization", "Optimizer",
                                         "Learning Rate", "MSE", "R²"])

# Сортировка по MSE
df_results_lstm_sorted = df_results_lstm.sort_values(by="MSE", ascending=True)

# Выводим таблицу с лучшими результатами
print("\nЛучшие результаты для LSTM:")
print(df_results_lstm_sorted.head())

# Сохраняем результаты в файл
df_results_lstm_sorted.to_csv("lstm_model_results.csv", index=False)

# Визуализация: MSE vs Epochs и Batch Size
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_results_lstm_sorted['Epochs'], y=df_results_lstm_sorted['MSE'], hue=df_results_lstm_sorted['Batch Size'],
                palette='coolwarm', size=df_results_lstm_sorted['Learning Rate'], sizes=(20, 200))
plt.title("MSE vs Epochs, Batch Size and Learning Rate for LSTM")
plt.xlabel("Количество эпох")
plt.ylabel("MSE")
plt.show()

# Визуализация: R² vs Epochs и Batch Size
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_results_lstm_sorted['Epochs'], y=df_results_lstm_sorted['R²'], hue=df_results_lstm_sorted['Batch Size'],
                palette='coolwarm', size=df_results_lstm_sorted['Learning Rate'], sizes=(20, 200))
plt.title("R² vs Epochs, Batch Size and Learning Rate for LSTM")
plt.xlabel("Количество эпох")
plt.ylabel("R²")
plt.show()

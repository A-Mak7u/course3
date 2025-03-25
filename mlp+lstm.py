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

# Загружаем данные
dataset = pd.read_csv("LST_final_TRUE.csv")

# Определяем признаки
geo_features = ["H", "TWI", "Aspect", "Hillshade", "Roughness", "Slope", "X", "Y"]
time_features = ["Temperature_merra_1000hpa", "Time", "DayOfYear"]
target = "T_rp5"

# Разделяем признаки
X_geo = dataset[geo_features].values  # Географические данные
X_time = dataset[time_features].values  # Временные данные
y = dataset[target].values

# Нормализация
scaler_geo = StandardScaler()
scaler_time = StandardScaler()
X_geo_scaled = scaler_geo.fit_transform(X_geo)
X_time_scaled = scaler_time.fit_transform(X_time)

# Формируем 3D-массив для LSTM (samples, time_steps, features)
time_steps = 5  # Увеличили количество временных шагов
X_time_lstm = np.array([X_time_scaled[i-time_steps:i] for i in range(time_steps, len(X_time_scaled))])
X_geo_final = X_geo_scaled[time_steps:]  # Синхронизация размерности
y_final = y[time_steps:]  # Целевая переменная

# Разделяем на обучающую и тестовую выборки
X_geo_train, X_geo_test, X_time_train, X_time_test, y_train, y_test = train_test_split(
    X_geo_final, X_time_lstm, y_final, test_size=0.2, random_state=42
)

# Создание LSTM-модели
lstm_input = keras.Input(shape=(time_steps, len(time_features)))
lstm_out = layers.LSTM(128, return_sequences=True)(lstm_input)
lstm_out = layers.LSTM(64, return_sequences=True)(lstm_out)
lstm_out = layers.LSTM(32)(lstm_out)
lstm_out = layers.Dropout(0.3)(lstm_out)

# Создание MLP-модели
mlp_input = keras.Input(shape=(len(geo_features),))
mlp_out = layers.Dense(128)(mlp_input)
mlp_out = layers.LeakyReLU()(mlp_out)  # Заменили ReLU на LeakyReLU
mlp_out = layers.BatchNormalization()(mlp_out)
mlp_out = layers.Dense(64, activation="relu")(mlp_out)
mlp_out = layers.Dropout(0.3)(mlp_out)

# Объединение потоков данных
merged = layers.concatenate([lstm_out, mlp_out])
merged = layers.Dense(64, activation="relu")(merged)
merged = layers.Dense(1)(merged)  # Выходной слой

# Создание общей модели
ensemble_model = keras.Model(inputs=[lstm_input, mlp_input], outputs=merged)

# Компиляция модели
ensemble_model.compile(optimizer="RMSprop", loss="mse", metrics=["mae"])

# Обучение модели
history = ensemble_model.fit(
    [X_time_train, X_geo_train], y_train,
    epochs=200, batch_size=32, validation_data=([X_time_test, X_geo_test], y_test), verbose=1
)

# Оценка модели
predictions = ensemble_model.predict([X_time_test, X_geo_test])
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Визуализация процесса обучения
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Эпохи")
plt.ylabel("MSE")
plt.title("График потерь улучшенной ансамблевой модели LSTM + MLP")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel("Эпохи")
plt.ylabel("MAE")
plt.title("График MAE улучшенной ансамблевой модели")
plt.legend()
plt.show()

# Визуализация предсказанных значений vs реальных
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=predictions.flatten(), alpha=0.5)
plt.xlabel("Реальная температура")
plt.ylabel("Предсказанная температура")
plt.title("Сравнение предсказанных и реальных значений (LSTM + MLP)")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')
plt.show()

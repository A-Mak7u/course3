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

dataset = pd.read_csv("LST_final_TRUE.csv")

geo_features = ["H", "TWI", "Aspect", "Hillshade", "Roughness", "Slope", "X", "Y"]
time_features = ["Temperature_merra_1000hpa", "Time", "DayOfYear"]
target = "T_rp5"

X_geo = dataset[geo_features].values  # географические данные
X_time = dataset[time_features].values  # временные данные
y = dataset[target].values

scaler_geo = StandardScaler()
scaler_time = StandardScaler()
X_geo_scaled = scaler_geo.fit_transform(X_geo)
X_time_scaled = scaler_time.fit_transform(X_time)

time_steps = 3
X_time_lstm = np.array([X_time_scaled[i-time_steps:i] for i in range(time_steps, len(X_time_scaled))])
X_geo_final = X_geo_scaled[time_steps:]
y_final = y[time_steps:]

X_geo_train, X_geo_test, X_time_train, X_time_test, y_train, y_test = train_test_split(
    X_geo_final, X_time_lstm, y_final, test_size=0.2, random_state=42
)

lstm_input = keras.Input(shape=(time_steps, len(time_features)))
lstm_output = layers.LSTM(64, activation="relu")(lstm_input)

mlp_input = keras.Input(shape=(len(geo_features),))
mlp_output = layers.Dense(64, activation="relu")(mlp_input)
mlp_output = layers.Dense(32, activation="relu")(mlp_output)

merged = layers.concatenate([lstm_output, mlp_output])
merged = layers.Dense(32, activation="relu")(merged)
merged = layers.Dense(1)(merged)  # Выходной слой

hybrid_model = keras.Model(inputs=[lstm_input, mlp_input], outputs=merged)

hybrid_model.compile(optimizer="adam", loss="mse", metrics=["mae"])

history = hybrid_model.fit(
    [X_time_train, X_geo_train], y_train,
    epochs=500, batch_size=32, validation_data=([X_time_test, X_geo_test], y_test), verbose=1
)

predictions = hybrid_model.predict([X_time_test, X_geo_test])
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Эпохи")
plt.ylabel("MSE")
plt.title("График потерь модели")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel("Эпохи")
plt.ylabel("MAE")
plt.title("График MAE модели")
plt.legend()
plt.show()

plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=predictions.flatten(), alpha=0.5)
plt.xlabel("Реальная температура")
plt.ylabel("Предсказанная температура")
plt.title("Сравнение предсказанных и реальных значений")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')
plt.show()

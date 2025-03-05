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

features = ["H", "TWI", "Aspect", "Hillshade", "Roughness", "Slope",
            "Temperature_merra_1000hpa", "Time", "DayOfYear", "X", "Y"]
target = "T_rp5"

X = dataset[features].values
y = dataset[target].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

lstm_model = keras.Sequential([
    layers.LSTM(64, activation="relu", return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    layers.LSTM(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1)
])

lstm_model.compile(optimizer="adam", loss="mse", metrics=["mae"])

history = lstm_model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_test, y_test), verbose=1)

predictions = lstm_model.predict(X_test)
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
plt.title("График потерь модели LSTM")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel("Эпохи")
plt.ylabel("MAE")
plt.title("График MAE модели LSTM")
plt.legend()
plt.show()

plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=predictions.flatten(), alpha=0.5)
plt.xlabel("Реальная температура")
plt.ylabel("Предсказанная температура")
plt.title("Сравнение предсказанных и реальных значений (LSTM)")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')
plt.show()

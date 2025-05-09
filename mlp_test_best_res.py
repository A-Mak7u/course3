import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 🔒 Фиксация сидов
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# 📥 Загрузка данных
dataset = pd.read_csv("LST_final_TRUE.csv")
features = ["H", "TWI", "Aspect", "Hillshade", "Roughness", "Slope",
            "Temperature_merra_1000hpa", "Time", "DayOfYear", "X", "Y"]
target = "T_rp5"

X = dataset[features].values
y = dataset[target].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Лучшие гиперпараметры
epochs = 300
batch_size = 32
neurons = (128, 64, 32)
activation = 'elu'
regularization = ''
optimizer = 'adam'
learning_rate = 0.001

# 🔁 Функция модели с расширенными метриками
def run_model(epochs, batch_size, neurons, activation, regularization, optimizer, learning_rate):
    model = keras.Sequential()
    model.add(layers.Dense(neurons[0], activation=activation, input_shape=(X_train.shape[1],)))
    if regularization == 'dropout':
        model.add(layers.Dropout(0.2))
    model.add(layers.Dense(neurons[1], activation=activation))
    if regularization == 'dropout':
        model.add(layers.Dropout(0.2))
    model.add(layers.Dense(neurons[2], activation=activation))
    if regularization == 'dropout':
        model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1))

    if optimizer == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        raise ValueError("Неизвестный оптимизатор")

    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_test, y_test), verbose=1)

    predictions = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    r2 = r2_score(y_test, predictions)

    return mse, rmse, mae, mape, r2, history

# 🚀 Запуск
mse, rmse, mae, mape, r2, history = run_model(epochs, batch_size, neurons, activation, regularization, optimizer, learning_rate)

# 📋 Вывод метрик
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R²: {r2:.4f}")

# 💾 Сохранение метрик
results_df = pd.DataFrame([{
    "Epochs": epochs,
    "Batch Size": batch_size,
    "Neurons": str(neurons),
    "Activation": activation,
    "Regularization": regularization,
    "Optimizer": optimizer,
    "Learning Rate": learning_rate,
    "MSE": mse,
    "RMSE": rmse,
    "MAE": mae,
    "MAPE": mape,
    "R²": r2
}])
results_df.to_csv("best_model_metrics.csv", index=False)

# 📈 Графики: MSE vs Epochs
plt.figure(figsize=(10, 6))
sns.lineplot(x=np.arange(1, epochs+1), y=history.history['loss'], label='MSE (Train)')
sns.lineplot(x=np.arange(1, epochs+1), y=history.history['val_loss'], label='MSE (Val)')
plt.title("MSE по эпохам")
plt.xlabel("Эпохи")
plt.ylabel("MSE")
plt.legend()
plt.show()

# 📈 Графики: MAE vs Epochs
plt.figure(figsize=(10, 6))
sns.lineplot(x=np.arange(1, epochs+1), y=history.history['mae'], label='MAE (Train)')
sns.lineplot(x=np.arange(1, epochs+1), y=history.history['val_mae'], label='MAE (Val)')
plt.title("MAE по эпохам")
plt.xlabel("Эпохи")
plt.ylabel("MAE")
plt.legend()
plt.show()


# Epochs,Batch Size,Neurons,Activation,Regularization,Optimizer,Learning Rate,MSE,RMSE,MAE,MAPE,R²
# 300,32,"(128, 64, 32)",elu,,adam,0.001,7.188101821599612,2.6810635616485508,2.011768555766132,11.193297449022605,0.8093724614557449


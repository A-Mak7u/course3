import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from tensorflow.keras.mixed_precision import set_global_policy
import concurrent.futures
from tqdm import tqdm
from datetime import datetime
import itertools
import random

# ------------------------
# CONFIG
# ------------------------
set_global_policy('mixed_float16')
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(8)

log_dir_base = "tensorboard_logs_lstm"
os.makedirs(log_dir_base, exist_ok=True)

# ------------------------
# LOAD DATA
# ------------------------
dataset = pd.read_csv("LST_final_TRUE.csv")
features = ["H", "TWI", "Aspect", "Hillshade", "Roughness", "Slope",
            "Temperature_merra_1000hpa", "Time", "DayOfYear", "X", "Y"]
target = "T_rp5"

X = dataset[features].values
y = dataset[target].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# LSTM требует 3D input: (samples, timesteps, features)
X_lstm = np.expand_dims(X_scaled, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_lstm, y, test_size=0.2, random_state=42)

# ------------------------
# GRID SEARCH CONFIG
# ------------------------
epochs_options = [100, 200]
batch_size_options = [32, 64]
neurons_options = [
    (64,),
    (128,),
    (64, 64),
    (128, 64)
]
activation_options = ['tanh', 'relu']
dropout_options = [0.0, 0.2]
batch_norm_options = [True, False]
l2_reg_options = [0.0, 1e-4]
learning_rate_options = [0.001, 0.0005]

all_combinations = list(itertools.product(
    epochs_options,
    batch_size_options,
    activation_options,
    neurons_options,
    dropout_options,
    batch_norm_options,
    l2_reg_options,
    learning_rate_options
))


print(f"Всего комбинаций: {len(all_combinations)}")
sampled_combinations = all_combinations  # полный перебор всех комбинаций

# ------------------------
# MODEL TRAINING
# ------------------------
def run_lstm_model(epochs, batch_size, activation, neurons, dropout, batch_norm, l2_reg, learning_rate):
    model = keras.Sequential()

    for i, units in enumerate(neurons):
        return_seq = i < len(neurons) - 1
        if i == 0:
            model.add(layers.LSTM(units,
                                  activation=activation,
                                  return_sequences=return_seq,
                                  kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None,
                                  input_shape=(X_train.shape[1], X_train.shape[2])))
        else:
            model.add(layers.LSTM(units,
                                  activation=activation,
                                  return_sequences=return_seq,
                                  kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None))

        if batch_norm:
            model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout))

    model.add(layers.Dense(1))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    log_dir = os.path.join(log_dir_base, datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
    early_stop_cb = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[tensorboard_cb, early_stop_cb],
                        verbose=0)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    return mse, rmse, mae, mape, r2


# ------------------------
# PARALLEL EXECUTION
# ------------------------
results = []


def process(params):
    e, b, a, n, d, bn, l2, lr = params
    metrics = run_lstm_model(e, b, a, n, d, bn, l2, lr)
    return (e, b, a, n, d, bn, l2, lr, *metrics)


with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(process, p) for p in sampled_combinations]

    # tqdm для отслеживания прогресса
    for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Обучение моделей"):
        results.append(f.result())


# ------------------------
# SAVE RESULTS
# ------------------------
df_results = pd.DataFrame(results, columns=[
    "Epochs", "Batch Size", "Activation", "Neurons", "Dropout",
    "BatchNorm", "L2", "Learning Rate", "MSE", "RMSE", "MAE", "MAPE", "R²"
])
df_results.sort_values(by="MSE", inplace=True)
df_results.to_csv("lstm_grid_results(full).csv", index=False)

# ------------------------
# VISUALIZATION
# ------------------------
metrics = ["MSE", "RMSE", "MAE", "MAPE", "R²"]
fig, axs = plt.subplots(3, 2, figsize=(14, 12))
axs = axs.flatten()
for i, metric in enumerate(metrics):
    sns.boxplot(x="Activation", y=metric, data=df_results, ax=axs[i])
    axs[i].set_title(f"{metric} by Activation")
axs[-1].axis('off')
plt.tight_layout()
plt.show()

# Correlation between metrics
sns.scatterplot(x="MSE", y="R²", data=df_results)
plt.title("MSE vs R²")
plt.show()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from tensorflow.keras.mixed_precision import set_global_policy
import concurrent.futures
from tqdm import tqdm
from datetime import datetime
import random
import itertools

set_global_policy('mixed_float16')
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(8)

log_dir_base = "tensorboard_logs"
os.makedirs(log_dir_base, exist_ok=True)

dataset = pd.read_csv("LST_final_TRUE.csv")
features = ["H", "TWI", "Aspect", "Hillshade", "Roughness", "Slope",
            "Temperature_merra_1000hpa", "Time", "DayOfYear", "X", "Y"]
target = "T_rp5"

X = dataset[features].values
y = dataset[target].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

epochs_options = [200, 300, 400]
batch_size_options = [32, 64, 128]
neurons_options = [
    (64, 64),
    (128, 128),
    (64, 64, 32),
    (128, 64),
    (64, 32, 16)
]
activation_options = ['elu', 'relu', 'tanh', 'selu', 'leaky_relu']
dropout_options = [0.0, 0.1, 0.2, 0.3]
learning_rate_options = [0.001, 0.0005, 0.0001]
batch_norm_options = [True, False]
l2_reg_options = [0.0, 1e-4, 1e-3]

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

sampled_combinations = random.sample(all_combinations, min(100, len(all_combinations)))

def run_model(epochs, batch_size, activation, neurons, dropout_rate, batch_norm, l2_reg, learning_rate):
    model = keras.Sequential()

    for units in neurons:
        # Выбор активации
        if activation == 'leaky_relu':
            model.add(layers.Dense(units, kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None))
            model.add(layers.LeakyReLU())
        else:
            model.add(layers.Dense(units, activation=activation,
                                   kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None))

        if batch_norm:
            model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    log_dir = os.path.join(log_dir_base, datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
    early_stop_cb = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=epochs, batch_size=batch_size,
                        callbacks=[early_stop_cb, tensorboard_cb],
                        verbose=0)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return mse, rmse, mae, mape, r2


results = []

def process_combination(params):
    e, b, a, n, d, bn, l2, lr = params
    metrics = run_model(e, b, a, n, d, bn, l2, lr)
    return (e, b, a, n, d, bn, l2, lr, *metrics)

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_combination, params) for params in sampled_combinations]
    for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        results.append(f.result())

df_results = pd.DataFrame(results, columns=["Epochs", "Batch Size", "Activation", "Neurons",
                                            "Dropout", "BatchNorm", "L2", "Learning Rate",
                                            "MSE", "RMSE", "MAE", "MAPE", "R²"])
df_results.sort_values(by="MSE", inplace=True)
df_results.to_csv("mlp_res_10april_extended.csv", index=False)

metrics = ["MSE", "RMSE", "MAE", "MAPE", "R²"]
fig, axs = plt.subplots(3, 2, figsize=(14, 12))
axs = axs.flatten()
for i, metric in enumerate(metrics):
    sns.boxplot(x="Activation", y=metric, data=df_results, ax=axs[i])
    axs[i].set_title(f"{metric} by Activation")
axs[-1].axis('off')
plt.tight_layout()
plt.show()

sns.scatterplot(x="MSE", y="R²", data=df_results)
plt.title("MSE vs R²")
plt.show()

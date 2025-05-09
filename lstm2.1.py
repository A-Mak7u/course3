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
import itertools
from multiprocessing import Pool, Manager
import time

# ------------------------
# CONFIG
# ------------------------
# Использование mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')  # Включение mixed precision (ускоряет работу на GPU)
tf.config.threading.set_intra_op_parallelism_threads(4)  # Оптимизация потоков CPU
tf.config.threading.set_inter_op_parallelism_threads(8)

# Убедитесь, что доступны GPU устройства
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"GPU детектировано: {len(physical_devices)} устройства(е).")
    # Ограничиваем использование только одного устройства, если их несколько
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    # Устанавливаем динамическое выделение памяти для GPU
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)

else:
    print("GPU не найдено, используется только CPU.")

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

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ------------------------
# RANDOM HYPERPARAM SEARCH CONFIG
# ------------------------
epochs_options = [50, 100]
batch_size_options = [32, 64]
activation_options = ['relu', 'elu']
neurons_options = [(64, 32), (128, 64)]
dropout_options = [0.0, 0.1]
learning_rate_options = [0.001, 0.0005]
batch_norm_options = [False, True]
l2_reg_options = [0.0, 0.01]

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

sampled_combinations = all_combinations


# ------------------------
# LSTM MODEL DEFINITION
# ------------------------
def run_model(epochs, batch_size, activation, neurons, dropout_rate, batch_norm, l2_reg, learning_rate):
    try:
        print(f"Запуск модели с параметрами: epochs={epochs}, batch_size={batch_size}, activation={activation}, "
              f"neurons={neurons}, dropout={dropout_rate}, batch_norm={batch_norm}, l2={l2_reg}, lr={learning_rate}",
              flush=True)

        model = keras.Sequential()
        model.add(layers.LSTM(128, input_shape=(X_train.shape[1], 1), return_sequences=True))
        model.add(layers.LSTM(64, return_sequences=False))

        for units in neurons:
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

        early_stop_cb = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                            epochs=epochs, batch_size=batch_size,
                            callbacks=[early_stop_cb],
                            verbose=1)

        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f"Готово: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%, R2={r2:.4f}", flush=True)

        return mse, rmse, mae, mape, r2

    except Exception as e:
        print(
            f"Ошибка при обучении модели с параметрами epochs={epochs}, batch_size={batch_size}, activation={activation}: {e}")
        return None


# ------------------------
# PARALLEL EXECUTION
# ------------------------
def process_combination(params, results):
    e, b, a, n, d, bn, l2, lr = params
    metrics = run_model(e, b, a, n, d, bn, l2, lr)
    if metrics is not None:
        results.append((e, b, a, n, d, bn, l2, lr, *metrics))
    else:
        print(f"Комбинация не удалась: {params}")


if __name__ == "__main__":
    start_time = time.time()

    print(f"Старт параллельного обучения ({len(sampled_combinations)} комбинаций)...", flush=True)

    # Используем Manager для безопасной работы с памятью между процессами
    with Manager() as manager:
        results = manager.list()  # Список для хранения результатов
        with Pool(processes=2) as pool:  # 2 процесса — оптимально для одной GPU
            pool.starmap(process_combination, [(params, results) for params in sampled_combinations])

        # Преобразуем результаты в DataFrame после завершения всех процессов
        if results:
            df_results = pd.DataFrame(list(results), columns=["Epochs", "Batch Size", "Activation", "Neurons",
                                                              "Dropout", "BatchNorm", "L2", "Learning Rate",
                                                              "MSE", "RMSE", "MAE", "MAPE", "R²"])

            duration = time.time() - start_time
            print(f"\n✅ Обучение завершено за {duration / 60:.2f} минут.")

            # Сортируем по MSE и сохраняем результат
            df_results.sort_values(by="MSE", inplace=True)
            df_results.to_csv("lstm2(full)2.csv", index=False)

            # ------------------------
            # Визуализация метрик с помощью боксплотов
            # ------------------------
            metrics = ["MSE", "RMSE", "MAE", "MAPE", "R²"]
            fig, axs = plt.subplots(3, 2, figsize=(14, 12))
            axs = axs.flatten()
            for i, metric in enumerate(metrics):
                sns.boxplot(x="Activation", y=metric, data=df_results, ax=axs[i])
                axs[i].set_title(f"{metric} по активации")
            axs[-1].axis('off')  # Убираем пустой график
            plt.tight_layout()
            plt.show()

            # ------------------------
            # График зависимости MSE от R² для выявления корреляции
            # ------------------------
            sns.scatterplot(x="MSE", y="R²", data=df_results)
            plt.title("Зависимость MSE от R²")
            plt.xlabel('MSE')
            plt.ylabel('R²')
            plt.show()
        else:
            print("Нет успешных результатов для анализа.")

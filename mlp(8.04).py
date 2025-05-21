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
from tensorflow.keras import layers, callbacks
from tensorflow.keras.mixed_precision import set_global_policy
import concurrent.futures
from tqdm import tqdm
from datetime import datetime

set_global_policy('mixed_float16')

tf.config.threading.set_intra_op_parallelism_threads(12)
tf.config.threading.set_inter_op_parallelism_threads(12)

dataset = pd.read_csv("LST_final_TRUE.csv")
features = ["H", "TWI", "Aspect", "Hillshade", "Roughness", "Slope",
            "Temperature_merra_1000hpa", "Time", "DayOfYear", "X", "Y"]
target = "T_rp5"

X = dataset[features].values
y = dataset[target].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

epochs_options = [100, 150]
batch_size_options = [32, 64]
neurons_options = [(128, 64, 32), (256, 128, 64)]
activation_options = ['relu', 'elu']
dropout_options = [0.0, 0.3]
learning_rate_options = [0.001, 0.0005]
batch_norm_options = [True, False]

log_dir_base = "tensorboard_logs"
os.makedirs(log_dir_base, exist_ok=True)

def run_model(epochs, batch_size, neurons, activation, dropout_rate, learning_rate, batch_norm):
    model = keras.Sequential()

    for i, units in enumerate(neurons):
        model.add(layers.Dense(units, activation=activation))
        if batch_norm:
            model.add(layers.BatchNormalization())
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    log_dir = os.path.join(log_dir_base, datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stop_cb = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

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
param_combinations = [(e, b, n, a, d, l, bn)
                      for e in epochs_options
                      for b in batch_size_options
                      for n in neurons_options
                      for a in activation_options
                      for d in dropout_options
                      for l in learning_rate_options
                      for bn in batch_norm_options]

def process_combination(params):
    e, b, n, a, d, l, bn = params
    metrics = run_model(e, b, n, a, d, l, bn)
    return (e, b, n, a, d, l, bn, *metrics)

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_combination, params) for params in param_combinations]
    for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        results.append(f.result())

df_results = pd.DataFrame(results, columns=["Epochs", "Batch Size", "Neurons", "Activation",
                                            "Dropout", "Learning Rate", "BatchNorm",
                                            "MSE", "RMSE", "MAE", "MAPE", "R²"])
df_results.sort_values(by="MSE", inplace=True)
df_results.to_csv("optimized_mlp_results.csv", index=False)

metrics = ["MSE", "RMSE", "MAE", "MAPE", "R²"]
fig, axs = plt.subplots(3, 2, figsize=(14, 12))
axs = axs.flatten()
for i, metric in enumerate(metrics):
    sns.boxplot(x="Activation", y=metric, data=df_results, ax=axs[i])
    axs[i].set_title(f"{metric} by Activation")
axs[-1].axis('off')
plt.tight_layout()
plt.show()


    # ✅ Преимущества текущей реализации
    # 1. Параллельный перебор гиперпараметров
    # Используется ThreadPoolExecutor для ускорения перебора, что значительно сокращает общее время подбора параметров на CPU.
    #
    # 2. Поддержка GPU и mixed precision
    # Включена поддержка mixed_float16, что ускоряет обучение на GPU (особенно RTX 4060), снижая потребление памяти и увеличивая скорость операций с тензорами.
    #
    # 3. Расширенный набор метрик
    # Выводятся 5 метрик качества (MSE, RMSE, MAE, MAPE, R²), что позволяет всесторонне оценивать модель и не полагаться только на MSE.
    #
    # 4. EarlyStopping и BatchNormalization
    # EarlyStopping предотвращает переобучение и экономит время.
    #
    # BatchNormalization ускоряет сходимость модели и делает обучение более стабильным.
    #
    # 5. TensorBoard для отслеживания обучения
    # Подключён TensorBoard для визуального анализа логов: изменение лосса, метрик, распределения весов.
    #
    # 6. Гибкий и расширяемый код
    # Код легко модифицируется: можно добавлять новые слои, активации, метрики и пр.
    #
    # ⚠️ Недостатки текущей реализации
    # 1. Большое количество комбинаций гиперпараметров
    # Несмотря на параллельность, перебор всех возможных вариантов (например, 128+) занимает много времени (до десятков часов) — особенно при небольшом ускорении на CPU.
    #
    # 2. Используется ThreadPoolExecutor, а не ProcessPoolExecutor
    # Это может ограничивать производительность при тяжёлых задачах обучения, поскольку потоки делят память и ресурсы (вместо запуска отдельных процессов).
    #
    # 3. Нет умного отбора параметров (GridSearch только)
    # Используется полный перебор (grid search), а не random search или optuna, которые работают быстрее и часто дают сопоставимые результаты.
    #
    # 4. Нет логирования времени или мониторинга загрузки
    # Не сохраняется время выполнения моделей или информация о загрузке CPU/GPU, что может быть полезно при оптимизации.
    #
    # 💡 Заключение
    # Данный код — сбалансированное и гибкое решение для подбора гиперпараметров нейронной сети MLP с упором на стабильность, расширяемость и хорошую информативность метрик. Однако для ускорения и масштабирования на больших объёмах данных следует использовать более интеллектуальные методы отбора гиперпараметров и лучше задействовать ресурсы GPU.
    #   7%|▋         | 9/128 [1:08:12<5:52:10, 177.57s/it]
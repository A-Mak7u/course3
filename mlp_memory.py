import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import itertools
from tqdm import tqdm

# Фиксация random seed
tf.random.set_seed(42)
np.random.seed(42)

# Проверка GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"GPU detected: {physical_devices}")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("No GPU detected, using CPU.")

# Загрузка датасета
data = pd.read_csv('LST_final_TRUE.csv')

# Проверка уникальности признаков
print("Unique values in features:")
print(data[['X', 'Y', 'H', 'TWI', 'Aspect', 'Hillshade', 'Roughness', 'Slope']].nunique())

# Удаление выбросов
data = data[data['T_rp5'].between(data['T_rp5'].quantile(0.05), data['T_rp5'].quantile(0.95))]

# Сортировка по времени
data = data.sort_values(by=['DayOfYear', 'Time'])

# Разделение на признаки и целевую переменную
X = data.drop('T_rp5', axis=1)
y = data['T_rp5']

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разбиение на train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Параметры для grid search
epochs_options = [200, 300]
batch_size_options = [32, 64]
activation_options = ['elu', 'relu']
neurons_options = [(128, 64, 32), (128, 64)]
dropout_options = [0.0, 0.2]

combinations = list(
    itertools.product(epochs_options, batch_size_options, activation_options, neurons_options, dropout_options))

# Списки для хранения результатов
results = []


# Функция для создания модели
def create_model(neurons, activation, dropout):
    model = Sequential()
    model.add(Dense(neurons[0], activation=activation, input_shape=(X.shape[1],)))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(neurons[1], activation=activation))
    if dropout > 0:
        model.add(Dropout(dropout))
    if len(neurons) > 2:
        model.add(Dense(neurons[2], activation=activation))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


# Grid search
print("Starting grid search...")
for epochs, batch_size, activation, neurons, dropout in tqdm(combinations, desc="Grid Search Progress"):
    model = create_model(neurons, activation, dropout)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0,
        workers=4,
        use_multiprocessing=True
    )

    y_pred = model.predict(X_test, verbose=0).flatten()

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)

    results.append({
        'Epochs': epochs,
        'Batch Size': batch_size,
        'Activation': activation,
        'Neurons': neurons,
        'Dropout': dropout,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R²': r2
    })

# Сохранение результатов
df_results = pd.DataFrame(results)
df_results.sort_values(by='MSE', inplace=True)
df_results.to_csv('mlp_grid_search_results.csv', index=False)

# Вывод лучших результатов
print("\nBest Grid Search Results:")
print(df_results.head())

# График лучших результатов
import matplotlib.pyplot as plt

plt.scatter(df_results['MSE'], df_results['R²'], c=df_results['Epochs'], cmap='viridis')
plt.xlabel('MSE')
plt.ylabel('R²')
plt.title('Grid Search Results')
plt.colorbar(label='Epochs')
plt.show()


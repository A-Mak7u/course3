import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm
import tensorflow as tf

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

# Настройка TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# Списки для хранения метрик
mse_scores = []
rmse_scores = []
mae_scores = []
mape_scores = []
r2_scores = []


# Функция для создания модели
def create_model():
    model = Sequential()
    model.add(Dense(128, activation='elu', input_shape=(X.shape[1],)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='elu'))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


# Callback для EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Кросс-валидация
print("Starting cross-validation...")
for fold, (train_index, test_index) in tqdm(enumerate(tscv.split(X_scaled), 1), total=tscv.n_splits,
                                            desc="Fold Progress"):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Проверка данных фолда
    print(f"\nFold {fold}:")
    print(f"Train T_rp5 stats: {y_train.describe()}")
    print(f"Test T_rp5 stats: {y_test.describe()}")

    # Создание и обучение модели
    model = create_model()
    model.fit(
        X_train, y_train,
        epochs=300,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0,
        workers=4,
        use_multiprocessing=True
    )

    # Предсказание
    y_pred = model.predict(X_test, verbose=0).flatten()

    # Вычисление метрик
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)

    mse_scores.append(mse)
    rmse_scores.append(rmse)
    mae_scores.append(mae)
    mape_scores.append(mape)
    r2_scores.append(r2)

# Вывод результатов
print("\nCross-Validation Results (Final):")
print(f"Mean MSE: {np.mean(mse_scores):.3f} ± {np.std(mse_scores):.3f}")
print(f"Mean RMSE: {np.mean(rmse_scores):.3f} ± {np.std(rmse_scores):.3f}")
print(f"Mean MAE: {np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}")
print(f"Mean MAPE: {np.mean(mape_scores):.3f}% ± {np.std(mape_scores):.3f}%")
print(f"Mean R²: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
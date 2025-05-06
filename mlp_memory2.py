import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ProgbarLogger
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

# Удаление выбросов (10%)
data = data[data['T_rp5'].between(data['T_rp5'].quantile(0.1), data['T_rp5'].quantile(0.9))]

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

# Создание модели
def create_model():
    model = Sequential()
    model.add(Dense(128, activation='elu', input_shape=(X.shape[1],)))
    model.add(Dense(64, activation='elu'))
    model.add(Dense(32, activation='elu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Callback для EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Обучение модели
model = create_model()
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping, ProgbarLogger()],
    verbose=1,
    workers=4,
    use_multiprocessing=True
)

# Сохранение модели
model.save('mlp_final_model.h5')

# Предсказание
y_pred = model.predict(X_test, verbose=0).flatten()

# Вычисление метрик
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
r2 = r2_score(y_test, y_pred)

# Вывод результатов
print("\nMLP Single Split Results (Final):")
print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"MAPE: {mape:.3f}%")
print(f"R²: {r2:.3f}")

# График потерь
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
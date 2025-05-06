import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

# Загрузка датасета
data = pd.read_csv('LST_final_TRUE.csv')

# Разделение на признаки и целевую переменную
X = data.drop('T_rp5', axis=1)
y = data['T_rp5']

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Инициализация модели
rf = RandomForestRegressor(random_state=42)

# Настройка TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# Списки для хранения метрик
mse_scores = []
r2_scores = []

# Кросс-валидация
for train_index, test_index in tscv.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Обучение модели
    rf.fit(X_train, y_train)

    # Предсказание
    y_pred = rf.predict(X_test)

    # Вычисление метрик
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mse_scores.append(mse)
    r2_scores.append(r2)

# Вывод средних метрик
print(f'Mean MSE: {np.mean(mse_scores):.3f} ± {np.std(mse_scores):.3f}')
print(f'Mean R²: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}')
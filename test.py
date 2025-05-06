import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')  # Неинтерактивный бэкенд для сохранения графиков
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Создание директории для сохранения результатов
output_dir = "data_analysis_results"
os.makedirs(output_dir, exist_ok=True)

# Загрузка датасета
data = pd.read_csv('LST_final_TRUE.csv')

# Удаление выбросов (5%)
data = data[data['T_rp5'].between(data['T_rp5'].quantile(0.05), data['T_rp5'].quantile(0.95))]

# Сортировка по времени
data = data.sort_values(by=['DayOfYear', 'Time'])

# Разделение на признаки и целевую переменную
X = data.drop('T_rp5', axis=1)
y = data['T_rp5']

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. Распределение T_rp5 по дням
plt.figure(figsize=(10, 6))
plt.scatter(data['DayOfYear'], data['T_rp5'], alpha=0.5)
plt.xlabel('DayOfYear')
plt.ylabel('T_rp5')
plt.title('T_rp5 vs DayOfYear')
plt.savefig(os.path.join(output_dir, 't_rp5_vs_dayofyear.png'))
plt.close()
t_rp5_by_day = data.groupby('DayOfYear')['T_rp5'].describe()
t_rp5_by_day.to_csv(os.path.join(output_dir, 't_rp5_by_day_stats.csv'))

# 2. Boxplot для T_rp5
plt.figure(figsize=(8, 6))
plt.boxplot(data['T_rp5'])
plt.title('Boxplot of T_rp5')
plt.ylabel('T_rp5')
plt.savefig(os.path.join(output_dir, 't_rp5_boxplot.png'))
plt.close()
t_rp5_stats = data['T_rp5'].describe()
t_rp5_stats.to_csv(os.path.join(output_dir, 't_rp5_stats.csv'))

# 3. Значимость признаков
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
rf.fit(X_scaled, y)
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)

# Вывод результатов
print("1. T_rp5 vs DayOfYear stats saved to 't_rp5_by_day_stats.csv'")
print(t_rp5_by_day)
print("\n2. T_rp5 stats saved to 't_rp5_stats.csv'")
print(t_rp5_stats)
print("\n3. Feature importance saved to 'feature_importance.csv'")
print(feature_importance)

# График значимости признаков
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance (Random Forest)')
plt.savefig(os.path.join(output_dir, 'feature_importance_plot.png'))
plt.close()
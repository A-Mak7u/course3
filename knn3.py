import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

# ------------------------
# ЗАГРУЗКА ДАННЫХ
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
# СЕТКА ГИПЕРПАРАМЕТРОВ
# ------------------------
neighbor_options = [3, 5, 7, 10, 15, 20, 25, 30, 35, 40]
weights_options = ['uniform', 'distance']
leaf_size_options = [10, 20, 30, 40, 50]
p_options = [1, 2]  # Манхэттенское или евклидово расстояние
all_combinations = [(n, w, l, p) for n in neighbor_options for w in weights_options for l in leaf_size_options for p in p_options]

# ------------------------
# ОБУЧЕНИЕ МОДЕЛИ
# ------------------------
results = []

for n_neighbors, weights, leaf_size, p in all_combinations:
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, leaf_size=leaf_size, p=p, n_jobs=-1)
    knn.fit(X_train, y_train)

    predictions = knn.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    results.append((n_neighbors, weights, leaf_size, p, mse, rmse, mae, mape, r2))

# ------------------------
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ------------------------
df_results = pd.DataFrame(results, columns=["Количество соседей", "Тип весов", "Размер листа", "P", "MSE", "RMSE", "MAE", "MAPE", "R²"])
df_results.sort_values(by="MSE", inplace=True)
df_results.to_csv("knn_full_results.csv", index=False)

# ------------------------
# БОКСПЛОТЫ ДЛЯ МЕТРИК
# ------------------------
metrics = ["MSE", "RMSE", "MAE", "MAPE", "R²"]
fig, axs = plt.subplots(3, 2, figsize=(14, 12))
axs = axs.flatten()

for i, metric in enumerate(metrics):
    sns.boxplot(x="Тип весов", y=metric, data=df_results, ax=axs[i])
    axs[i].set_title(f"{metric} по типу весов")
axs[-1].axis('off')
plt.tight_layout()
plt.show()

# ------------------------
# РАСПРОСТРАНЕНИЕ ДЛЯ КАЖДОЙ МЕТРИКИ VS R²
# ------------------------
for metric in metrics:
    if metric != "R²":
        sns.scatterplot(x=metric, y="R²", hue="Тип весов", style="Тип весов", data=df_results, palette="Set2", s=100)
        plt.title(f"{metric} vs R² (KNN)")
        plt.grid(True)
        plt.show()

# ------------------------
# ПАРНЫЙ ГРАФИК МЕТРИК
# ------------------------
sns.pairplot(df_results[metrics], diag_kind="kde")
plt.suptitle("Парный график метрик оценки", y=1.02)
plt.show()

# ------------------------
# ТЕПЛОВАЯ КАРТА КОРРЕЛЯЦИИ ПАРАМЕТРОВ
# ------------------------
# убираем категориальный 'Тип весов'
df_numeric = df_results.drop(columns=['Тип весов'])
corr = df_numeric.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Тепловая карта корреляции: гиперпараметры и метрики")
plt.show()

# ------------------------
# БАРЧАРТ ТОП-КОНФИГУРАЦИЙ
# ------------------------
top_n = 10
plt.figure(figsize=(12, 6))
sns.barplot(
    x=df_results.head(top_n)["MSE"],
    y=[f'N={n}, W={w}, L={l}, P={p}' for n, w, l, p in zip(df_results.head(top_n)["Количество соседей"], df_results.head(top_n)["Тип весов"], df_results.head(top_n)["Размер листа"], df_results.head(top_n)["P"])],
    palette='viridis', hue=df_results.head(top_n)["Тип весов"]
)
plt.xlabel("MSE")
plt.title(f"Топ {top_n} конфигураций KNN по MSE")
plt.tight_layout()
plt.show()

# ------------------------
# ВИОЛИНОВОЙ ГРАФИК РАСПРЕДЕЛЕНИЯ МЕТРИК
# ------------------------
fig, axs = plt.subplots(3, 2, figsize=(14, 12))
axs = axs.flatten()

for i, metric in enumerate(metrics):
    sns.violinplot(y=metric, data=df_results, ax=axs[i])
    axs[i].set_title(f"Распределение {metric}")
axs[-1].axis('off')
plt.tight_layout()
plt.show()

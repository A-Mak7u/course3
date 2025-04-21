import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_absolute_percentage_error, confusion_matrix
from collections import Counter
from tqdm import tqdm  # Импортируем tqdm для прогресс-бара

# ------------------------
# СОБСТВЕННЫЙ КЛАСС KNN ДЛЯ КЛАССИФИКАЦИИ
# ------------------------

class KNNClassifier:
    def __init__(self, n_neighbors=3, weights='uniform', leaf_size=30, p=2):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.leaf_size = leaf_size
        self.p = p

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        # Расстояния от x до всех точек в обучающем наборе
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        # Сортируем индексы по расстоянию
        k_indices = np.argsort(distances)[:self.n_neighbors]
        # Получаем метки для k ближайших соседей
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Возвращаем наиболее часто встречающийся класс
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))


# ------------------------
# ЗАГРУЗКА ДАННЫХ
# ------------------------

dataset = pd.read_csv("LST_final_TRUE.csv")
features = ["H", "TWI", "Aspect", "Hillshade", "Roughness", "Slope",
            "Temperature_merra_1000hpa", "Time", "DayOfYear", "X", "Y"]
target = "T_rp5"

X = dataset[features].values
y = dataset[target].values

# Преобразуем целевую переменную в категориальную для классификации
threshold = np.median(y)  # Преобразуем на основе медианы
y = (y > threshold).astype(int)  # Если больше медианы - 1, иначе - 0

# Масштабируем данные
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

# Добавляем прогресс-бар с помощью tqdm
for n_neighbors, weights, leaf_size, p in tqdm(all_combinations, desc="Обучение модели", ncols=100):
    knn = KNNClassifier(n_neighbors=n_neighbors, weights=weights, leaf_size=leaf_size, p=p)
    knn.fit(X_train, y_train)

    predictions = knn.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)

    # Матрица ошибок
    cm = confusion_matrix(y_test, predictions)

    results.append((n_neighbors, weights, leaf_size, p, accuracy, mae, mape, cm))

# ------------------------
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ------------------------

df_results = pd.DataFrame(results, columns=["Количество соседей", "Тип весов", "Размер листа", "P", "Точность", "MAE", "MAPE", "Матрица ошибок"])
df_results.sort_values(by="Точность", ascending=False, inplace=True)
df_results.to_csv("knn_classification_results.csv", index=False)

# ------------------------
# БОКСПЛОТЫ ДЛЯ МЕТРИК
# ------------------------

metrics = ["Точность", "MAE", "MAPE"]
fig, axs = plt.subplots(2, 2, figsize=(14, 12))
axs = axs.flatten()

for i, metric in enumerate(metrics):
    sns.boxplot(x="Тип весов", y=metric, data=df_results, ax=axs[i])
    axs[i].set_title(f"{metric} по типу весов")
axs[-1].axis('off')
plt.tight_layout()
plt.show()

# ------------------------
# РАСПРОСТРАНЕНИЕ ДЛЯ КАЖДОЙ МЕТРИКИ VS Точность
# ------------------------

for metric in metrics:
    if metric != "Точность":
        sns.scatterplot(x=metric, y="Точность", hue="Тип весов", style="Тип весов", data=df_results, palette="Set2", s=100)
        plt.title(f"{metric} vs Точность (KNN)")
        plt.grid(True)
        plt.show()

# ------------------------
# ПАРНЫЙ ГРАФИК МЕТРИК
# ------------------------

sns.pairplot(df_results[metrics], diag_kind="kde")
plt.suptitle("Парный график метрик классификации", y=1.02)
plt.show()

# ------------------------
# ТЕПЛОВАЯ КАРТА КОРРЕЛЯЦИИ ПАРАМЕТРОВ
# ------------------------

# убираем категориальный 'Тип весов'
df_numeric = df_results.drop(columns=['Тип весов', 'Матрица ошибок'])
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
    x=df_results.head(top_n)["Точность"],
    y=[f'N={n}, W={w}, L={l}, P={p}' for n, w, l, p in zip(df_results.head(top_n)["Количество соседей"], df_results.head(top_n)["Тип весов"], df_results.head(top_n)["Размер листа"], df_results.head(top_n)["P"])],
    palette='viridis', hue=df_results.head(top_n)["Тип весов"]
)
plt.xlabel("Точность")
plt.title(f"Топ {top_n} конфигураций KNN по точности")
plt.tight_layout()
plt.show()

# ------------------------
# ВИОЛИНОВОЙ ГРАФИК РАСПРЕДЕЛЕНИЯ МЕТРИК
# ------------------------

fig, axs = plt.subplots(2, 2, figsize=(14, 12))
axs = axs.flatten()

for i, metric in enumerate(metrics):
    sns.violinplot(y=metric, data=df_results, ax=axs[i])
    axs[i].set_title(f"Распределение {metric}")
axs[-1].axis('off')
plt.tight_layout()
plt.show()

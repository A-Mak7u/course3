import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # Импортируем tqdm для прогресс-бара

# ------------------------
# СОБСТВЕННЫЙ КЛАСС KNN ДЛЯ КЛАССИФИКАЦИИ
# ------------------------

class KNNClassifier:
    def __init__(self, n_neighbors=3, weights='uniform', p=2):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p  # P=1 -> Манхэттенское расстояние, P=2 -> Евклидово расстояние

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in tqdm(X_test, desc="Прогнозирование", unit="объект")]
        return np.array(predictions)

    def _predict(self, x):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.n_neighbors]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        if self.weights == 'uniform':
            most_common = Counter(k_nearest_labels).most_common(1)
            return most_common[0][0]
        elif self.weights == 'distance':
            distances_k = [distances[i] for i in k_indices]
            weighted_labels = [label / dist if dist != 0 else label for label, dist in zip(k_nearest_labels, distances_k)]
            most_common = Counter(weighted_labels).most_common(1)
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
# ОБУЧЕНИЕ МОДЕЛИ
# ------------------------

knn = KNNClassifier(n_neighbors=5, weights='uniform', p=2)
knn.fit(X_train, y_train)

# Прогнозируем и оцениваем точность
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.4f}")

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
print("Матрица ошибок:\n", cm)

# Визуализация матрицы ошибок
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Низкая температура", "Высокая температура"], yticklabels=["Низкая температура", "Высокая температура"])
plt.title("Матрица ошибок (Confusion Matrix)")
plt.xlabel("Предсказание")
plt.ylabel("Истинные значения")
plt.show()

# ------------------------
# ВИЗУАЛИЗАЦИЯ РАСПРЕДЕЛЕНИЯ
# ------------------------

sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=y_pred, palette="Set1", style=y_pred, markers=["o", "s"])
plt.title("Распределение точек в 2D (с классификацией по KNN)")
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.show()

# ------------------------f
# ROC-Кривая и AUC
# ------------------------

fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Ложноположительный риск')
plt.ylabel('Истинноположительный риск')
plt.title('ROC-Кривая')
plt.legend(loc="lower right")
plt.show()

# ------------------------
# График точности vs k
# ------------------------

k_values = range(1, 21)  # Пробуем значения k от 1 до 20
accuracies = []

for k in k_values:
    knn = KNNClassifier(n_neighbors=k, weights='uniform', p=2)
    knn.fit(X_train, y_train)
    y_pred_k = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred_k))

plt.figure(figsize=(8, 6))
plt.plot(k_values, accuracies, marker='o', color='b', label='Точность модели')
plt.title('Зависимость точности от числа соседей (k)')
plt.xlabel('Число соседей (k)')
plt.ylabel('Точность')
plt.grid(True)
plt.legend()
plt.show()

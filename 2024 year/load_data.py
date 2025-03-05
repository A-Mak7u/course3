import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('LST_final_TRUE.csv')

print("Информация о датасете:")
print(dataset.info())

print("\nПервые строки данных:")
print(dataset.head())

print("\nПроверка на пропуски:")
print(dataset.isnull().sum())

print("\nСтатистические характеристики данных:")
pd.set_option('display.max_columns', None)
print(dataset.describe())

print("\nТипы данных в столбцах:")
print(dataset.dtypes)

print("\nРаспределение данных для числовых признаков:")
dataset.hist(figsize=(12, 10), bins=50)
plt.tight_layout()
plt.show()

print("\nАнализ выбросов с помощью boxplot:")
numerical_columns = dataset.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(12, 8))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(3, 4, i)
    sns.boxplot(dataset[col])
    plt.title(col)
plt.tight_layout()
plt.show()

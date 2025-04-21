import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from tqdm import tqdm

# ------------------------
# LOAD DATA
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
# HYPERPARAMETER GRID
# ------------------------
neighbor_options = [3, 5, 7, 10, 15, 20, 25, 30]
weights_options = ['uniform', 'distance']
all_combinations = [(n, w) for n in neighbor_options for w in weights_options]

# ------------------------
# MODEL TRAINING
# ------------------------
results = []

for n_neighbors, weights in tqdm(all_combinations, desc="KNN Grid Search"):
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, n_jobs=-1)
    knn.fit(X_train, y_train)

    predictions = knn.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    results.append((n_neighbors, weights, mse, rmse, mae, mape, r2))

# ------------------------
# SAVE RESULTS
# ------------------------
df_results = pd.DataFrame(results, columns=["Neighbors", "Weights", "MSE", "RMSE", "MAE", "MAPE", "R²"])
df_results.sort_values(by="MSE", inplace=True)
df_results.to_csv("knn_results.csv", index=False)

# ------------------------
# PLOT RESULTS
# ------------------------
metrics = ["MSE", "RMSE", "MAE", "MAPE", "R²"]
fig, axs = plt.subplots(3, 2, figsize=(14, 12))
axs = axs.flatten()

for i, metric in enumerate(metrics):
    sns.boxplot(x="Weights", y=metric, data=df_results, ax=axs[i])
    axs[i].set_title(f"{metric} by Weights")
axs[-1].axis('off')
plt.tight_layout()
plt.show()

# ------------------------
# SCATTER PLOTS FOR METRICS
# ------------------------
for metric in metrics:
    if metric != "R²":  # для R² отдельно по убыванию
        sns.scatterplot(x=metric, y="R²", hue="Weights", style="Weights", data=df_results, palette="Set2", s=100)
        plt.title(f"{metric} vs R² (KNN)")
        plt.grid(True)
        plt.show()

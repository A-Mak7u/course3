import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tqdm import tqdm
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"GPU detected: {physical_devices}")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("No GPU detected, using CPU.")

data = pd.read_csv('LST_final_TRUE.csv')

data = data[data['T_rp5'].between(data['T_rp5'].quantile(0.07), data['T_rp5'].quantile(0.93))]

data = data.groupby(['DayOfYear', 'Time']).agg({
    'T_rp5': 'mean',
    'X': 'first', 'Y': 'first', 'H': 'first', 'TWI': 'first',
    'Temperature_merra_1000hpa': 'first'
}).reset_index()

X = data.drop(['T_rp5', 'Roughness', 'Aspect', 'Hillshade', 'Slope'], axis=1, errors='ignore')
y = data['T_rp5']

print("Unique values in features:")
print(X[['X', 'Y', 'H', 'TWI', 'Temperature_merra_1000hpa', 'Time', 'DayOfYear']].nunique())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

mse_scores = []
rmse_scores = []
mae_scores = []
mape_scores = []
r2_scores = []

def create_model():
    model = Sequential()
    model.add(Dense(128, activation='tanh', input_shape=(X.shape[1],), kernel_regularizer=l2(0.0005)))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='tanh', kernel_regularizer=l2(0.0005)))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("Starting cross-validation...")
for fold, (train_index, test_index) in tqdm(enumerate(kf.split(X_scaled), 1), total=kf.n_splits, desc="Fold Progress"):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    print(f"\nFold {fold}:")
    print(f"Train T_rp5 stats: {y_train.describe()}")
    print(f"Test T_rp5 stats: {y_test.describe()}")

    model = create_model()
    model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
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

    mse_scores.append(mse)
    rmse_scores.append(rmse)
    mae_scores.append(mae)
    mape_scores.append(mape)
    r2_scores.append(r2)

print("\nCross-Validation Results (Aggregated):")
print(f"Mean MSE: {np.mean(mse_scores):.3f} ± {np.std(mse_scores):.3f}")
print(f"Mean RMSE: {np.mean(rmse_scores):.3f} ± {np.std(rmse_scores):.3f}")
print(f"Mean MAE: {np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}")
print(f"Mean MAPE: {np.mean(mape_scores):.3f}% ± {np.std(mape_scores):.3f}%")
print(f"Mean R²: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")

results = pd.DataFrame({
    'MSE': mse_scores,
    'RMSE': rmse_scores,
    'MAE': mae_scores,
    'MAPE': mape_scores,
    'R²': r2_scores
})
results.to_csv('mlp_cross_validation_aggregated_results.csv', index=False)
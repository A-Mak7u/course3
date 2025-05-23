import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import itertools
import os
from torch.cuda.amp import GradScaler, autocast
import uuid
import concurrent.futures
import random

# Фиксация seed для воспроизводимости
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Проверка CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Кастомный датасет
class CustomDataset(TensorDataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # [N, 11] для Linear слоя
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Модель LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, activation):
        super(LSTMModel, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)  # Преобразование 11 признаков
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.ln = nn.LayerNorm(hidden_size)  # LayerNorm для стабильности
        self.fc = nn.Linear(hidden_size, 1)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:  # leaky_relu
            self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x.unsqueeze(1))  # [N, 1, hidden_size]
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.ln(out[:, -1, :])  # LayerNorm на последнем шаге
        out = self.dropout(out)
        out = self.activation(out)
        out = self.fc(out)
        return out

# Функция для загрузки данных
def load_data(file_path, batch_size):
    dataset = pd.read_csv(file_path)
    features = ["H", "TWI", "Aspect", "Hillshade", "Roughness", "Slope",
                "Temperature_merra_1000hpa", "Time", "DayOfYear", "X", "Y"]
    target = "T_rp5"
    X = dataset[features].values
    y = dataset[target].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    # num_workers=2 для снижения нагрузки на CPU; уменьшите до 0, если CPU перегружен
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)
    return train_loader, test_loader

# Функция для вычисления метрик
def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, mape, r2

# Функция обучения
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs, patience):
    scaler = GradScaler()
    best_loss = float("inf")
    patience_counter = 0
    best_model_path = f"best_model_{uuid.uuid4()}.pth"

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        # Валидация
        model.eval()
        val_loss = 0.0
        predictions, actuals = [], []
        with torch.no_grad(), autocast():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()
                predictions.extend(outputs.squeeze().cpu().numpy())
                actuals.extend(targets.cpu().numpy())

        val_loss /= len(test_loader)
        train_loss /= len(train_loader)
        scheduler.step(val_loss)

        # Метрики
        mse, rmse, mae, mape, r2 = compute_metrics(np.array(actuals), np.array(predictions))
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, R2: {r2:.4f}")

        # Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Загрузка лучшей модели
    model.load_state_dict(torch.load(best_model_path))
    os.remove(best_model_path)
    return mse, rmse, mae, mape, r2, model

# Функция для запуска одного эксперимента
def run_experiment(params, file_path):
    batch_size, activation, hidden_size, num_layers, dropout, lr, epochs = params
    print(f"Testing: bs={batch_size}, act={activation}, hs={hidden_size}, nl={num_layers}, do={dropout}, lr={lr}, ep={epochs}")

    train_loader, test_loader = load_data(file_path, batch_size)

    # Инициализация модели
    model = LSTMModel(input_size=11, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, activation=activation).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)

    # Обучение
    mse, rmse, mae, mape, r2, trained_model = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs, patience=5)
    return params + (mse, rmse, mae, mape, r2), trained_model

# Основная функция
def main():
    # Расширенный набор гиперпараметров
    param_grid = {
        "batch_size": [32, 64, 128],
        "activation": ["relu", "tanh", "leaky_relu"],
        "hidden_size": [64, 128, 256, 512],
        "num_layers": [1, 2, 3],
        "dropout": [0.0, 0.1, 0.2],
        "learning_rate": [0.01, 0.001, 0.0005],
        "epochs": [50, 100]
    }
    all_combinations = list(itertools.product(*param_grid.values()))
    results = []
    best_mse = float("inf")
    best_model = None
    target_mse = 5.557594  # Целевой MSE для сравнения

    # Загрузка данных
    file_path = "LST_final_TRUE.csv"

    # Параллельное выполнение экспериментов
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_experiment, params, file_path) for params in all_combinations]
        for future in concurrent.futures.as_completed(futures):
            params_metrics, model = future.result()
            results.append(params_metrics)
            mse = params_metrics[7]  # MSE находится на 7-й позиции
            if mse < best_mse:
                best_mse = mse
                best_model = model

    # Сохранение результатов
    df_results = pd.DataFrame(results, columns=["Batch Size", "Activation", "Hidden Size", "Num Layers", "Dropout",
                                               "Learning Rate", "Epochs", "MSE", "RMSE", "MAE", "MAPE", "R2"])
    df_results.sort_values(by="MSE", inplace=True)
    df_results.to_csv("lstm3_1.csv", index=False)

    # Сохранение лучшей модели
    mse_threshold = 0.1
    if abs(best_mse - target_mse) < mse_threshold:
        final_model_path = "final_best_lstm.pth"
        torch.save(best_model.state_dict(), final_model_path)
        print(f"Best model saved to {final_model_path} with MSE={best_mse:.6f}")
    else:
        print(f"Best MSE={best_mse:.6f} not close enough to target MSE={target_mse:.6f}, model not saved.")

if __name__ == "__main__":
    main()



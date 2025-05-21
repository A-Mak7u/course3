import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import concurrent.futures
import os
from torch.cuda.amp import GradScaler, autocast
import uuid
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
# кто тут?
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class CustomDataset(TensorDataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, activation):
        super(LSTMModel, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        lstm_dropout = dropout if num_layers > 1 else 0
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=lstm_dropout)
        self.ln = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x.unsqueeze(1))
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.ln(out[:, -1, :])
        out = self.dropout(out)
        out = self.activation(out)
        out = self.fc(out)
        return out

def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, mape, r2

def train_and_evaluate_fold(model, train_loader, test_loader, epochs, learning_rate, patience):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)
    scaler = GradScaler()

    best_loss = float("inf")
    patience_counter = 0
    best_model_path = f"best_model_fold_{uuid.uuid4()}.pth"

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

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(torch.load(best_model_path))
    os.remove(best_model_path)
    mse, rmse, mae, mape, r2 = compute_metrics(np.array(actuals), np.array(predictions))
    return mse, rmse, mae, mape, r2

def cross_validate_lstm(X, y, param_combinations, n_splits=5, epochs=100):
    kf = KFold(n_splits=n_splits, shuffle=False)  # не перемешиваем из-за временного ряда
    results = []

    for i, params in enumerate(param_combinations):
        print(f"Processing combination {i + 1}/{len(param_combinations)}: {params}")
        batch_size, activation, hidden_size, num_layers, dropout, learning_rate = params[:6]
        epochs = params[6]

        fold_metrics = {'mse': [], 'rmse': [], 'mae': [], 'mape': [], 'r2': []}

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            print(f"  Fold {fold + 1}/{n_splits}")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            train_dataset = CustomDataset(X_train, y_train)
            test_dataset = CustomDataset(X_test, y_test)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                      num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

            model = LSTMModel(input_size=X.shape[1], hidden_size=hidden_size, num_layers=num_layers,
                              dropout=dropout, activation=activation).to(device)

            mse, rmse, mae, mape, r2 = train_and_evaluate_fold(model, train_loader, test_loader, epochs, learning_rate,
                                                               patience=5)
            fold_metrics['mse'].append(mse)
            fold_metrics['rmse'].append(rmse)
            fold_metrics['mae'].append(mae)
            fold_metrics['mape'].append(mape)
            fold_metrics['r2'].append(r2)

        avg_metrics = {
            'batch_size': batch_size,
            'activation': activation,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'mse_mean': np.mean(fold_metrics['mse']),
            'mse_std': np.std(fold_metrics['mse']),
            'rmse_mean': np.mean(fold_metrics['rmse']),
            'rmse_std': np.std(fold_metrics['rmse']),
            'mae_mean': np.mean(fold_metrics['mae']),
            'mae_std': np.std(fold_metrics['mae']),
            'mape_mean': np.mean(fold_metrics['mape']),
            'mape_std': np.std(fold_metrics['mape']),
            'r2_mean': np.mean(fold_metrics['r2']),
            'r2_std': np.std(fold_metrics['r2'])
        }
        results.append(avg_metrics)

    return pd.DataFrame(results)

def main():
    file_path = "LST_final_TRUE.csv"
    dataset = pd.read_csv(file_path)
    features = ["H", "TWI", "Aspect", "Hillshade", "Roughness", "Slope",
                "Temperature_merra_1000hpa", "Time", "DayOfYear", "X", "Y"]
    target = "T_rp5"
    X = dataset[features].values
    y = dataset[target].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    param_combinations = [
        (32, "leaky_relu", 384, 3, 0.1, 0.0003, 100),  # лучший из 216 комбинаций
        (32, "leaky_relu", 192, 2, 0.1, 0.0007, 100),  # лучший из 144 комбинаций
        (64, "relu", 256, 3, 0.1, 0.0005, 100)
    ]

    results_df = cross_validate_lstm(X_scaled, y, param_combinations, n_splits=5, epochs=100)

    print(results_df)
    results_df.to_csv("lstm_cv.csv", index=False)

if __name__ == "__main__":
    main()
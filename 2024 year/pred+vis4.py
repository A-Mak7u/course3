import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


# 1
def load_data(file_path):
    dataset = pd.read_csv(file_path)
    return dataset


# 2
def preprocess_data(X_train, X_test, method):
    if method == "None":
        return X_train, X_test
    elif method == "StandardScaler":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif method == "MinMaxScaler":
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif method == "PolynomialFeatures":
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train = poly.fit_transform(X_train)
        X_test = poly.transform(X_test)
    return X_train, X_test


# 3
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2


# 4
def main():
    file_path = 'LST_final_TRUE.csv'
    data = load_data(file_path)

    X = data.drop(columns=['T_rp5'])
    y = data['T_rp5']

    preprocessing_methods = ["None", "StandardScaler", "MinMaxScaler", "PolynomialFeatures"]
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "K-Neighbors Regressor": KNeighborsRegressor(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Support Vector Regressor": SVR()
    }

    results = []

    # обучающая и тестовая
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for preproc in preprocessing_methods:
        X_train_proc, X_test_proc = preprocess_data(X_train, X_test, preproc)

        for model_name, model in models.items():
            model.fit(X_train_proc, y_train)

            mse, r2 = evaluate_model(model, X_test_proc, y_test)
            results.append({
                "Preprocessing": preproc,
                "Model": model_name,
                "MSE": mse,
                "R^2": r2
            })

    results_df = pd.DataFrame(results)
    print(results_df)

    results_df.to_csv("model_results_with_visualization.csv", index=False)

    models_unique = results_df['Model'].unique()

    for model in models_unique:
        model_data = results_df[results_df['Model'] == model]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_data['Preprocessing'], model_data['MSE'], color='skyblue')
        plt.title(f"Метрика MSE для модели: {model}")
        plt.xlabel("Метод предобработки")
        plt.ylabel("MSE")
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

        plt.savefig(f"{model}_MSE_results.png")
        plt.show()

        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_data['Preprocessing'], model_data['R^2'], color='salmon')
        plt.title(f"Метрика R² для модели: {model}")
        plt.xlabel("Метод предобработки")
        plt.ylabel("R²")
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

        plt.savefig(f"{model}_R2_results.png")
        plt.show()


if __name__ == "__main__":
    main()

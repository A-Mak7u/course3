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


def load_data(file_path):
    dataset = pd.read_csv(file_path)
    return dataset


def preprocess_data(data, method):
    features = data.drop(columns=['T_rp5'])
    target = data['T_rp5']

    if method == "None":
        return features, target
    elif method == "StandardScaler":
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    elif method == "MinMaxScaler":
        scaler = MinMaxScaler()
        features = scaler.fit_transform(features)
    elif method == "PolynomialFeatures":
        poly = PolynomialFeatures(degree=2, include_bias=False)
        features = poly.fit_transform(features)

    return features, target


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2


def main():
    file_path = 'LST_final_TRUE.csv'
    data = load_data(file_path)

    preprocessing_methods = ["None", "StandardScaler", "MinMaxScaler", "PolynomialFeatures"]
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(),
        "K-Neighbors Regressor": KNeighborsRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "Support Vector Regressor": SVR()
    }

    results = []

    for preproc in preprocessing_methods:
        X, y = preprocess_data(data, preproc)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            mse, r2 = evaluate_model(model, X_test, y_test)
            results.append({
                "Preprocessing": preproc,
                "Model": model_name,
                "MSE": mse,
                "R^2": r2
            })

    results_df = pd.DataFrame(results)

    results_df.to_csv("model_results_with_visualization.csv", index=False)

    for preproc in preprocessing_methods:
        preproc_data = results_df[results_df['Preprocessing'] == preproc]

        plt.figure(figsize=(12, 6))
        for i, model_name in enumerate(models.keys()):
            model_data = preproc_data[preproc_data['Model'] == model_name]
            plt.bar(model_name, model_data['MSE'].values[0], label=model_name)

        plt.title(f"Метрика MSE для метода предобработки: {preproc}")
        plt.xlabel("Модели")
        plt.ylabel("MSE")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(f"{preproc}_MSE_results.png")
        plt.show()

        plt.figure(figsize=(12, 6))
        for i, model_name in enumerate(models.keys()):
            model_data = preproc_data[preproc_data['Model'] == model_name]
            plt.bar(model_name, model_data['R^2'].values[0], label=model_name)

        plt.title(f"Метрика R² для метода предобработки: {preproc}")
        plt.xlabel("Модели")
        plt.ylabel("R²")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(f"{preproc}_R2_results.png")
        plt.show()


if __name__ == "__main__":
    main()

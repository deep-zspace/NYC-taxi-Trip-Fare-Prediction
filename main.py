import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, median_absolute_error, explained_variance_score
from xgboost import XGBRegressor
import numpy as np
from tqdm import tqdm
from tabulate import tabulate

class TaxiFarePredictor:
    def __init__(self, file_path, n_samples=100_000, verbose=False):
        self.file_path = file_path
        self.n_samples = n_samples
        self.verbose = verbose
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_and_preprocess_data(self):
        print("Loading data...")
        self.df = pd.read_csv(self.file_path)
        if self.df.empty:
            print("No data loaded. Please check the dataset file.")
            return

        print("Data loaded successfully!")
        print(f"Initial shape of the data: {self.df.shape}")

        print("Dropping rows with missing values...")
        self.df.dropna(inplace=True)
        print(f"Shape after dropping missing values: {self.df.shape}")

        print("Removing rows with negative fare amounts...")
        self.df = self.df[self.df['total_amount'] >= 0]
        print(f"Shape after removing negative fare amounts: {self.df.shape}")

        print("Removing outliers...")
        self.df = self.remove_outliers(self.df, 'trip_distance')
        print(f"Shape after removing outliers from 'trip_distance': {self.df.shape}")
        self.df = self.remove_outliers(self.df, 'total_amount')
        print(f"Shape after removing outliers from 'total_amount': {self.df.shape}")
        self.df = self.remove_outliers(self.df, 'trip_duration')
        print(f"Shape after removing outliers from 'trip_duration': {self.df.shape}")

        if self.df.empty:
            print("No data left after preprocessing! Please check the dataset and preprocessing steps.")
            return

        print("Sampling the data...")
        n_samples = min(self.n_samples, len(self.df))
        if n_samples == 0:
            print("No data left to sample after preprocessing! Please check the dataset and preprocessing steps.")
            return
        self.df = self.df.sample(n=n_samples, random_state=42)
        print(f"Shape after sampling: {self.df.shape}")

        print("Scaling the features...")
        features = ['day', 'hour', 'trip_duration', 'trip_distance']
        scaler = StandardScaler()
        self.df[features] = scaler.fit_transform(self.df[features])

        print("Splitting the data into training and testing sets...")
        X = self.df[features]
        y = self.df['total_amount']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Data split into training and testing sets successfully!")

        print("Generating exploratory plots...")
        self.generate_exploratory_plots()
        print("Exploratory plots generated successfully!")

    def remove_outliers(self, df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    def generate_exploratory_plots(self):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        sns.histplot(self.df['trip_distance'], bins=30, kde=True)
        plt.title('Trip Distance')

        plt.subplot(1, 3, 2)
        sns.histplot(self.df['total_amount'], bins=30, kde=True)
        plt.title('Total Amount')

        plt.subplot(1, 3, 3)
        sns.histplot(self.df['trip_duration'], bins=30, kde=True)
        plt.title('Trip Duration')

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 8))
        correlation_matrix = self.df[['day', 'hour', 'trip_duration', 'trip_distance', 'total_amount']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.show()
    
    def train_model(self, model, model_params, model_name):
        print(f"Training {model_name} model...")
        model.set_params(**model_params)
        with tqdm(total=model_params.get("n_estimators", 100), desc=f"Training {model_name}") as pbar:
            for i in range(1, model_params.get("n_estimators", 100) + 1):
                model.n_estimators = i
                model.fit(self.X_train, self.y_train)
                pbar.update(1)
        return model
    
    def evaluate_model(self, model, model_name):
        print(f"Evaluating {model_name}...")
        y_pred = model.predict(self.X_test)

        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = mean_squared_error(self.y_test, y_pred, squared=False)
        r2 = r2_score(self.y_test, y_pred)
        mape = mean_absolute_percentage_error(self.y_test, y_pred)
        medae = median_absolute_error(self.y_test, y_pred)
        explained_variance = explained_variance_score(self.y_test, y_pred)

        headers = ["Metric", "Value"]
        table = [
            ["MAE", mae],
            ["RMSE", rmse],
            ["R²", r2],
            ["MAPE", mape],
            ["Median AE", medae],
            ["Explained Variance", explained_variance]
        ]
        print(f"\n{model_name} Evaluation Metrics:")
        print(tabulate(table, headers, tablefmt="grid"))

        # Plot Distribution of Predictions vs Actual Values
        print(f"Plotting Distribution of Predictions vs Actual Values for {model_name}...")
        plt.figure(figsize=(10, 5))
        sns.histplot(self.y_test, color='blue', label='Actual Values', kde=True, stat='density')
        sns.histplot(y_pred, color='red', label='Predicted Values', kde=True, stat='density')
        plt.title(f'{model_name} - Distribution of Predictions vs Actual Values')
        plt.xlabel('Fare Amount')
        plt.ylabel('Density')
        plt.legend()
        plt.show()

        # Residuals Plot
        print(f"Plotting Residuals Distribution for {model_name}...")
        residuals = self.y_test - y_pred
        plt.figure(figsize=(10, 5))
        sns.histplot(residuals, bins=30, kde=True)
        plt.title(f'{model_name} - Residuals Distribution')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.show()

        # Feature Importance
        if hasattr(model, 'feature_importances_'):
            print(f"Plotting Feature Importance for {model_name}...")
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(10, 5))
            plt.title("Feature Importance")
            plt.bar(range(self.X_train.shape[1]), importances[indices], align="center")
            plt.xticks(range(self.X_train.shape[1]), [self.X_train.columns[i] for i in indices])
            plt.xlabel('Feature')
            plt.ylabel('Importance')
            plt.show()

        # Prediction vs Actual Scatter Plot
        print(f"Plotting Prediction vs Actual Scatter Plot for {model_name}...")
        plt.figure(figsize=(10, 5))
        plt.scatter(self.y_test, y_pred, alpha=0.5, color='red', label='Predicted Values')
        plt.scatter(self.y_test, self.y_test, alpha=0.5, color='blue', label='Actual Values')
        plt.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)], color='green', linestyle='--', label='Ideal Line')
        plt.title(f'{model_name} - Prediction vs Actual')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.show()

        # Residuals vs Predicted Values
        print(f"Plotting Residuals vs Predicted Values for {model_name}...")
        plt.figure(figsize=(10, 5))
        plt.scatter(y_pred, residuals, alpha=0.5, color='purple', label='Residuals')
        plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), color='red', linestyle='--', label='Zero Error Line')
        plt.title(f'{model_name} - Residuals vs Predicted Values')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.legend()
        plt.show()

        return {
            "MAE": mae,
            "RMSE": rmse,
            "R²": r2,
            "MAPE": mape,
            "Median AE": medae,
            "Explained Variance": explained_variance
        }

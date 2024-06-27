import os
import glob
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from neuralnet import NeuralNetwork
from main import TaxiFarePredictor

class TaxiFarePredictorNN(TaxiFarePredictor):
    def __init__(self, file_path, n_samples=100_000):
        super().__init__(file_path, n_samples)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NeuralNetwork().to(self.device)

    def load_model(self, model_path='best_model_0.pth'):
        if not os.path.exists(model_path):
            print(f"Model file {model_path} does not exist")
            return
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        print(f"Model loaded from {model_path}")
    
    def predict(self, day, hour, trip_duration, trip_distance):
        self.model.eval()
        input_features = torch.tensor([[day, hour, trip_duration, trip_distance]], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_features)
        return prediction.cpu().numpy()[0][0]

    def load_and_preprocess_data(self):
        self.df = pd.read_csv(self.file_path)
        self.df.dropna(inplace=True)
        self.df = self.df[self.df['total_amount'] >= 0]

        self.df = self.remove_outliers(self.df, 'trip_distance')
        self.df = self.remove_outliers(self.df, 'total_amount')
        self.df = self.remove_outliers(self.df, 'trip_duration')

        n_samples = min(self.n_samples, len(self.df))
        self.df = self.df.sample(n=n_samples, random_state=42)

        features = ['day', 'hour', 'trip_duration', 'trip_distance']
        scaler = StandardScaler()
        self.df[features] = scaler.fit_transform(self.df[features])

        X = self.df[features]
        y = self.df['total_amount']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Convert to PyTorch tensors
        X_train = torch.tensor(self.X_train.values, dtype=torch.float32).to(self.device)
        X_test = torch.tensor(self.X_test.values, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(self.y_train.values, dtype=torch.float32).to(self.device).view(-1, 1)
        y_test = torch.tensor(self.y_test.values, dtype=torch.float32).to(self.device).view(-1, 1)

        # Create DataLoader
        self.train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
        self.test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

    def predict_fare_with_user_input(self):
        def get_user_input(prompt, input_type, valid_range=None):
            while True:
                try:
                    user_input = input_type(input(prompt))
                    if valid_range and user_input not in valid_range:
                        raise ValueError
                    return user_input
                except ValueError:
                    print(f"Invalid input. Please enter a valid {input_type.__name__} value.")

        day = get_user_input("Enter day of the week (0-6, where 0=Sunday, 1=Monday, ..., 6=Saturday): ", int, range(0, 7))
        print(f"Day: {day}")
        hour = get_user_input("Enter hour of the day (0-23): ", int, range(0, 24))
        print(f"Hour: {hour}")
        trip_duration = get_user_input("Enter trip duration in minutes: ", float)
        print(f"Trip Duration: {trip_duration} minutes")
        trip_distance = get_user_input("Enter trip distance in miles: ", float)
        print(f"Trip Distance: {trip_distance} miles")

        predicted_fare = self.predict(day, hour, trip_duration, trip_distance)
        print(f"Predicted Fare: ${predicted_fare:.2f}")

        # Calculate prediction accuracy metrics using the test data
        self.load_and_preprocess_data()
        actuals = []
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                outputs = self.model(inputs)
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(targets.cpu().numpy())

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))  # Manually compute RMSE
        mape = mean_absolute_percentage_error(actuals, predictions) * 100  # Convert to percentage

        print(f"Prediction Accuracy Metrics:")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")


def main():
    # Load the dataset and initialize the predictor
    dataset_directory = 'dataset/'
    if not os.path.exists(dataset_directory):
        print(f"Dataset directory {dataset_directory} does not exist.")
        return
    csv_files = glob.glob(os.path.join(dataset_directory, 'main.csv'))

    if not csv_files:
        print("No main.csv file found in the dataset directory.")
    else:
        csv_file_path = csv_files[0]
        predictor = TaxiFarePredictorNN(
            csv_file_path, 
            n_samples=1_000  
        )
        
        model_file = 'train_model/best_model_0.pth'
        model_number = input("Enter the model number to use (0-3, default is 0): ").strip()
        
        if model_number not in ['0', '1', '2', '3']:
            print("Invalid input. Using default model: best_model_0.pth")
        else:
            model_file = f'train_model/best_model_{model_number}.pth'
        
        if not os.path.exists(model_file):
            print(f"Model file {model_file} does not exist. Please train the model first and try again!!! :)")
            return
        
        predictor.load_model(model_path=model_file)

        # Predict fare with user input
        predictor.predict_fare_with_user_input()

if __name__ == "__main__":
    main()

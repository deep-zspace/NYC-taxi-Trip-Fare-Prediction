import os
import glob
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from neuralnet import NeuralNetwork
from main import TaxiFarePredictor

class TaxiFarePredictorNN(TaxiFarePredictor):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NeuralNetwork().to(self.device)
        self.std_residuals = None

    def load_model(self, model_path='best_model_0.pth'):
        if not os.path.exists(model_path):
            print(f"Model file {model_path} does not exist")
            return
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {model_path}")

    def predict(self, day, hour, trip_duration, trip_distance):
        self.model.eval()
        input_features = torch.tensor([[day, hour, trip_duration, trip_distance]], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_features)
        return prediction.cpu().numpy()[0][0]

    def calculate_residuals(self, X_train, y_train):
        self.model.eval()
        batch_size = 32  
        train_loader = DataLoader(TensorDataset(X_train, y_train.unsqueeze(1)), batch_size=batch_size, shuffle=True)
        residuals = []
        with torch.no_grad():
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                residuals.append((targets - outputs).cpu())

        residuals = torch.cat(residuals).numpy().flatten()
        self.std_residuals = np.std(residuals)

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

        if self.std_residuals is not None:
            z_score = 1.96  # For 95% confidence
            interval = z_score * self.std_residuals  # Prediction interval margin

            pred_lower = predicted_fare - interval
            pred_upper = predicted_fare + interval
            print(f"Prediction Interval: ${pred_lower:.2f} to ${pred_upper:.2f}")
        else:
            print("Prediction confidence interval cannot be calculated because residuals are not available.")

def main():
    # Load the dataset and initialize the predictor
    dataset_directory = 'dataset/'
    csv_files = glob.glob(os.path.join(dataset_directory, 'main.csv'))

    if not csv_files:
        print("No main.csv file found in the dataset directory.")
        return
    
    csv_file_path = csv_files[0]
    predictor = TaxiFarePredictorNN()
        
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

    # Load and preprocess data for residual calculation
    df = pd.read_csv(csv_file_path)
    df.dropna(inplace=True)
    df = df[df['total_amount'] >= 0]

    df = predictor.remove_outliers(df, 'trip_distance')
    df = predictor.remove_outliers(df, 'total_amount')
    df = predictor.remove_outliers(df, 'trip_duration')

    n_samples = min(100_000, len(df))
    df = df.sample(n=n_samples, random_state=42)

    features = ['day', 'hour', 'trip_duration', 'trip_distance']
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    X = df[features]
    y = df['total_amount']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(predictor.device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(predictor.device)

    predictor.calculate_residuals(X_train_tensor, y_train_tensor)

    # Predict fare with user input
    predictor.predict_fare_with_user_input()

if __name__ == "__main__":
    main()

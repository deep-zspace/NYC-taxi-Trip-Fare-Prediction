import os
import glob
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, median_absolute_error, explained_variance_score
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from neuralnet import NeuralNetwork
from main import TaxiFarePredictor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tabulate import tabulate

class TaxiFarePredictorNN(TaxiFarePredictor):
    def __init__(self, file_path, n_samples=100_000, batch_size=32, epochs=10, learning_rate=0.001, 
                 weight_decay=1e-5, patience=3, factor=0.5, early_stopping=True, early_stopping_patience = 10):
        super().__init__(file_path, n_samples)
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.factor = factor
        self.early_stopping = early_stopping
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NeuralNetwork().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=self.patience, factor=self.factor, verbose=True)
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_counter = 0
        self.best_loss = float('inf')

    def load_and_preprocess_data(self):
        super().load_and_preprocess_data()
        
        # Prepare DataLoader
        features = ['day', 'hour', 'trip_duration', 'trip_distance']
        X = self.df[features].values
        y = self.df['total_amount'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device).view(-1, 1)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(self.device).view(-1, 1)
        
        # Create DataLoader
        self.train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=self.batch_size, shuffle=False)
        print("Data preprocessing completed successfully!")
    
    def train_model(self):
        print("Training the model...")
        train_losses = []
        val_losses = []
        progress_bar = tqdm(total=self.epochs, desc="Training Progress")
        
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, targets in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(self.train_loader.dataset)
            train_losses.append(epoch_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in self.test_loader:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
            val_loss /= len(self.test_loader.dataset)
            val_losses.append(val_loss)
            
            # Check for early stopping
            if self.early_stopping:
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.early_stopping_counter = 0
                    torch.save(self.model.state_dict(), 'best_model.pth')
                    print(f"Epoch {epoch+1}/{self.epochs}: Validation loss improved to {val_loss:.4f}. Saving the best model.")
                else:
                    self.early_stopping_counter += 1
                    print(f"Epoch {epoch+1}/{self.epochs}: No improvement in validation loss. Early stopping counter: {self.early_stopping_counter}/{self.early_stopping_patience}.")
                    if self.early_stopping_counter >= self.early_stopping_patience:
                        print(f"Early stopping triggered at epoch {epoch+1}.")
                        break

            self.scheduler.step(val_loss)
            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")
            progress_bar.update(1)
        
        progress_bar.close()
        print("Generating plots...")
        self.plot_loss(train_losses, val_losses)
        print("Model training completed successfully! The best model has been saved as 'best_model.pth'")
    
    def evaluate_model(self):
        print("Evaluating the model...")
        self.model.load_state_dict(torch.load('train_model/best_model.pth'))
        self.model.eval()
        predictions = []
        actuals = []
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                outputs = self.model(inputs)
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(targets.cpu().numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        mae = mean_absolute_error(actuals, predictions)
        rmse = mean_squared_error(actuals, predictions, squared=False)
        r2 = r2_score(actuals, predictions)
        mape = mean_absolute_percentage_error(actuals, predictions)
        medae = median_absolute_error(actuals, predictions)
        explained_variance = explained_variance_score(actuals, predictions)
        
        metrics = [
            ["Mean Absolute Error", mae],
            ["Root Mean Squared Error", rmse],
            ["R² Score", r2],
            ["Mean Absolute Percentage Error", mape],
            ["Median Absolute Error", medae],
            ["Explained Variance Score", explained_variance]
        ]
        
        print(tabulate(metrics, headers=["Metric", "Value"], tablefmt="grid"))
        
        print("generating plot...")
        self.plot_predictions_vs_actuals(actuals, predictions)
        self.plot_residuals(actuals, predictions)

    def plot_loss(self, train_losses, val_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss', color='blue', linestyle='-', marker='o')
        plt.plot(val_losses, label='Validation Loss', color='orange', linestyle='-', marker='x')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title('Training and Validation Loss Over Epochs', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.show()

    def plot_predictions_vs_actuals(self, actuals, predictions):
        plt.figure(figsize=(10, 5))
        plt.scatter(actuals, actuals, alpha=0.5, label='Actual Values', color='blue', marker='o')
        plt.scatter(actuals, predictions, alpha=0.5, label='Predicted Values', color='green', marker='x')
        plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2, label='Perfect Prediction Line')
        plt.xlabel('Actual Fare Amount ($)', fontsize=14)
        plt.ylabel('Predicted Fare Amount ($)', fontsize=14)
        plt.title('Actual vs Predicted Fare Amounts', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.show()


    def plot_residuals(self, actuals, predictions):
        residuals = actuals - predictions
        plt.figure(figsize=(10, 5))
        plt.scatter(predictions, residuals, alpha=0.5, label='Residuals', color='purple')
        plt.hlines(y=0, xmin=predictions.min(), xmax=predictions.max(), colors='r', linestyles='dashed', label='Zero Error Line')
        plt.xlabel('Predicted Fare Amount ($)', fontsize=14)
        plt.ylabel('Residuals ($)', fontsize=14)
        plt.title('Residuals vs Predicted Fare Amounts', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.show()

    def load_model(self, model_path='best_model.pth'):
        print("Loading model ...")
        if not os.path.exists(model_path):
            print(f"Model file {model_path} does not exist")
            return
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print(f"Model loaded successfully from {model_path}")
    
    def predict(self, day, hour, trip_duration, trip_distance):
        self.model.eval()
        input_features = torch.tensor([[day, hour, trip_duration, trip_distance]], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_features)
        return prediction.cpu().numpy()[0][0]

    def predict_fare_with_user_input(self):
        day = int(input("Enter day of the week (0-6, where 0=Sunday, 1=Monday, ..., 6=Saturday): "))
        print(f"Day: {day}")
        hour = int(input("Enter hour of the day (0-23): "))
        print(f"Hour: {hour}")
        trip_duration = float(input("Enter trip duration in Minutes: "))
        print(f"Trip Duration: {trip_duration} Minutes")
        trip_distance = float(input("Enter trip distance in miles: "))
        print(f"Trip Distance: {trip_distance} miles")

        predicted_fare = self.predict(day, hour, trip_duration, trip_distance)
        print(f"Predicted Fare: ${predicted_fare:.2f}")

        # Calculate prediction accuracy metrics using the test data
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
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mape = mean_absolute_percentage_error(actuals, predictions) * 100 

        print(f"Prediction Accuracy Metrics:")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")


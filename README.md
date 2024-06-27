
# NYC Taxi Trip Fare Prediction

## Project Overview

This project aims to predict taxi fares in New York City using the Yellow Taxi Trip Records dataset. The project employs both traditional machine learning algorithms and neural networks to build predictive models.

## Project Structure

```
NYC-taxi-Trip-Fare-Prediction/
│
├── dataset/                     # Directory containing the dataset
│   └── main.csv                 # CSV file with taxi trip data
│
├── train_model/                 # Directory containing trained models
│   ├── best_model_0.pth
│   ├── best_model_1.pth
│   ├── best_model_2.pth
│   ├── best_model_3.pth
│
├── main.py                      # Main script for traditional ML model training
├── main_nn.py                   # Script for training neural network models
├── neuralnet.py                 # Definition of the neural network architecture
├── predict_fare.py              # Script to load a model and predict fares based on user input
├── results_data.ipynb           # Notebook for analyzing results
├── run_model.ipynb              # Notebook for running and evaluating models
├── run_nn_data_50k.ipynb        # Notebook for running neural network models with 50k samples
├── run_nn_data_550k.ipynb       # Notebook for running neural network models with 550k samples
├── run_nn_data_1000k.ipynb      # Notebook for running neural network models with 1000k samples
├── run_nn_data_2900k.ipynb      # Notebook for running neural network models with 2900k samples
├── ML project proposal.pdf      # Project proposal document
├── README.md                    # This README file
```

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/deep-zspace/NYC-taxi-Trip-Fare-Prediction.git
    cd NYC-taxi-Trip-Fare-Prediction
    ```

2. **Create a virtual environment and activate it**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The dataset is located in the `dataset/` directory. It contains NYC Yellow Taxi trip records. To download the full dataset, visit the [NYC Taxi & Limousine Commission Trip Record Data page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

## Usage

### Training the Model

#### Traditional Machine Learning Algorithms

To train a model using traditional machine learning algorithms, run the `main.py` script:

```bash
python main.py
```

This script loads the dataset, preprocesses it, trains a regression model, and saves the trained model to the `train_model/` directory.

#### Neural Networks

To train a model using a neural network, run the `main_nn.py` script:

```bash
python main_nn.py
```

This script uses PyTorch to define and train a neural network model. The trained model is saved in the `train_model/` directory.

### Predicting Taxi Fare

To predict taxi fare using a trained model, run the `predict_fare.py` script. This script prompts the user to input details such as the day of the week, hour of the day, trip duration, and trip distance. It then loads the selected model and outputs the predicted fare.

```bash
python predict_fare.py
```

When prompted, enter the model number (0-3) you wish to use. If no input is given, the script defaults to using `best_model_0.pth`.

Example input and output:
```
Enter the model number to use (0-3, default is 0): 0
Model loaded from train_model/best_model_0.pth
Enter day of the week (0-6, where 0=Sunday, 1=Monday, ..., 6=Saturday): 3
Day: 3
Enter hour of the day (0-23): 14
Hour: 14
Enter trip duration in minutes: 15
Trip Duration: 15.0 minutes
Enter trip distance in miles: 3.5
Trip Distance: 3.5 miles
Predicted Fare: $65.14
Prediction Accuracy Metrics:
Mean Absolute Error (MAE): 2.77
Root Mean Squared Error (RMSE): 3.90
Mean Absolute Percentage Error (MAPE): 13.23%
```

### Notebooks

Various Jupyter notebooks are provided for exploring, training, and evaluating the models:

- `results_data.ipynb`: Analyzing the results of model predictions.
- `run_model.ipynb`: Running and evaluating different models.
- `run_nn_data_50k.ipynb`: Running neural network models with 50k samples.
- `run_nn_data_550k.ipynb`: Running neural network models with 550k samples.
- `run_nn_data_1000k.ipynb`: Running neural network models with 1000k samples.
- `run_nn_data_2900k.ipynb`: Running neural network models with 2900k samples.

## Contributors

- Deep Kotadiya (kotadiya.d@northeastern.edu)
- Nitin Somashekhar (somashekhar.n@northeastern.edu)
- Girish Raut (raut.g@northeastern.edu)

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

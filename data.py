import pandas as pd

# Load data
data = pd.read_csv('dataset/yellow_tripdata_2024-01.csv')

# Convert datetime columns to datetime type
data['tpep_pickup_datetime'] = pd.to_datetime(data['tpep_pickup_datetime'])
data['tpep_dropoff_datetime'] = pd.to_datetime(data['tpep_dropoff_datetime'])

# Extract day of the week and hour of the day from pickup datetime
data['pickup_day_of_week'] = data['tpep_pickup_datetime'].dt.dayofweek
data['pickup_hour'] = data['tpep_pickup_datetime'].dt.hour

# Calculate total fare as the sum of fare amount, extra, and MTA tax
data['total_fare'] = data['fare_amount'] + data['extra'] + data['mta_tax']

# Select relevant features
selected_features = data[['pickup_day_of_week', 'pickup_hour', 'trip_distance', 'total_fare']]

# Save the resulting dataframe to a new CSV file
selected_features.to_csv('main_new.csv', index=False)

print("Processed data saved to 'main_new.csv'")

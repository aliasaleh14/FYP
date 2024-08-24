import os
import glob
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import time

# Define base path
base_path = "C:/Users/ALIA/Desktop/model/"

# Define file paths
rf_model_path = os.path.join(base_path, "RF_model.bin")
feature_scaler_path = os.path.join(base_path, "feature_scaler.bin")
target_scaler_path = os.path.join(base_path, "target_scaler.bin")
weather_data_path = os.path.join(base_path, "Weather Station/*/*_nict_weather.csv")

# Function to get the latest file
def get_latest_file(path):
    list_of_files = glob.glob(path)
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getmtime)
    return latest_file

# Load the Random Forest model
rf_model = joblib.load(rf_model_path)

# Load the scalers
feature_scaler = joblib.load(feature_scaler_path)
target_scaler = joblib.load(target_scaler_path)

while True:
    # Get the latest CSV file
    latest_csv_file = get_latest_file(weather_data_path)
    
    if latest_csv_file:
        # Load and preprocess sample data
        sample_data = pd.read_csv(latest_csv_file)
        sample_data['datetime'] = pd.to_datetime(sample_data['datetime'])

        # Filter for the latest data
        latest_data = sample_data.sort_values(by='datetime', ascending=False).head(1)
        required_columns = ['AccumulatedPrecipitation', 'AirPressure', 'Humidity', 'Temperature']
        sample_data = latest_data[required_columns]

        # Normalize the sample data using the loaded feature scaler
        normalized_sample_data = feature_scaler.transform(sample_data)

        # Convert normalized sample data back to DataFrame with original column names
        normalized_sample_data_df = pd.DataFrame(normalized_sample_data, columns=required_columns)

        # Random Forest Predictions
        rf_predictions = rf_model.predict(normalized_sample_data_df)
        rf_predictions_actual = target_scaler.inverse_transform(rf_predictions.reshape(-1, 1))

        # Print predictions
        print("GWL (Prediction):", rf_predictions_actual.flatten())
    
    else:
        print("No new files found. Waiting for new data...")
    
    # Wait for some time before checking again (e.g., every 10 seconds)
    time.sleep(10)
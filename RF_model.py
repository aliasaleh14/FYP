import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import joblib
import pickle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Define base path
base_path = "C:/Users/ALIA/Desktop/model/"

# Define file paths
weather_data_path = "C:/Users/ALIA/Desktop/Study/Sem 7/FYP/FYP/Weather Station/*/*_nict_weather.csv"
GWL_data_path = "C:/Users/ALIA/Desktop/Study/Sem 7/FYP/FYP/Ground Sensor/*/Borehole 3/*_nict_water_lvl3.csv"
all_weather_path = os.path.join(base_path, 'all_weather.bin')
all_GWL_path = os.path.join(base_path, 'all_GWL.bin')
merged_data_path = os.path.join(base_path, 'merged.bin')
model_save_path = os.path.join(base_path, 'RF_model.bin')
feature_scaler_path = os.path.join(base_path, 'feature_scaler.bin')
target_scaler_path = os.path.join(base_path, 'target_scaler.bin')

# Combine all weather data
desired_weather_columns = ["datetime", "AccumulatedPrecipitation", "AirPressure", "Humidity", "Temperature"]
weather_dfs = []

for file in glob.glob(weather_data_path):
    try:
        df = pd.read_csv(file)
        df = df[desired_weather_columns]
        weather_dfs.append(df)
    except (FileNotFoundError, pd.errors.ParserError, Exception):
        continue

if weather_dfs:
    combined_weather_df = pd.concat(weather_dfs, ignore_index=True)
    with open(all_weather_path, 'wb') as f:
        pickle.dump(combined_weather_df, f)

# Combine all GWL data
desired_GWL_columns = ["datetime", "depthResult"]
GWL_dfs = []

for file in glob.glob(GWL_data_path):
    try:
        df = pd.read_csv(file)
        if 'depthResult' in df.columns:
            df = df.dropna(subset=['depthResult'])
            df = df[desired_GWL_columns]
            GWL_dfs.append(df)
    except (FileNotFoundError, pd.errors.ParserError, Exception):
        continue

if GWL_dfs:
    combined_GWL_df = pd.concat(GWL_dfs, ignore_index=True)
    with open(all_GWL_path, 'wb') as f:
        pickle.dump(combined_GWL_df, f)

# Load and merge data
with open(all_weather_path, 'rb') as f:
    data_df = pickle.load(f)
with open(all_GWL_path, 'rb') as f:
    truth_df = pickle.load(f)

data_df['datetime'] = pd.to_datetime(data_df['datetime'])
truth_df['datetime'] = pd.to_datetime(truth_df['datetime'])
matching_truth_df = truth_df[truth_df['datetime'].isin(data_df['datetime'])]
merged_data = pd.merge(data_df, matching_truth_df, on='datetime')
with open(merged_data_path, 'wb') as f:
    pickle.dump(merged_data, f)

# Min-Max Normalization
with open(merged_data_path, 'rb') as f:
    df = pickle.load(f)

target_column = 'depthResult'
features = df.drop(columns=[target_column])
target = df[target_column]
numeric_features = features.select_dtypes(include=[float, int])
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
normalized_features = feature_scaler.fit_transform(numeric_features)
normalized_target = target_scaler.fit_transform(target.values.reshape(-1, 1))
df_normalized = pd.DataFrame(normalized_features, columns=numeric_features.columns)
df_normalized[target_column] = normalized_target

# Save the scalers to binary files
joblib.dump(feature_scaler, feature_scaler_path)
joblib.dump(target_scaler, target_scaler_path)

# Random Forest Model
numeric_columns = [col for col in desired_weather_columns if col != "datetime"]
X = df_normalized[numeric_columns]
y = df_normalized["depthResult"]

if X.isnull().values.any() or y.isnull().any():
    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    non_nan_indices = ~y.isnull()
    X = X.loc[non_nan_indices]
    y = y[non_nan_indices]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=45)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=45)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

model = RandomForestRegressor(random_state=45)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Save the Random Forest model to a binary file
joblib.dump(best_model, model_save_path)
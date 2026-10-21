# Code 1:
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (MinMaxScaler, Normalizer, RobustScaler, StandardScaler)

# Fetch a real-world dataset automatically (California Housing Dataset)
data = fetch_california_housing(as_frame=True)
df = data.frame

# Split into features (X) and target (y)
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']

# Define the scaling techniques to compare
scalers = {'Unscaled (Raw)': None, 'MinMax Scaler': MinMaxScaler(), 'Standard Scaler': StandardScaler(), 'Robust Scaler': RobustScaler(), 'Normalizer (Row-wise)': Normalizer(),}

results = []

# Train model and evaluate performance for each scaling method
for name, scaler in scalers.items():
  # Split data into training and testing sets to prevent data leakage
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  if scaler is not None:
    # Fit ONLY on the training data, then transform both train and test sets
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
  else:
    X_train_scaled, X_test_scaled = X_train, X_test

  # Train a model (Linear Regression)
  model = LinearRegression()
  model.fit(X_train_scaled, y_train)

  # Predict on test set and evaluate
  y_pred = model.predict(X_test_scaled)
  mse = mean_squared_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)

  results.append({'Scaler': name, 'MSE': mse, 'R2 Score': r2})

# Display the comparative results
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Code 2:
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load real-world dataset
data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

# Normalization (Min-Max)
norm_scaler = MinMaxScaler()
X_train_norm = norm_scaler.fit_transform(X_train)
X_test_norm = norm_scaler.transform(X_test)

# Standardization (Z-score)
std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_test_std = std_scaler.transform(X_test)

# Evaluate KNN Performance
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print("Unscaled Accuracy:", accuracy_score(y_test, knn.predict(X_test)))

knn.fit(X_train_norm, y_train)
print("Normalized Accuracy:", accuracy_score(y_test, knn.predict(X_test_norm)))

knn.fit(X_train_std, y_train)
print("Standardized Accuracy:", accuracy_score(y_test, knn.predict(X_test_std)))


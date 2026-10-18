import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
from tensorflow import keras

print(f"NumPy: {np.__version__}\nPandas: {pd.__version__}\nScikit-Learn: {sklearn.__version__}\nTensorFlow: {tf.__version__}")

Import Dataset: https://archive.ics.uci.edu/ml/datasets/automobile:
===================================================================
!pip install ucimlrepo -q

from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

# Load Automobile dataset (ID 10) from UCI
automobile = fetch_ucirepo(id=10)
X = automobile.data.features
y = automobile.data.targets

# Combine features and target, dropping rows with missing values
df = pd.concat([X, y], axis=1).dropna()
print(f"Automobile dataset loaded: {df.shape[0]} vehicles, {df.shape[1]} technical specs")

Numpy:
======
import numpy as np

# Convert engine and power specs to a raw matrix
specs_matrix = df[['engine-size', 'horsepower']].to_numpy()

# 1. Vectorized Outlier Removal: Calculate Z-scores manually
means = np.mean(specs_matrix, axis=0)
stds = np.std(specs_matrix, axis=0)
z_scores = (specs_matrix - means) / stds

# Filter out outlier vehicles exceeding 3 standard deviations
clean_mask = np.all(np.abs(z_scores) <= 3, axis=1)
filtered_specs = specs_matrix[clean_mask]

print(f"Vehicles before: {specs_matrix.shape[0]} -> Vehicles after cleaning: {filtered_specs.shape[0]}")

# 2. Linear Algebra: Calculate the covariance matrix via matrix multiplication
centered_data = filtered_specs - np.mean(filtered_specs, axis=0)
covariance_matrix = (centered_data.T @ centered_data) / (filtered_specs.shape[0] - 1)
print(f"Engine vs Horsepower Covariance Matrix:\n{covariance_matrix}")

Pandas:
=======
import pandas as pd

# Create a binary classification target (1 for Luxury/Premium, 0 for Standard)
price_threshold = df['price'].median()
df['Is_Premium'] = np.where(df['price'] > price_threshold, 1, 0)

# Multi-level aggregation profiling vehicle performance by drive wheels and aspiration (std vs turbo)
car_summary = df.groupby(['drive-wheels', 'aspiration']).agg({
    'horsepower': ['mean', 'max'],
    'city-mpg': ['mean', 'median'],
    'engine-size': lambda x: (x > 150).sum() # Count of vehicles with large engines
})

# Flatten multi-index header structures
car_summary.columns = ['_'.join(col).strip() for col in car_summary.columns.values]
print("--- Aggregated Automobile Metrics ---")
print(car_summary.head())

# Filter for lightweight cars with high horsepower that are classified as standard price
power_weight_ratio_cut = df['horsepower'].quantile(0.80)
sport_budget_cars = df[(df['horsepower'] > power_weight_ratio_cut) & (df['Is_Premium'] == 0)]
print(f"\nHigh-power budget vehicles: {sport_budget_cars.shape[0]}")

Matplotlib

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Distribution of curb weight with an average marker
axes[0].hist(df['curb-weight'], bins=15, color='crimson', edgecolor='black', alpha=0.7)
axes[0].axvline(df['curb-weight'].mean(), color='blue', linestyle='--', linewidth=2,
                label=f"Avg Weight: {df['curb-weight'].mean():.1f} lbs")
axes[0].set_title('Vehicle Curb Weight Distribution')
axes[0].set_xlabel('Curb Weight')
axes[0].set_ylabel('Vehicle Count')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Subplot 2: Error bars for horsepower across different body styles
power_stats = df.groupby('body-style')['horsepower'].agg(['mean', 'std']).reset_index()
axes[1].errorbar(power_stats['body-style'], power_stats['mean'], yerr=power_stats['std'],
                 fmt='-o', color='darkgreen', ecolor='orange', elinewidth=2, capsize=5)
axes[1].set_title('Horsepower Variance by Car Body Style')
axes[1].set_xlabel('Body Style')
axes[1].set_ylabel('Horsepower')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

Scikit-Learn:
=============
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Isolate features from the engineered binary target variable
X_features = df.drop(columns=['price', 'Is_Premium'])
y_target = df['Is_Premium']

numeric_cols = ['wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg']
categorical_cols = ['make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'engine-type', 'num-of-cylinders', 'fuel-system']

# Define structural pipeline transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.25, random_state=42, stratify=y_target)

# Execute the preprocessor transformer matrix splits
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# Train a structured Random Forest to evaluate premium classifications
model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)
print("--- Premium Segment Classification Report ---")
print(classification_report(y_test, predictions))

Tensorflow & Keras:
===================
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks

def build_automotive_nn(input_dimension):
    network = models.Sequential([
        layers.Input(shape=(input_dimension,)),

        # Layer 1: He Uniform initialization with weight regularizers
        layers.Dense(64, activation='relu',
                     kernel_initializer='he_uniform',
                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Layer 2
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        # Output probability mapping via sigmoid function
        layers.Dense(1, activation='sigmoid')
    ])

    network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return network

# Instantiating the network directly on the pipeline dimensions mapped out via Scikit-learn
nn_classifier = build_automotive_nn(input_dimension=X_train_scaled.shape[1])

early_halt = callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
lr_decay = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0)

# Run model optimization cycles
history = nn_classifier.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=40,
    batch_size=16,
    callbacks=[early_halt, lr_decay],
    verbose=1
)

test_loss, test_accuracy = nn_classifier.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\n🚀 Final Automotive Network Intent Metric -> Test Accuracy: {test_accuracy:.4f}")
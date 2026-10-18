import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to save plot directly
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Dataset
iris = load_iris(as_frame=True)
dataset = iris.frame
dataset.rename(columns={'target': 'Class'}, inplace=True)

# Create an imbalanced version of Iris dataset
imbalanced_indices = np.hstack([
    np.where(dataset['Class'] == 0)[0],
    np.where(dataset['Class'] == 1)[0],
    np.where(dataset['Class'] == 2)[0][:10]
])
dataset = dataset.iloc[imbalanced_indices].reset_index(drop=True)

# Separate features and target
X = dataset.drop('Class', axis=1)
y = dataset['Class']

# Split train and test sets before resampling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (Keep column names to avoid sklearn feature warnings)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# Manual Random Over-Sampling using Pandas
train_df = X_train_scaled.copy()
train_df['Class'] = y_train.values

max_class_size = train_df['Class'].value_counts().max()
resampled_dfs = []

for cls, group in train_df.groupby('Class'):
    resampled_dfs.append(group.sample(max_class_size, replace=True, random_state=42))

resampled_df = pd.concat(resampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

X_res = resampled_df.drop('Class', axis=1)
y_res = resampled_df['Class']

# Fit Logistic Regression (Setosa vs. Non-Setosa)
y_train_binary = (y_res == 0).astype(int)
y_test_binary = (y_test == 0).astype(int)

model = LogisticRegression()
model.fit(X_res, y_train_binary)

# Model Performance
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test_binary, y_pred):.2f}\n")

# Interpret Model Parameters
feature_names = X.columns
intercept = model.intercept_[0]
coefficients = model.coef_[0]

print("--- Model Parameters ---")
print(f"Intercept (Bias): {intercept:.4f}")
for name, coef in zip(feature_names, coefficients):
    odds_ratio = np.exp(coef)
    print(f"Feature: {name:20s} | Weight: {coef:8.4f} | Odds Ratio: {odds_ratio:.4f}")

# Generate and Plot the Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z_values = np.dot(X_res, coefficients) + intercept
sorted_indices = np.argsort(z_values)
z_sorted = z_values[sorted_indices]

z_curve = np.linspace(-10, 10, 300)
sig_curve = sigmoid(z_curve)

plt.figure(figsize=(8, 5))
plt.plot(z_curve, sig_curve, color='red', linewidth=2, label='Sigmoid Function')
plt.scatter(z_sorted, y_train_binary.iloc[sorted_indices], color='blue', alpha=0.5, label='Samples (Probability)')
plt.axhline(0.5, color='gray', linestyle='--', label='Decision Threshold (0.5)')
plt.axvline(0, color='gray', linestyle=':')
plt.title('Logistic Regression: Sigmoid Function & Decision Boundary')
plt.xlabel('Log-Odds / Linear Combination (z = w·X + b)')
plt.ylabel('Predicted Probability P(Y=1)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save figure to file instead of showing interactive GUI window
plt.savefig('sigmoid_plot.png', dpi=300)
print("\nPlot saved successfully as 'sigmoid_plot.png'")

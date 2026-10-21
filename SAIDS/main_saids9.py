import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import RandomOverSampler, SMOTE

# Load Dataset
dataset = pd.read_csv('creditcard.csv')

print(f"Total dataset samples: {len(dataset)}")
class_counts = dataset['Class'].value_counts()
print(f"Class 0        : {round(class_counts[0] / len(dataset) * 100, 2)}% of total")
print(f"Class 1 (Fraud): {round(class_counts[1] / len(dataset) * 100, 2)}% of total\n")

# Separate features and target
X = dataset.drop('Class', axis=1)
y = dataset['Class']

# Split train and test sets before resampling to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features for distance-based estimators like SMOTE
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Helper function to print resampled class balance
def print_resampled_stats(name, y_resampled):
    counts = y_resampled.value_counts()
    total = len(y_resampled)
    print(f"--- {name} ---")
    print(f"Total Train Samples: {total}")
    print(f"Class 0        : {round(counts[0] / total * 100, 2)}%")
    print(f"Class 1 (Fraud): {round(counts[1] / total * 100, 2)}%\n")

# Apply Random Under-Sampling
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X_train_scaled, y_train)
print_resampled_stats("Random Under-Sampling", y_rus)

# Apply Tomek Links Under-Sampling
tl = TomekLinks()
X_tl, y_tl = tl.fit_resample(X_train_scaled, y_train)
print_resampled_stats("Tomek Links Under-Sampling", y_tl)

# Apply Random Over-Sampling
ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(X_train_scaled, y_train)
print_resampled_stats("Random Over-Sampling", y_ros)

# Apply SMOTE Over-Sampling
sm = SMOTE(random_state=42)
X_sm, y_sm = sm.fit_resample(X_train_scaled, y_train)
print_resampled_stats("SMOTE Over-Sampling", y_sm)

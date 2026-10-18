import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
 
# Load the standard diabetes regression dataset
data = load_diabetes()
X = data.data
y = data.target
print("Dataset Shape:", X.shape)
 
# Initialize K-Fold cross-validation splitter with 5 splits and a fixed seed
kf = KFold(n_splits=5, shuffle=True, random_state=42)
 
# Define the dictionary of regression models to evaluate
models = {
    "Linear Regression": LinearRegression(), 
    "Ridge Regression": Ridge(alpha=1.0), 
    "Lasso Regression": Lasso(alpha=0.1, max_iter=10000)
}
 
# Loop through each model to perform cross-validation and display individual fold metrics
results = {}
for name, model in models.items():
    scores = cross_val_score(
        model,
        X,
        y,
        cv=kf,
        scoring="r2"
    )
    results[name] = scores
    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)
 
    # Print the performance score for each validation fold
    for i, score in enumerate(scores, start=1):
        print(f"Fold {i} Validation Score: {score:.4f}")
 
    # Calculate and print overall summary metrics for the model
    mean_score = scores.mean()
    std_score = scores.std()
    print(f"Mean Validation Score: {mean_score:.4f}")
    print(f"Standard Deviation: {std_score:.4f}")
 
# Structure cross-validation performance results into a structured list
comparison = []
for name, scores in results.items():
    comparison.append([
        name,
        scores[0],
        scores[1],
        scores[2],
        scores[3],
        scores[4],
        scores.mean(),
        scores.std()
    ])
 
# Convert performance list into a pandas DataFrame with distinct headers
comparison_df = pd.DataFrame(comparison, columns=["Model", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Mean Validation Score", "Standard Deviation"])
 
# Print the final side-by-side performance summary table to the console
print("\n\n")
print("=" * 100)
print("MODEL COMPARISON")
print("=" * 100)
print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
 
# Extract average scores for visualization
model_names = list(results.keys())
mean_scores = [
    results[name].mean()
    for name in model_names
]
 
# Generate a bar chart comparing the mean R² scores across all tested models
plt.figure(figsize=(8, 5))
plt.bar(model_names, mean_scores)
plt.xlabel("Machine Learning Model")
plt.ylabel("Mean Validation R² Score")
plt.title("K-Fold Cross Validation Model Comparison")
plt.ylim(0, 0.6)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load & inspect the dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
print(X.head())
print('Shape of the feature matrix:', X.shape)

# Standardise the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA (reduces 4 features to 2 components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print('Explained variance ratio:', pca.explained_variance_ratio_)
print('Total variance reatained:', sum(pca.explained_variance_ratio_) * 100)

# Visualize before vs after PCA
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.xlabel('Standardized Feature 1 (sepal length)')
plt.ylabel('Standardized Feature 2 (sepal width)')
plt.title('Before PCA')

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('After PCA')

plt.tight_layout()
plt.show()

# Train a classifier on the reduced data and evaluate
X_train, X_test, y_train, y_test = train_test_split(
X_pca, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (after PCA)')
plt.show()

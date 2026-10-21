import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris Dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add Species Column
df['Species'] = iris.target
df['Species'] = df['Species'].map({
    0: 'Setosa',
    1: 'Versicolor',
    2: 'Virginica'
})

# First 5 Rows
print(df.head())

# Dataset Info
print(df.info())

# Dataset Shape
print("Shape:", df.shape)

# Check Missing Values
print(df.isnull().sum())

# Statistical Summary
print(df.describe())

# Histogram Plot
df.hist(figsize=(8,6))
plt.show()

# Box Plot
plt.figure(figsize=(8,5))
sns.boxplot(data=df.iloc[:,:4])
plt.show()

# Pair Plot
sns.pairplot(df, hue="Species")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(6,5))
sns.heatmap(df.iloc[:,:4].corr(), annot=True, cmap="coolwarm")
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Read and display dataset
df = pd.read_csv('salary_data.csv')
print("First 5 Rows of Dataset:")
print(df.head())

# Clean missing values in feature and target columns
df = df.dropna(subset=['Years of Experience', 'Salary'])

# Select only ONE employee per experience level (removes vertical stacked dots)
df = df.drop_duplicates(subset=['Years of Experience'])

# Prepare the data for simple linear regression
X_linear = df[['Years of Experience']]
y_linear = df['Salary']

# Create and train the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_linear, y_linear)

# Make predictions
y_pred_linear = linear_model.predict(X_linear)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X_linear, y_linear, color='blue', alpha=0.8, label='Actual Data')
plt.plot(X_linear, y_pred_linear, color='red', linewidth=2, label='Linear Regression Line')
plt.title('Simple Linear Regression: Salary vs. Years of Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)
plt.show()

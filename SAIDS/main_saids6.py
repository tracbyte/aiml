import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Prepare data (X must be a 2D column, y is 1D)
X = np.array([10, 20, 30, 40, 50, 60, 70, 90, 100]).reshape(-1, 1)
y = np.array([15, 25, 35, 42, 56, 62, 71, 92, 105])

# Calculate Correlation Coefficient
correlation = np.corrcoef(X.flatten(), y)[0, 1]
print(f"Correlation Coefficient: {correlation:.2f}")

# rain Linear Regression Model
model = LinearRegression()
model.fit(X, y)

print(f"Slope (Coefficient): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# Predict sales for 80 spend
predicted_sales = model.predict([[80]])
print(f"Predicted Sales for 80 Spend: {predicted_sales[0]:.2f}")

# Plot the results
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.xlabel("Ad Spend")
plt.ylabel("Sales")
plt.legend()
plt.show()


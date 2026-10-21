import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate data inline using NumPy arrays
X = np.array([[1], [2], [3], [4], [5], [6]])  # Years of Experience
y = np.array([30000, 42000, 51000, 68000, 75000, 89000])  # Salary

# Initialize and train the model
model = LinearRegression()
model.fit(X, y)

# Predict and display visual results
y_pred = model.predict(X)
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
plt.legend()
plt.show()

import numpy as np
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2.5 * X + np.random.randn(100, 1) * 2

import pandas as pd
df = pd.DataFrame(data=np.hstack([X, y]), columns=['Feature', 'Target'])
print(df.head())

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

import matplotlib.pyplot as plt
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression Model')
plt.show()

import tensorflow as tf
from tensorflow import keras
nn_model = keras.Sequential([keras.layers.Dense(1, input_shape=[1])])
nn_model.compile(optimizer='sgd', loss='mean_squared_error')
nn_model.fit(X, y, epochs=10, verbose=0)
y_pred_nn = nn_model.predict(X)

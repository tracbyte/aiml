import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

print("1. DATA LOADING & PREPARATION")

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target
df["flower_name"] = df.target.apply(lambda x: iris.target_names[x])

print("Dataset Directory:", dir(iris))
print("Feature Names:", iris.feature_names)
print("Target Names:", iris.target_names)
print("\nFirst 5 rows of the DataFrame:")
print(df.head())

df0 = df[df.target == 0]
df1 = df[df.target == 1]
df2 = df[df.target == 2]

print("2. DATA VISUALIZATION")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.scatter("sepal length (cm)", "sepal width (cm)", data=df0, color="green", marker="+", label=iris.target_names[0])
plt.scatter("sepal length (cm)", "sepal width (cm)", data=df1, color="blue", marker=".", label=iris.target_names[1])
plt.legend()

plt.subplot(1, 2, 2)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.scatter("petal length (cm)", "petal width (cm)", data=df0, color="green", marker="+", label=iris.target_names[0])
plt.scatter("petal length (cm)", "petal width (cm)", data=df1, color="blue", marker=".", label=iris.target_names[1])
plt.legend()
plt.tight_layout()
plt.show()

print("3. TRAIN-TEST SPLIT")

X = df.drop(["target", "flower_name"], axis="columns")
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

print("4. BASE MODEL TRAINING & PREDICTION")

model = SVC()
model.fit(X_train, y_train)

base_score = model.score(X_test, y_test)
print(f"\nBase Model Score: {base_score}")

sample1 = pd.DataFrame([[4.8, 3.0, 1.5, 0.3]], columns=iris.feature_names)
sample2 = pd.DataFrame([[6.5, 3.0, 5.5, 2.0]], columns=iris.feature_names)
sample3 = pd.DataFrame([[5.5, 2.8, 4.2, 1.3]], columns=iris.feature_names)

pred_1 = model.predict(sample1)
pred_2 = model.predict(sample2)
pred_3 = model.predict(sample3)

print(f"Prediction 1 (Expected Setosa): {pred_1}")
print(f"Prediction 2 (Expected Virginica): {pred_2}")
print(f"Prediction 3 (Expected Versicolor): {pred_3}")

print("5. HYPERPARAMETER TUNING")

model_C100 = SVC(C=100)
model_C100.fit(X_train, y_train)
print(f"Model (C=100) Score: {model_C100.score(X_test, y_test)}")

model_C10 = SVC(C=10)
model_C10.fit(X_train, y_train)
print(f"Model (C=10) Score: {model_C10.score(X_test, y_test)}")

model_linear_kernel = SVC(kernel="linear")
model_linear_kernel.fit(X_train, y_train)
print(f"Model (Linear Kernel) Score: {model_linear_kernel.score(X_test, y_test)}")

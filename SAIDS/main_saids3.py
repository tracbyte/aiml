import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom, poisson, t, chi2, weibull_min

plt.style.use('ggplot')

# Generate normal random numbers
data = np.random.normal(loc=50, scale=10, size=1000)

plt.figure(figsize=(6,4))
plt.hist(data, bins=30, density=True, color='skyblue', edgecolor='black')
plt.title("Random Variable - Normal Distribution")
plt.xlabel("Values")
plt.ylabel("Probability Density")
plt.show()

# Normal Distribution curve
x = np.linspace(-4, 4, 200)
y = norm.pdf(x, 0, 1)

plt.figure(figsize=(6,4))
plt.plot(x, y, color='blue')
plt.title("Normal Distribution")
plt.xlabel("X")
plt.ylabel("Probability Density")
plt.grid(True)
plt.show()

# Binomial Distribution
n = 10
p = 0.5
x = np.arange(0, n+1)
y = binom.pmf(x, n, p)

plt.figure(figsize=(6,4))
plt.bar(x, y)
plt.title("Binomial Distribution")
plt.xlabel("Number of Successes")
plt.ylabel("Probability")
plt.show()

# Poisson Distribution
lam = 4
x = np.arange(0, 15)
y = poisson.pmf(x, lam)

plt.figure(figsize=(6,4))
plt.bar(x, y)
plt.title("Poisson Distribution")
plt.xlabel("Number of Events")
plt.ylabel("Probability")
plt.show()

# Student's t Distribution
x = np.linspace(-4, 4, 100)
y = t.pdf(x, df=10)

plt.figure(figsize=(6,4))
plt.plot(x, y)
plt.title("Student's t Distribution")
plt.xlabel("Random Variable")
plt.ylabel("Probability Density")
plt.grid(True)
plt.show()

# Chi-Square Distribution
x = np.linspace(0, 20, 100)
y = chi2.pdf(x, df=4)

plt.figure(figsize=(6,4))
plt.plot(x, y)
plt.title("Chi-Square Distribution")
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.grid(True)
plt.show()

# Weibull Distribution
x = np.linspace(0, 3, 100)
shape = 1.5
y = weibull_min.pdf(x, shape)

plt.figure(figsize=(6,4))
plt.plot(x, y)
plt.title("Weibull Distribution")
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.grid(True)
plt.show()

# Generate standard normal random numbers
data = np.random.normal(loc=0, scale=1, size=1000)

plt.figure(figsize=(6,4))
plt.hist(data, bins=30, density=True)
plt.title("Random Variables from Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

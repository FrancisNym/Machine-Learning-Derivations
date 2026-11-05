import numpy as np
import matplotlib.pyplot as plt

# Using a seed this time so results are reproducible
rng = np.random.default_rng(28)

# Make some random training and test data (in the future I will also test my algorithm on some real world data problems)
# Method for making random data:
# Fill our "X" array with random numbers
# Find the probility of being in each class based on our pre-set ideal "w" and "b"
# Pass this through a random binomial function to make the data imperfect (like the real world)
def make_data(n=500, true_w = np.array([2.0, 0.5]), true_b = 0.6):
    n_features = len(true_w)

    X = rng.normal(size=(n,n_features))
    prob = (1 / (1 + np.exp(-(X@true_w + true_b))))
    y = rng.binomial(1, prob)
    return X, y
X, y = make_data()

# Create training and test sets, to see how good our model is
split = int(0.8 * len(y))
X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]


print(round(0.6,0))
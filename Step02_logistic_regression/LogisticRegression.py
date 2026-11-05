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

def sigmoid(z):
    return (1 / (1 + np.exp(-(z))))

def predict(X, w, b):
    prob = sigmoid(X@w + b)
    return prob

def gradients(w, b, X, y):
    y_hat = predict(X, w, b)
    gradw = X.T @ (y-y_hat) / len(y)
    gradb = np.mean((y - y_hat))
    return gradw, gradb


def train():
    n_features = X.shape[1]

    w = np.zeros(n_features)
    b = 0.0
    learning_rate = 0.1
    mingrad = 0.0001
    steps = 0
    gradw = 1
    gradb = 1
    while steps < 10000 and (np.max(np.abs(gradw))>mingrad or abs(gradb) > mingrad):
        gradw, gradb = gradients(w, b, X_train, y_train)
        b = b - (gradb*learning_rate)
        w = w - gradw*learning_rate
        steps += 1
        if not (np.all(np.isfinite(w)) and np.isfinite(b)):
            break
    return w, b
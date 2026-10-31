import numpy as np
import matplotlib.pyplot as plt


# Generate random data with noise, so that data doesn't form a perfectly straight line
def make_data(n=30, true_w = 0.5, true_b = 2.0, std = 0.3,):
    xvals = np.random.uniform(0, 5, size=n)
    noise = np.random.normal(0, std, size=n)
    yvals = ((true_w*xvals)+true_b+noise)
    return xvals, yvals

# Generates predictions for "y" based on current w, b.
def predict(w, b, x):
    return ((w*x)+b)

# Finds the mean squared error for current w, b against all data points. 
# Not actually needed for training, (as shown in the derivation pdf) 
# but useful for interesting plots at the end of training
def loss(w, b, x, y):
    predcited = predict(w, b, x)
    error = predcited - y
    return np.mean(error**2)


# Finds the current gradients of w, b with respect to loss function. 
# (Applied partial derivative expression as shown in derivation pdf)
def gradients(w, b, x, y):
    wgrad = 2*(np.mean(x*(predict(w, b, x)-y)))
    bgrad = 2*(np.mean(predict(w, b, x)-y))
    return wgrad, bgrad


# The actual training loop
# Lots of this function is obsolete for finding actual values, such as "history" but is useful for plots
# Can experiment with different learning rates
# Follows the standard gradient descent update formula as shown in derivations pdf
def train(x, y, learning_rate = 0.05):
    w = 0.0
    b = 0.0
    steps = 0
    mingrad = 0.0001
    gradw = 1
    gradb = 1
    history = {"loss": [loss(w, b, x, y)], "w": [w], "b": [b]}
    while steps < 1000 and (abs(gradw) > mingrad or abs(gradb) > mingrad):
        gradw, gradb = gradients(w, b, x, y)
        w -= gradw*learning_rate
        b -= gradb*learning_rate
        steps += 1
        history["loss"].append(loss(w, b, x, y))
        history["w"].append(w)
        history["b"].append(b)
        if not (np.isfinite(w) and np.isfinite(b)):
            break
    return w, b, steps, history

# Plots 3 graphs of: 
# 1) data vs real line of best fit vs our fitted line
# 2) Shows our fitted loss vs the mathematical minimum loss calculated by np.polyfit
#    its possible for this because a simple line such as this can be mathematically solved
#    but the whole point of this code is to learn gradient descent, that is why I use gradient descent not polyfit.
# 3) How the parameters change over the total number of steps

def plots(w, b, x, y, history, true_w=0.5, true_b=2.0):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    xs = np.array([x.min(), x.max()])
    axes[0].scatter(x, y, label="data")
    axes[0].plot(xs, predict(w, b, xs), color="red", label="fitted line")
    axes[0].plot(xs, true_w * xs + true_b, "--", color="green", label="true line")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Fit")
    axes[0].legend()

    pw, pb = np.polyfit(x, y, 1)
    axes[1].plot(history["loss"])
    axes[1].axhline(loss(pw, pb, x, y), linestyle="--", color="gray", label="lowest possible (polyfit)")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("loss")
    axes[1].set_title("Loss")
    axes[1].legend()

    axes[2].plot(history["w"], label="w")
    axes[2].plot(history["b"], label="b")
    axes[2].axhline(true_w, linestyle="--", color="gray")
    axes[2].axhline(true_b, linestyle="--", color="gray")
    axes[2].set_xlabel("step")
    axes[2].set_title("Parameters")
    axes[2].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    xvals, yvals = make_data()
    w, b, steps, history = train(xvals, yvals)
    print(w, b, steps)
    print(np.polyfit(xvals, yvals, 1))
    plots(w, b, xvals, yvals, history)
import numpy as np
import matplotlib.pyplot as plt

def make_data(n=30, true_w = 0.5, true_b = 2.0, std = 0.3,):
    xvals = np.random.uniform(0, 5, size=n)
    noise = np.random.normal(0, std, size=n)
    yvals = ((true_w*xvals)+true_b+noise)
    return xvals, yvals

def predict(w, b, x):
    return ((w*x)+b)

def loss(w, b, x, y):
    predcited = predict(w, b, x)
    error = predcited - y
    return np.mean(error**2)

def gradients(w, b, x, y):
    wgrad = 2*(np.mean(x*(predict(w, b, x)-y)))
    bgrad = 2*(np.mean(predict(w, b, x)-y))
    return wgrad, bgrad

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
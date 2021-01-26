import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List


def generate_linear_data(N: int, mA: list, mB: list, sigmaA: float, sigmaB: float, target_values = [-1, 1]):
    covA = np.zeros((2, 2))
    np.fill_diagonal(covA, sigmaA)
    covB = np.zeros((2, 2))
    np.fill_diagonal(covB, sigmaB)

    classA = np.random.multivariate_normal(mA, covA, N)
    classB = np.random.multivariate_normal(mB, covB, N)
    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((target_values[0] * np.ones(classA.shape[0]), target_values[1] * np.ones(classB.shape[0])))

    inputs = np.append(inputs, np.ones((inputs.shape[0], 1)), axis = 1)

    indices = np.arange(inputs.shape[0])
    np.random.shuffle(indices)
    inputs = inputs[indices]
    targets = targets[indices]

    return {"inputs": inputs.T, "targets": np.atleast_2d(targets)}


def generate_nonlinear_data(N: int, mA: list, mB: list, sigmaA: float, sigmaB: float, target_values = [-1, 1]):
    covA = np.zeros((2, 2))
    np.fill_diagonal(covA, sigmaA)
    covB = np.zeros((2, 2))
    np.fill_diagonal(covB, sigmaB)

    classA_1 = np.random.multivariate_normal(mA, covA, N // 2)
    mA_2 = np.array(mA) * [-1, 1]
    classA_2 = np.random.multivariate_normal(mA_2, covA, N // 2)
    classA = np.concatenate((classA_1, classA_2))

    classB = np.random.multivariate_normal(mB, covB, N)
    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((target_values[0] * np.ones(classA.shape[0]), target_values[1] * np.ones(classB.shape[0])))

    inputs = np.append(inputs, np.ones((inputs.shape[0], 1)), axis = 1)

    indices = np.arange(inputs.shape[0])
    np.random.shuffle(indices)
    inputs = inputs[indices]
    targets = targets[indices]
    return {"inputs": inputs.T, "targets": np.atleast_2d(targets)}


def train_test_split(x: np.array, y: np.array, split: float, split_valance = None):
    if split_valance is None:
        split_valance = [0.5, 0.5]
    if sum(split_valance) != 1:
        raise ValueError("The valance should sum to 1")

    split = int((1 - split) * x.shape[1])
    x_valance, y_valance = split_valance
    x_train, x_val = x[:, :split], x[:, split:]
    y_train, y_val = y[:, :split], y[:, split:]

    return x_train, x_val, y_train, y_val


def plot_losses(losses: Dict, title: str) -> None:
    plt.plot(losses["val_losses"], label = "Validation loss")
    plt.plot(losses["epoch_losses"], label = "Train loss")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error loss")
    plt.legend()
    plt.title(title)
    plt.show()

    plt.plot(losses["val_accuracies"], label = "Validation accuracy")
    plt.plot(losses["epoch_accuracies"], label = "Train accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(title)
    plt.show()


def generate_gauss_data(x_range: Dict, y_range: Dict) -> Dict:
    """
    Generates data from the function f(x,y) = e^[−(x2+y2)/10] −0.5

    Parameters
    ----------
    x_range : Dictionary {"start": start, "end" : end, "steps" : steps}
    y_range : Dictionary {"start": start, "end" : end, "steps" : steps}

    Returns
    -------
    Return dictionary with x, y range as input and the results of the gaussian as targets --> {"inputs": inputs, "targets": targets}
    """

    x = np.atleast_2d(np.arange(x_range['start'], x_range['end'] + x_range['steps'], x_range['steps']))
    y = np.atleast_2d(np.arange(y_range['start'], y_range['end'] + x_range['steps'], y_range['steps']))
    z = np.exp(-x ** 2 / 10) * np.exp(-y.T ** 2 / 10) - 0.5

    targets = np.reshape(z, (1, x.shape[1] ** 2))
    xx, yy = np.meshgrid(x, y)
    inputs = np.append(np.reshape(xx, (1, x.shape[1] ** 2)), np.reshape(yy, (1, y.shape[1] ** 2)), axis = 0)
    return {"inputs": inputs, "targets": targets}


def plot_decision_boundary(x, y, model, steps=1000, cmap="Paired"):
    """
    Function to plot the decision boundary and data points of a model.
    Data points are colored based on their actual label.
    """
    cmap = plt.get_cmap(cmap)

    # Define region of interest by data limits
    x_min, x_max = x[0].min() - 1, x[0].max() + 1
    y_min, y_max = x[1].min() - 1, x[1].max() + 1
    x_span = np.linspace(x_min, x_max, steps)
    y_span = np.linspace(y_min, y_max, steps)
    xx, yy = np.meshgrid(x_span, y_span)

    # Make predictions across region of interest
    input_data = np.c_[xx.ravel(), yy.ravel()]
    input_data = np.append(input_data, np.ones((input_data.shape[0], 1)), axis=1).T
    labels = model.pred(input_data)

    # Plot decision boundary in region of interest
    z = labels.reshape(xx.shape)
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, z, cmap=cmap, alpha=0.5)

    # Plot data points.
    plt.scatter(x[0], x[1], c=list(map(lambda x: "r" if x == 1 else "b", y[0])))
    plt.title("Decision boundary for network.")

    plt.show()

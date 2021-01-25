import numpy as np
import matplotlib.pyplot as plt
from typing import Dict


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
    return {"inputs": inputs.T, "targets": targets}


def train_test_split(x: np.array, y: np.array, split: float):
    split = int((1-split)*x.shape[1])
    x_train, x_val = x[:, :split], x[:, split:]
    y_train, y_val = y[:, :split], y[:, split:]

    return x_train, x_val, y_train, y_val


def plot_losses(losses: Dict) -> None:
    plt.plot(losses["val_losses"], label="Validation loss")
    plt.plot(losses["epoch_losses"], label="Train loss")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error loss")
    plt.legend()
    plt.show()

    plt.plot(losses["val_accuracies"], label="Validation accuracy")
    plt.plot(losses["epoch_accuracies"], label="Train accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

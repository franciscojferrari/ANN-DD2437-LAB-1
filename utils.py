import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Dict, List
import glob
import os
import natsort
import imageio


def generate_linear_data(N: int, mA: list, mB: list, sigmaA: float, sigmaB: float, target_values = [-1, 1]):
    covA = np.zeros((2, 2))
    np.fill_diagonal(covA, sigmaA)
    covB = np.zeros((2, 2))
    np.fill_diagonal(covB, sigmaB)

    classA = np.random.multivariate_normal(mA, covA, N)
    classB = np.random.multivariate_normal(mB, covB, N)
    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((target_values[0] * np.ones(classA.shape[0]), target_values[1] * np.ones(classB.shape[0])))

    # inputs = np.append(inputs, np.ones((inputs.shape[0], 1)), axis = 1)

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

    # inputs = np.append(inputs, np.ones((inputs.shape[0], 1)), axis = 1)

    indices = np.arange(inputs.shape[0])
    np.random.shuffle(indices)
    inputs = inputs[indices]
    targets = targets[indices]
    return {"inputs": inputs.T, "targets": np.atleast_2d(targets)}


def train_test_split(x: np.array, y: np.array, split: float):
    split = int((1 - split) * x.shape[1])

    indices = np.arange(x.shape[1])
    np.random.shuffle(indices)

    train_indices, val_indices = sorted(indices[:split]), sorted(indices[split:])
    x_train, x_val = x[:, train_indices], x[:, val_indices]
    y_train, y_val = y[:, train_indices], y[:, val_indices]

    return x_train, x_val, y_train, y_val


def train_test_split_class(x: np.array, y: np.array, split: float, split_valance = None, special_case = False):
    if split_valance is None:
        split_valance = [0.5, 0.5]
    if sum(split_valance) != 1:
        raise ValueError("The valance should sum to 1")

    a_value, b_value = np.unique(y)

    indices_a = np.argwhere(y.flatten() == a_value).T.flatten()
    indices_b = np.argwhere(y.flatten() == b_value).T.flatten()
    # np.random.shuffle(indices_a)
    # np.random.shuffle(indices_b)

    split_a, split_b = int((1 - split * split_valance[0] * 2) * indices_a.shape[0]), int(
        (1 - split * split_valance[1] * 2) * indices_b.shape[0])

    train_indices_a, val_indices_a = indices_a[:split_a], indices_a[split_a:]
    train_indices_b, val_indices_b = indices_b[:split_b], indices_b[split_b:]

    x_train, x_val = np.concatenate((x[:, train_indices_a], x[:, train_indices_b]), axis = 1), np.concatenate(
        (x[:, val_indices_a], x[:, val_indices_a]), axis = 1)

    y_train, y_val = np.concatenate((y[:, train_indices_a], y[:, train_indices_b]), axis = 1), np.concatenate(
        (y[:, val_indices_a], y[:, val_indices_a]), axis = 1)

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
    return {"inputs": inputs, "targets": targets, "xx": xx, "yy": yy, "z": z, "size": x.shape[1]}


def plot_decision_boundary(x, y, model, steps = 1000, cmap = "Paired"):
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
    input_data = np.append(input_data, np.ones((input_data.shape[0], 1)), axis = 1).T
    labels = model.pred(input_data)

    # Plot decision boundary in region of interest
    z = labels.reshape(xx.shape)
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, z, cmap = cmap, alpha = 0.5)

    # Plot data points.
    plt.scatter(x[0], x[1], c = list(map(lambda x: "r" if x == 1 else "b", y[0])))
    plt.title("Decision boundary for network.")

    plt.show()


def plot_gaussian(input_data, predicted_z, title = "", gif = None):
    Z = input_data['z']
    norm = plt.Normalize(Z.min(), Z.max())
    colors = cm.viridis(norm(Z))
    colors[:, :, 3] = 0.5
    rcount, ccount, _ = colors.shape

    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    surf = ax.plot_surface(input_data['xx'], input_data['yy'], Z, rcount = rcount, ccount = ccount,
                           facecolors = colors, shade = False)
    surf.set_facecolor((0, 0, 0, 0))
    ax.plot_surface(input_data['xx'], input_data['yy'], predicted_z, color = "red", alpha = 0.5)
    plt.title(title)
    if gif is None:
        plt.show()
    else:
        filename = f'images/{title}_{gif["epoch"]}_{gif["seq"]}.png'
        plt.savefig(filename)
        plt.close()


def plot_gif(outputName, repeat_frames = 5):
    # filenames = sorted(glob.glob('images/*.png'))
    all_filenames = natsort.natsorted(glob.glob("images/*.png"))
    step = int((1 / repeat_frames) if repeat_frames < 1 else 1)
    repeat_frames = int(repeat_frames if repeat_frames >= 1 else 1)
    filenames = [item for item in all_filenames for i in range(0, repeat_frames)][::step]

    with imageio.get_writer(f"images/{outputName}.gif", mode = "I") as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    for filename in set(all_filenames):
        os.remove(filename)

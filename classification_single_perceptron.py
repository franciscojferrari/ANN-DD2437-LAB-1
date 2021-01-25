import numpy as np
from matplotlib import pyplot as plt
import imageio
import random
import glob
import os
import natsort


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

    return {"inputs": inputs.T, "targets": targets}


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


def perceptron_learning_batch(inputs, targets, learning_rate = 0.1, epochs = 6):
    W = np.random.normal(0, 1, inputs.shape[0])
    plot_hyperplane(inputs, W, targets, "perceptron_learning_batch")
    for epoch in range(epochs):
        prediction = W @ inputs
        prediction[prediction >= 0] = 1
        prediction[prediction < 0] = -1
        error = (targets - prediction)
        W = W + inputs @ (learning_rate * error)
        plot_hyperplane(inputs, W, targets, "perceptron_learning_batch")
        print(W)

    return prediction


def perceptron_learning_sequential(X, T, learning_rate = 0.001, epochs = 5):
    W = np.random.normal(0, 1, X.shape[0])
    for epoch in range(epochs):
        sum_error = 0
        for i, x in enumerate(X.T):
            y_i = W @ x
            y_i = 1 if y_i >= 0 else 0
            error = y_i - T[i]
            sum_error += error ** 2
            W = W - learning_rate * (error * x.T)
            print(W)


def delta_learning_batch(X, T, learning_rate = 0.001, epochs = 5):
    W = np.random.normal(0, 1, X.shape[0])
    plot_hyperplane(X, W, T, f"0delta_learning_batch", gif = {"epoch": "00", "seq": 0})
    squared_error = []
    for epoch in range(epochs):
        prediction = W @ X
        error = (prediction - T)
        squared_error.append((error @ error.T) / 2)
        W = W - learning_rate * (error) @ X.T
        plot_hyperplane(X, W, T, f"delta_learning_batch - epoch:{epoch}", gif = {"epoch": epoch, "seq": 0})
    print(squared_error)
    plot_gif("delta_learning_batch", repeat_frames = 5)


def delta_learning_sequential(X, T, learning_rate = 0.001, epochs = 5):
    W = np.random.normal(0, 1, X.shape[0])
    squared_error = []
    plot_hyperplane(X, W, T, f"0delta_learning_sequential", gif = {"epoch": "00", "seq": 0})
    for epoch in range(epochs):
        sum_error = 0
        for i, x in enumerate(X.T):
            y_i = W @ x
            error = y_i - T[i]
            sum_error += error ** 2
            W = W - learning_rate * (error * x.T)
            plot_hyperplane(X, W, T, f"delta_learning_sequential - epoch:{epoch}", gif = {"epoch": epoch, "seq": i})
            print(W)
        # plot_hyperplane(X, W, T, f"delta_learning_sequential - epoch:{epoch}", gif = {"epoch": epoch, "seq": i})
        squared_error.append(sum_error)
    plot_gif("delta_learning_sequential", repeat_frames = 1)
    print(squared_error)


def plot_gif(outputName, repeat_frames = 5):
    # filenames = sorted(glob.glob('images/*.png'))
    filenames = natsort.natsorted(glob.glob('images/*.png'))
    filenames = [item for item in filenames for i in range(repeat_frames)]
    with imageio.get_writer(f'images/{outputName}.gif', mode = 'I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    for filename in set(filenames):
        os.remove(filename)


def plot_hyperplane(data, weights, targets, title, gif = None):
    plt.scatter(data[0], data[1], c = list(map(lambda x: 'r' if x == 1 else 'b', targets)))
    plt.title(title)
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    xx = np.linspace(-2, 2)
    a = -weights[0] / weights[1]
    bias = -weights[2] / weights[1]
    yy = a * xx + bias
    plt.plot(xx, yy, 'k-')
    plt.ylim([ymin * 1.2, ymax * 1.2])
    plt.xlim([xmin * 1.2, xmax * 1.2])

    if gif is not None:
        filename = f'images/{title}_{gif["epoch"]}_{gif["seq"]}.png'
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def exe_3_1_2():
    np.random.seed(42)
    n = 100
    mA = [2.0, 0.5]
    sigmaA = 0.5
    mB = [-2.0, -1.5]
    sigmaB = 0.5
    data = generate_linear_data(n, mA, mB, sigmaA, sigmaB, target_values = [1, -1])
    inputs, targets = data['inputs'], data['targets']

    # perceptron_learning_batch(inputs, targets)
    # delta_learning_batch(inputs, targets, learning_rate = 0.001, epochs = 30)
    # perceptron_learning_sequential(inputs, targets)
    delta_learning_sequential(inputs, targets, learning_rate = 0.001, epochs = 10)


def exe_3_1_3():
    np.random.seed(42)
    n = 100
    mA = [1.0, 0.3]
    sigmaA = 0.2
    mB = [0, -0.1]
    sigmaB = 0.3
    data_non_linear = generate_nonlinear_data(n, mA, mB, sigmaA, sigmaB, target_values = [1, -1])
    inputs_non_linear, targets_non_linear = data_non_linear['inputs'], data_non_linear['targets']


def main():
    exe_3_1_3()


# exe_3_1_2()


if __name__ == '__main__':
    main()

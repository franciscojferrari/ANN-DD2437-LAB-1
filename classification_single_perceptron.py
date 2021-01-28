import numpy as np
from matplotlib import pyplot as plt

from utils import generate_linear_data, generate_nonlinear_data, plot_gif
from sklearn.metrics import mean_squared_error, accuracy_score


def perceptron_learning_batch(inputs, targets, learning_rate = 0.1, epochs = 6, plot_gifs = False):
    inputs = np.append(inputs.T, np.ones((inputs.shape[1], 1)), axis = 1).T
    W = np.random.randn(1, inputs.shape[0])

    if plot_gifs:
        plot_hyperplane(inputs, W, targets, f"0perceptron_learning_batch", gif = {"epoch": "00", "seq": 0})

    errors, accuracies = [], []
    for epoch in range(epochs):
        prediction = W @ inputs
        prediction[prediction >= 0] = 1
        prediction[prediction < 0] = -1
        error = targets - prediction

        W = W + learning_rate * (error @ inputs.T)
        errors.append(mean_squared_error(targets.flatten(), prediction.flatten()))
        accuracies.append(accuracy_score(targets.flatten(), prediction.flatten()))
        if plot_gifs:
            plot_hyperplane(inputs, W, targets, f"perceptron_learning_batch - {epoch}",
                            gif = {"epoch": epoch, "seq": 0})
    if plot_gifs:
        plot_gif("perceptron_learning_batch", repeat_frames = 0.5)

    return {"epoch_errors": errors, "epoch_accuracies": accuracies}


def perceptron_learning_sequential(X, T, learning_rate = 0.001, epochs = 5):
    X = np.append(X.T, np.ones((X.shape[1], 1)), axis = 1).T
    W = np.random.randn(1, X.shape[0])

    plot_hyperplane(X, W, T, f"0perceptron_learning_sequential", gif = {"epoch": "00", "seq": 0})
    errors, accuracies = [], []
    for epoch in range(epochs):
        sum_error = 0
        for i, x in enumerate(X.T):
            y_i = W @ x
            y_i = 1 if y_i >= 0 else 0
            error = T[0][i] - y_i
            W = W + learning_rate * (error * x.T)

        prediction = W @ X
        prediction[prediction >= 0] = 1
        prediction[prediction < 0] = -1

        plot_hyperplane(X, W, T, f"perceptron_learning_sequential - {epoch}",
                        gif = {"epoch": epoch, "seq": 0})

        errors.append(mean_squared_error(T.flatten(), prediction.flatten()))
        accuracies.append(accuracy_score(T.flatten(), prediction.flatten()))

    plot_gif("perceptron_learning_sequential", repeat_frames = 0.5)
    return {"epoch_errors": errors, "epoch_accuracies": accuracies}


def step_function_pred(X):
    """Added for plotting purposes."""
    x = np.copy(X)
    x[x >= 0] = 1
    x[x < 0] = 0
    return x


def delta_learning_batch(X, T, learning_rate = 0.001, epochs = 5, plot_gifs = False):
    X = np.append(X.T, np.ones((X.shape[1], 1)), axis = 1).T
    W = np.random.normal(0, 1, X.shape[0])
    if plot_gifs:
        plot_hyperplane(X, W, T, f"0delta_learning_batch", gif = {"epoch": "00", "seq": 0})
    squared_error = []
    errors, accuracies = [], []

    for epoch in range(epochs):
        prediction = W @ X
        error = T - prediction
        W = W + learning_rate * (error @ X.T)
        results = step_function_pred(prediction.flatten())
        errors.append(mean_squared_error(T.flatten(), results))
        accuracies.append(accuracy_score(T.flatten(), results))

        if plot_gifs:
            plot_hyperplane(X, W, T, f"delta_learning_batch - epoch:{epoch}", gif = {"epoch": epoch, "seq": 0}, )

    if plot_gifs:
        plot_gif("delta_learning_batch", repeat_frames = 1)

    return {"epoch_errors": errors, "epoch_accuracies": accuracies}


def delta_learning_sequential(X, T, learning_rate = 0.001, epochs = 5):
    W = np.random.normal(0, 1, X.shape[0])
    squared_error = []
    plot_hyperplane(
        X, W, T, f"0delta_learning_sequential", gif = {"epoch": "00", "seq": 0}
    )
    for epoch in range(epochs):
        sum_error = 0
        for i, x in enumerate(X.T):
            y_i = W @ x
            error = y_i - T[i]
            sum_error += error ** 2
            W = W - learning_rate * (error * x.T)
            plot_hyperplane(
                X,
                W,
                T,
                f"delta_learning_sequential - epoch:{epoch}",
                gif = {"epoch": epoch, "seq": i},
            )
            print(W)
        # plot_hyperplane(X, W, T, f"delta_learning_sequential - epoch:{epoch}", gif = {"epoch": epoch, "seq": i})
        squared_error.append(sum_error)
    plot_gif("delta_learning_sequential", repeat_frames = 1)
    print(squared_error)


def plot_hyperplane(data, weights, targets, title, gif = None):
    plt.scatter(
        data[0], data[1], c = list(map(lambda x: "r" if x == 1 else "b", targets[0]))
    )
    plt.title(title)
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    xx = np.linspace(-2, 2)
    a = -weights[0][0] / weights[0][1]
    bias = -weights[0][2] / weights[0][1]
    yy = a * xx + bias
    plt.plot(xx, yy, "k-")
    plt.ylim([ymin * 1.2, ymax * 1.2])
    plt.xlim([xmin * 1.2, xmax * 1.2])

    if gif is not None:
        filename = f'images/{title}_{gif["epoch"]}_{gif["seq"]}.png'
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_errors(losses, title):
    plt.plot(losses["epoch_errors"], label = "MSE")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error loss")
    plt.legend()
    plt.title(title)
    plt.show()

    plt.plot(losses["epoch_accuracies"], label = "Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(title)
    plt.show()


def exe_3_1_2():
    np.random.seed(999)
    n = 100
    mA = [1.0, 0.5]
    sigmaA = 0.5
    mB = [-1.0, 0]
    sigmaB = 0.5
    data = generate_linear_data(n, mA, mB, sigmaA, sigmaB, target_values = [1, -1])
    inputs, targets = data["inputs"], data["targets"]

    # perceptron_learning_batch_results = perceptron_learning_batch(inputs, targets, learning_rate = 0.001, epochs = 200)
    # plot_errors(perceptron_learning_batch_results, "perceptron_learning_batch")

    delta_learning_batch_results = delta_learning_batch(inputs, targets, learning_rate = 0.001, epochs = 200)

    # perceptron_learning_sequential_results = perceptron_learning_sequential(inputs, targets, epochs = 200)
    # plot_errors(perceptron_learning_sequential_results, "perceptron_learning_batch")
    # delta_learning_sequential(inputs, targets, learning_rate = 0.001, epochs = 10)


def exe_3_1_3():
    np.random.seed(42)
    n = 100
    mA = [1.0, 0.3]
    sigmaA = 0.2
    mB = [0, -0.1]
    sigmaB = 0.3
    data_non_linear = generate_nonlinear_data(
        n, mA, mB, sigmaA, sigmaB, target_values = [1, -1]
    )
    inputs_non_linear, targets_non_linear = (
        data_non_linear["inputs"],
        data_non_linear["targets"],
    )


def main():
    # exe_3_1_3()
    exe_3_1_2()


if __name__ == "__main__":
    main()

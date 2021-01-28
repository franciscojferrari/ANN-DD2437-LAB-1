import numpy as np
import matplotlib.pyplot as plt
from utils import generate_linear_data, generate_nonlinear_data, train_test_split, plot_losses, plot_decision_boundary, \
    train_test_split_class
from sklearn.metrics import mean_squared_error, accuracy_score


class NueralNet:
    def __init__(self, x, y, hidden_layer_size, output_layer_size, lr = 0.001, is_binary = True):
        self.X = np.append(x.T, np.ones((x.shape[1], 1)), axis = 1).T
        self.Y = np.atleast_2d(y)
        self.Yp = np.zeros((1, self.Y.shape[1]))

        self.is_binary = is_binary
        # (input, hidden, output)
        self.dims = [self.X.shape[0], hidden_layer_size, output_layer_size]
        self.param = {}
        self.ch = {}

        self.loss = []
        self.lr = lr
        self.momentum = 0.9
        self.samples = self.Y.shape[1]

        self.sequential = False  # Using batch or sequential learning.
        self.batch_size = 1 if self.sequential else self.samples
        self.initWeights()

    @staticmethod
    def sigmoid(X):
        return (2 / (1 + np.exp(-X))) - 1

    @staticmethod
    def sigmoid_prime(X):
        return ((1 + X) * (1 - X)) / 2

    @staticmethod
    def step_function(X):
        x = np.copy(X)
        x[x >= 0] = 1
        x[x < 0] = -1
        return x

    @staticmethod
    def step_function_pred(X):
        """Added for plotting purposes."""
        x = np.copy(X)
        x[x >= 0] = 1
        x[x < 0] = 0
        return x

    def initWeights(self):
        # W shape (V size, N features) or (hidden layer size, X shape)
        self.W = np.random.randn(self.dims[1], self.dims[0])
        # V shape (target shape, hidden layer dimension)
        self.V = np.random.randn(self.dims[2], self.dims[1] + 1)
        # self.V = np.append(self.V, np.ones((self.V.shape[0], 1)), axis = 1)
        self.dw = None

    def corss_entropy_loss(self, Yp):
        loss = (1. / self.samples) * (((self.Y + 1) / 2) @ np.log(Yp).T - (
                (1 - ((self.Y + 1) / 2)) @ np.log(1 - Yp).T))
        return loss

    def fowardPass(self, X = None, Y = None, include_bias = False):
        X = (self.X if X is None else X)
        if include_bias:
            X = np.append(X.T, np.ones((X.shape[1], 1)), axis = 1).T

        hin = self.W @ X
        hout = NueralNet.sigmoid(hin)
        hout = np.append(hout.T, np.ones((hout.shape[1], 1)), axis = 1).T  # maybe here
        self.ch['hin'], self.ch['hout'] = hin, hout

        oin = self.V @ hout
        out = NueralNet.sigmoid(oin)
        self.ch['oin'], self.ch['out'] = oin, out

        self.Yp = out
        loss = mean_squared_error(self.Y if Y is None else Y, out)

        return {"Yp": out, "loss": loss}

    def backwardsPass(self, X = None, Y = None):
        delta_o = (self.Yp - (self.Y if Y is None else Y)) * NueralNet.sigmoid_prime(self.Yp)
        delta_h = (self.V.T @ delta_o) * NueralNet.sigmoid_prime(self.ch['hout'])
        delta_h = delta_h[:-1, :]

        if self.dw is None:
            self.dw = delta_h @ (self.X if X is None else X).T
            self.dv = delta_o @ self.ch['hout'].T
        else:
            self.dw = (self.momentum * self.dw) - ((1 - self.momentum) * (delta_h @ (self.X if X is None else X).T))
            self.dv = (self.momentum * self.dv) - ((1 - self.momentum) * (delta_o @ self.ch['hout'].T))

        self.W = self.W + (self.lr * self.dw)
        self.V = self.V + (self.lr * self.dv)

    def train_network(self, epochs, x_val = None, y_val = None):
        batch_losses, batch_accuracies, batch_out, epoch_losses, epoch_accuracies, val_losses, val_accuracies = [], [], [], [], [], [], []
        Xbatches = np.array_split(self.X, self.samples // self.batch_size, axis = 1)
        Ybatches = np.array_split(self.Y, self.samples // self.batch_size, axis = 1)

        for epoch in range(epochs):

            for Xbatch, Ybatch in list(zip(Xbatches, Ybatches)):
                forward = self.fowardPass(Xbatch, Ybatch)
                self.backwardsPass(Xbatch, Ybatch)

                batch_losses.append(forward['loss'])
                if self.is_binary:
                    predictions = NueralNet.step_function(forward['Yp'])
                    batch_accuracies.append(accuracy_score(Ybatch[0], predictions[0]))

            forward = self.fowardPass()
            epoch_losses.append(forward['loss'])

            predictions = forward['Yp']
            if self.is_binary:
                predictions = NueralNet.step_function(predictions)
                epoch_accuracies.append(accuracy_score(self.Y[0], predictions[0]))

            batch_out.append(predictions)

            if type(x_val) and type(y_val) is np.ndarray:
                val_results = self.predict(x_val, y_val)
                val_accuracies.append(val_results['acc'])
                val_losses.append(val_results['loss'])

        return {"batch_losses": batch_losses, "batch_accuracies": batch_accuracies,
                "epoch_losses": epoch_losses, "epoch_accuracies": epoch_accuracies,
                "val_accuracies": val_accuracies, "val_losses": val_losses,
                "batch_out": batch_out}

    def predict(self, X, y):
        forward = self.fowardPass(X, y, include_bias = True)
        predictions = forward['Yp']
        accuracy = [0]
        if self.is_binary:
            predictions = NueralNet.step_function(forward['Yp'])
            accuracy = accuracy_score(y[0], predictions[0])

        return {"pred": predictions, "acc": accuracy, "loss": forward['loss']}

    def forward_pred(self, x):
        """Added for plotting."""
        hin = self.W @ x
        hout = self.sigmoid(hin)
        hout = np.append(hout.T, np.ones((hout.shape[1], 1)), axis = 1).T
        self.ch["hin"], self.ch["hout"] = hin, hout

        oin = self.V @ hout
        out = NueralNet.sigmoid(oin)
        self.ch["oin"], self.ch["out"] = oin, out

        return out

    def pred(self, x):
        """Added for plotting."""
        forward = self.forward_pred(x)
        predictions = NueralNet.step_function_pred(forward)

        return predictions


def exe_3_2_1():
    """Experiment varying the number of hidden layers."""

    # Params.
    epochs = 500
    n = 110
    mA, mB = [2.0, 0.5], [-2.0, -1.5]
    sigmaA, sigmaB = 0.5, 0.5
    output_layer_size = 1

    # Experiment 1 - Changing number of hidden nodes

    train_accuracies, train_losses = [], []

    # Generate data.
    data = generate_linear_data(n, mA, mB, sigmaA, sigmaB, target_values = [1, -1])
    inputs, targets = data['inputs'], data['targets']

    x_train, x_val, y_train, y_val = train_test_split_class(inputs, targets, split = 0.2, split_valance = [0.5, 0.5])

    # Create and train network with different number of hidden layers.
    for hidden_layer_shape in range(1, 30):
        model = NueralNet(x_train, y_train, hidden_layer_size = hidden_layer_shape,
                          output_layer_size = output_layer_size)
        losses = model.train_network(epochs, x_val, y_val)
        train_accuracies.append(losses["epoch_accuracies"][-1])
        train_losses.append(losses["epoch_losses"][-1])

    # Plot results.
    plt.plot(train_losses, label = "Train losses")
    plt.xlabel("Number of hidden nodes.")
    plt.ylabel("Mean Squared Error loss")
    plt.show()

    plt.plot(train_accuracies, label = "Train Accuracies")
    plt.xlabel("Number of hidden nodes.")
    plt.ylabel("Mean Squared Error loss")
    plt.show()

    # Experiment 2 - Changing train test split ratio

    split_ratios = [0.2, 0.4, 0.6, 0.8]
    hidden_layer_shape = 8

    for split in split_ratios:
        # Generate data.
        # data = generate_linear_data(n, mA, mB, sigmaA, sigmaB, target_values = [1, -1])
        data = generate_nonlinear_data(n, mA, mB, sigmaA, sigmaB, target_values = [1, -1])
        inputs, targets = data['inputs'], data['targets']
        # x_train, x_val, y_train, y_val = train_test_split(inputs, targets, split = split)
        x_train, x_val, y_train, y_val = train_test_split_class(inputs, targets, split = split,
                                                                split_valance = [0.5, 0.5])

        model = NueralNet(x_train, y_train, hidden_layer_size = hidden_layer_shape,
                          output_layer_size = output_layer_size)
        losses = model.train_network(epochs, x_val, y_val)

        plot_losses(losses, f"Data split: {split}")

    # Experiment 3  - Change both train test split ratio and number of hidden nodes.

    # Experiment 4 - Check difference between sequential and batch learning.

    # Experiment 5 - Plot decision boundary.
    hidden_layer_shape = 18
    output_layer_size = 1
    epochs = 500

    data = generate_nonlinear_data(n, mA, mB, sigmaA, sigmaB, target_values = [1, -1])
    inputs, targets = data["inputs"], data["targets"]
    x_train, x_val, y_train, y_val = train_test_split(inputs, targets, split = 0.2)
    model = NueralNet(
        x_train,
        y_train,
        hidden_layer_size = hidden_layer_shape,
        output_layer_size = output_layer_size,
    )
    model.train_network(epochs)
    plot_decision_boundary(inputs, targets, model)


def main():
    np.random.seed(42)
    exe_3_2_1()

    # # Set hyperparameters.
    # n = 110
    # mA = [2.0, 0.5]
    # sigmaA = 0.5
    # mB = [-2.0, -1.5]
    # sigmaB = 0.5
    #
    # # Generate data.
    # data = generate_linear_data(n, mA, mB, sigmaA, sigmaB, target_values = [1, -1])
    # inputs, targets = data['inputs'], data['targets']
    #
    # # TODO: Proper dataset split like 3.1.2 (class valance, etc)
    # x_train, x_val, y_train, y_val = train_test_split(inputs, targets, split = 0.33)
    # aa = NueralNet(x_train, y_train, hidden_layer_size = 5, output_layer_size = 1)
    # losses = aa.train_network(100, x_val, y_val)
    # plot_losses(losses)


if __name__ == '__main__':
    main()

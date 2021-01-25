import numpy as np
import matplotlib as plt
from utils import generate_linear_data, generate_nonlinear_data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


class NueralNet:
    def __init__(self, x, y):
        self.X = x
        self.Y = np.atleast_2d(y)
        self.Yp = np.zeros((1, self.Y.shape[1]))

        self.L = 2
        # (input, hidden, output)
        self.dims = [self.X.shape[0], 5, 1]
        self.param = {}
        self.ch = {}
        self.batch_size = 7

        self.loss = []
        self.lr = 0.001
        self.momentum = 0.9
        self.samples = self.Y.shape[1]
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

    def initWeights(self):
        np.random.seed(42)
        # W shape (V size, N features) or (hidden layer size, X shape)
        self.W = np.random.randn(self.dims[1], self.dims[0])
        # V shape (target shape, hidden layer dimension)
        self.V = np.random.randn(self.dims[2], self.dims[1] + 1)
        # self.V = np.append(self.V, np.ones((self.V.shape[0], 1)), axis = 1)
        self.dw = None

    def corss_entropy_loss(self, Yp):
        loss = (1. / self.samples) * (((self.Y + 1) / 2) @ np.log(Yp).T - ((1 - ((self.Y + 1) / 2)) @ np.log(1 - Yp).T))
        return loss

    def fowardPass(self, X = None, Y = None):
        hin = self.W @ (self.X if X is None else X)
        hout = NueralNet.sigmoid(hin)
        hout = np.append(hout.T, np.ones((hout.shape[1], 1)), axis = 1).T
        self.ch['hin'], self.ch['hout'] = hin, hout

        oin = self.V @ hout
        out = NueralNet.sigmoid(oin)
        self.ch['oin'], self.ch['out'] = oin, out

        self.Yp = out
        loss = mean_squared_error(self.Y if Y is None else Y, out)
        return {"Yp": self.Yp, "loss": loss}

    def backwardsPass(self, X = None, Y = None):
        delta_o = ((self.Y if Y is None else Y) - self.Yp) * NueralNet.sigmoid_prime(self.ch['oin'])
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
        batch_losses, batch_accuracies, epoch_losses, epoch_accuracies, val_losses, val_accuracies = [], [], [], [], [], []
        Xbatches = np.array_split(self.X, self.samples // self.batch_size, axis = 1)
        Ybatches = np.array_split(self.Y, self.samples // self.batch_size, axis = 1)

        for epoch in range(epochs):
            for Xbatch, Ybatch in list(zip(Xbatches, Ybatches)):
                forward = self.fowardPass(Xbatch, Ybatch)
                self.backwardsPass(Xbatch, Ybatch)

                batch_losses.append(forward['loss'])
                predictions = NueralNet.step_function(forward['Yp'])
                batch_accuracies.append(accuracy_score(Ybatch[0], predictions[0]))

            forward = self.fowardPass()
            epoch_losses.append(forward['loss'])
            predictions = NueralNet.step_function(forward['Yp'])
            epoch_accuracies.append(accuracy_score(self.Y[0], predictions[0]))
            if x_val and y_val is not None:
                self.predict(x_val, y_val)

        return {"batch_losses": batch_losses, "batch_accuracies": batch_accuracies, "epoch_losses": epoch_losses,
                "epoch_accuracies": epoch_accuracies}

    def predict(self, X, y):
        forward = self.fowardPass(X)
        predictions = NueralNet.step_function(forward['Yp'])
        accuracy = accuracy_score(y[0], predictions[0])
        return {"pred": predictions, "acc": accuracy, "loss": forward['loss']}


def main():
    # np.random.seed(42)
    # n = 100
    # mA = [1.0, 0.3]
    # sigmaA = 0.2
    # mB = [0, -0.1]
    # sigmaB = 0.3
    # data_non_linear = generate_linear_data(n, mA, mB, sigmaA, sigmaB, target_values = [1, -1])
    # inputs_non_linear, targets_non_linear = data_non_linear['inputs'], data_non_linear['targets']
    np.random.seed(42)
    n = 100
    mA = [2.0, 0.5]
    sigmaA = 0.5
    mB = [-2.0, -1.5]
    sigmaB = 0.5
    data = generate_linear_data(n, mA, mB, sigmaA, sigmaB, target_values = [1, -1])
    inputs, targets = data['inputs'], data['targets']

    aa = NueralNet(inputs, targets)
    aa.train_network(100)


if __name__ == '__main__':
    main()

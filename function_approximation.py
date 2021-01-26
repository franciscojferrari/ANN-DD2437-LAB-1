from utils import generate_gauss_data, train_test_split
import classification_two_layers as nn
import numpy as np
import matplotlib.pyplot as plt


def main():
    x = y = {"start": -5, "end": 5, "steps": 0.5}
    data = generate_gauss_data(x, y)
    inputs, targets = data["inputs"], data['targets']
    x_train, x_val, y_train, y_val = train_test_split(inputs, targets, 0.20)
    # x_train = inputs
    # y_train = targets

    network = nn.NueralNet(x_train, y_train, hidden_layer_size = 20, output_layer_size = 1, is_binary = False)
    losses = network.train_network(epochs = 1000, x_val = x_val, y_val = y_val)

    # Plot results.
    plt.plot(losses['batch_losses'], label = "Train losses")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error loss")
    plt.show()


if __name__ == '__main__':
    main()

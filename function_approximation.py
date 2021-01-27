from utils import generate_gauss_data, train_test_split, plot_gaussian, plot_gif
import classification_two_layers as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def main():
    x = y = {"start": -5, "end": 5, "steps": 0.5}
    data = generate_gauss_data(x, y)
    inputs, targets = data["inputs"], data['targets']
    x_train, x_val, y_train, y_val = train_test_split(inputs, targets, 0.20)
    # x_train = inputs
    # y_train = targets

    losses = []

    for layer_size in range(1, 25):
        network = nn.NueralNet(x_train, y_train, hidden_layer_size = layer_size, output_layer_size = 1,
                               is_binary = False)
        nnTrainResults = network.train_network(epochs = 1000)

        results = network.fowardPass(inputs, targets, include_bias = True)
        losses.append(results['loss'])

        batch_out = np.reshape(results["Yp"], (data['size'], data['size']))

        plot_gaussian(data, batch_out, f"Gaussian Out - hidden_layer_size {layer_size}",
                      gif = {"epoch": 1000, "seq": 0})

    # Plot results.
    plt.plot(losses, label = "Losses")
    plt.xlabel("Number of hidden nodes.")
    plt.ylabel("Mean Squared Error loss")
    plt.show()


# for epoch, batch in enumerate(nnTrainResults["batch_out"]):
#     batch_out = np.reshape(batch, (data['size'], data['size']))
#     plot_gaussian(data, batch_out, f"Gaussian Out - epoch:{epoch}", gif = {"epoch": epoch, "seq": 0})
# plot_gif("gaussian_batch", repeat_frames = 1)


if __name__ == '__main__':
    main()

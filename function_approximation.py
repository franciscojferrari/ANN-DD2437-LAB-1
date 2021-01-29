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

    #################### NETWORK SIZE ANALYSIS #####################
    # losses, batch_losses = [], []
    # for layer_size in range(1, 25):
    #     network = nn.NueralNet(x_train, y_train, hidden_layer_size = layer_size, output_layer_size = 1,
    #                            is_binary = False)
    #     nnTrainResults = network.train_network(epochs = 400)
    #
    #     results = network.fowardPass(inputs, targets, include_bias = True)
    #     losses.append(results['loss'])
    #
    #     batch_out = np.reshape(results["Yp"], (data['size'], data['size']))
    #     # plot_gaussian(data, batch_out, f"Gaussian Out - hidden_layer_size {layer_size}",
    #     #               gif = {"epoch": 1000, "seq": 0})
    #     batch_losses.append(nnTrainResults['batch_losses'])
    #
    # for i in [2, 4, 5, 7, 10, 15, 18, 23]:
    #     # Plot results.
    #     plt.plot(batch_losses[i], label = f" N. Hidden Layer {i}")
    #     plt.xlabel("Epochs")
    #     plt.ylabel("Mean Squared Error loss")
    #     plt.legend(loc = 'best')
    # plt.show()

    #################### SPLIT ANALYSIS #########################
    split_ratios = [0.8]
    hidden_layer_shape = 15

    for split in split_ratios:
        x_train, x_val, y_train, y_val = train_test_split(inputs, targets, split)
        network = nn.NueralNet(x_train, y_train, hidden_layer_size = hidden_layer_shape, output_layer_size = 1,
                               is_binary = False)
        losses = network.train_network(1000, inputs, targets)

        plt.plot(losses["val_losses"], label = "Validation loss")
        plt.plot(losses["epoch_losses"], label = "Train loss")
        plt.xlabel("Epochs")
        plt.ylabel("Mean Squared Error loss")
        plt.legend()
        plt.title(f"Data Split - Training: {round((1 - split) * 100)}%")
        plt.show()

    ############# LEARNING RATE ANALYSIS ###############
    # hidden_layer_shape = 15
    # lrs = [0.001, 0.005, 0.01, 0.05, 0.1]
    #
    # for lr in lrs:
    #     x_train, x_val, y_train, y_val = train_test_split(inputs, targets, 0.2)
    #     network = nn.NueralNet(x_train, y_train, hidden_layer_size = hidden_layer_shape, output_layer_size = 1,
    #                            is_binary = False, lr = lr)
    #     losses = network.train_network(500, inputs, targets)
    #
    #     plt.plot(losses["batch_losses"], label = f"Learning Rate: {lr}")
    #     plt.xlabel("Epochs")
    #     plt.ylabel("Mean Squared Error loss")
    #     plt.legend()
    #     plt.title(f"MSE by Learning Rate")
    # plt.show()

    #################### PLOT 3d #####################
    # x_train = inputs
    # y_train = targets
    # hidden_layer_shape = 8
    #
    # network = nn.NueralNet(x_train, y_train, hidden_layer_size = hidden_layer_shape, output_layer_size = 1,
    #                        is_binary = False, lr = 0.001)
    # nnTrainResults = network.train_network(epochs = 1000)
    #
    # results = network.fowardPass(inputs, targets, include_bias = True)
    #
    # for epoch, batch in enumerate(nnTrainResults["batch_out"]):
    #     batch_out = np.reshape(batch, (data['size'], data['size']))
    #     plot_gaussian(data, batch_out, f"Gaussian Out - epoch:{epoch}", gif = {"epoch": epoch, "seq": 0})
    # plot_gif("gaussian_batch", repeat_frames = 1)


if __name__ == '__main__':
    main()

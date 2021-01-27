#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 11:07:35 2021

@author: aleix

the goal of this program is to find which are the best and worse architecures
for the mackey glass time - series problem. It corresponds to the tasks 1 and 2
of section 4.3 in the pdf.

We split data into training and validation. We are interested in the validation loss.
We will use a neural network with two hidden layers. 

Because the inits of each neural networks are random, we need to perform statistics
and run them multiple times. We will return the mean value of the training and 
validation losses with their standard deviations. We will save the results in
a file architecture_vs_loss.txt. 

"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import mean_squared_error

from data_builder import InputsTargetsBuilder

def get_train_val_test(frac_train=0.8):
    # the assignment says we will work with the data between t=301 and t=1500
    mackey_glass = np.loadtxt("mackey_glass.txt", usecols=1)[301:1500]
    num_test = 200 # the assignment says the last 200 points are for testing
    num_train = int((mackey_glass.size - num_test) * 0.8)
    num_val = mackey_glass.size - num_test - num_train
    builder = InputsTargetsBuilder()
    return builder.build_train_val_test(mackey_glass, num_train, num_val)    
 

def build_architecture(two_hidden_layers, nodes_input=5, nodes_output=1):
    # the assignment says just two hidden layers: tuple of ints (n1, n2)
    model = Sequential()
    model.add(Input(shape=nodes_input))
    model.add(Dense(two_hidden_layers[0], activation="sigmoid"))
    model.add(Dense(two_hidden_layers[1], activation="sigmoid"))
    model.add(Dense(nodes_output, activation="relu"))
    # the activation functions are taken as specified in the assignment
    return model

def train_model(model, train_data, val_data, figure_file,
                lambda_regul=None, num_epochs=300, batch_size=16, l_rate=0.05):
                                                                  
    if lambda_regul:    
        callback = None
    else:
        callback = EarlyStopping(monitor="val_loss", min_delta=0.001, 
                                                       patience=30, mode="min")
    model.compile(loss="mean_squared_error", optimizer=SGD(lr=l_rate))
    history = model.fit(x=train_data[0], y=train_data[1], batch_size=batch_size, 
       epochs=num_epochs, validation_data=val_data, callbacks=[callback], 
                                                                 verbose=False) 
    plt.figure()
    plt.xlabel("epoch")
    plt.ylabel("mean_squared_error")
    plt.plot(history.history["loss"], label="train_mse")
    plt.plot(history.history["val_loss"], label="val_mse")
    plt.legend()
    plt.savefig(figure_file)

    last_train_mse = history.history["loss"][-1]
    last_val_mse = history.history["val_loss"][-1]
    return last_train_mse, last_val_mse
    

def train_multiple_times(two_hidden_layers, train_data, val_data, num_runs=10,
                                                            lambda_regul=None):
    last_train_mses = np.zeros(num_runs)
    last_val_mses = np.zeros(num_runs)
    for i in range(num_runs):
        figure_file = "Plots/%d_%d_run%d.png" % (two_hidden_layers[0], 
                                                       two_hidden_layers[1], i)
        model = build_architecture(two_hidden_layers)
        train_mse, val_mse = train_model(model, train_data, val_data, figure_file, 
                                                     lambda_regul=lambda_regul)
        last_train_mses[i] = train_mse
        last_val_mses[i] = val_mse
    return last_train_mses, last_val_mses
        

def try_different_architectures(train_data, val_data, lambda_regul=None,
       first_layer_tries=[3, 4, 5], second_layer_tries=[2, 4, 6], num_runs=10):
    
    f_out = open("ResultsPart2/architectures.txt", "w")
    f_out.write("# %d runs statistics\n" % num_runs)
    f_out.write("# nodes_hidden_layer1  nodes_hidden_layer2  mean_train_mse" 
        "  std_train_mse  mean_val_mse std_val_mse\n")
    for h1 in first_layer_tries:
        for h2 in second_layer_tries:
            print(h1, h2)
            last_train_mses, last_val_mses = train_multiple_times(
                (h1, h2), 
                train_data, 
                val_data, 
                num_runs=num_runs, 
                lambda_regul=lambda_regul)
            mean_train = np.mean(last_train_mses)
            std_train = np.std(last_train_mses)
            
            mean_val = np.mean(last_val_mses)
            std_val = np.std(last_val_mses)
            f_out.write("%d  %d  %.8e  %.8e  %.8e  %.8e\n" % 
                            (h1, h2, mean_train, std_train, mean_val, std_val))
    f_out.close()
    return

def predict_multiple_times(two_hidden_layers, train_data, val_data, test_data, 
    num_runs=10, lambda_regul=None, num_epochs=300, batch_size=16, l_rate=0.05): 
    
    
    if lambda_regul:    
        callback = None
    else:
        callback = EarlyStopping(monitor="val_loss", min_delta=0.001, 
                                                       patience=30, mode="min")
    
    num_test = test_data[0].shape[0]
    predictions = np.zeros(shape=(num_runs, num_test))
    last_train_mses = np.zeros(num_runs)
    last_val_mses = np.zeros(num_runs)
    for i in range(num_runs):
        model = build_architecture(two_hidden_layers)
        model.compile(loss="mean_squared_error", optimizer=SGD(lr=l_rate))
        history = model.fit(x=train_data[0], y=train_data[1], batch_size=batch_size, 
           epochs=num_epochs, validation_data=val_data, callbacks=[callback], 
                                                                     verbose=False)
        last_train_mses[i] = history.history["loss"][-1]
        last_val_mses[i] = history.history["val_loss"][-1]
        predictions[i, :] = model.predict(test_data[0])[:, 0]
    return last_train_mses, last_val_mses, predictions







train_data, val_data, test_data = get_train_val_test()
print("num_train + num_val = ", train_data[1].size + val_data[1].size)

worse_architecture = (3, 2)
last_train_mses, last_val_mses, predictions = predict_multiple_times(
    worse_architecture, train_data, val_data, test_data, num_runs=10) 

print(last_train_mses)
print(last_val_mses)

good_preds = predictions[last_val_mses < 0.2]
num_good_preds = good_preds.shape[0]

average_pred = np.sum(good_preds, axis=0) / good_preds.shape[0]
pred_std = np.std(good_preds, axis=0)

num_points = average_pred.shape[0]
time=np.arange(1325, 1325+num_points)
plt.figure()
plt.title("Worse architecture: %d runs average" % good_preds.shape[0])
plt.xlabel("time")
plt.ylabel("mackey glass")
plt.plot(time, test_data[1], label="test signal")
plt.plot(time, average_pred, color="red", label="averaged prediction")
plt.fill_between(x=time, y1=average_pred-pred_std, 
                 y2=average_pred+pred_std, label="pred_std", color="darkorange")
plt.legend()


print("test_mse", mean_squared_error(test_data[1], average_pred))

#first_layer_tries = [3, 4, 5]
#second_layer_tries = [2, 4, 6]
#try_different_architectures(train_data, 
#                            val_data, 
#                            first_layer_tries=first_layer_tries, 
#                            second_layer_tries=second_layer_tries)

















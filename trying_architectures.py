#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 15:35:00 2021

@author: aleix

This program identifies the best and worse architectures by saving the stats
of each architecture in a file.
"""

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import RandomNormal

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
    model.add(Dense(two_hidden_layers[0], activation="sigmoid", kernel_initializer=RandomNormal()))
    model.add(Dense(two_hidden_layers[1], activation="sigmoid", kernel_initializer=RandomNormal()))
    model.add(Dense(nodes_output, activation="relu", kernel_initializer=RandomNormal()))
    # the activation functions are taken as specified in the assignment
    return model

# architecture will be a tuple of two integers indicating the num of hidden layers
class EarlyStoppingPredictor(object):
    def __init__(self, architecture):
        self.architecture = architecture
        self.callback = EarlyStopping(monitor="val_loss", min_delta=min_delta, 
                                                 patience=patience, mode="min")
    
    def init_and_train_model(self):
        model = build_architecture(self.architecture)
        model.compile(loss="mean_squared_error", optimizer=SGD(lr=l_rate))
        history = model.fit(x=x_train, 
              y=y_train, 
              batch_size=batch_size, 
              epochs=num_epochs,
              validation_data=(x_val, y_val),
              shuffle=False,
              verbose=verbose,
              callbacks=[self.callback])
        
        return model, history
    
    # sucessful predictions are assessed by looking at the validation loss
    def get_suceessful_predictions(self, num_preds, max_attempts=20):
        attempts = 0
        num_sucess = 0
        train_mses = np.zeros(num_preds)
        val_mses = np.zeros(num_preds)
        predictions = np.zeros(shape=(num_preds, test_length))
        while num_sucess < num_preds and attempts < max_attempts:
            model, history = self.init_and_train_model()
            train_mse = history.history["loss"][-1]
            val_mse = history.history["val_loss"][-1]
            converged = val_mse < val_threshold
            if converged:
                # saving results
                pred = model.predict(x_test)[:, 0] # getting rid of shape (N, 1)
                train_mses[num_sucess] = train_mse
                val_mses[num_sucess] = val_mse
                predictions[num_sucess, :] = pred
                num_sucess += 1
            attempts += 1
        return train_mses, val_mses, predictions
    

class GridSearch(object):
    def __init__(self, layer1_nodes, layer2_nodes):
        self.layer1_nodes = layer1_nodes 
        self.layer2_nodes = layer2_nodes
    
    def try_different_architectures(self, num_runs):
        for h1 in self.layer1_nodes:
            for h2 in self.layer2_nodes:
                predsfile = "%d_%d_preds" % (h1, h2)
                loss_file = "%d_%d_loss.txt" % (h1, h2)
                val_loss = "%d_%d_val_loss.txt" % (h1, h2)
                predictor = EarlyStoppingPredictor((h1, h2))
                train_mses, val_mses, preds = predictor.get_suceessful_predictions(num_runs)
                np.save(predsfile, preds)
                np.savetxt(loss_file, train_mses)
                np.savetxt(val_loss, val_mses)
        return
    

# Set of global variables
# for early stopping
min_delta = 0.0001

patience = 30
val_threshold = 0.1
#data
(x_train, y_train), (x_val, y_val), (x_test, y_test) = get_train_val_test()
np.save("test", y_test)
test_length = y_test.shape[0]
#hiperparams
l_rate = 0.05
batch_size = 16
num_epochs = 30
# printing options
verbose = True

num_preds = 2
h1 = [3]
h2 = [6]
grid = GridSearch(h1, h2)
grid.try_different_architectures(num_preds)





# 
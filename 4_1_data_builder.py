#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:06:26 2021

@author: aleix

This program receives a time-series and generates training, validation and test
data. It builds the inputs and the targets as specified in the assignment. That
is, according to a delay: 
An input x = [signal[I], signal[I-delay], ... signal[I-(size-1)delay]] 
(The code works with the flipped order given above)

A target y = signal[I+delay] 

We will store all x samples in a numpy array of shape (num_samples, sample_size)
We will store all targets in a numpy array of shape (num_samples)
In that way, the data can be directly sent to the Keras library, which requires
this structure.
"""

import numpy as np
import sys


class InputsTargetsBuilder(object):
    def __init__(self, sample_size=5, delay_step=5):
        self.sample_size = sample_size
        self.delay_step = delay_step

    # given num_train, num_val, the reamining data will be assumed to be test data
    def build_train_val_test(self, signal, num_train, num_val):
        train_signal = signal[0:num_train]
        val_signal = signal[num_train : num_train + num_val]
        test_signal = signal[num_train + num_val :]

        x_train, y_train = self.build_inputs_targets(train_signal)
        x_val, y_val = self.build_inputs_targets(val_signal)
        x_test, y_test = self.build_inputs_targets(test_signal)

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    def build_inputs_targets(self, signal):
        min_signal_index = (self.sample_size - 1) * self.delay_step
        # we can not start building a sample x from a lower index, otherwise
        # we would encounter a negative index in the past
        max_signal_index = signal.size - 1 - self.delay_step
        # we can not build targets from a higer index,
        # otherwise we would surpass the last index size-1
        num_samples = max_signal_index - min_signal_index + 1
        if num_samples < 0:
            print("not enough data for the given sample size and delay step")
            sys.exit()

        x_inputs = np.zeros(shape=(num_samples, self.sample_size))
        y_targets = np.zeros(shape=num_samples)
        for i in range(min_signal_index, max_signal_index + 1):
            x_sample = signal[
                i
                - (self.sample_size - 1) * self.delay_step : i
                + self.delay_step : self.delay_step
            ]
            # i-min_signal_index is just a sample counter (0 to num_samples-1)
            x_inputs[i - min_signal_index, :] = x_sample
            y_targets[i - min_signal_index] = signal[i + self.delay_step]
        return x_inputs, y_targets


# Here I provide a small example
"""
signal = np.array([-1, 3, 6, 8, 10, 34, 56, 58])
builder = InputsTargetsBuilder(sample_size=3, delay_step=2)
print(builder.build_inputs_targets(signal))
"""

"""
Dense Layer Module
By: Abel Yohannes
Website: https://github.com/abelyo252/
"""

import numpy as np


class Dense:
    """
    Dense (fully-connected) layer for a neural network.

    Args:
        units: Number of neurons in this layer.
        input_shape: Shape of the input (required for the first layer).
        activation: Name of activation (stored but not applied — use separate
                     activation layers).
    """

    def __init__(self, units, input_shape=None, activation=None):
        self.units = units
        self.input_shape = input_shape
        self.activation = activation
        self.prev_layer = None
        self.layer_name = "dense"
        self.trainable = True

        if self.input_shape is None and self.prev_layer is not None:
            self.input_shape = self.prev_layer.output_shape

        if self.input_shape:
            input_dim = np.prod(self.input_shape)
            self.weights = np.random.randn(input_dim, self.units) * np.sqrt(2 / input_dim)
            self.biases = np.zeros(self.units)
        else:
            self.weights = None
            self.biases = None

    def __str__(self):
        return f"Dense(units={self.units}, input_shape={self.input_shape}, activation={self.activation})"

    def forward(self, inputs, training=True):
        self.inputs = inputs

        if self.weights is None or self.biases is None:
            if self.input_shape is None:
                self.input_shape = inputs.shape[1:]
            if self.input_shape is None:
                raise ValueError("Weights and biases are not initialized. Please provide input_shape.")

            input_dim = np.prod(self.input_shape)
            # Xavier initialization
            limit = np.sqrt(6 / (input_dim + self.units))
            self.weights = np.random.uniform(-limit, limit, (input_dim, self.units))
            self.biases = np.zeros(self.units)


        self.output = np.dot(inputs, self.weights) + self.biases

        return self.output

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0)
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs

    def get_parameters(self):
        return self.weights, self.biases

    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases

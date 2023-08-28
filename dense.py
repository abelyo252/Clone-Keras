"""
Dense Layer Module
By: Abel Yohannes
Website: https://github.com/abelyo252/
"""


# Import standard modules.
import sys
import os

# Import non-standard modules.
import numpy as np

class Dense:
    """
        DenseLayer class that represents a dense layer in a neural network
        and display the weights and biases matrix for that layer,
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
            # He initialization
            #self.weights = np.random.randn(input_dim, self.units) * np.sqrt(2 / input_dim)
            #self.biases = np.zeros(self.units)
            # Xavier initialization
            limit = np.sqrt(6 / (input_dim + self.units))
            self.weights = np.random.uniform(-limit, limit, (input_dim, self.units))
            self.biases = np.zeros(self.units)


        self.output = np.dot(inputs, self.weights) + self.biases

        return self.output

    def backward(self, dvalues):
        if self.prev_layer is not None:
            self.dweights = np.dot(self.prev_layer.output.T, dvalues)
            self.dbiases = np.sum(dvalues, axis=0)
            self.dinputs = np.dot(dvalues, self.weights.T)
            #print("Dense Dinput : ", self.dinputs)

        else:
            self.dweights = np.dot(self.inputs.T, dvalues)
            self.dbiases = np.sum(dvalues, axis=0)
            self.dinputs = np.dot(dvalues, self.weights.T)
            #print("Finish : ", self.dinputs)

        return self.dinputs

    def get_parameters(self):
        return self.weights , self.biases

    def set_parameters(self, weights,biases):
        self.weights = weights
        self.biases = biases


def main():
    # Creating a NeuralNetwork object
    dense = Dense(units=5, input_shape=(3,), activation='relu')
    print(dense.weights)



if __name__ == "__main__":
    main()

# See Github or Telegram Address for help at https://github.com/abelyo252/
# https://t.me/benyohanan

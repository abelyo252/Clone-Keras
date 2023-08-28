"""
Activation Layer Module
By: Abel Yohannes
Website: https://github.com/abelyo252/
"""


# Import standard modules.
import sys
import os

# Import non-standard modules.
import numpy as np


class Activation_ReLU:
    def __init__(self):
        self.input = None
        self.output = None
        self.layer_name = "relu"
        self.trainable = False

    def __str__(self):
        return f"ReLU(X={self.input}, Output={self.output})"

    def forward(self, inputs, training=True):
        self.input = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, dvalues):
        #print("Relu receive dvalue ", dvalues)
        self.dinputs = dvalues.copy()
        self.dinputs[self.input <= 0] = 0

        return self.dinputs

    def prediction(self, outputs):
        return np.where(outputs > 0, 1, 0)


## Activation Softmax
class Activation_Softmax:
    def __init__(self):
        self.input = None
        self.output = None
        self.trainable = False

    def __str__(self):
        return f"Softmax(X={self.input}, Output={self.output})"

    def forward(self, inputs, training=True):
        self.input = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

        #print("Softmax Dinput ", self.dinputs)

        return self.dinputs


## Activation Sigmoid
class Activation_Sigmoid:

    def __init__(self):
        self.input = None
        self.output = None
        self.trainable = False

    def __str__(self):
        return f"Sigmoid(X={self.input}, Output={self.output})"

    # Implement Method
    def forward(self,inputs, training):
        # Remember Input Value
        self.inputs = inputs
        # Calculate Output
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output

    def backward(self, dvalues):
        # Derivative - calculates from output of the sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output
        return self.dinputs

    def prediction(self,outputs):
        return (outputs > 0.5) * 1


class Activation_Linear:
    def __init__(self):
        self.input = None
        self.output = None
        self.layer_name = "linear"
        self.trainable = False

    def __str__(self):
        return f"Linear(X={self.input}, Output={self.output})"

    def forward(self, inputs, training=True):
        self.inputs = inputs
        self.output = inputs
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues
        self.dinputs

    def prediction(self, outputs):
        return outputs



class Activation_Tanh:
    def __init__(self):
        self.input = None
        self.output = None
        self.layer_name = "tanh"
        self.trainable = False

    def __str__(self):
        return f"Tanh(X={self.input}, Output={self.output})"

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.tanh(inputs)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs = dvalues * (1 - np.power(self.output, 2))
        return self.dinputs

    def prediction(self, outputs):
        return np.where(outputs > 0, 1, -1)


# See Github or Telegram Address for help at https://github.com/abelyo252/
# https://t.me/benyohanan
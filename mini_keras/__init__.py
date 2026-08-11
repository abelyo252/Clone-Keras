"""
Mini-Keras: A minimal Keras-like deep learning framework built from scratch.

This package provides a simplified implementation of neural network building blocks
for educational purposes, including layers, activations, losses, and optimizers.

Usage:
    from mini_keras import Sequential, Dense
    from mini_keras.activations import ReLU, Softmax
    from mini_keras.optimizers import Adam, SGD
    from mini_keras.losses import CategoricalCrossentropy
    from mini_keras.datasets import load_mnist
    from mini_keras.utils import to_categorical
"""

from mini_keras.model import Sequential
from mini_keras.layers import Dense
from mini_keras.activations import (
    Activation_ReLU,
    Activation_Softmax,
    Activation_Sigmoid,
    Activation_Linear,
    Activation_Tanh,
    ReLU,
    Softmax,
    Sigmoid,
    Linear,
    Tanh,
)
from mini_keras.losses import (
    Loss_CategoricalCrossEntropy,
    CategoricalCrossentropy,
)
from mini_keras.optimizers import (
    Optimizer_SGD,
    Optimizer_SGD_Momentum,
    OptimizerAdaGrad,
    Optimizer_RMSprop,
    Optimizer_Adam,
    SGD,
    Adam,
    RMSprop,
    Adagrad,
)

__version__ = "0.1.0"
__author__ = "Abel Yohannes"

__all__ = [
    # Model
    "Sequential",
    # Layers
    "Dense",
    # Activations (clean names)
    "ReLU",
    "Softmax",
    "Sigmoid",
    "Linear",
    "Tanh",
    # Activations (legacy names)
    "Activation_ReLU",
    "Activation_Softmax",
    "Activation_Sigmoid",
    "Activation_Linear",
    "Activation_Tanh",
    # Losses
    "CategoricalCrossentropy",
    "Loss_CategoricalCrossEntropy",
    # Optimizers (clean names)
    "SGD",
    "Adam",
    "RMSprop",
    "Adagrad",
    # Optimizers (legacy names)
    "Optimizer_SGD",
    "Optimizer_SGD_Momentum",
    "OptimizerAdaGrad",
    "Optimizer_RMSprop",
    "Optimizer_Adam",
]

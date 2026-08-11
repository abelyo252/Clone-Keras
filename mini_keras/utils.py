"""
Utility functions for Mini-Keras.
"""

import numpy as np


def to_categorical(y, num_classes):
    """
    Convert class labels to one-hot encoded format.

    Args:
        y: Array of integer class labels.
        num_classes: Total number of classes.

    Returns:
        One-hot encoded numpy array of shape (len(y), num_classes).
    """
    one_hot = np.zeros((y.shape[0], num_classes), dtype='float32')
    one_hot[np.arange(y.shape[0]), y] = 1.0
    return one_hot

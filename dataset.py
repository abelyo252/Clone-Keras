"""
Dataset Creater Module
By: Abel Yohannes
Website: https://github.com/abelyo252/
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random


def create_mnist(train_data_path, train_labels_path, test_data_path, test_labels_path, test_size=0.2):
    # Load training data
    train_data = pd.read_csv(train_data_path)
    train_labels = pd.read_csv(train_labels_path)

    # Convert the data and labels to NumPy arrays
    X_train = train_data.values.T
    Y_train = train_labels.values.T

    # Load testing data
    test_data = pd.read_csv(test_data_path)
    test_labels = pd.read_csv(test_labels_path)

    # Convert the data and labels to NumPy arrays
    X_test = test_data.values.T
    Y_test = test_labels.values.T

    return X_train, Y_train, X_test, Y_test

def visualize_mnist_images(images, labels, digit_labels, num_images=10):
    # Select a subset of images to visualize
    random_indices = np.random.choice(images.shape[0], num_images, replace=False)
    selected_images = images[random_indices]
    selected_labels = labels[random_indices]

    # Visualize the images and labels
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    axes = axes.flatten()

    for i in range(num_images):
        axes[i].imshow(selected_images[i].reshape(28, 28), cmap='gray')
        axes[i].set_title('Label: ' + digit_labels[np.argmax(selected_labels[i])])
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()




def create_data(samples, classes, test_size=0.2):
    train_samples = int(samples * (1 - test_size))
    test_samples = samples - train_samples

    X_train = np.zeros((train_samples * classes, 2))
    y_train = np.zeros(train_samples * classes, dtype='uint8')
    X_test = np.zeros((test_samples * classes, 2))
    y_test = np.zeros(test_samples * classes, dtype='uint8')

    for class_number in range(classes):
        train_start = train_samples * class_number
        train_end = train_samples * (class_number + 1)
        test_start = test_samples * class_number
        test_end = test_samples * (class_number + 1)

        r_train = np.linspace(0.0, 1, train_samples)
        t_train = np.linspace(class_number * 4, (class_number + 1) * 4, train_samples) + np.random.randn(train_samples) * 0.2

        X_train[train_start:train_end] = np.c_[r_train * np.sin(t_train * 2.5), r_train * np.cos(t_train * 2.5)]
        y_train[train_start:train_end] = class_number

        r_test = np.linspace(0.0, 1, test_samples)
        t_test = np.linspace(class_number * 4, (class_number + 1) * 4, test_samples) + np.random.randn(test_samples) * 0.2

        X_test[test_start:test_end] = np.c_[r_test * np.sin(t_test * 2.5), r_test * np.cos(t_test * 2.5)]
        y_test[test_start:test_end] = class_number

    return X_train, y_train, X_test, y_test

def visualize_dataset(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Dataset Visualization')
    plt.show()

# See Github or Telegram Address for help at https://github.com/abelyo252/
# https://t.me/benyohanan
"""
Dataset Module
By: Abel Yohannes
Website: https://github.com/abelyo252/
"""

import numpy as np
import gzip
import struct
import os
import urllib.request


def _download_if_needed(url, filepath):
    """Download a file from url if it doesn't already exist locally."""
    if not os.path.exists(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, filepath)
    return filepath


def _read_idx_images(filepath):
    """Read IDX image file format used by MNIST."""
    with gzip.open(filepath, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num_images, rows, cols)
    return data


def _read_idx_labels(filepath):
    """Read IDX label file format used by MNIST."""
    with gzip.open(filepath, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def load_mnist(data_dir='mnist_data'):
    """
    Download and load the MNIST dataset (pure numpy, no tensorflow).

    Args:
        data_dir: Directory to cache downloaded files.

    Returns:
        (X_train, y_train), (X_test, y_test) — images as uint8 arrays,
        labels as uint8 arrays.
    """
    base_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz',
    }

    paths = {}
    for key, filename in files.items():
        paths[key] = _download_if_needed(base_url + filename, os.path.join(data_dir, filename))

    X_train = _read_idx_images(paths['train_images'])
    y_train = _read_idx_labels(paths['train_labels'])
    X_test = _read_idx_images(paths['test_images'])
    y_test = _read_idx_labels(paths['test_labels'])

    return (X_train, y_train), (X_test, y_test)


def create_spiral_data(samples, classes, test_size=0.2):
    """
    Create a spiral dataset for classification.

    Args:
        samples: Number of samples per class.
        classes: Number of classes.
        test_size: Fraction of data to use for testing.

    Returns:
        X_train, y_train, X_test, y_test
    """
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
    """Plot a 2D dataset with color-coded classes."""
    import matplotlib.pyplot as plt
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Dataset Visualization')
    plt.show()


def visualize_mnist_images(images, labels, digit_labels, num_images=10):
    """Visualize a subset of MNIST images."""
    import matplotlib.pyplot as plt
    random_indices = np.random.choice(images.shape[0], num_images, replace=False)
    selected_images = images[random_indices]
    selected_labels = labels[random_indices]

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    axes = axes.flatten()

    for i in range(num_images):
        axes[i].imshow(selected_images[i].reshape(28, 28), cmap='gray')
        axes[i].set_title('Label: ' + digit_labels[np.argmax(selected_labels[i])])
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# Legacy alias
create_data = create_spiral_data

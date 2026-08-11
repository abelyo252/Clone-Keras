"""
Try saved Model using mini_keras library
By: Abel Yohannes
Website: https://github.com/abelyo252/
"""

import numpy as np
import matplotlib.pyplot as plt
from mini_keras import Sequential
from mini_keras.datasets import load_mnist, visualize_mnist_images
from mini_keras.utils import to_categorical


def main():
    # Load and preprocess the MNIST data
    (X_train, y_train), (X_test, y_test) = load_mnist()

    # Reshape and normalize input data
    X_train_flat = X_train.reshape(-1, 28 * 28).astype('float32') / 255.0
    X_test_flat = X_test.reshape(-1, 28 * 28).astype('float32') / 255.0

    # One-hot encode targets
    num_classes = 10
    Y_train = to_categorical(y_train, num_classes)
    Y_test = to_categorical(y_test, num_classes)

    # Define digit labels
    digit_labels = [str(i) for i in range(10)]

    # Load saved model (.ab extension)
    loaded_model = Sequential.load_model('model/mnist.ab')
    print("Model loaded successfully from model/mnist.ab")

    # Predict single sample
    sample_index = 0
    data = np.reshape(X_test_flat[sample_index], (1, -1))
    prediction = loaded_model.predict(data)
    predicted_label = np.argmax(prediction)

    print("Predicted label:", predicted_label)
    print("True label:     ", y_test[sample_index])

    # Visualize test image
    sample_idx_visual = 4
    image_data = X_test[sample_idx_visual]
    label_onehot = Y_test[sample_idx_visual]

    plt.imshow(image_data, cmap='gray')
    plt.title(f"Label: {np.argmax(label_onehot)}")
    plt.axis('off')
    plt.show()

    # Visualize a subset of MNIST training images
    visualize_mnist_images(X_train_flat, Y_train, digit_labels, num_images=10)


if __name__ == '__main__':
    main()
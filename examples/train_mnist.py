"""
Train MNIST with Mini-Keras
Example script demonstrating how to use the mini_keras library.
"""

import numpy as np
import matplotlib.pyplot as plt
from mini_keras import Sequential, Dense
from mini_keras.activations import ReLU, Softmax
from mini_keras.datasets import load_mnist
from mini_keras.utils import to_categorical


def plot_loss_and_accuracy(history):
    loss = history['loss']
    accuracy = history['accuracy']
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'b-', label='Loss')
    plt.plot(epochs, accuracy, 'r-', label='Accuracy')
    plt.title('Loss and Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    # Load and preprocess the MNIST data
    (X_train, y_train), (X_test, y_test) = load_mnist()

    # Reshape the input data
    X_train = X_train.reshape(-1, 28 * 28)
    X_test = X_test.reshape(-1, 28 * 28)

    # Normalize the pixel values to the range [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Convert the labels to one-hot encoded format
    num_classes = 10
    Y_train = to_categorical(y_train, num_classes)
    Y_test = to_categorical(y_test, num_classes)

    model = Sequential()

    # Add layers to the model
    model.add(Dense(units=128, input_shape=(28 * 28,)))
    model.add(ReLU())
    model.add(Dense(units=64))
    model.add(ReLU())
    model.add(Dense(units=10))
    model.add(Softmax())

    # Compile model with loss, optimizer, and metrics
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Training
    num_epochs = 15
    history = model.fit(X_train, Y_train, epochs=num_epochs, batch_size=64)

    # Plotting accuracy and loss
    plot_loss_and_accuracy(history)

    # Save the model
    model.save_model('model/mnist.ab')
    print("Model saved to model/mnist.ab")


if __name__ == '__main__':
    main()

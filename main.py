"""
Implement Your Archetecture
By: Abel Yohannes
Website: https://github.com/abelyo252/
"""


import numpy as np
from activation import Activation_ReLU , Activation_Softmax
from loss import Loss_CategoricalCrossEntropy
from optimizer import Optimizer_Adam
from dataset import create_data , visualize_dataset , create_mnist
from dense import Dense
from tensorflow import keras
import matplotlib.pyplot as plt
from model import Sequential

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

    #X_train, y_train, X_test, y_test = create_data(samples=5000, classes=3, test_size=0.2)
    #visualize_dataset(X_train, y_train)

    # Load and preprocess the MNIST data
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # Reshape the input data
    X_train = X_train.reshape(-1, 28 * 28)
    X_test = X_test.reshape(-1, 28 * 28)

    # Normalize the pixel values to the range [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Convert the labels to one-hot encoded format
    num_classes = 10
    Y_train = keras.utils.to_categorical(y_train, num_classes)
    Y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()

    # Add layers to the model
    model.add(Dense(units=128, input_shape=(28*28,), activation='relu'))
    model.add(Activation_ReLU())
    model.add(Dense(units=64, activation='relu'))
    model.add(Activation_ReLU())
    model.add(Dense(units=10, activation='relu'))
    model.add(Activation_Softmax())

    # Define loss, optimizer, and metrics
    loss = Loss_CategoricalCrossEntropy()
    # optimizer = Optimizer_SGD(learning_rate=0.05)
    optimizer = Optimizer_Adam(learning_rate=0.01, decay=5e-7)
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    # User-defined number of epochs
    num_epochs = 15
    # Training
    history = model.fit(X_train, Y_train, epochs=num_epochs, batch_size=64)
    # Plotting accuracy and loss
    plot_loss_and_accuracy(history)
    # Save the model with .ab extension
    model.save_model('model/mnist.ab')





if __name__=='__main__':
    main()

# See Github or Telegram Address for help at https://github.com/abelyo252/
# https://t.me/benyohanan
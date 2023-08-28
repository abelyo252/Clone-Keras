"""
Try saved Model
By: Abel Yohannes
Website: https://github.com/abelyo252/
"""



from dataset import create_data , visualize_dataset,create_mnist,visualize_mnist_images
import numpy as np
from model import Sequential
from tensorflow import keras
import matplotlib.pyplot as plt

def main():

    #X_train, y_train, X_test, y_test = create_data(samples=1000, classes=3, test_size=0.2)
    #visualize_dataset(X_test, y_test)

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

    # Define the labels for each digit
    digit_labels = [str(i) for i in range(10)]



    model = Sequential()

    # Load the model with .ab extension
    loaded_model = Sequential.load_model('model/mnist.ab')
    # Make predictions using the model
    data = X_test[0]
    test_label = Y_test[0]


    # Reshape the data to have shape (1, input_shape)
    data = np.reshape(data, (1, -1))
    prediction = loaded_model.predict(data)
    predicted_label = np.argmax(prediction)

    # Print the predicted label
    print("Predicted label:", predicted_label)
    # Select the first test sample and its corresponding label
    data = X_test[4]
    label = Y_test[4]

    # Reshape the data to its original shape (28x28)
    data = data.reshape((28, 28))

    # Plot the image
    plt.imshow(data, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

    # Visualize a subset of MNIST training images
    visualize_mnist_images(X_train, Y_train, digit_labels, num_images=10)


if __name__=='__main__':
    main()

# See Github or Telegram Address for help at https://github.com/abelyo252/
# https://t.me/benyohanan
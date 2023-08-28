"""
Sequential Model Module
By: Abel Yohannes
Website: https://github.com/abelyo252/
"""


from activation import Activation_ReLU, Activation_Softmax,Activation_Tanh,Activation_Linear,Activation_Sigmoid
from loss import Loss_CategoricalCrossEntropy
from optimizer import Optimizer_Adam , Optimizer_SGD
from dense import Dense
import numpy as np
from tqdm import tqdm
import pickle

class Sequential:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None
        self.metrics = []

    def add(self, layer):
        if self.layers:
            # Connect the layer to the previous layer
            layer.prev_layer = self.layers[-1]

        self.layers.append(layer)

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        self.output = output
        return self.output

    def backward(self, targets):
        dvalues = targets
        for layer in reversed(self.layers):
            dvalues = layer.backward(dvalues)
        return dvalues

    def update_params(self):
        for layer in self.layers:
            if layer.trainable:
                self.optimizer.pre_update_params()
                self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

    def compile(self, loss, optimizer, metrics=None):
        if loss == "categorical_crossentropy":
            self.loss = Loss_CategoricalCrossEntropy()
        else:
            self.loss = loss

        if optimizer == "sgd":
            self.optimizer = Optimizer_SGD(learning_rate=0.05)
        elif optimizer == "adam":
            self.optimizer = Optimizer_Adam(learning_rate=0.01, decay=5e-7)
        else:
            self.optimizer = optimizer

        if metrics is not None:
            self.metrics = metrics

    # Update the fit() method
    from tqdm import tqdm

    def fit(self, X, y, epochs=10, batch_size=None):
        accuracy_history = []
        loss_history = []

        # Determine the batch size
        if batch_size is None:
            batch_size = X.shape[0]  # Use the entire dataset as a batch
            num_batches = 1  # Only one batch for full-batch optimization
        else:
            num_samples = X.shape[0]
            num_batches = num_samples // batch_size

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_accuracy = 0.0

            # Shuffle the data for each epoch
            indices = np.random.permutation(num_samples)
            X = X[indices]
            y = y[indices]

            # Create the progress bar
            progress_bar = tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")

            for batch in range(num_batches):
                # Extract the current mini-batch
                start_idx = batch * batch_size
                end_idx = start_idx + batch_size
                X_batch = X[start_idx:end_idx]
                y_batch = y[start_idx:end_idx]

                # Forward propagation
                output = self.forward(X_batch.copy())
                loss_batch = self.loss.forward(output, y_batch.copy())
                epoch_loss += np.sum(loss_batch)


                predictions = np.argmax(output, axis=1)
                if len(y_batch.shape) == 2:
                    y_batch = np.argmax(y_batch, axis=1)
                epoch_accuracy += np.sum(predictions == y_batch)

                # Backpropagation
                grads = self.loss.backward(output, y_batch)
                self.backward(grads)

                # Update parameters
                self.update_params()

                # Update the progress bar description
                progress_bar.set_postfix(loss=f"{np.mean(loss_batch):.4f}",
                                         acc=f"{np.mean(predictions == y_batch):.4f}")
                progress_bar.update(1)

            # Calculate average loss and accuracy for the epoch
            data_loss = epoch_loss / num_samples
            accuracy = epoch_accuracy / num_samples

            # History Record
            accuracy_history.append(accuracy)
            loss_history.append(data_loss)

            # Close the progress bar for the epoch
            progress_bar.close()

        history = {'accuracy': accuracy_history, 'loss': loss_history}
        return history

    def predict(self, X):
        # Forward propagation
        output = self.forward(X.copy())

        return output

    def evaluate(self, X_test, y_test):
        output = self.forward(X_test.copy())
        loss = self.loss.forward(output, y_test.copy())
        data_loss = np.mean(loss)

        predictions = np.argmax(output, axis=1)
        if len(y_test.shape) == 2:
            y_test = np.argmax(y_test, axis=1)
        accuracy = np.mean(predictions == y_test)

        return accuracy, data_loss

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, filepath):
        with open(filepath, 'rb') as f:
            loaded_model = pickle.load(f)

        # Create a new instance of Sequencial
        model = Sequential()

        # Add layers to the network based on the loaded model
        for layer in loaded_model.layers:
            if isinstance(layer, Dense):
                # Create a new instance of the Dense layer and add it to the network
                new_layer = Dense(units=layer.units)
            elif isinstance(layer, Activation_ReLU):
                # Create a new instance of the Activation_ReLU layer and add it to the network
                new_layer = Activation_ReLU()
            elif isinstance(layer, Activation_Softmax):
                # Create a new instance of the Activation_Softmax layer and add it to the network
                new_layer = Activation_Softmax()
            elif isinstance(layer, Activation_Sigmoid):
                # Create a new instance of the Activation_Sigmoid layer and add it to the network
                new_layer = Activation_Sigmoid()
            elif isinstance(layer, Activation_Linear):
                # Create a new instance of the Activation_Linear layer and add it to the network
                new_layer = Activation_Linear()
            elif isinstance(layer, Activation_Tanh):
                # Create a new instance of the Activation_Tanh layer and add it to the network
                new_layer = Activation_Tanh()
            else:
                raise ValueError(f"Unsupported layer type: {type(layer)}")

            model.add(new_layer)

        return model

# See Github or Telegram Address for help at https://github.com/abelyo252/
# https://t.me/benyohanan
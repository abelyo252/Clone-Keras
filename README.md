# Clone-Keras: Framework for Deep Learning 

Clone-Keras is a framework designed for deep learning tasks. Inspired by the popular Keras library, help students to know the underlined principle for Artificial neural network. This framework simplifies the process of creating neural networks. With a concise and intuitive syntax identical to keras aim developers can easily define ANN architectures using high-level building blocks, such as layers, activation functions, and optimizers. This is Mini version of Keras-API and help student how forward and backward propagation work.


 ![Image](https://github.com/abelyo252/Clone-Keras/blob/main/XD%20File/clone-keras.png)

This repository serves as the development hub for dummy who want to build Deep learning from the scratch.


## About Clone-Keras

Clone-Keras is a Python-based profound learning system planned to encourage the advancement of artificial neural systems from the scratch.
It is built on mathematical point of view so apprentice able to get it the concept from the scratch and clarify designers almost how maths
is run the framework, external code utilized for running ANN code take after keras so designers will not confounded by learning this system.
This repo engages engineers and analysts to require full advantage of the adaptability
and cross-platform capabilities arithmetic of manufactured neural organize.


**The goal of Keras Clone is to empower developers about how math work in creating machine learning-powered applications.**


## Optimizers
Gradient Descent is an optimization algorithm that iteratively updates the parameters of a function by moving in the direction of steepest descent. The parameter update formula is as follows:
<p align="center"> <img src="https://github.com/abelyo252/Clone-Keras/blob/main/XD%20File/grad_descent.png"> </p>

The algorithm starts with an initial guess for the parameters and then repeatedly updates them by taking steps proportional to the negative gradient of the function at that point. By moving in the direction opposite to the gradient, the algorithm gradually descends towards the minimum of the function.

The general formula for the parameter update in gradient descent is as follows:
```python
parameter = parameter - learning_rate * gradient
```

<p align="center"> <img src="https://github.com/abelyo252/Clone-Keras/blob/main/optimization/grad.gif"> </p>

# Forward and Backward Propagation
Forward and Backpropagation can be effectively performed using a computational graph, which helps visualize and organize the computations involved in the forward and backward passes. As a beginner, it recognizes the gradients in the forward pass, calculates the required derivatives, and updates the network's parameters based on the optimization algorithm.i learnt from karpathy [Click His youtube tourial]([https://www.google.com](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)). you can find this on mini-keras folder

Import Neccesary Python Modules
```python
from data import Data
from visualization import visualize_computational_graph,concatenate_images
import matplotlib.pyplot as plt
```

```python
# inputs x1,x2
x1 = Data(2.0, label='x1')
x2 = Data(0.0, label='x2')
# weights w1,w2
w1 = Data(-3.0, label='w1')
w2 = Data(1.0, label='w2')
# bias of the neuron
b = Data(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
y = x1w1x2w2 + b; y.label = 'y'
# visualized L graph now
viz_Y = visualize_computational_graph(y)
Data.backward(y)
viz_back_y = visualize_computational_graph(y) # visualized final grad of all object

# Concatenate the images horizontally
concatenated_image = concatenate_images(viz_Y, viz_back_y, axis='horizontal')
```

This is the result of Linear Regression Backprop
<p align="center"> <img src="https://github.com/abelyo252/Clone-Keras/blob/main/mini-keras/backprop.png"> </p>

## Installation

To install **Mini-Keras** directly as a library via `pip`:

```bash
pip install git+https://github.com/abelyo252/Clone-Keras.git
```

Or install locally in editable mode:

```bash
git clone https://github.com/abelyo252/Clone-Keras.git
cd Clone-Keras
pip install -e .
```

---

## Quickstart & Example Usage

The core data structures of Mini-Keras mimic Keras API with a `Sequential` model stacking layers.

### Building a Model

```python
from mini_keras import Sequential, Dense
from mini_keras.activations import ReLU, Softmax

model = Sequential()

model.add(Dense(units=128, input_shape=(28*28,)))
model.add(ReLU())
model.add(Dense(units=64))
model.add(ReLU())
model.add(Dense(units=10))
model.add(Softmax())
```

<p align="center">
  <img src="https://github.com/abelyo252/Clone-Keras/blob/main/ann_arch.png" alt="Image" width="638" height="374">
</p>

---

### Compiling and Training

Configure learning process with `.compile()`:

```python
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
```

Train on data and save the model:

```python
from mini_keras.datasets import load_mnist
from mini_keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = load_mnist()

X_train = X_train.reshape(-1, 28 * 28).astype('float32') / 255.0
Y_train = to_categorical(y_train, num_classes=10)

model.fit(X_train, Y_train, epochs=15, batch_size=64)
model.save_model('model/mnist.ab')
```

### Loading and Predicting

Generate predictions on new data using a saved model:

```python
import numpy as np
from mini_keras import Sequential

# Load model
loaded_model = Sequential.load_model('model/mnist.ab')

# Predict
sample_data = np.reshape(X_test[0], (1, -1))
prediction = loaded_model.predict(sample_data)
predicted_label = np.argmax(prediction)

print("Predicted label:", predicted_label)
```
---
## Support

You can ask questions and join the development discussion:

- @ Telegram t.me/@benyohanan

---

## Opening an issue

You can also post **bug reports and feature requests** (only)
in [GitHub issues](https://github.com/ab).


---

## Opening a PR

I'm welcome for contributions! Before opening a PR, please read
[contributor guide](https://github.com/blob/master/CONTRIBUTING.md)


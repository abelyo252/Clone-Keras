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

To install the most recent version of Clone-Keras, just follow these simple instructions. I use Python 3.11.4 for this project; you can download 3.11.4 from [here](https://www.python.org/ftp/python/3.11.4/python-3.11.4-amd64.exe) ,if the two are incompatible, try another version by searching online. If git wasn't installed on your Windows PC, get it from `https://gitforwindows.org/` or install it on linux using `sudo apt-get install git` 

`git clone https://github.com/abelyo252/Clone-Keras.git`<br>
`cd Clone-Keras/`<br>
`pip install -r requirements.txt`<br>

```python
from model import Sequential
```

---


## Run Code for training after define your achtecture in main.py

`$ python main.py`<br>


---

## First contact with This Framework

The core data structures of Clone-Keras are the same as keras
The only sort of show is the [`Sequential` model](https://keras.io/guides/sequential_model/), a direct stack of layers. This System
is still in demo form, so we didnt consolidate utilitarian api, too this repo didnt claim that clone entire keras thing from the scratch, we as it were actualize ANN as it were

Here is the `Sequential` model:

```python
from model import Sequential

model = Sequential()
```

Stacking layers is as easy as `.add()`:

```python
from model import Sequential
from dense import Dense
from activation import Activation_ReLU , Activation_Softmax

model = Sequential()

model.add(Dense(units=128, input_shape=(28*28,), activation='relu'))
model.add(Activation_ReLU())
model.add(Dense(units=64, activation='relu'))
model.add(Activation_ReLU())
model.add(Dense(units=10, activation='relu'))
model.add(Activation_Softmax())

```

Once your model looks good, configure its learning process with `.compile()`:

```python
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=None)
```

You can now iterate on your training data in batches and also save model using <yourmodel>.ab:

```python
# x_train and y_train are Numpy arrays.
model.fit(X_train, Y_train, epochs=num_epochs, batch_size=64)
model.save_model('model/<yourmodel>.ab')
```



generate predictions on new data using saved model:

```python
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


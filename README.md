<div align="center">

# 🧠 Mini-Keras (Clone-Keras)

### *A Minimal, From-Scratch Deep Learning Framework in Pure Python & NumPy*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abelyo252/Clone-Keras/blob/main/tutorials/mini_keras_demo.ipynb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-NumPy%20Only-orange.svg)](https://numpy.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/abelyo252/Clone-Keras/pulls)

<p align="center">
  <img src="https://raw.githubusercontent.com/abelyo252/Clone-Keras/main/XD%20File/clone-keras.png" alt="Mini-Keras Banner" width="700">
</p>

</div>

---

## 📌 Overview

**Mini-Keras** is an educational, lightweight deep learning framework designed to demonstrate the inner workings of artificial neural networks from first principles. Built entirely on top of **NumPy**, it mirrors the high-level API design of [Keras](https://keras.io/) while exposing clear, readable implementations of forward propagation, backpropagation, gradient calculation, parameter updates, and model serialization.

Whether you are a student learning neural network mathematics or a developer curious about building deep learning libraries from scratch, **Mini-Keras** bridges the gap between high-level ML APIs and foundational calculus.

---

## ⚡ Try Instantly in Google Colab

Click the badge below to open the complete end-to-end interactive demo directly in **Google Colab** (no local setup required):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abelyo252/Clone-Keras/blob/main/tutorials/mini_keras_demo.ipynb)

---

## ✨ Features

- **🚀 Zero Deep Learning Dependencies**: Implemented using pure NumPy & standard libraries (no PyTorch, TensorFlow, or JAX needed).
- **📦 Clean Keras-like API**: Define neural networks using intuitive `Sequential()`, `Dense()`, and `.add()` interfaces.
- **⚡ Multiple Optimization Algorithms**: Includes `Adam`, `SGD`, `SGD with Momentum`, `RMSprop`, and `Adagrad`.
- **🧪 Diverse Activation Functions**: Supports `ReLU`, `Softmax`, `Sigmoid`, `Tanh`, and `Linear`.
- **📊 Dataset Utilities**: Built-in, dependency-free MNIST dataset loader and synthetic spiral data generator.
- **💾 Model Serialization**: Simple `.save_model()` and `.load_model()` functionality using custom binary formats (`.ab`).
- **📚 Educational Computational Graph Engine**: Includes scalar-level backprop engine & computational graph visualization inspired by Andrej Karpathy.

---

## 🛠️ Installation

### Option 1: Install Directly via `pip` from GitHub

```bash
pip install git+https://github.com/abelyo252/Clone-Keras.git
```

### Option 2: Local Development Setup

```bash
# Clone the repository
git clone https://github.com/abelyo252/Clone-Keras.git
cd Clone-Keras

# Install in editable mode
pip install -e .
```

---

## 🚀 Quickstart & Usage

### 1. Build and Train a Neural Network

```python
import numpy as np
from mini_keras import Sequential, Dense
from mini_keras.activations import ReLU, Softmax
from mini_keras.datasets import load_mnist
from mini_keras.utils import to_categorical

# 1. Load and preprocess data
(X_train, y_train), (X_test, y_test) = load_mnist()
X_train_flat = X_train.reshape(-1, 28 * 28).astype('float32') / 255.0
X_test_flat = X_test.reshape(-1, 28 * 28).astype('float32') / 255.0

Y_train = to_categorical(y_train, num_classes=10)
Y_test = to_categorical(y_test, num_classes=10)

# 2. Construct Sequential Model
model = Sequential()
model.add(Dense(units=128, input_shape=(784,)))
model.add(ReLU())
model.add(Dense(units=64))
model.add(ReLU())
model.add(Dense(units=10))
model.add(Softmax())

# 3. Compile Model with Loss & Optimizer
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 4. Train Model
history = model.fit(X_train_flat, Y_train, epochs=15, batch_size=64)

# 5. Save Trained Model
model.save_model('model/mnist.ab')
```

<p align="center">
  <img src="https://raw.githubusercontent.com/abelyo252/Clone-Keras/main/ann_arch.png" alt="ANN Architecture" width="600">
</p>

---

### 2. Load Model & Make Predictions

```python
import numpy as np
from mini_keras import Sequential
from mini_keras.datasets import load_mnist

# Load saved model
model = Sequential.load_model('model/mnist.ab')

# Run inference on test data
(_, _), (X_test, y_test) = load_mnist()
sample = X_test[0].reshape(1, 784).astype('float32') / 255.0

prediction = model.predict(sample)
predicted_digit = np.argmax(prediction)

print(f"Predicted Digit: {predicted_digit}")
print(f"Actual Ground Truth: {y_test[0]}")
```

---

## 🔬 Forward and Backward Propagation (Computational Graph)

Forward and Backpropagation can be effectively performed using a computational graph, which helps visualize and organize the computations involved in the forward and backward passes. Learn how gradients flow step-by-step through mathematical nodes (inspired by Andrej Karpathy's `micrograd`). You can find this inside the `mini-keras/` directory.

### Importing Necessary Modules

```python
from data import Data
from visualization import visualize_computational_graph, concatenate_images
import matplotlib.pyplot as plt
```

### Computational Graph Construction & Visualization

```python
# inputs x1, x2
x1 = Data(2.0, label='x1')
x2 = Data(0.0, label='x2')

# weights w1, w2
w1 = Data(-3.0, label='w1')
w2 = Data(1.0, label='w2')

# bias of the neuron
b = Data(6.8813735870195432, label='b')

# x1*w1 + x2*w2 + b
x1w1 = x1 * w1; x1w1.label = 'x1*w1'
x2w2 = x2 * w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
y = x1w1x2w2 + b; y.label = 'y'

# Visualize forward graph
viz_Y = visualize_computational_graph(y)

# Perform backpropagation
Data.backward(y)

# Visualize graph with updated gradients
viz_back_y = visualize_computational_graph(y)

# Concatenate the images horizontally
concatenated_image = concatenate_images(viz_Y, viz_back_y, axis='horizontal')
```

### Linear Regression Backpropagation Result

<p align="center">
  <img src="https://raw.githubusercontent.com/abelyo252/Clone-Keras/main/mini-keras/backprop.png" alt="Linear Regression Backprop Graph" width="750">
</p>

---

## 🧮 Theoretical Background & Optimization

### Gradient Descent
Gradient Descent iteratively updates model parameters $(\theta)$ in the direction of steepest loss reduction:

$$\theta = \theta - \eta \cdot \nabla_{\theta} L$$

<p align="center">
  <img src="https://raw.githubusercontent.com/abelyo252/Clone-Keras/main/XD%20File/grad_descent.png" alt="Gradient Descent Equation" width="400">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/abelyo252/Clone-Keras/main/optimization/grad.gif" alt="Gradient Descent Animation" width="500">
</p>

---

## 📦 Supported Building Blocks

| Category | Component | Description |
| :--- | :--- | :--- |
| **Model** | `Sequential` | Linear stack of layers |
| **Layers** | `Dense` | Fully-connected dense layer with He / Xavier weight initialization |
| **Activations** | `ReLU`, `Softmax`, `Sigmoid`, `Tanh`, `Linear` | Non-linearities with exact backward gradients |
| **Optimizers** | `Adam`, `SGD`, `SGDMomentum`, `RMSprop`, `Adagrad` | Adaptive & momentum-based learning rate algorithms |
| **Losses** | `CategoricalCrossentropy` | Stable cross-entropy loss with numerical clipping |

---

## 📁 Repository Structure

```
Clone-Keras/
├── pyproject.toml              # Build & package metadata for pip
├── README.md                   # Project documentation
├── mini_keras/                 # Core Mini-Keras framework package
│   ├── __init__.py             # Public API exports
│   ├── model.py                # Sequential container class
│   ├── layers.py               # Dense layer implementation
│   ├── activations.py          # Activation functions & derivatives
│   ├── optimizers.py           # Optimization algorithms
│   ├── losses.py               # Loss functions & gradients
│   ├── datasets.py             # Pure-NumPy MNIST & synthetic dataset helpers
│   └── utils.py                # Encoding & helper utilities
├── mini-keras/                 # Scalar-level computational graph engine
│   ├── data.py                 # Data scalar class with backward autograd
│   └── visualization.py        # Computational graph visualizer
├── examples/                   # Standalone Python example scripts
│   ├── train_mnist.py          # MNIST training script
│   └── load_saved_model.py     # Inference script
└── tutorials/                  # Interactive Jupyter Notebook tutorials
    └── mini_keras_demo.ipynb   # Complete walkthrough notebook ([Open in Colab](https://colab.research.google.com/github/abelyo252/Clone-Keras/blob/main/tutorials/mini_keras_demo.ipynb))
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/abelyo252/Clone-Keras/issues).

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 💬 Community & Support

- **Author**: Abel Yohannes
- **GitHub**: [@abelyo252](https://github.com/abelyo252/)
- **Telegram**: [@benyohanan](https://t.me/benyohanan)

---

## 📜 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

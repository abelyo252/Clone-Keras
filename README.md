# Clone-Keras: Framework for Deep Learning 
![Clone-Keras logo]([https://](https://github.com/abelyo252/Clone-Keras/blob/main/XD%20File/clone-keras.png))

This repository serves as the development hub for dummy who want to build Deep learning from the scratch.


## About Clone-Keras

Keras Clone is a Python-based profound learning system planned to encourage the advancement of artificial neural systems from the scratch.
It is built on mathematical point of view so apprentice able to get it the concept from the scratch and clarify designers almost how maths
is run the framework, external code utilized for running ANN code take after keras so designers will not confounded by learning this system.
This repo engages engineers and analysts to require full advantage of the adaptability
and cross-platform capabilities arithmetic of manufactured neural organize.


**The goal of Keras Clone is to empower developers about how math work in creating machine learning-powered applications.**

## Installation

To install the most recent version of Clone-Keras, just follow these simple instructions. You must install Python versions 3.6.x to 3.9.x; we are using Python 3.6 for this project; you can download 3.6.8 from [here](https://www.python.org/ftp/python/3.6.8/python-3.6.8-amd64.exe) ,if the two are incompatible, try another version by searching online. If git wasn't installed on your Windows PC, get it from `https://gitforwindows.org/` or install it on linux using `sudo apt-get install git` 

`git clone https://github.com/abelyo252/Clone-Keras.git`<br>
`cd Clone-Keras/`<br>
`pip install -r requirements.txt`<br>

```python
from model import sequential
```

---


## Run Code

`$ python pyNetwork.py`<br>


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
model = Sequential()

model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
```

Once your model looks good, configure its learning process with `.compile()`:

```python
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```

You can now iterate on your training data in batches:

```python
# x_train and y_train are Numpy arrays.
model.fit(x_train, y_train, epochs=5, batch_size=32)
```



generate predictions on new data:

```python
classes = model.predict(x_test, batch_size=128)
```

For more in-depth youtube tutorials about this framework, you can check out our youtube playlist:

-   [Introduction to Clone-Keras for engineers](https://youtube.com/intro_to_keras_for_engineers/)

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

We welcome contributions! Before opening a PR, please read
[our contributor guide](https://github.com/keras-team/keras/blob/master/CONTRIBUTING.md),
and the [API design guideline](https://github.com/keras-team/governance/blob/master/keras_api_design_guidelines.md).

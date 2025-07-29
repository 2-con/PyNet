<p align="center">
  <img width="658" height="165" alt="image" src="https://github.com/user-attachments/assets/0dcf4931-a993-42a9-9c94-96e9afb6d0e3" />
</p>

# PyNet Version 1.0.0

PyNet is an Artificial Intelligence and Machine Learning framework entirely built from Python.

Unlike Tensorflow or PyTorch, PyNet is intended for small-scale educational and experimental purposes, its meant to lower the barrier of entry for people intrested in machine learning and Artificial Intelligence using an easy-to-learn language while still implimenting key concepts. But because everything is entirely built on Python, training can be quite slow depending on the machine and therefore, is not advised to serve as a general-purpose library; Other, more proffesionally developed libraries may suite the task at hand.

# Contents

At its core, PyNet is a Library full of layers, activation functions, models and APIs, PyNet also has some unorthodox layers that are mainly there for experimental purposes, more information in the documentation provided.

## NetCore API

The main API for PyNet, guarenteed to continue receiving constant updates and fixes. It is the most up-to-date standard within this framework.

**NetCore Features:**
  - Multichannel Convolution
  - Dense
  - Localunit (Locally Connected Neurons)
  - AFS (Adaptive Feature Scaler) - Experimental layer
  - Maxpooling
  - Meanpooling
  - Flatten
  - Reshape
  - Operation (normalization and activation functions)
  - RecurrentBlock
  - RNN
  - LSTM
  - GRU

## NetFlash API

An experimental API meant to optimize NetCore to allow for greater utility. it is currently under development.

## PyNet Alpha

The first implimentation of PyNet, a module containing functions for propagating, backpropagating and updating a neural network; the predecessor to PyNet. Be aware that advanced features such as optimizers and parametric functions are not available for this implimentation by default.

**PyNet Alpha Features**
  - Initialize
  - Propagate
  - Backpropagate
  - Update
  - Default training function (not required, user can make their own)

## Other Models

**Regressors**
- Linear Regression
- Polynomial Regression
- Logistic Regression
- Exponential Regression
- Power Regression
- Logarithmic Regression
- Sinusoidal Regression (external model, does not follow PyNet API)

**Classifiers**
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Naive Bayes
- SVM

**Datasets**
- Cluster (classification)
- Image (variations of the MNIST dataset as a python list)
- Regression
- Text

**Tools**
- Arraytools
- Logic
- Math
- Scaler
- Utility
- Visual

Many more featurs are present, but they are not as important as the features listed above or are internal features.

# Installation

Dependancies
- NumPy
- Math
- Random
- Itertools
- Time
- Matplotlib

# Liscence

**Apache License 2.0**
This project is licensed under the MIT License - see [LICENSE](https://github.com/2-con/PyNet/blob/main/LICENSE) for more details.

# Configuration

While PyNet has an internal configuration file, it is not very extensive and is meant to provide default values to background processes, PyNet is all about transparancy and configurability per model and does not rely heavly on a central file.

# Contributors

Maintainer
- 2-Con

Contributors
- None

# Contact

The email below is an organisation email, external emails might get flagged and/or discarded.

Email: 72212@jisedu.or.id

# Development status

PyNet is almost completed, just some additional features and fixing and it should be **mostly** completed by August.

estimated date: August 2025 (uncertian + not guarenteed)

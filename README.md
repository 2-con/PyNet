<p align="center">
  <img width="1047" height="226" alt="Screenshot 2025-08-14 211650" src="https://github.com/user-attachments/assets/bae68280-f298-4578-82c5-6a12700adee3" />
</p>

# PyNet

A free and open framework for AI & Machine Learning

PyNet is a free and open-source, Python-based framework designed to demystify machine learning and AI. Unlike complex libraries like TensorFlow or PyTorch, PyNet focuses on small-scale educational projects and experiments. It provides full access to the source code, lowering the barrier to entry for enthusiasts.

## Core APIs

# NetFlash API

A high-performance API built around the JAX ecosystem, leveraging JNP operations and JIT-compiled systems to boost calculations up to 5x the speed thanks to the XLA compiler. Everything is designed to be modular, so a custom layer can be passed into the sequential class as long as it adheres to NetFlash spesifications. More information on the documentation.

**NetFlash Features:**
  - Dense
  - Localunit (Locally Connected Layer)
  - Multichannel Convolution
  - Multichannel Deconvolution
  - Maxpooling
  - Meanpooling
  - Flatten
  - Operation
  - Reshape
  - Dropout
  - Recurrent Layer
  - GRU (Gated Recurrent Unit) Layer
  - LSTM (Long Sort Term Memory) Layer
  - Multiheaded Self-Attention

# NetCore API

The main and first API built for the PyNet framework, everthing is built entirely on python with object-oriented programming as the main programming paradigm for each layer. But because everything is entirely built on Python, training can be quite slow depending on the machine and therefore, is not advised to serve as a general-purpose library; Other, more proffesionally developed libraries may suite the task at hand. 

**NetCore Features:**
  - Dense
  - Localunit (Locally Connected Layer)
  - Operation
  - Multichannel Convolution
  - Maxpooling
  - Meanpooling
  - Flatten
  - Reshape
  - RecurrentBlock
  - RNN
  - LSTM
  - GRU

# PyNet Alpha

The predecessor to PyNet's main APIs. This module contains basic functions for propagation, backpropagation, and updating a neural network. It's a great starting point for understanding the core mechanics before diving into the main APIs. Advanced features such as optimizers and parametric functions are not available for this implimentation by default.

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
- Sinusoidal Regression (external model, does not use the PyNet framework)

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

Many more featurs are present, but they are not listed above since they are either internal features or configuration.

# Installation

Dependancies
- NumPy
- Math
- Random
- Itertools
- Time
- Matplotlib
- Typing

# Licence

**Apache License 2.0**
This project is licensed under the MIT License - see [LICENSE](https://github.com/2-con/PyNet/blob/main/LICENSE) for more details.

# Configuration

While PyNet has an internal configuration file, it is not very extensive and is meant to provide default values to background processes, PyNet is all about transparancy and configurability per model and does not rely heavly on a central file.

# Contributors

Maintainer
- 2-Con

Contributors
- None

# Development status

PyNet is almost completed, just some additional features and fixing and it should be **mostly** completed by August.

estimated date: August 2025 (uncertian + not guarenteed)


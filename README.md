<p align="center">
  <img width="1047" height="226" alt="Screenshot 2025-08-14 211650" src="https://github.com/user-attachments/assets/bae68280-f298-4578-82c5-6a12700adee3" />
</p>


<div style="border: 2px solid #ffffffff; border-radius: 100px;">
</div>


<div style="border: 2px solid #ffffffff; border-radius: 15px; margin: 20px; padding: 0px 20px 0px 20px;">
  <p align="center" style="font-size:70px">
    <b>
      PyNet
    </b>
  </p>

  <p align="center" style="font-size:30px; padding: 0px 0px 20px 0px;">
    <i>
      A free and open framework for AI & Machine Learning
    </i>
  </p>  
</div>


<div style="border: 2px solid #ffffffff; border-radius: 15px; margin: 20px; padding: 0px 20px 0px 20px;">
  <p align="center" style="font-size:40px">
    <b>
      Table of Contents
    </b>

  <p align="center" style="font-size:20px">
    ❓ What is PyNet ❓
  </p>

  <p align="center" style="font-size:20px">
    🚀 Getting Started 🚀
  </p>

  <p align="center" style="font-size:20px">
    ⚡ Core Features ⚡
  </p>

  <p align="center" style="font-size:20px">
    ⚒️ Additional Features ⚒️
  </p>

  <p align="center" style="font-size:20px">
    📚 Documentation 📚
  </p>

  <p align="center" style="font-size:20px">
    📲 Installation 📲
  </p>

  <p align="center" style="font-size:20px">
    ⚖️ Licence ⚖️
  </p>

  <p align="center" style="font-size:20px">
    🤝 Contributors 🤝
  </p>

</div>


<div style="border: 2px solid #ffffffff; border-radius: 15px; margin: 20px; padding: 0px 20px 0px 20px;">
  <p align="center" style="font-size:40px">
    <b>
      What is Pynet
    </b>

  <p align="center" style="font-size:20px">
    PyNet is a free and open-source software framework for machine learning and artificial intelligence. Unlike professional-grade libraries, PyNet focuses on small-scale educational projects and experiments by providing multiple APIs and spaces for testing, experimenting, and debugging model architectures. PyNet also allows full access to the source code, lowering the barrier to entry for enthusiasts.
  </p>

  <p align="center" style="font-size:20px">
    PyNet features distinct APIs and a suite of tools that cater to different needs that allows users to explore core ML concepts from the ground up in multiple programming paradigms. Users can directly modify and test new ideas by injecting code directly into the framework itself to allow for greater control, PyNet is client-based so modified code only affect the local file and nothing else.
  </p>

  <p align="center" style="font-size:20px">
    PyNet is commited to democratizing and opening up machine learning and artificial intelligence to the world, so from the newest student to the most experienced scientist, we are determined to share a new technological revolution is in the horizon.
  </p>
</div>


<div style="border: 2px solid #ffffffff; border-radius: 15px; margin: 20px; padding: 0px 20px 0px 20px;">
  <p align="center" style="font-size:40px">
    <b>
      TITLE
    </b>

  <p align="center" style="font-size:20px">
    TEXT

</div>

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


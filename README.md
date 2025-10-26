<p align="center" style="font-size:70px">
  <img width="1047" height="226" alt="Screenshot 2025-08-14 211650" src="https://github.com/user-attachments/assets/bae68280-f298-4578-82c5-6a12700adee3" />
  <b>PyNet</b>
</p>

---

# **PyNet** 🧠

### **A free and open framework for Machine Learning**

---

### What is PyNet ❓
PyNet is a free and open-source software framework for machine learning and artificial intelligence. Unlike professional-grade libraries, PyNet focuses on small-scale educational projects and curiosity. Everything in PyNet, even the calculation logic, is written in pure python therefore there will be a lot of overhead that can slow down code.

PyNet features a main API and a suite of tools that to assist in model building. However, it is reccomended to use PyNet's successor JXNet which is the direct continuation of this repository.

PyNet is committed to democratizing and opening up machine learning and artificial intelligence to the world, so from the newest student to the most experienced scientist, we are determined to share a new technological revolution that is on the horizon.

---

### Getting Started 🚀

PyNet's interface are designed to be intuitive. Although the usage may vary depending on the model, the core approach is still the same

#### For Core APIs
All the core models have similar method names and procedures with the exception of alpha which has its own system.

    # create a model instance
    MODEL = sequential(
      layer1,
      layer2,
      ...
    )

    # model must be compiled
    MODEL.compile(
      ...
    )

    # train the model. make sure the inputs are of the appropriate datatype
    MODEL.fit(feature,labels)

    # the model can be used after training.
    MODEL.push(input)
  
#### PyNet Alpha
PyNet Alpha is found in the API folder, its an API in the sense that it is completely independent from PyNet by using its own management system and interface to interact with itself. PyNet Alpha is also the earliest functional model that is meant to be an introductory model to be disected and studied.

    # create a list of weights
    MODEL = Initialize(...)

    # use a built-in method...
    train(network, features, targets, ...)

    # or build your own method (as long as the methods are being used properly)
    for epoch in range(epochs):
      for feature, target in zip(features, targets):
        activations, weighted_sums = propagate(network, feature, ReLU)
        error = backpropegate(network, activations, weighted_sums, target, ReLU_derivative)
        
        update(network, activations, error, learning_rate)

      if epoch % 10 == 0:
        print(f"Epoch {epoch:5} | Error {mse(target, activations[-1])}")

#### Other models
PyNet also have other models for regression, classification and clustering under the "models" folder.

    # using a linear regression as an example
    MODEL = Linear(...)

    # simmilar to 
    MODEL.compile(...)

    # only some models have a "fit" method, some like K-Nearest Neighbors dont have this method at all.
    MODEL.fit(features, labels)

    # simmilar to the "push" method from the core APIs
    MODEL.predict(...)

---

### Core Features ⚡
At its core, PyNet is a framework full of layers and processes that require complex setup to run properly, hence, prebuilt APIs are made in order to streamline this process. Currently, Pynet hosts four main APIs that could be used to abstract processes and make development easier and faster.

---

### Active PyNet APIs
---
#### StaticNet API
The main and first API built for the PyNet framework, everything is built entirely on python with object-oriented programming as the main programming paradigm for each layer. But because everything is entirely built on Python, training can be quite slow depending on the machine and therefore, is not advised to serve as a general-purpose library; Other, more professionally developed libraries may suit the task at hand.

**Learnable layers:**
- Fully Connected
- Locally Connected
- Multichannel Convolution
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- Simple Recurrent (Recurrent)

**Functional layers:**
- Recurrent Block
- Operation (Multipurpose)
- Multichannel MaxPooling
- Multichannel MeanPooling
- Flatten
- Dropout
- Reshape

---
#### LiteNet
The predecessor to PyNet's main APIs. This module contains basic functions for propagation, backpropagation, and updating a neural network. It's a great starting point for understanding the core mechanics before diving into the main APIs. Advanced features such as optimizers and parametric functions are not available for this implementation by default.

**Provided Methods:**
- Initialize
- Propagate
- Backpropagate
- Update
- Complimentary Training Method

For a more detailed description on specific APIs or layers, refer to the documentation
---

### Additional Features ⚒️

Aside from APIs and layers, PyNet also contains other features that could aid in your project or experiment.  

#### Arraytools
Tools for dealing with tensors and lists in python. While not as extensive as NumPy or JNP, it is still quite useful for custom implimentations.

#### Math
A place to store general functions that dosent fit into any category (loss, activation, etc...). PyNet aims to organize the code to the best of our ability.

#### Scaler
Simmilar to the Math file, this contains functions that deal with vectors (1D lists), unlike ArrayTools, these functions are not linear algebra oriented and more statistics-heavy

#### Utility
General-use functions including wrappers and iterators. 

#### Visual
Functions used to display and visualize PyNet objects useful for debugging.

---
### Regressors

PyNet's regression models provide a diverse set of tools for predicting continuous values. These models are self-contained and easy to use, making them ideal for understanding the fundamentals of curve fitting and trend analysis.

- Linear Regression
- Polynomial Regression
- Logistic Regression
- Exponential Regression
- Power Regression
- Logarithmic Regression
---
### Classifiers

The classification suite offers a range of models for predicting discrete categories. These algorithms are perfect for learning about different approaches to pattern recognition and decision-making.

- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Naive Bayes
- SVM

---

### Installation 📲
Despite self-sufficiency as a core philosophy, PyNet still needs some modules for core operations and processing.
<br>

**Dependencies**
<br>

- NumPy
- Math
- Time
- Matplotlib
- Typing

---

### License ⚖️
This project is licensed under the Apache License 2.0 (January 2004) for distribution and modification.

**[Apache License 2.0 (January 2004)](https://github.com/2-con/PyNet/blob/main/LICENSE)**

---

### Contributors 🤝
PyNet is a project that started in January 2025 and has been receiving updates ever since from one person, any help is much appreciated.

**Maintainer**
2-Con

**Contributors**
None

"""
PyNet Alpha
==========
  A simple implementation of a Neural Network in Python, the predecessor to PyNet. A bare-bones version of Synapse API
  with a lot of features absent. What it lacks in complexity it makes up for in speed.
  
  And unlike the other APIs, this version is entirely comprised of fully connected layers, requiring the input neurons
  to be defined during initialization.
"""

#######################################################################################################
#                                               Imports                                               #
#######################################################################################################

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random

from core.static.activations import ReLU
from core.static.derivatives import ReLU_derivative
from core.static.losses import Mean_squared_error as mse

################################################################################################
#                                           Main code                                          #
################################################################################################

def initialize(*args: int) -> list:
  """
  Initialize
  -----
    Generates a randomized neural network model based on spesifications.
    Input neurons must be defined.
  -----
  Args
  -----
  - Args : numerical arguments representing the amount of neurons per layer
  -----
  Returns
  -----
  - A list of layers (non-PyNet layers)
  """

  network = []  # list of layers

  for a in range(len(args)-1):
    layer = []

    for _ in range(args[a+1]):
      neuron = {
        'weights': [random.uniform(0.1, 1) for _ in range(args[a])],
        'bias': random.uniform(-1, 1)
      }
      layer.append(neuron)
    network.append(layer)

  return network

def propagate(model, input, activation_function) -> tuple[list, list]:
  """
  Propagate
  -----
    Performs one forward pass through the entire network
  -----
  Args
  -----
  - model (model) : The neural network model (list of layers)
  - input (list or tuple) : The input values for the first layer
  - activation_function (function) : The activation function to use for all neurons
  -----
  Returns
  -----
  - A tuple containing:
    - activations: A list of activations for each layer
    - weighted_sums: A list of weighted sums for each layer
  """
  
  current_input = input[:] # copy the input layer, because it will be modified.
  activations = [current_input] # list of layer outputs
  weighted_sums = [] # all of the weighted sums 

  for layer in model:
    layer_output = []
    layer_weighted_sums = []

    for neuron in layer:
      weighted = 0

      for weight, input_value in zip(neuron['weights'], current_input):
        weighted += weight * input_value
      weighted += neuron['bias']

      layer_weighted_sums.append(weighted)
      layer_output.append( activation_function(weighted) )

    activations.append(layer_output[:]) #copy to prevent aliasing.
    current_input = layer_output[:] #copy to prevent aliasing.
    weighted_sums.append(layer_weighted_sums[:])
  
  #return the very last layer (output) and the activations
  return activations, weighted_sums

def backpropegate(model, activations:list, weighted_sums:list, target:list, activation_derivative) -> list:
  """
  Backpropegate
  -----
    Performs one backward pass through the entire network, fetching the error gradients.
  -----
  Args
  -----
  - model (model) : The neural network model
  - activations (tuple) : Activations  from the forward pass
  - activations (tuple) : Weighted sums from the forward pass
  - target (tuple or list) :  The target output values
  -----
  Returns
  -----
  - A list containing:
    - lists of errors for each neuron in the layer
  """
  
  errors = [([0.0] * len(layer)) for layer in model] # initialize list
  output_errors = [target[i] - activations[-1][i] for i in range(len(target))]
  errors[-1] = [error * activation_derivative(weightedsum) for error, weightedsum in zip(output_errors, weighted_sums[-1])]

  # goes backwards from the input
  for reversed_layer_index in reversed(range(len(model) - 1)):
    layer = model[reversed_layer_index]
    next_layer_errors = errors[reversed_layer_index + 1]
    current_errors = []

    # goes over all the enuron in the layer
    for neuron_index, neuron in enumerate(layer):

      # goes over the all the neurons in the previous layer
      error = sum(next_neuron['weights'][neuron_index] * next_error for next_neuron, next_error in zip(model[reversed_layer_index + 1], next_layer_errors))
      error *= activation_derivative(weighted_sums[reversed_layer_index][neuron_index])# derivative goes here
      current_errors.append(error)
      
    errors[reversed_layer_index] = current_errors[:]
  return errors

def update(model, activations: list, error: list, learning_rate: float) -> None:
  """
  Update
  -----
    Performs one update pass through the entire network without any optimizer. This function does not return anything
    as it only updates the current values inside the model.
  -----
  Args
  -----
  - network       (model) : The neural network model
  - activations   (tuple) : The activations from forward propagation
  - error         (tuple) : The error gradients from backpropagation
  - learning_rate (float) : The learning rate
  -----
  Returns
  -----
  - None
  """
  #print(f"Network layer sizes: {[len(layer) for layer in network]}") # debug print

  # iterate over layers
  for layer_index in range(len(model)):
    
    # iterate over each neuron
    for neuron_index, neuron in enumerate(model[layer_index]):
      
      # iterate over each weights
      for weight_index in range(len(neuron['weights'])):
        neuron['weights'][weight_index] += learning_rate * error[layer_index][neuron_index] * activations[layer_index][weight_index]
      
      # bias
      neuron['bias'] += learning_rate * error[layer_index][neuron_index]

################################################################################################
#                             prebuilt trainer for convenience                                 #
################################################################################################

def train(network, features, targets, learning_rate: float, epochs: int, **kwargs):
  """
  Train
  -----
    Trains the network, this should only be a demonstration of how to use the functions present in this file.
    feel free to copy the source code and use it in your own code.
  -----
  Args
  -----
  - network       (model) : list of layers, which is a list of neurons
  - features      (list) : the features to use
  - targets       (list) : the corresponding targets to use
  - learning_rate (float) : learning rate
  - epochs        (int) : how many rounds of training the network will get
  -----
  Source Code
  -----
  >>>
  for epoch in range(epochs):
    for feature, target in zip(features, targets):
      activations, weighted_sums = propagate(network, feature, ReLU)
      error = backpropegate(network, activations, weighted_sums, target, ReLU_derivative)
      update(network, activations, error, learning_rate)
    if epoch % 10 == 0:
      print(f"Epoch {epoch:5} | Error {mse(target, activations[-1])}")
  """
  for epoch in range(epochs):
    for feature, target in zip(features, targets):
      activations, weighted_sums = propagate(network, feature, ReLU)
      error = backpropegate(network, activations, weighted_sums, target, ReLU_derivative)
      
      update(network, activations, error, learning_rate)

    if epoch % 10 == 0:
      print(f"Epoch {epoch:5} | Error {mse(target, activations[-1])}")

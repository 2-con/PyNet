"""
Regression
=====

"""

import sys
import os
current_script_dir = os.path.dirname(__file__)
pynet_root_dir = os.path.abspath(os.path.join(current_script_dir, '..'))
sys.path.append(pynet_root_dir)

import math
from core.activation import Sigmoid
import core.optimizer as optimizer
import tools.utility as utility
import random
from api.synapse import Key
from tools.math import sgn
import itertools
import numpy as np

Optimizer = optimizer.Optimizer # set global object

class Linear:
  def __init__(self, input_size, output_size):
    """
    Linear Regression
    -----
      Create a multidimentional linear regression model with custom hyperparameters
    -----
    Args
    -----
    - input_size  (int) : number of input features
    - output_size (int) : number of output features
    """
    
    # defined during compiling
    self.optimizer      = None
    self.loss           = None
    self.metrics        = None
    self.learning_rate  = None
    self.epochs         = None

    self.is_compiled    = False # if the model is already compiled
    self.is_trained     = False # if the model is already fitted
    self.input_size     = input_size
    self.output_size    = output_size
    self.error_logs = []

    self.optimizer_instance = Optimizer()
    
    self.neurons = [
      {
      'weights': [random.uniform(-1, 1) for _ in range(input_size)],
      'bias': random.uniform(-1, 1)
      }
      for _ in range(output_size)
    ]
    
  def compile(self, optimizer, loss, learning_rate, epochs, **kwargs):
    """
    Compile
    -----
      Compiles the model to be ready for training.
      the PyNet commpiler will automatically take care of hyperparameters and fine tuning under the hood
      unless explicitly defined
    -----
    Args
    -----
    - optimizer                  (str)   : optimizer to use
    - loss                       (str)   : loss function to use
    - metrics                    (list)  : metrics to use
    - learning_rate              (float) : learning rate to use
    - epochs                     (int)   : number of epochs to train for
    - (Optional) batchsize       (int)   : batch size, defaults to 1
    - (Optional) initialization  (int)   : weight initialization
    - (Optional) experimental    (str)   : experimental settings to use

    Optimizer hyperparameters
    -----
    - (Optional) alpha    (float)
    - (Optional) beta     (float)
    - (Optional) epsilon  (float)
    - (Optional) gamma    (float)
    - (Optional) delta    (float)
    -----
    Optimizers
    - ADAM        (Adaptive Moment Estimation)
    - RMSprop     (Root Mean Square Propagation)
    - Adagrad
    - Amsgrad
    - Adadelta
    - Gradclip    (Gradient Clipping)
    - Adamax
    - SGNDescent  (Sign Gradient Descent)
    - Default     (PyNet descent)
    - Variational Momentum
    - Momentum
    - None        (Gradient Descent)

    Losses
    - mean squared error
    - mean abseloute error
    - total squared error
    - total abseloute error
    - categorical crossentropy
    - binary cross entropy
    - sparse categorical crossentropy
    - hinge loss

    """
    
    self.optimizer      = optimizer.lower()
    self.loss           = loss.lower()
    self.learning_rate  = learning_rate
    self.epochs         = epochs
    self.batchsize      = kwargs.get('batchsize', 1)
    self.experimental   = kwargs.get('experimental', [])

    self.alpha    = kwargs.get('alpha', None) # momentum decay
    self.beta     = kwargs.get('beta', None)
    self.epsilon  = kwargs.get('epsilon', None) # zerodivison prevention
    self.gamma    = kwargs.get('gamma', None)
    self.delta    = kwargs.get('delta', None)
    
    self.is_compiled = True
    
  def fit(self, features, targets, **kwargs):
    """
    Fit
    -----
      Trains the model to fit into the given features and targets. it is reccomended to keep the verbosity to 0, 3 or 4
      as the progressbar is detrimental to efficiency
    -----
    Args
    -----
    - features (list)  : the features to use
    - targets  (list)  : the corresponding targets to use

    - (optional) verbose       (int) : whether to show anything during training
    - (optional) regularity    (int) : how often to show training stats
    """
    def Backpropagate(predicted, target):
      
      if self.loss == 'total squared error':
        errors = [2 * (pred - true) for pred, true in zip(predicted, target)]

      elif self.loss == 'mean abseloute error':
        errors = [sgn(pred - true) / len(target) for pred, true in zip(predicted, target)]

      elif self.loss == 'total abseloute error':
        errors = [sgn(pred - true) for pred, true in zip(predicted, target)]

      elif self.loss == 'hinge loss':
        errors = [-true if 1-true*pred > 0 else 0 for true, pred in zip(target, predicted)]

      else: # defaults to MSE if loss is ambiguous
        errors = [((pred - true)) / len(target) for pred, true in zip(predicted, target)]
      
      return errors
    
    def Update(error, input, timestep, batchsize):
      
      alpha = self.alpha
      beta = self.beta
      epsilon = self.epsilon
      gamma = self.gamma
      delta = self.delta

      optimize = Key.OPTIMIZER.get(self.optimizer)
      learning_rate = self.learning_rate
      param_id = 0 # must be a positive integer
      
      for neuron_index, neuron in enumerate(self.neurons):

        for weight_index, weight in enumerate(neuron['weights']):

          # calculate universal gradient
          weight_gradient = error[neuron_index] * input[weight_index]

          if (self.optimizer != 'none') and ('fullgrad' not in self.experimental):
            weight_gradient /= batchsize

          # Update weights
          param_id += 1
          neuron['weights'][weight_index] = optimize(learning_rate, weight, weight_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)

        # Updating bias
        bias_gradient = error[neuron_index]

        if (self.optimizer != 'none') and ('fullgrad' not in self.experimental):
          bias_gradient /= batchsize

        param_id += 1
        neuron['bias'] = optimize(learning_rate, neuron['bias'], bias_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)

    epochs = self.epochs + 1
    verbose = kwargs.get('verbose', 0)
    regularity = kwargs.get('regularity', 1)
    timestep = 0
    errors = []
    
    for epoch in utility.progress_bar(range(epochs), "> Training", "Complete", decimals=2, length=70, empty=' ') if verbose==1 else range(epochs):
      epoch_loss = 0
      for base_index in utility.progress_bar(range(0, len(features), self.batchsize), "> Processing Batch", f"Epoch {epoch+1 if epoch == 0 else epoch}/{epochs-1} ({round( ((epoch+1)/epochs)*100 , 2)})%", decimals=2, length=70, empty=' ') if verbose==2 else range(0, len(features), self.batchsize):

        for batch_index in range(self.batchsize):

          if base_index + batch_index >= len(features):
            continue

          activations = self.predict(features[base_index + batch_index])
          errors.append(Backpropagate(activations, targets[base_index + batch_index]))
          
          epoch_loss += Key.ERROR[self.loss](targets[base_index + batch_index], activations)
        
        timestep += 1
        
        if 'frozenTS' in self.experimental:
          timestep = 1
        
        for error in errors:
          Update(error, features[base_index + batch_index], timestep, self.batchsize)
          
        errors = []

      self.error_logs.append(epoch_loss)

      if epoch % regularity == 0 and verbose>=3:
        prefix = f"Epoch {epoch+1 if epoch == 0 else epoch}/{epochs-1} ({round( ((epoch+1)/epochs)*100 , 2)}%) "
        suffix = f"| Loss: {str(epoch_loss):25} |"

        rate = f" ROC: {epoch_loss - self.error_logs[epoch-1] if epoch > 0 else 0}"

        pad = ' ' * ( len(f"Epoch {epochs}/{epochs-1} (100.0%) ") - len(prefix))
        print(prefix + pad + suffix + rate if verbose == 4 else prefix + pad + suffix)
  
  def predict(self, point):

    if len(point) != self.input_size:
      raise ValueError(f"input must have {self.input_size} elements")

    return [
    (
      sum(input_val * weight_val for input_val, weight_val in zip(point, _neuron['weights'])) + _neuron['bias']
    )
    for _neuron in self.neurons
    ]

class Polynomial:
  def __init__(self, input_size, output_size, degree):
    """
    Polynomial Regression
    -----
      Create a multidimentional polynomial regression model with custom hyperparameters
    -----
    Args
    -----
    - input_size  (int) : number of input features
    - output_size (int) : number of output features
    - degree      (int) : degree of the polynomial
    """
    
    # defined during compiling
    self.optimizer      = None
    self.loss           = None
    self.metrics        = None
    self.learning_rate  = None
    self.epochs         = None
    self.is_compiled    = False # if the model is already compiled
    self.is_trained     = False # if the model is already fitted
    self.input_size_raw = input_size
    self.input_size     = (math.factorial( input_size + degree ) / (math.factorial(degree) * math.factorial( input_size ))) - 1
    self.output_size    = output_size
    self.error_logs = []
    self.degree = degree
    self.optimizer_instance = Optimizer()
    
    self.neurons = [
      {
      'weights': [random.uniform(-1, 1) for _ in range(input_size)],
      'bias': random.uniform(-1, 1)
      }
      for _ in range(output_size)
    ]
    
  def compile(self, optimizer, loss, learning_rate, epochs, **kwargs):
    """
    Compile
    -----
      Compiles the model to be ready for training.
      the PyNet commpiler will automatically take care of hyperparameters and fine tuning under the hood
      unless explicitly defined
    -----
    Args
    -----
    - optimizer                  (str)   : optimizer to use
    - loss                       (str)   : loss function to use
    - metrics                    (list)  : metrics to use
    - learning_rate              (float) : learning rate to use
    - epochs                     (int)   : number of epochs to train for
    - (Optional) batchsize       (int)   : batch size, defaults to 1
    - (Optional) initialization  (int)   : weight initialization
    - (Optional) experimental    (str)   : experimental settings to use

    Optimizer hyperparameters
    -----
    - (Optional) alpha    (float)
    - (Optional) beta     (float)
    - (Optional) epsilon  (float)
    - (Optional) gamma    (float)
    - (Optional) delta    (float)
    -----
    Optimizers
    - ADAM        (Adaptive Moment Estimation)
    - RMSprop     (Root Mean Square Propagation)
    - Adagrad
    - Amsgrad
    - Adadelta
    - Gradclip    (Gradient Clipping)
    - Adamax
    - SGNDescent  (Sign Gradient Descent)
    - Default     (PyNet descent)
    - Variational Momentum
    - Momentum
    - None        (Gradient Descent)

    Losses
    - mean squared error
    - mean abseloute error
    - total squared error
    - total abseloute error
    - categorical crossentropy
    - binary cross entropy
    - sparse categorical crossentropy
    - hinge loss

    """
    
    self.optimizer      = optimizer.lower()
    self.loss           = loss.lower()
    self.learning_rate  = learning_rate
    self.epochs         = epochs
    self.batchsize      = kwargs.get('batchsize', 1)
    self.experimental   = kwargs.get('experimental', [])

    self.alpha    = kwargs.get('alpha', None) # momentum decay
    self.beta     = kwargs.get('beta', None)
    self.epsilon  = kwargs.get('epsilon', None) # zerodivison prevention
    self.gamma    = kwargs.get('gamma', None)
    self.delta    = kwargs.get('delta', None)
    
    self.is_compiled = True
    
  def fit(self, features, targets, **kwargs):
    """
    Fit
    -----
      Trains the model to fit into the given features and targets. it is reccomended to keep the verbosity to 0, 3 or 4
      as the progressbar is detrimental to efficiency
    -----
    Args
    -----
    - features (list)  : the features to use
    - targets  (list)  : the corresponding targets to use

    - (optional) verbose       (int) : whether to show anything during training
    - (optional) regularity    (int) : how often to show training stats
    """
    def Backpropagate(predicted, target):
      
      if self.loss == 'total squared error':
        errors = [2 * (pred - true) for pred, true in zip(predicted, target)]

      elif self.loss == 'mean abseloute error':
        errors = [sgn(pred - true) / len(target) for pred, true in zip(predicted, target)]

      elif self.loss == 'total abseloute error':
        errors = [sgn(pred - true) for pred, true in zip(predicted, target)]

      elif self.loss == 'hinge loss':
        errors = [-true if 1-true*pred > 0 else 0 for true, pred in zip(target, predicted)]

      else: # defaults to MSE if loss is ambiguous
        errors = [((pred - true)) / len(target) for pred, true in zip(predicted, target)]
      
      return errors
    
    def Update(error, input, timestep, batchsize):
      
      alpha = self.alpha
      beta = self.beta
      epsilon = self.epsilon
      gamma = self.gamma
      delta = self.delta

      optimize = Key.OPTIMIZER.get(self.optimizer)
      learning_rate = self.learning_rate
      param_id = 0 # must be a positive integer
      
      for neuron_index, neuron in enumerate(self.neurons):

        for weight_index, weight in enumerate(neuron['weights']):

          # calculate universal gradient
          weight_gradient = error[neuron_index] * input[weight_index]

          if (self.optimizer != 'none') and ('fullgrad' not in self.experimental):
            weight_gradient /= batchsize

          # Update weights
          param_id += 1
          neuron['weights'][weight_index] = optimize(learning_rate, weight, weight_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)

        # Updating bias
        bias_gradient = error[neuron_index]

        if (self.optimizer != 'none') and ('fullgrad' not in self.experimental):
          bias_gradient /= batchsize

        param_id += 1
        neuron['bias'] = optimize(learning_rate, neuron['bias'], bias_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)

    epochs = self.epochs + 1
    verbose = kwargs.get('verbose', 0)
    regularity = kwargs.get('regularity', 1)
    timestep = 0
    errors = []
    
    for epoch in utility.progress_bar(range(epochs), "> Training", "Complete", decimals=2, length=70, empty=' ') if verbose==1 else range(epochs):
      epoch_loss = 0
      for base_index in utility.progress_bar(range(0, len(features), self.batchsize), "> Processing Batch", f"Epoch {epoch+1 if epoch == 0 else epoch}/{epochs-1} ({round( ((epoch+1)/epochs)*100 , 2)})%", decimals=2, length=70, empty=' ') if verbose==2 else range(0, len(features), self.batchsize):

        for batch_index in range(self.batchsize):

          if base_index + batch_index >= len(features):
            continue

          activations = self.predict(features[base_index + batch_index])
          errors.append(Backpropagate(activations, targets[base_index + batch_index]))
          
          epoch_loss += Key.ERROR[self.loss](targets[base_index + batch_index], activations)
        
        timestep += 1
        
        if 'frozenTS' in self.experimental:
          timestep = 1
        
        for error in errors:
          Update(error, features[base_index + batch_index], timestep, self.batchsize)
          
        errors = []

      self.error_logs.append(epoch_loss)

      if epoch % regularity == 0 and verbose>=3:
        prefix = f"Epoch {epoch+1 if epoch == 0 else epoch}/{epochs-1} ({round( ((epoch+1)/epochs)*100 , 2)}%) "
        suffix = f"| Loss: {str(epoch_loss):25} |"

        rate = f" ROC: {epoch_loss - self.error_logs[epoch-1] if epoch > 0 else 0}"

        pad = ' ' * ( len(f"Epoch {epochs}/{epochs-1} (100.0%) ") - len(prefix))
        print(prefix + pad + suffix + rate if verbose == 4 else prefix + pad + suffix)
  
  def predict(self, point):

    def poly_features(input_values, degree):
      num_original_features = len(input_values)
      polynomial_features = [] # Start with the constant (bias) term of 1.0

      # Iterate through each desired degree, from 1 up to the specified maximum degree
      for d in range(1, degree + 1):
        for combo_indices in itertools.combinations_with_replacement(range(num_original_features), d):
          term_value = 1
          # Calculate the product of the input values corresponding to the chosen indices
          for index in combo_indices:
            term_value *= input_values[index]
            
          polynomial_features.append(term_value)

      return polynomial_features
    
    if len(poly_features(point, self.degree)) != self.input_size:
      raise ValueError(f"input must have {self.input_size_raw} elements")
    
    return [
    (
      sum(input_val * weight_val for input_val, weight_val in zip(poly_features(point, self.degree), _neuron['weights'])) + _neuron['bias']
    )
    for _neuron in self.neurons
    ]

class Logistic:
  def __init__(self, input_size, output_size, **kwargs):
    """
    Polynomial Regression
    -----
      Create a multidimentional polynomial regression model with custom hyperparameters
    -----
    Args
    -----
    - input_size        (int) : number of input features
    - output_size       (int) : number of output features
    - (optional) degree (int) : degree of the logits, default is 1
    """
    
    # defined during compiling
    self.optimizer      = None
    self.loss           = None
    self.metrics        = None
    self.learning_rate  = None
    self.epochs         = None
    
    degree = kwargs.get('degree', 1)
    self.is_compiled    = False # if the model is already compiled
    self.is_trained     = False # if the model is already fitted
    self.input_size_raw = input_size
    self.input_size     = (math.factorial( input_size + degree ) / (math.factorial(degree) * math.factorial( input_size ))) - 1
    self.output_size    = output_size
    self.error_logs = []
    self.degree = degree
    self.optimizer_instance = Optimizer()
    
    self.neurons = [
      {
      'weights': [random.uniform(-1, 1) for _ in range(input_size)],
      'bias': random.uniform(-1, 1)
      }
      for _ in range(output_size)
    ]
    
  def compile(self, optimizer, loss, learning_rate, epochs, **kwargs):
    """
    Compile
    -----
      Compiles the model to be ready for training.
      the PyNet commpiler will automatically take care of hyperparameters and fine tuning under the hood
      unless explicitly defined
    -----
    Args
    -----
    - optimizer                  (str)   : optimizer to use
    - loss                       (str)   : loss function to use
    - metrics                    (list)  : metrics to use
    - learning_rate              (float) : learning rate to use
    - epochs                     (int)   : number of epochs to train for
    - (Optional) batchsize       (int)   : batch size, defaults to 1
    - (Optional) initialization  (int)   : weight initialization
    - (Optional) experimental    (str)   : experimental settings to use

    Optimizer hyperparameters
    -----
    - (Optional) alpha    (float)
    - (Optional) beta     (float)
    - (Optional) epsilon  (float)
    - (Optional) gamma    (float)
    - (Optional) delta    (float)
    -----
    Optimizers
    - ADAM        (Adaptive Moment Estimation)
    - RMSprop     (Root Mean Square Propagation)
    - Adagrad
    - Amsgrad
    - Adadelta
    - Gradclip    (Gradient Clipping)
    - Adamax
    - SGNDescent  (Sign Gradient Descent)
    - Default     (PyNet descent)
    - Variational Momentum
    - Momentum
    - None        (Gradient Descent)

    Losses
    - mean squared error
    - mean abseloute error
    - total squared error
    - total abseloute error
    - categorical crossentropy
    - binary cross entropy
    - sparse categorical crossentropy
    - hinge loss
    """
    
    self.optimizer      = optimizer.lower()
    self.loss           = loss.lower()
    self.learning_rate  = learning_rate
    self.epochs         = epochs
    self.batchsize      = kwargs.get('batchsize', 1)
    self.experimental   = kwargs.get('experimental', [])

    self.alpha    = kwargs.get('alpha', None) # momentum decay
    self.beta     = kwargs.get('beta', None)
    self.epsilon  = kwargs.get('epsilon', None) # zerodivison prevention
    self.gamma    = kwargs.get('gamma', None)
    self.delta    = kwargs.get('delta', None)
    
    self.is_compiled = True
    
  def fit(self, features, targets, **kwargs):
    """
    Fit
    -----
      Trains the model to fit into the given features and targets. it is reccomended to keep the verbosity to 0, 3 or 4
      as the progressbar is detrimental to efficiency
    -----
    Args
    -----
    - features (list)  : the features to use
    - targets  (list)  : the corresponding targets to use

    - (optional) verbose       (int) : whether to show anything during training
    - (optional) regularity    (int) : how often to show training stats
    """
    def Backpropagate(predicted, target):
      
      if self.loss == 'total squared error':
        errors = [2 * (pred - true) for pred, true in zip(predicted, target)]

      elif self.loss == 'mean abseloute error':
        errors = [sgn(pred - true) / len(target) for pred, true in zip(predicted, target)]

      elif self.loss == 'total abseloute error':
        errors = [sgn(pred - true) for pred, true in zip(predicted, target)]

      elif self.loss == 'categorical crossentropy':

        errors = [-true / pred if pred != 0 else -1000 for true, pred in zip(target, predicted)]

      elif self.loss == 'sparse categorical crossentropy':

        errors = [-(true == i) / pred if pred != 0 else -1000 for i, pred, true in zip(range(len(predicted)), predicted, target)]

      elif self.loss == 'binary crossentropy':

        errors = [
                            -1 / pred if true == 1 else 1 / (1 - pred) if pred < 1 else 1000
                            for true, pred in zip(target, predicted)
          ]
      
      elif self.loss == 'hinge loss':
        errors = [-true if 1-true*pred > 0 else 0 for true, pred in zip(target, predicted)]

      else: # defaults to MSE if loss is ambiguous
        errors = [((pred - true)) / len(target) for pred, true in zip(predicted, target)]
      
      return errors
    
    def Update(error, input, timestep, batchsize):
      
      alpha = self.alpha
      beta = self.beta
      epsilon = self.epsilon
      gamma = self.gamma
      delta = self.delta

      optimize = Key.OPTIMIZER.get(self.optimizer)
      learning_rate = self.learning_rate
      param_id = 0 # must be a positive integer
      
      for neuron_index, neuron in enumerate(self.neurons):

        for weight_index, weight in enumerate(neuron['weights']):

          # calculate universal gradient
          weight_gradient = error[neuron_index] * input[weight_index]

          if (self.optimizer != 'none') and ('fullgrad' not in self.experimental):
            weight_gradient /= batchsize

          # Update weights
          param_id += 1
          neuron['weights'][weight_index] = optimize(learning_rate, weight, weight_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)

        # Updating bias
        bias_gradient = error[neuron_index]

        if (self.optimizer != 'none') and ('fullgrad' not in self.experimental):
          bias_gradient /= batchsize

        param_id += 1
        neuron['bias'] = optimize(learning_rate, neuron['bias'], bias_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)

    epochs = self.epochs + 1
    verbose = kwargs.get('verbose', 0)
    regularity = kwargs.get('regularity', 1)
    timestep = 0
    errors = []
    
    for epoch in utility.progress_bar(range(epochs), "> Training", "Complete", decimals=2, length=70, empty=' ') if verbose==1 else range(epochs):
      epoch_loss = 0
      for base_index in utility.progress_bar(range(0, len(features), self.batchsize), "> Processing Batch", f"Epoch {epoch+1 if epoch == 0 else epoch}/{epochs-1} ({round( ((epoch+1)/epochs)*100 , 2)})%", decimals=2, length=70, empty=' ') if verbose==2 else range(0, len(features), self.batchsize):

        for batch_index in range(self.batchsize):

          if base_index + batch_index >= len(features):
            continue

          activations = self.predict(features[base_index + batch_index])
          errors.append(Backpropagate(activations, targets[base_index + batch_index]))
          
          epoch_loss += Key.ERROR[self.loss](targets[base_index + batch_index], activations)
        
        timestep += 1
        
        if 'frozenTS' in self.experimental:
          timestep = 1
        
        for error in errors:
          Update(error, features[base_index + batch_index], timestep, self.batchsize)
          
        errors = []

      self.error_logs.append(epoch_loss)

      if epoch % regularity == 0 and verbose>=3:
        prefix = f"Epoch {epoch+1 if epoch == 0 else epoch}/{epochs-1} ({round( ((epoch+1)/epochs)*100 , 2)}%) "
        suffix = f"| Loss: {str(epoch_loss):25} |"

        rate = f" ROC: {epoch_loss - self.error_logs[epoch-1] if epoch > 0 else 0}"

        pad = ' ' * ( len(f"Epoch {epochs}/{epochs-1} (100.0%) ") - len(prefix))
        print(prefix + pad + suffix + rate if verbose == 4 else prefix + pad + suffix)
  
  def predict(self, point):

    def poly_features(input_values, degree):
      num_original_features = len(input_values)
      polynomial_features = [] # Start with the constant (bias) term of 1.0

      # Iterate through each desired degree, from 1 up to the specified maximum degree
      for d in range(1, degree + 1):
        for combo_indices in itertools.combinations_with_replacement(range(num_original_features), d):
          term_value = 1
          # Calculate the product of the input values corresponding to the chosen indices
          for index in combo_indices:
            term_value *= input_values[index]
            
          polynomial_features.append(term_value)

      return polynomial_features
    
    if len(poly_features(point, self.degree)) != self.input_size:
      raise ValueError(f"input must have {self.input_size_raw} elements")
    
    return [
    Sigmoid(
      sum(input_val * weight_val for input_val, weight_val in zip(poly_features(point, self.degree), _neuron['weights'])) + _neuron['bias']
    )
    for _neuron in self.neurons
    ]

class Exponential:
  ...

class Sinusoidal:
  # AI made this as an example on what to do
  # this is not mine

  def __init__(self, input_size: int, num_harmonics: int = 1):
    self.input_size = input_size
    self.num_harmonics = num_harmonics
    self.weights = np.zeros((input_size, num_harmonics))
    self.inner_weights = np.zeros((input_size, num_harmonics))
    self.inner_biases = np.zeros((input_size, num_harmonics))
    self.bias = 0

  def fit(self, features, targets, learning_rate: float = 0.01, epochs: int = 1000):
    for _ in range(epochs):
      for point, target in zip(features, targets):
        prediction = self.predict(point)
        error = target - prediction
        self.bias += learning_rate * error
        for i in range(self.input_size):
          for j in range(self.num_harmonics):
            self.weights[i, j] += learning_rate * error * math.sin(self.inner_weights[i, j] * point[i] + self.inner_biases[i, j])
            self.inner_weights[i, j] += learning_rate * error * self.weights[i, j] * math.cos(self.inner_weights[i, j] * point[i] + self.inner_biases[i, j]) * point[i]
            self.inner_biases[i, j] += learning_rate * error * self.weights[i, j] * math.cos(self.inner_weights[i, j] * point[i] + self.inner_biases[i, j])

  def predict(self, point):
    if len(point) != self.input_size:
      raise ValueError(f"point must have {self.input_size} elements")

    weighted_sum = 0
    for i in range(self.input_size):
      for j in range(self.num_harmonics):
        weighted_sum += math.sin(self.inner_weights[i, j] * point[i] + self.inner_biases[i, j]) * self.weights[i, j]
    return weighted_sum + self.bias

class Power:
  ...

class Logarithmic:
  ...
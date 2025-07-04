"""
Regression
=====

"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.activation import Sigmoid
import core.optimizer as optimizer
import tools.utility as utility
from api.synapse import Key
import random, numpy as np, math, itertools
from tools.math import sgn

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
    
    self.is_compiled = False

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
    - (optional) verbose         (int)   : whether to show anything during training
    - (optional) logging         (int)   : how often to show training stats

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
    self.verbose        = kwargs.get('verbose', 0)
    self.logging        = kwargs.get('logging', 1)

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
    verbose = self.verbose
    regularity = self.logging
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
    
    self.is_compiled = False
    
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
    - (optional) verbose         (int)   : whether to show anything during training
    - (optional) logging         (int)   : how often to show training stats

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
    self.verbose        = kwargs.get('verbose', 0)
    self.logging        = kwargs.get('logging', 1)

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
    verbose = self.verbose
    regularity = self.logging
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
    Logistic Regression
    -----
      Create a multidimentional logistic regression model with custom hyperparameters
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
    
    self.is_compiled = False
    
    degree = kwargs.get('degree', 1)
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
    - (optional) verbose         (int)   : whether to show anything during training
    - (optional) logging         (int)   : how often to show training stats

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
    self.verbose        = kwargs.get('verbose', 0)
    self.logging        = kwargs.get('logging', 1)

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
    verbose = self.verbose
    regularity = self.logging
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
  def __init__(self, input_size, output_size):
    """
    Exponential Regression
    -----
      Create a multidimentional Exponential regression model with custom hyperparameters, 
      following the formula y = a * e ^ (b * x) + c where e = 2.71828 (Euler's number)
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
    
    self.is_compiled = False

    self.input_size     = input_size
    self.output_size    = output_size
    self.error_logs = []

    self.optimizer_instance = Optimizer()
    
    self.neurons = [
      {
      'weights'  : [random.uniform(-1, 1) for _ in range(input_size)],
      'intercept': random.uniform(-1,1),
      'factor'   : random.uniform(-1,1),
      'bias'     : random.uniform(-1,1),
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
      
      Since exponential regressions tend to create exploding gradients, it is highly reccomended to use a low learning rate
      or to use an adaptive optimizer
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
    - (optional) verbose         (int)   : whether to show anything during training
    - (optional) logging         (int)   : how often to show training stats

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
    self.verbose        = kwargs.get('verbose', 0)
    self.logging        = kwargs.get('logging', 1)

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
        
        total_x = sum([weight * input[weight_index] for weight_index, weight in enumerate(neuron['weights'])])

        for weight_index, weight in enumerate(neuron['weights']):

          # calculate universal gradient
          weight_gradient = error[neuron_index] * input[weight_index] * neuron['factor'] * math.exp(neuron['factor'] * total_x)

          if (self.optimizer != 'none') and ('fullgrad' not in self.experimental):
            weight_gradient /= batchsize

          # Update weights
          param_id += 1
          neuron['weights'][weight_index] = optimize(learning_rate, weight, weight_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)

        # Updating bias, intercept and factor
        bias_gradient      = error[neuron_index]
        intercept_gradient = error[neuron_index] * 2.71828 ** (neuron['factor'] * total_x)
        factor_gradient    = error[neuron_index] * 2.71828 ** (neuron['factor'] * total_x) * total_x * neuron['intercept']

        if (self.optimizer != 'none') and ('fullgrad' not in self.experimental):
          bias_gradient /= batchsize
          intercept_gradient /= batchsize
          factor_gradient /= batchsize

        param_id += 1
        neuron['bias'] = optimize(learning_rate, neuron['bias'], bias_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)
        
        param_id += 1
        neuron['intercept'] = optimize(learning_rate, neuron['intercept'], bias_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)
        
        param_id += 1
        neuron['factor'] = optimize(learning_rate, neuron['factor'], bias_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)

    epochs = self.epochs + 1
    verbose = self.verbose
    regularity = self.logging
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
      _neuron['intercept'] * math.exp( _neuron['factor'] * sum( input_val * weight_val for input_val, weight_val in zip(point, _neuron['weights']) ) ) + _neuron['bias']
    )
    for _neuron in self.neurons
    ]

# class Sinusoidal:
#   # AI made this as an example on what to do
#   # this is not mine

#   def __init__(self, input_size: int, num_harmonics: int = 1):
#     self.input_size = input_size
#     self.num_harmonics = num_harmonics
#     self.weights = np.zeros((input_size, num_harmonics))
#     self.inner_weights = np.zeros((input_size, num_harmonics))
#     self.inner_biases = np.zeros((input_size, num_harmonics))
#     self.bias = 0

#   def fit(self, features, targets, learning_rate: float = 0.01, epochs: int = 1000):
#     for _ in range(epochs):
#       for point, target in zip(features, targets):
#         prediction = self.predict(point)
#         error = target - prediction
#         self.bias += learning_rate * error
#         for i in range(self.input_size):
#           for j in range(self.num_harmonics):
#             self.weights[i, j] += learning_rate * error * math.sin(self.inner_weights[i, j] * point[i] + self.inner_biases[i, j])
#             self.inner_weights[i, j] += learning_rate * error * self.weights[i, j] * math.cos(self.inner_weights[i, j] * point[i] + self.inner_biases[i, j]) * point[i]
#             self.inner_biases[i, j] += learning_rate * error * self.weights[i, j] * math.cos(self.inner_weights[i, j] * point[i] + self.inner_biases[i, j])

#   def predict(self, point):
#     if len(point) != self.input_size:
#       raise ValueError(f"point must have {self.input_size} elements")

#     weighted_sum = 0
#     for i in range(self.input_size):
#       for j in range(self.num_harmonics):
#         weighted_sum += math.sin(self.inner_weights[i, j] * point[i] + self.inner_biases[i, j]) * self.weights[i, j]
#     return weighted_sum + self.bias

class Power:
  def __init__(self, input_size, output_size):
    """
    Power Regression
    -----
      Create a multidimentional power regression model with custom hyperparameters, 
      following the formula y = a * x^b + c
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
    
    self.is_compiled = False

    self.input_size     = input_size
    self.output_size    = output_size
    self.error_logs = []

    self.optimizer_instance = Optimizer()
    
    self.neurons = [
      {
      'weights'   : [random.uniform(-1, 1) for _ in range(input_size)],
      'multiplier': random.uniform(-1, 1),
      'exponent'  : random.uniform(-1, 1),
      'bias'      : random.uniform(-1, 1)
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
    - (optional) verbose         (int)   : whether to show anything during training
    - (optional) logging         (int)   : how often to show training stats

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
    self.verbose        = kwargs.get('verbose', 0)
    self.logging        = kwargs.get('logging', 1)

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
        
        full_x = sum([weight * input[index] for index, weight in enumerate(neuron['weights'])])

        for weight_index, weight in enumerate(neuron['weights']):

          # calculate universal gradient
          weight_gradient = error[neuron_index] * input[weight_index] * (neuron['exponent'] * full_x ** (neuron['exponent'] - 1))

          if (self.optimizer != 'none') and ('fullgrad' not in self.experimental):
            weight_gradient /= batchsize

          # Update weights
          param_id += 1
          neuron['weights'][weight_index] = optimize(learning_rate, weight, weight_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)

        # Updating bias, multiplier and exponent
        bias_gradient = error[neuron_index]
        multiplier_gradient = error[neuron_index] * full_x ** neuron['exponent']
        exponent_gradient = error[neuron_index]   * (full_x ** neuron['exponent']) * math.log(full_x) * neuron['multiplier']

        if (self.optimizer != 'none') and ('fullgrad' not in self.experimental):
          bias_gradient /= batchsize
          multiplier_gradient /= batchsize
          exponent_gradient /= batchsize

        param_id += 1
        neuron['bias'] = optimize(learning_rate, neuron['bias'], bias_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)

        param_id += 1
        neuron['multiplier'] = optimize(learning_rate, neuron['multiplier'], bias_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)

        param_id += 1
        neuron['exponent'] = optimize(learning_rate, neuron['exponent'], bias_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)

    epochs = self.epochs + 1
    verbose = self.verbose
    regularity = self.logging
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
      _neuron['multiplier'] * sum(input_val * weight_val for input_val, weight_val in zip(point, _neuron['weights'])) ** _neuron['exponent'] + _neuron['bias']
    )
    for _neuron in self.neurons
    ]

class Logarithmic:
  def __init__(self, input_size, output_size):
    """
    Logarithmic Regression
    -----
      Create a multidimentional logarithmic regression model with custom hyperparameters, 
      following the formula y = a * ln(x) + b
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
    
    self.is_compiled = False

    self.input_size     = input_size
    self.output_size    = output_size
    self.error_logs = []

    self.optimizer_instance = Optimizer()
    
    self.neurons = [
      {
      'weights'   : [random.uniform(-1, 1) for _ in range(input_size)],
      'multiplier': random.uniform(-1, 1),
      'bias'      : random.uniform(-1, 1)
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
    - (optional) verbose         (int)   : whether to show anything during training
    - (optional) logging         (int)   : how often to show training stats

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
    self.verbose        = kwargs.get('verbose', 0)
    self.logging        = kwargs.get('logging', 1)

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
        
        full_x = sum([weight * input[index] for index, weight in enumerate(neuron['weights'])])

        for weight_index, weight in enumerate(neuron['weights']):

          # calculate universal gradient
          weight_gradient = error[neuron_index] * input[weight_index] * (1/full_x)

          if (self.optimizer != 'none') and ('fullgrad' not in self.experimental):
            weight_gradient /= batchsize

          # Update weights
          param_id += 1
          neuron['weights'][weight_index] = optimize(learning_rate, weight, weight_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)
        
        # Updating bias, multiplier and exponent
        bias_gradient = error[neuron_index]
        multiplier_gradient = error[neuron_index] * math.log(full_x)

        if (self.optimizer != 'none') and ('fullgrad' not in self.experimental):
          bias_gradient /= batchsize
          multiplier_gradient /= batchsize

        param_id += 1
        neuron['bias'] = optimize(learning_rate, neuron['bias'], bias_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)

        param_id += 1
        neuron['multiplier'] = optimize(learning_rate, neuron['multiplier'], bias_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)

    epochs = self.epochs + 1
    verbose = self.verbose
    regularity = self.logging
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
      _neuron['multiplier'] * math.log( sum(input_val * weight_val for input_val, weight_val in zip(point, _neuron['weights'])) ) + _neuron['bias']
    )
    for _neuron in self.neurons
    ]

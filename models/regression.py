"""
Regression
=====

"""
import math
import numpy as np
from pynet.core.activation import Sigmoid
import pynet.core.optimizer as optimizer
import pynet.tools.utility as utility

Optimizer = optimizer.Optimizer # set global object

class Linear:
  def __init__(self):
    """
    Linear Regression
    """
    
    # defined during compiling
    self.optimizer      = None # name
    self.loss           = None # name
    self.metrics        = None # name
    self.learning_rate  = None
    self.epochs         = None

    self.is_compiled    = False # if the model is already compiled
    self.is_trained     = False # if the model is already fitted

    self.error_logs = []

    self.optimizer_instance = Optimizer()
    self.weights = [0]
    self.bias = 0
    
  def compile(self, optimizer, loss, learning_rate, epochs, metrics, **kwargs):
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

    Metrics
    - accuracy
    - precision
    - recall
    - f1_score
    - roc_auc
    - log_loss

    - categorical crossentropy
    - binary crossentropy
    - sparse categorical crossentropy
    - hinge loss

    - mean squared error
    - mean abseloute error
    - total squared error
    - total abseloute error
    - root mean squared error
    - r2_score

    """
    
    self.optimizer = optimizer.lower()
    self.loss = loss.lower()
    self.metrics = [m.lower() for m in metrics]
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.batchsize = kwargs.get('batchsize', 1)

    self.alpha = kwargs.get('alpha', None) # momentum decay
    self.beta = kwargs.get('beta', None)
    self.epsilon = kwargs.get('epsilon', None) # zerodivison prevention
    self.gamma = kwargs.get('gamma', None)
    self.delta = kwargs.get('delta', None)
    
    self.verbose = kwargs.get('verbose', 0)
    self.regularity = kwargs.get('regularity', 1)
    
    self.is_compiled = True

  def fit(self, features, targets, **kwargs):
    
    
    
    for epoch in utility.progress_bar(range(self.epochs), "> Training", "Complete", decimals=2, length=70, empty=' ') if self.verbose==1 else range(self.epochs):
      epoch_loss = 0
      for base_index in utility.progress_bar(range(0, len(features), self.batchsize), "> Processing Batch", f"Epoch {epoch+1 if epoch == 0 else epoch}/{self.epochs-1} ({round( ((epoch+1)/self.epochs)*100 , 2)})%", decimals=2, length=70, empty=' ') if self.verbose==2 else range(0, len(features), self.batchsize):
        errors = []

        for batch_index in range(self.batchsize):

          if base_index + batch_index >= len(features):
            continue

          point = features[base_index + batch_index]
          target = targets[base_index + batch_index]
          prediction = self.predict(point)

          error = target - prediction

          errors.append(error)

        # update(activations, weighted_sums, errors, learning_rate)

      self.error_logs.append(epoch_loss)

      if epoch % self.regularity == 0 and self.verbose>=3:
        prefix = f"Epoch {epoch+1 if epoch == 0 else epoch}/{self.epochs-1} ({round( ((epoch+1)/self.epochs)*100 , 2)}%) "
        suffix = f"| Loss: {str(epoch_loss):25} |"

        rate = f" ROC: {epoch_loss - self.error_logs[epoch-1] if epoch > 0 else 0}"

        pad = ' ' * ( len(f"Epoch {self.epochs}/{self.epochs-1} (100.0%) ") - len(prefix))
        print(prefix + pad + suffix + rate if self.verbose == 4 else prefix + pad + suffix)
  
  def predict(self, point):

    if len(point) != self.input_size:
      raise ValueError(f"input must have {self.input_size} elements")

    return sum([point * weight for point, weight in zip(point, self.weights)]) + self.bias

class Polynomial_Regression:
  ...

class Logistic_Regression:
  ...

class Exponential_Regression:
  ...

class Sinusoidal_Regression:
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

class Power_Regression:
  ...
class Logarithmic_Regression:
  ...
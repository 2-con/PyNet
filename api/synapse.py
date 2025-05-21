"""
Synapse API
=====
  A high-level API for sequential models of neural networks, Synapse automatically handles the learning
  process along with hyperparameters unless specified. things such as input neurons do not need
  to be defined and are handled automatically.
-----
Provides
-----
  (Learnable layers)
  1. Convolution layers
  2. Dense layers

  (Utility layers)
  1. Maxpooling layers
  2. Meanpooling layers
  3. Flatten layers
  4. Reshape layers
  5. AFS (Adaptive Feature Scaler) layers
  6. Operation (normalization and activation functions)

"""
#######################################################################################################
#                                    File Information and Handling                                    #
#######################################################################################################

__version__ = "1.0.0"

__package__ = "pynet"

if __name__ == "__main__":
  print("""
        This file is not meant to be run as a main file.
        More information can be found about PyNet's Synapse API on the documentation.
        'docs.txt' or https://www.pynet.com/api_docs/python/synapse
        """)
  exit()

#######################################################################################################
#                                               Imports                                               #
#######################################################################################################

import random
import numpy as np

from pynet.tools import arraytools, scaler, utility, visual
from pynet.tools import math as math2

from pynet.core import activation as Activation
from pynet.core import derivative as Derivative
from pynet.core import loss as Error
from pynet.core import metric as Metrics
from pynet.core import initialization as Initialization
import pynet.core.optimizer as optimizer

Optimizer = optimizer.Optimizer # set global object

#######################################################################################################
#                                               Extra                                                 #
#######################################################################################################

""" Notes

2/5/2025

  there is something peculiar with the AFS layer, spesifically when it comes to its backpropagation algorithm.
  typically you would've pass the weighted sum through the activation derivative function. but when the AFS layer
  is fed its activation instead, the result becomes more consistent and accurate? might have to investigate this further.
  its a real anomaly!
  (reffering to propagate > weighted sums)

  even more so, for the update policy, the APS layer seems to me much more accurate when it mimicks a Dense layer.
  multiplying its activation... it did return better results
  (reffering to update > AFS > weight)

3/5/2025

  even more intresting stuff is happening, in line 1313, instead of averaging out the bias gradients,
  i feel like i'll get better results if i dont, yet locially, the opposite should be true. as its
  industry practice to average it out...

5/5/2025

  thinking of adding a 'settings' feature that can get the weights and biases of a layer as a list that can be
  inserted to a compatible layer, essencially importing/exporting a layer and its pre-tuned weights. ill do this
  after completing the LCN (localunit) layer

8/5/2025

  thinking of adding customizable weight initialization, probably xavier/glorot initialization

13/5/2025

  convolution backpropagation is wrong. error gradients from the versions are always 0, it might originated from the
  backpropagation since the activations are working correctly

"""

""" ideas list:

  - code trace the training process, the training process should be flawlessly optimized and streamlined
  since pure python is notoriously slow

"""

#######################################################################################################
#                                         Internal Classes                                            #
#######################################################################################################

class Key:

  ACTIVATION = {
    # rectifiers
    'relu': Activation.ReLU,
    'softplus': Activation.Softplus,
    'mish': Activation.Mish,
    'swish': Activation.Swish,
    'leaky relu': Activation.Leaky_ReLU,
    'elu': Activation.ELU,
    'gelu': Activation.GELU,
    'selu': Activation.SELU,
    'reeu': Activation.ReEU,
    'none': Activation.Linear,
    'tandip': Activation.TANDIP,

    # normalization functions
    'binary step': Activation.Binary_step,
    'softsign': Activation.Softsign,
    'sigmoid': Activation.Sigmoid,
    'tanh': Activation.Tanh,
  }

  ACTIVATION_DERIVATIVE = {
    # rectifiers
    'relu': Derivative.ReLU_derivative,
    'softplus': Derivative.Softplus_derivative,
    'mish': Derivative.Mish_derivative,
    'swish': Derivative.Swish_derivative,
    'leaky relu': Derivative.Leaky_ReLU_derivative,
    'elu': Derivative.ELU_derivative,
    'gelu': Derivative.GELU_derivative,
    'selu': Derivative.SELU_derivative,
    'reeu': Derivative.ReEU_derivative,
    'none': Derivative.Linear_derivative,
    'tandip': Derivative.TANDIP_derivative,

    # normalization functions
    'binary step': Derivative.Binary_step_derivative,
    'softsign': Derivative.Softsign_derivative,
    'sigmoid': Derivative.Sigmoid_derivative,
    'tanh': Derivative.Tanh_derivative,
  }

  SCALER = {
    'standard scaler': scaler.standard_scaler,
    'min max scaler': scaler.min_max_scaler,
    'max abs scaler': scaler.max_abs_scaler,
    'robust scaler': scaler.robust_scaler,
    'softmax': scaler.softmax,
  }

  SCALER_DERIVATIVE = {
    'standard scaler': Derivative.Standard_scaler_derivative,
    'min max scaler': Derivative.Min_max_scaler_derivative,
    'max abs scaler': Derivative.Max_abs_scaler_derivative,
    'robust scaler': Derivative.Robust_scaler_derivative,
    'softmax': Derivative.Softmax_derivative,
  }

  ERROR = {
    # continuous error
    'mean squared error': Error.Mean_squared_error,
    'mean abseloute error': Error.Mean_absolute_error,
    'total squared error': Error.Total_squared_error,
    'total abseloute error': Error.Total_absolute_error,

    # categorical error
    'categorical crossentropy': Error.Categorical_crossentropy,
    'sparse categorical crossentropy': Error.Sparse_categorical_crossentropy,
    'binary crossentropy': Error.Binary_crossentropy,
    'hinge loss': Error.Hinge_loss,
    'l1 loss': Error.l1_loss,
  }

  OPTIMIZER = {
    'adam': Optimizer.Adam,
    'rmsprop': Optimizer.RMSprop,
    'adagrad': Optimizer.Adagrad,
    'amsgrad': Optimizer.Amsgrad,
    'adadelta': Optimizer.Adadelta,
    'gradclip': Optimizer.Gradclip,
    'adamax': Optimizer.Adamax,
    'sgnd': Optimizer.SGND,
    'default': Optimizer.Default,
    'none': Optimizer.Default,
    'variational momentum': Optimizer.Variational_Momentum,
    'momentum': Optimizer.Momentum,
  }

  METRICS = {

    # classification metrics
    'accuracy': Metrics.Accuracy,
    'precision': Metrics.Precision,
    'recall': Metrics.Recall,
    'f1 score': Metrics.F1_score,
    'roc auc': Metrics.ROC_AUC,
    'log loss': Metrics.Log_Loss,

    # categorical error
    'categorical crossentropy': Error.Categorical_crossentropy,
    'binary crossentropy': Error.Binary_crossentropy,
    'hinge loss': Error.Hinge_loss,

    # continuous error
    'mean squared error': Error.Mean_squared_error,
    'mean abseloute error': Error.Mean_absolute_error,
    'total squared error': Error.Total_squared_error,
    'total abseloute error': Error.Total_absolute_error,
    'root mean squared error': Metrics.Root_Mean_Squared_Error,
    'r2 score': Metrics.R2_Score,

  }

  INITIALIZATION = {
    'xavier uniform in': Initialization.Xavier_uniform_in,
    'xavier uniform out': Initialization.Xavier_uniform_out,
    'he uniform': Initialization.He_uniform,
    'glorot uniform': Initialization.Glorot_uniform,
    'lecun uniform': Initialization.Lecun_uniform,
    
    'he normal': Initialization.He_normal,
    'glorot normal': Initialization.Glorot_normal,
    'lecun normal': Initialization.Glorot_normal,
    
    'default': Initialization.Default,
    'none': Initialization.Default,
  }
  
#######################################################################################################
#                                          Sequential Model                                           #
#######################################################################################################

class Sequential:

  # pre-processing

  def __init__(self, *args):
    """
    Sequential Model
    ======
      Must contain Synapse layers to be added to the model. either directly through the constructor or through the add() method
    -----

    Available layers:
    - Convolution
    - Maxpooling
    - Meanpooling
    - Flatten
    - Reshape
    - Dense
    - AFS (Adaptive Feature Scaler)

    Additional info:
    - Input neurons are not neccecary, they are automatically handled during model fitting


    """
    self.layers = list(args) if args != None else []

    # defined during compiling
    self.optimizer      = None # name
    self.loss           = None # name
    self.metrics        = None # name
    self.learning_rate  = None
    self.epochs         = None

    self.is_compiled    = False # if the model is already compiled
    self.is_trained     = False # if the model is already fitted
    self.is_validated   = False # if the model is already validated

    self.error_logs = []

    self.optimizer_instance = Optimizer()

  def add(self, layer):
    """
    Add
    -----
      Adds a specified layer to the network.
    -----
    Args
    -----
    - layer (synapse object) : the layer to add to the model
    """
    if type(layer) not in (Convolution, Dense, Maxpooling, Meanpooling, Flatten, Reshape, AFS, Operation, Localunit):
      raise ValueError("layer must be of type Convolution, Dense, Maxpooling, Meanpooling, Flatten, Reshape, or AFS")

    self.layers.append(layer)

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

    self.experimental = kwargs.get('experimental', 'none')

    self.is_compiled = True

    # Structure - error prevention
    if self.layers == [] or self.layers is None:
      raise ValueError("No layers in model")
    for layer_index, layer in enumerate(self.layers):
      if type(layer) not in (Dense, Convolution, Maxpooling, Meanpooling, Flatten, Reshape, AFS, Operation, Localunit, Recurrent, LTSM, GRU):
        raise TypeError(f"Unknown layer type '{layer.__class__.__name__ }' at layer {layer_index+1}")

    # Kwargs - error prevention
    if learning_rate <= 0:
      raise ValueError("learning_rate must be greater than 0")
    if type(learning_rate) != float:
      raise TypeError("learning_rate must be a float")
    if type(epochs) != int:
      raise TypeError("epochs must be an int")
    if epochs <= 0:
          raise ValueError("epochs must be greater than 0")

    # Compiler - error prevention
    if self.optimizer is None:
      raise ValueError("Optimizer not set")
    if self.optimizer not in Key.OPTIMIZER:
      raise ValueError(f"Invalid optimizer {self.optimizer}")
    if self.loss is None:
      raise ValueError("Loss not set")
    if self.metrics == None:
      raise ValueError("Metrics not set")

  # processing

  def fit(self, features, targets, **kwargs):
    """
    Args
    -----
    - features (list)  : the features to use
    - targets  (list)  : the corresponding targets to use

    - (optional) verbose       (int) : whether to show anything during training
    - (optional) regularity    (int) : how often to show training stats
    - (optional) decimals      (int) : how many decimals to show
    """
    
    def update(activations, weighted_sums, errors, learning_rate):
      """
      Update
      -----
      updates the weights of the entire model, excluding the utility layers
      """

      alpha = self.alpha
      beta = self.beta
      epsilon = self.epsilon
      gamma = self.gamma
      delta = self.delta

      optimize = Key.OPTIMIZER.get(self.optimizer)

      param_id = 0

      for this_layer_index in reversed(range(len(self.layers))):

        layer = self.layers[this_layer_index]
        prev_activations = activations[this_layer_index]
        this_WS = weighted_sums[this_layer_index]
        #this_layer_error = errors[this_layer_index]

        if type(layer) == Dense and layer.learnable:

          for neuron_index, neuron in enumerate(layer.neurons):

            for weight_index in range(len(neuron['weights'])):

              # calculate universal gradient
              weight_gradient = 0

              for error_versions in errors:
                weight_gradient += error_versions[this_layer_index][neuron_index] * prev_activations[weight_index]

              weight_gradient /= len(errors)

              # Update weights
              param_id += 1
              neuron['weights'][weight_index] = optimize(learning_rate, neuron['weights'][weight_index], weight_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id)

            # Updating bias
            bias_gradient = 0

            for error_versions in errors:
              bias_gradient += error_versions[this_layer_index][neuron_index]

            if self.optimizer not in ('none', ''):
              bias_gradient /= len(errors)

            param_id += 1
            neuron['bias'] = optimize(learning_rate, neuron['bias'], bias_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id)

        elif type(layer) == Convolution and layer.learnable:

          weight_gradient = arraytools.generate_array(len(layer.kernel[0]), len(layer.kernel), value=0)
          kernel = arraytools.mirror(layer.kernel[::-1], 'Y') # flipped kernel
          bias_gradient = 0

          for error_versions in errors:

            # Rows of prev_errors (0 to 1)
            for a in range((len(prev_activations) - len(layer.kernel)) + 1):

              # Columns of prev_errors (0 to 1)
              for b in range((len(prev_activations[0]) - len(layer.kernel[0])) + 1):

                # Apply the flipped kernel to the corresponding region in layer_errors
                for kernel_row in range(len(layer.kernel)):

                  for kernel_col in range(len(layer.kernel[0])):
                    # Calculate the corresponding position in layer_errors

                    conv_row = kernel_row + a
                    conv_col = kernel_col + b

                    derivative = Key.ACTIVATION_DERIVATIVE[layer.activation](this_WS[a][b])
                    
                    weight_gradient[kernel_row][kernel_col] += prev_activations[conv_row][conv_col] * error_versions[this_layer_index][a][b] * derivative

            for row in range(len(error_versions[this_layer_index])):
              for col in range(len(error_versions[this_layer_index][0])):
                bias_gradient += error_versions[this_layer_index][row][col]

          bias_gradient /= len(prev_activations)**2

          # update kernel weights
          for a in range(len(layer.kernel)):
            for b in range(len(layer.kernel[a])):

              param_id += 1

              if self.optimizer not in ('none', ''):
                weight_gradient[a][b] /= len(errors)

              layer.kernel[a][b] = optimize(learning_rate, layer.kernel[a][b], weight_gradient[a][b], self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id)

          if layer.use_bias:
            param_id += 1

            if self.optimizer not in ('none', ''):
              bias_gradient /= len(errors)

            layer.bias = optimize(learning_rate, layer.bias, bias_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id)

        elif type(layer) == AFS and layer.learnable:

          for this_neuron_index, _ in enumerate(layer.neurons):

            # calculate weight gradient

            weight_gradient = 0

            for error_versions in errors:
              weight_gradient += error_versions[this_layer_index][this_neuron_index] * prev_activations[this_neuron_index]

            weight_gradient /= len(errors)

            param_id += 1
            layer.neurons[this_neuron_index]['weight'] = optimize(learning_rate, layer.neurons[this_neuron_index]['weight'], weight_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id)

            # calculate bias gradient
            if layer.use_bias:
              bias_gradient = 0

              for error_versions in errors:
                bias_gradient += error_versions[this_layer_index][this_neuron_index]

              if self.optimizer not in ('none', ''):
                bias_gradient /= len(errors)

              param_id += 1
              layer.neurons[this_neuron_index]['bias'] = optimize(learning_rate, layer.neurons[this_neuron_index]['bias'], bias_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id)

        elif type(layer) == Localunit and layer.learnable:

          for neuron_index, neuron in enumerate(layer.neurons):

            for weight_index in range(len(neuron['weights'])):

              # calculate universal gradient
              weight_gradient = 0

              for error_versions in errors:
                weight_gradient += error_versions[this_layer_index][neuron_index] * prev_activations[weight_index]

              weight_gradient /= len(errors)

              # Update weights
              param_id += 1
              neuron['weights'][weight_index] = optimize(learning_rate, neuron['weights'][weight_index], weight_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id)

            # Updating bias

            bias_gradient = 0

            for error_versions in errors:
              bias_gradient += error_versions[this_layer_index][neuron_index]

            if self.optimizer not in ('none', ''):
              bias_gradient /= len(errors)

            param_id += 1
            neuron['bias'] = optimize(learning_rate, neuron['bias'], bias_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id)

        elif type(layer) == Recurrent and layer.learnable:
          Wa_gradient = 0
          Wb_gradient = 0
          B_gradient  = 0
          
          for error_version in errors:
            Wa_gradient += error_version[this_layer_index][0]
            Wb_gradient += error_version[this_layer_index][1]
            B_gradient  += error_version[this_layer_index][2]

          if self.optimizer not in ('none', ''):
            Wa_gradient /= len(errors)
            Wb_gradient /= len(errors)
            B_gradient  /= len(errors)

          param_id += 1
          layer.carry_weight = optimize(learning_rate, layer.carry_weight, Wa_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id)
          param_id += 1
          layer.input_weight = optimize(learning_rate, layer.input_weight, Wb_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id)
          param_id += 1
          layer.bias         = optimize(learning_rate, layer.bias, B_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id)
        
        # elif type(layer) == LTSM and layer.learnable:
        #   pass
        
        # elif type(layer) == GRU and layer.learnable:
        #   pass
       
    self.is_trained = True

    # variables
    learning_rate = self.learning_rate
    epochs = self.epochs + 1

    verbose = kwargs.get('verbose', 0)
    regularity = kwargs.get('regularity', 1)

    if True: # Error Prevention

      # Function args - error prevention
      if not type(features) in (list, tuple):
        raise TypeError("features must be a list")
      if not type(targets) in (list, tuple):
        raise TypeError("targets must be a list")
      if len(features) == 0 or len(targets) == 0:
        raise ValueError("features or targets must not be empty")
      if len(features[0]) == 0 or len(targets[0]) == 0:
        raise ValueError("feature or target must not be empty")

    # neural network fitting
    x = features[0]
    sizes = [arraytools.shape(x)]
    calibrate_dense = False
    calibrate_afs = False
    calibrate_localunit = False
    calibrate_operation = False
    
    # calibration and structure intialization
    for _ in range(2):
      
      x = features[0]
      
      for layer_index, layer in enumerate(self.layers):

        if type(layer) == Maxpooling:
          x = layer.apply(x)
          sizes.append(arraytools.shape(x))

        elif type(layer) == Meanpooling:
          x = layer.apply(x)
          sizes.append(arraytools.shape(x))

        elif type(layer) == Convolution:
          x = layer.apply(x)
          sizes.append(arraytools.shape(x))

        elif type(layer) == Flatten:
          x = layer.apply(x)
          layer.set_length(len(x))
          sizes.append(arraytools.shape(x))

        elif type(layer) == Reshape:
          x = layer.apply(x)
          sizes.append(arraytools.shape(x))

        elif type(layer) == AFS:

          if not calibrate_afs:
            
            if layer_index == 0: # first layer
              layer.reshape_input_shape(len(features[0]), (1,1))

            else: # every other layer
              layer.reshape_input_shape(len(self.layers[layer_index-1].neurons), (1,1))
            
            x = layer.apply(x)
            sizes.append(arraytools.shape(x))
          
          else:
            
            if layer_index == len(self.layers) - 1: # last
              layer.reshape_input_shape(len(self.layers[layer_index-1].neurons), sizes[layer_index])

            else: # every other layer
              layer.reshape_input_shape(len(self.layers[layer_index-1].neurons), sizes[layer_index+1])

        elif type(layer) == Dense:
          if not calibrate_dense:
            
            if layer_index == 0: # first layer
              layer.reshape_input_shape(len(features[0]), (1,1))

            else: # every other layer
              layer.reshape_input_shape(len(self.layers[layer_index-1].neurons), (1,1))
            
            x = layer.apply(x)
            sizes.append(arraytools.shape(x))
          
          else:
            
            if layer_index == len(self.layers) - 1: # last
              layer.reshape_input_shape(len(self.layers[layer_index-1].neurons), sizes[layer_index])

            else: # every other layer
              layer.reshape_input_shape(len(self.layers[layer_index-1].neurons), sizes[layer_index+1])
          
        elif type(layer) == Operation:

          if not calibrate_operation:
            
            if layer_index == 0: # first layer
              layer.reshape_input_shape(len(features[0]), (1,1))

            else: # every other layer
              layer.reshape_input_shape(len(self.layers[layer_index-1].neurons), (1,1))
            
            x = layer.apply(x)
            sizes.append(arraytools.shape(x))
          
          else:
            
            if layer_index == len(self.layers) - 1: # last
              layer.reshape_input_shape(len(self.layers[layer_index-1].neurons), sizes[layer_index])

            else: # every other layer
              layer.reshape_input_shape(len(self.layers[layer_index-1].neurons), sizes[layer_index+1])

        elif type(layer) == Localunit:

          if not calibrate_localunit:
            
            if layer_index == 0: # first layer
              layer.reshape_input_shape(len(features[0]), (1,1))

            else: # every other layer
              layer.reshape_input_shape(len(self.layers[layer_index-1].neurons), (1,1))
            
            x = layer.apply(x)
            sizes.append(arraytools.shape(x))
          
          else:
            
            if layer_index == len(self.layers) - 1: # last
              layer.reshape_input_shape(len(self.layers[layer_index-1].neurons), sizes[layer_index])

            else: # every other layer
              layer.reshape_input_shape(len(self.layers[layer_index-1].neurons), sizes[layer_index+1])

      calibrate_dense = True
      calibrate_afs = True
      calibrate_localunit = True
      calibrate_operation = True
      
      self.sizes = sizes
    
    self.RNN  = any(type(layer) == Recurrent for layer in self.layers)
    self.LTSM = any(type(layer) == LTSM for layer in self.layers)
    self.GRU  = any(type(layer) == GRU for layer in self.layers)
    
    # main training loop - iterate over the epochs
    for epoch in utility.progress_bar(range(epochs), "> Training", "Complete", decimals=2, length=70, empty=' ') if verbose==1 else range(epochs):
      epoch_loss = 0
      for base_index in utility.progress_bar(range(0, len(features), self.batchsize), "> Processing Batch", f"Epoch {epoch+1 if epoch == 0 else epoch}/{epochs-1} ({round( ((epoch+1)/epochs)*100 , 2)})%", decimals=2, length=70, empty=' ') if verbose==2 else range(0, len(features), self.batchsize):
        errors = []

        for batch_index in range(self.batchsize):

          if base_index + batch_index >= len(features):
            continue

          activations, weighted_sums = self.Propagate(features[base_index + batch_index])
          
          if self.RNN:
            skibidi = 0
            target = []
            for layer in self.layers:
              if layer.return_output == False:
                target.append(0)
              else:
                if skibidi >= len(targets[base_index + batch_index]):
                  raise IndexError("not enough elements in the targets to verify")
                target.append(targets[base_index + batch_index][skibidi])
                skibidi += 1
          
          else:
            target = targets[base_index + batch_index]
          
          errors.append(self.Backpropagate(activations, weighted_sums, target))

          if self.RNN:
            predicted = [cell[1] for cell in activations[1:]]

          else:
            predicted = activations[-1]
            target = targets[base_index + batch_index]
          
          epoch_loss += Key.ERROR[self.loss](target, predicted)

        update(activations, weighted_sums, errors, learning_rate)
        errors = []

      self.error_logs.append(epoch_loss)

      if epoch % regularity == 0 and verbose>=3:
        prefix = f"Epoch {epoch+1 if epoch == 0 else epoch}/{epochs-1} ({round( ((epoch+1)/epochs)*100 , 2)}%) "
        suffix = f"| Loss: {str(epoch_loss):25} |"

        rate = f" ROC: {epoch_loss - self.error_logs[epoch-1] if epoch > 0 else 0}"

        pad = ' ' * ( len(f"Epoch {epochs}/{epochs-1} (100.0%) ") - len(prefix))
        print(prefix + pad + suffix + rate if verbose == 4 else prefix + pad + suffix)

  def Propagate(self, input):
    """
    Forward pass / Propagate
    -----
    gets the activations of the entire model, excluding the utility layers
    """

    x = input[:]
    activations = [x]
    weighted_sums = []

    if self.RNN:
      
      input_index = 0
      carry = 0
      carryWS = 0
      for layer in self.layers: # get activations
        
        if layer.accept_input:
          
          output, carry = layer.apply(x[input_index], carry)
          
          activations.append([x[input_index], output])
          
          outputWS, carryWS = layer.get_weighted_sum(x[input_index], carryWS)
          
          input_index += 1
          
        else:
          output, carry = layer.apply(0, carry)
          activations.append([0, output])
          
        weighted_sums.append(outputWS)
        
    else:
      x = input[:]
      xWS = 0
      for layer in self.layers:
        
        if type(layer) in (Convolution, Dense, AFS, Operation, Localunit):
          xWS = layer.get_weighted_sum(x)
          weighted_sums.append(xWS)

        else:
          weighted_sums.append(x)
          
        x = layer.apply(x)
        activations.append(x)  

    return activations, weighted_sums

  def Backpropagate(self, activations, weighted_sums, target):
    """
    Backward pass / Backpropagate
    -----
    gets the errors of the entire model, excluding the utility layers
    """

    def index_corrector(index):
      if index < 0:
        return None
      elif index > prev_layer.receptive_field - 1:
        return None
      else:
        return index

    errors = [0] * ( len(self.layers) )
    initial_gradient = []
    
    if self.RNN:
      predicted = [cell[1] for cell in activations[1:]]
      
    else:
      predicted = activations[-1]
    
    if True: # calculate the initial gradient

      if self.loss == 'total squared error':
        initial_gradient = [(pred - true) for pred, true in zip(predicted, target)]

      elif self.loss == 'mean abseloute error':
        initial_gradient = [math2.sgn(pred - true) / len(target) for pred, true in zip(predicted, target)]

      elif self.loss == 'total abseloute error':
        initial_gradient = [math2.sgn(pred - true) for pred, true in zip(predicted, target)]

      elif self.loss == 'categorical crossentropy':

        initial_gradient = [-true / pred if pred != 0 else -1000 for true, pred in zip(target, predicted)]

      elif self.loss == 'sparse categorical crossentropy':

        initial_gradient = [-(true == i) / pred if pred != 0 else -1000 for i, pred, true in zip(range(len(predicted)), predicted, target)]

      elif self.loss == 'binary crossentropy':

        initial_gradient = [
                            -1 / pred if true == 1 else 1 / (1 - pred) if pred < 1 else 1000
                            for true, pred in zip(target, predicted)
        ]

      elif self.loss == 'hinge loss':
        initial_gradient = [-true if 1-true*pred > 0 else 0 for true, pred in zip(target, predicted)]

      else: # defaults to MSE if loss is ambiguous
        initial_gradient = [(2 * (pred - true)) / len(target) for pred, true in zip(predicted, target)]

      output_layer_errors = []

      if type(self.layers[-1]) == Operation:

        this_WS = weighted_sums[-1]
        this_activations = activations[-1]
        this_layer = self.layers[-1]
        derivative_list = []

        if self.layers[-1].operation == 'min max scaler':
          derivative_list = [Key.SCALER_DERIVATIVE[this_layer.operation](a, min=this_layer.minimum, max=this_layer.maximum) for a in this_WS]

          output_layer_errors = [error * derivative for error, derivative in zip(initial_gradient, derivative_list)]

        elif self.layers[-1].operation == 'standard scaler':
          derivative_list = [Key.SCALER_DERIVATIVE[this_layer.operation](a, std=this_layer.std) for a in this_WS]

          output_layer_errors = [error * derivative for error, derivative in zip(initial_gradient, derivative_list)]

        elif self.layers[-1].operation == 'max abs scaler':
          derivative_list = [Key.SCALER_DERIVATIVE[this_layer.operation](a, max=this_layer.maximum) for a in this_WS]

          output_layer_errors = [error * derivative for error, derivative in zip(initial_gradient, derivative_list)]

        elif self.layers[-1].operation == 'robust scaler':
          derivative_list = [Key.SCALER_DERIVATIVE[this_layer.operation](a, q1=this_layer.q1, q3=this_layer.q3) for a in this_WS]

          output_layer_errors = [error * derivative for error, derivative in zip(initial_gradient, derivative_list)]

        elif self.layers[-1].operation == 'softmax':
          derivative_list = Key.SCALER_DERIVATIVE[this_layer.operation](this_activations)

          sqrt_size = int(len(derivative_list)**0.5)
          derivative_list = arraytools.reshape(derivative_list, sqrt_size, sqrt_size)

          num_outputs = len(initial_gradient)
          num_inputs = len(derivative_list[0]) # Assuming derivative_list is already reshaped

          for j in range(num_inputs): # Iterate through the columns of the Jacobian (inputs of softmax)
            result = 0
            for i in range(num_outputs): # Iterate through the rows of the Jacobian (outputs of softmax) and elements of initial_gradient
              result += initial_gradient[i] * derivative_list[i][j]
            output_layer_errors.append(result)

        else: # if its a regular operation

          derivative_list = [Key.ACTIVATION_DERIVATIVE[this_layer.operation](a) for a in this_WS]

          output_layer_errors = [error * derivative for error, derivative in zip(initial_gradient, derivative_list)]

      elif type(self.layers[-1]) in (Dense, AFS, Localunit): # if its not an operation layer

        for error, weighted in zip(initial_gradient, weighted_sums[-1]):
          derivative = Key.ACTIVATION_DERIVATIVE[self.layers[-1].activation](weighted)
          output_layer_errors.append(error * derivative)

      elif type(self.layers[-1]) == Flatten: # if its a flatten layer

        sizex, sizey = self.layers[-1].input_shape

        output_layer_errors = arraytools.reshape(initial_gradient[:], sizex, sizey)

      elif type(self.layers[-1]) == Recurrent: # if its a recurrent layer
        recurrent_output_errors = initial_gradient[:]
        
        error_C = initial_gradient[-1]
        error_D = 0
        total_error = error_C + error_D
        
        derivative = Key.ACTIVATION_DERIVATIVE[self.layers[-1].activation](weighted_sums[-1])
        
        error_Wa = derivative * total_error * activations[-1][0]
        error_Wb = derivative * total_error * activations[-1][1]
        error_B  = derivative * total_error
        error_a  = derivative * total_error * self.layers[-1].carry_weight
        
        output_layer_errors = [error_Wa, error_Wb, error_B, error_a]
        
    errors[-1] = output_layer_errors

    for this_layer_index in reversed(range(len(self.layers)-1)):
      # FRONT | next layer --> this layer --> previous layer | BACK

      this_layer = self.layers[this_layer_index]
      this_activations = activations[this_layer_index]

      this_WS = weighted_sums[this_layer_index]

      prev_errors = errors[this_layer_index + 1]
      prev_layer = self.layers[this_layer_index + 1]

      layer_errors = []

      if type(this_layer) in (Dense, AFS, Localunit):

        for this_neuron_index, _ in enumerate(this_layer.neurons):

          neuron_error = []

          derivative = Key.ACTIVATION_DERIVATIVE[this_layer.activation](this_WS[this_neuron_index])

          if type(prev_layer) == Dense:

            for prev_neuron_index, prev_neuron in enumerate(prev_layer.neurons):
              neuron_error.append(prev_errors[prev_neuron_index] * prev_neuron['weights'][this_neuron_index])

            layer_errors.append( derivative * sum(neuron_error) )

          elif type(prev_layer) == AFS:

            layer_errors.append( derivative * prev_errors[this_neuron_index] * prev_layer.neurons[this_neuron_index]['weight'] )

          elif type(prev_layer) == Operation:

            layer_errors.append( derivative * prev_errors[this_neuron_index] )

          elif type(prev_layer) == Localunit:

            for prev_neuron_index, prev_neuron in enumerate(prev_layer.neurons):

              if index_corrector(this_neuron_index-(prev_neuron_index)) != None:
                neuron_error.append(prev_errors[prev_neuron_index] * prev_neuron['weights'][this_neuron_index-prev_neuron_index])

            layer_errors.append( derivative * sum(neuron_error) )

      elif type(this_layer) == Convolution:

        kernel = this_layer.kernel[:]

        layer_errors = arraytools.generate_array(value=0, *this_layer.input_shape)

        for a in range(len(layer_errors) - (len(this_layer.kernel) - 1)):
          for b in range(len(layer_errors[0]) - (len(this_layer.kernel[0]) - 1)):

            # Apply the flipped kernel to the corresponding region in layer_errors
            for kernel_row in range(len(kernel)):
              for kernel_col in range(len(kernel[0])):
                
                # Calculate the corresponding position in layer_errors
                output_row = a + kernel_row
                output_col = b + kernel_col

                # calculate the derivative
                #derivative = Key.ACTIVATION_DERIVATIVE[this_layer.activation](this_WS[a][b])

                # Check if the calculated position is within the bounds of the input
                # if 0 <= output_row < len(layer_errors) and 0 <= output_col < len(layer_errors[0]):
                #   layer_errors[output_row][output_col] += prev_errors[a][b] * kernel[kernel_row][kernel_col]
                
                # Accumulate the weighted error
                layer_errors[output_row][output_col] += prev_errors[a][b] * kernel[kernel_row][kernel_col] #* derivative

          #visual.numerical_display(prev_WS, title='next WS')
          #visual.numerical_display(layer_errors, title='layer errors')

      elif type(this_layer) == Flatten:

        for this_neuron_index in range(len(this_layer.neurons)):

          neuron_error = []

          if type(prev_layer) == Dense:
            for prev_neuron_index, prev_neuron in enumerate(prev_layer.neurons):

              neuron_error.append(prev_errors[prev_neuron_index] * prev_neuron['weights'][this_neuron_index])

          elif type(prev_layer) == AFS:
            for prev_neuron_index, prev_neuron in enumerate(prev_layer.neurons):

              neuron_error.append(prev_errors[prev_neuron_index] * prev_neuron['weight'])

          elif type(prev_layer) == Operation:
            for prev_neuron_index, prev_neuron in enumerate(prev_layer.neurons):

              neuron_error.append(prev_errors[prev_neuron_index])

          elif type(prev_layer) == Localunit:
            for prev_neuron_index, prev_neuron in enumerate(prev_layer.neurons):

              if index_corrector(this_neuron_index-(prev_neuron_index)) != None:
                neuron_error.append(prev_errors[prev_neuron_index] * prev_neuron['weights'][this_neuron_index-prev_neuron_index])

          layer_errors.append( sum(neuron_error) )

        sizex, sizey = this_layer.input_shape

        layer_errors = arraytools.reshape(layer_errors[:], sizex, sizey)

      elif type(this_layer) == Reshape:

        sizex, sizey = this_layer.input_shape

        layer_errors = arraytools.reshape(prev_errors[:], sizex, sizey)

      elif type(this_layer) == Maxpooling:
        thisx, thisy = this_layer.input_size

        layer_errors = arraytools.generate_array(thisx, thisy, value=0)

        skibidi = -1

        # iterate over all the layers
        for a in range(0, len(weighted_sums[this_layer_index-1]), this_layer.stride):

          skibidi += 1
          toilet = -1

          # # iterate over all the elements in the layer
          for b in range(0, len(weighted_sums[this_layer_index-1][a]), this_layer.stride):
            toilet += 1

            # # control the vertical
            # for c in range(this_layer.size):

            #   # control the horizontal
            #   for d in range(this_layer.size):

            #     if a+c < len(weighted_sums[this_layer_index-1]) and b+d < len(weighted_sums[this_layer_index-1][a]):

            #       # if the current element is the maximum value, send it all the error gradient

            #       if (weighted_sums[this_layer_index-1][a+c][b+d] == this_WS[skibidi][toilet]):
            #         layer_errors[a+c][b+d] += prev_errors[skibidi][toilet]

            # ==========

            max_value = this_WS[skibidi][toilet]  # Get the maximum value
            count = 0  # Count of elements with the maximum value

            # Find all elements with the maximum value in the pooling window
            for c in range(this_layer.size):
              for d in range(this_layer.size):
                if a + c < len(weighted_sums[this_layer_index - 1]) and b + d < len(weighted_sums[this_layer_index - 1][a]):
                  if weighted_sums[this_layer_index - 1][a + c][b + d] == max_value:
                    count += 1

            # Distribute the gradient equally among the maximum elements
            for c in range(this_layer.size):
              for d in range(this_layer.size):
                if a + c < len(weighted_sums[this_layer_index - 1]) and b + d < len(weighted_sums[this_layer_index - 1][a]):
                  if weighted_sums[this_layer_index - 1][a + c][b + d] == max_value:
                    layer_errors[a + c][b + d] += prev_errors[skibidi][toilet] / count  # Divide by count

      elif type(this_layer) == Meanpooling:

        thisx, thisy = this_layer.input_size

        layer_errors = arraytools.generate_array(thisx, thisy, value=0)

        # for meanpooling, we need to distribute the error over the kernel size
        # this is done by dividing the error by the kernel size (averaging it over the kernel size)

        skibidi = -1

        # iterate over all the layers
        for a in range(0, len(weighted_sums[this_layer_index-1]), this_layer.stride):

          skibidi += 1
          toilet = -1

          # iterate over all the elements in the layer
          for b in range(0, len(weighted_sums[this_layer_index-1][a]), this_layer.stride):
            toilet += 1

            # control the vertical
            for c in range(this_layer.size):

              # control the horizontal
              for d in range(this_layer.size):

                if a+c < len(weighted_sums[this_layer_index-1]) and b+d < len(weighted_sums[this_layer_index-1][a]):

                  # distribute the error over the kernel size

                  layer_errors[a+c][b+d] += prev_errors[skibidi][toilet] / (this_layer.size**2)

      elif type(this_layer) == Operation:

        if this_layer.operation == 'min max scaler':
          derivative_list = [Key.SCALER_DERIVATIVE[this_layer.operation](a, min=this_layer.minimum, max=this_layer.maximum) for a in this_WS]

        elif this_layer.operation == 'standard scaler':
          derivative_list = [Key.SCALER_DERIVATIVE[this_layer.operation](a, std=this_layer.std) for a in this_WS]

        elif this_layer.operation == 'max abs scaler':
          derivative_list = [Key.SCALER_DERIVATIVE[this_layer.operation](a, max=this_layer.maximum) for a in this_WS]

        elif this_layer.operation == 'robust scaler':
          derivative_list = [Key.SCALER_DERIVATIVE[this_layer.operation](a, q1=this_layer.q1, q3=this_layer.q3) for a in this_WS]

        elif this_layer.operation == 'softmax':
          derivative_list = Key.SCALER_DERIVATIVE[this_layer.operation](this_activations)

        for this_neuron_index in range(len(this_layer.neurons)):

          neuron_error = []

          if this_layer.operation in Key.ACTIVATION_DERIVATIVE:
            derivative = Key.ACTIVATION_DERIVATIVE[this_layer.operation](this_WS[this_neuron_index])
          else:
            derivative = derivative_list[this_neuron_index]

          if type(prev_layer) == Dense:

            for prev_neuron_index, prev_neuron in enumerate(prev_layer.neurons):
              neuron_error.append(prev_errors[prev_neuron_index] * prev_neuron['weights'][this_neuron_index])

            layer_errors.append( derivative * sum(neuron_error) )

          elif type(prev_layer) == AFS:

            layer_errors.append( derivative * prev_errors[this_neuron_index] * prev_layer.neurons[this_neuron_index]['weight'] )

          elif type(prev_layer) == Operation:
            layer_errors.append( derivative * prev_errors[this_neuron_index] )

          elif type(prev_layer) == Localunit:

            for prev_neuron_index, prev_neuron in enumerate(prev_layer.neurons):

              if index_corrector(this_neuron_index-(prev_neuron_index)) != None:
                neuron_error.append(prev_errors[prev_neuron_index] * prev_neuron['weights'][this_neuron_index-prev_neuron_index])

            layer_errors.append( derivative * sum(neuron_error) )

          elif type(prev_layer) == Reshape:

            layer_errors.append( prev_errors[this_neuron_index] )

      elif type(this_layer) == Recurrent:
        
        if type(prev_layer) != Recurrent:
          raise SystemError(f'Recurrent layer must be preceded by a Recurrent layer and not {type(prev_layer)}')
        
        error_C = recurrent_output_errors[this_layer_index]
        error_D = prev_errors[3]
        total_error = error_C + error_D
        
        derivative = Key.ACTIVATION_DERIVATIVE[this_layer.activation](this_WS)
        
        error_Wa = derivative * total_error * this_activations[0]
        error_Wb = derivative * total_error * this_activations[1]
        error_B  = derivative * total_error
        error_a  = derivative * total_error * this_layer.carry_weight
        
        layer_errors = [error_Wa, error_Wb, error_B, error_a]
        
      errors[this_layer_index] = layer_errors[:]

    return errors

  # post-processing

  def evaluate(self, features, targets, **kwargs) -> None:
    """
    Evaluate
    -----
      Validates the model based on the given validation data and prints out the results. this assumes an already compiled model.
    -----
    Args
    -----
    - features (list)  : the features to use
    - targets  (list)  : the corresponding targets to use

    - (optional) stats    (bool)  : show training stats
    """
    self.is_validated = True

    stats = kwargs.get('stats', False)

    for metric in self.metrics:

      correct = 0

      for i in (utility.progress_bar(range(len(features)), f"Evaluating with {metric}", "Complete", decimals=2, length=50, empty=' ') if not stats else range(len(features))):

        predicted = self.push(features[i])
        actual = targets[i]

        correct += int(scaler.argmax(predicted) == scaler.argmax(actual))

        # if scaler.argmax(predicted) != scaler.argmax(actual):
        #   visual.image_display(features[i])
        #   print(f"actual: {actual}")
        #   print(f"predicted: {predicted}")
        #   print(f"actual: {scaler.argmax(actual)}")
        #   print(f"predicted: {scaler.argmax(predicted)}")

      print(f"accuracy: {100*(correct/len(features))}%")

    self.validation_loss = 100

  def push(self, x):
    """
    Args
    -----
    x (list) : the input to the model
    """

    if self.RNN:

      carry = 0      
      input_index = 0
      answer = []
      
      for layer in self.layers:

        if layer.accept_input:
          output, carry = layer.apply(x[input_index], carry)
          input_index += 1
          
        else:
          output, carry = layer.apply(0, carry)

        if layer.return_output:
          answer.append(output)
      
      return answer
      
    else:
    
      for layer in self.layers:

        if type(layer) == Convolution:
          x = layer.apply(x)

        elif type(layer) == Dense:
          x = layer.apply(x)

        elif type(layer) == Maxpooling:
          x = layer.apply(x)

        elif type(layer) == Meanpooling:
          x = layer.apply(x)

        elif type(layer) == Flatten:
          x = layer.apply(x)

        elif type(layer) == Reshape:
          x = layer.apply(x)

        elif type(layer) == AFS:
          x = layer.apply(x)

        elif type(layer) == Operation:
          x = layer.apply(x)

        elif type(layer) == Localunit:
          x = layer.apply(x)

        else:
          raise TypeError("Unknown layer type")

      return x

  def peek(self) -> None:
    """
    Peek
    -----
      returns all the weights and biases of all the layers
    """

    for layer in range(len(self.layers)):
      print(f"layer {layer+1}")

      if type(self.layers[layer]) == Convolution:
        visual.numerical_display(self.layers[layer].kernel, title=f'Kernel bias: {self.layers[layer].bias}')
      elif type(self.layers[layer]) in (Dense, AFS, Localunit):
        for index, neuron in enumerate(self.layers[layer].neurons):
          print("  │")
          print(f"  ├─ Neuron  {index+1}")
          print(f"  │  Weights {neuron['weights']}")
          print(f"  │  Bias    {neuron['bias']}")
        print("  │")
        print(f"  └─ Activation Function: {self.layers[layer].activation}")
      else:
        print("  └─ No weights to peek")
      print()

  def summary(self, **kwargs) -> None:
    """
    Summary
    """

    counter = 0
    traniable_params = 0
    non_traniable_params = 0

    print("| Model Summary:")
    print("|")
    print(f"|   Status    : {'Uncompiled' if self.optimizer is None or self.loss is None else 'Compiled':25} | Training   : {'Untrained!' if self.is_trained==False else 'Trained':15}")
    print(f"|   Optimizer : {str(self.optimizer):25} | Validation : {'Validated' if self.is_validated else 'Unvalidated'}")
    print(f"|   Loss      : {str(self.loss):25} | Final Loss : {self.error_logs[-1]:15} ")
    print(f"|   Metrics   : {', '.join(self.metrics)}")
    print("|")
    print("| Model Overview:")
    print("|")
    print(f"|   Layer No.  | Layer Type  | Learnable | Input Shape | Output Shape | Layer Name")
    print(f"|              |             |           |             |              |")

    for layer in self.layers:
      counter += 1

      if type(layer) in (Dense, AFS, Localunit):
        traniable_params += len(layer.neurons) * len(layer.neurons[0]['weights']) + 1 if layer.learnable else 0
        non_traniable_params += len(layer.neurons) * len(layer.neurons[0]['weights']) + 1 if not layer.learnable else 0

      elif type(layer) == Convolution:
        traniable_params += len(layer.kernel) * len(layer.kernel[0]) + 1 if layer.learnable else 0
        non_traniable_params += len(layer.kernel) * len(layer.kernel[0]) + 1 if not layer.learnable else 0

      if type(layer) in (Dense, Convolution, AFS, Localunit):
        learnable = True if layer.learnable else False
        print(f"|   {counter:10} | {layer.__class__.__name__:11} | {str(learnable):10}| {str(self.sizes[counter-1]):11} | {str(self.sizes[counter]):12} | {layer.name:20}")

      else:
        print(f"|   {counter:10} | {layer.__class__.__name__:11} |           | {str(self.sizes[counter-1]):11} | {str(self.sizes[counter]):12} | {layer.name:20}")

    print("|")
    print("| Parameter Breakdown:")
    print("|")
    print(f"|   Trainable Parameters     : {traniable_params}")
    print(f"|   Non-Trainable Parameters : {non_traniable_params}")
    print(f"|   Total Parameters         : {traniable_params + non_traniable_params}")
    print()

#######################################################################################################
#                                      Sequential Model Classes                                       #
#######################################################################################################

# neural layers

class Convolution:
  def __init__(self, kernel, activation, **kwargs):
    """
    Convolution
    -----
      Convolution layer with valid padding, accepts and returns 2D arrays.
    -----
    Args
    -----
    - kernel                (2D Array)      : the kernel to apply
    - activation            (string)        : the activation function
    - (Optional) bias       (bool)          : weither to use bias, default is False
    - (Optional) learnable  (boolean)       : whether or not the kernel is learnable, defaults to True
    - (Optional) name       (string)        : the name of the layer
    -----
    Activation functions
    - ReLU
    - Softplus
    - Mish
    - Swish
    - Leaky ReLU
    - ELU
    - GELU
    - SELU
    - ReEU
    - None
    - TANDIP

    Normalization functions
    - Binary Step
    - Softsign
    - Sigmoid
    - Tanh
    """
    self.kernel = kernel
    self.activation = activation.lower()
    self.learnable = kwargs.get('learnable', True)
    self.use_bias = kwargs.get('bias', False)
    self.bias = kwargs.get('bias_value', 0)
    self.name = kwargs.get('name', 'convolution')
    self.input_shape = 0

  def apply(self, input):
    answer = []
    
    self.input_shape = (len(input[0]), len(input))
    # iterate over all layers
    for a in range( len(input) - (len(self.kernel) - 1) ):

      layer_output = []

      # iterate over all the elements in the layer
      for b in range( len(input[a]) - (len(self.kernel[0]) - 1) ):
        dot_product = 0

        # iterate over the kernel layers
        for c in range(len(self.kernel)):

          # iterate over the kernel elements
          for d in range(len(self.kernel[c])):

            # apply the kernel to the input
            dot_product += self.kernel[c][d] * input[a+c][b+d]

        layer_output.append(Key.ACTIVATION[self.activation](dot_product) + self.bias)

      answer.append(layer_output[:])
    return answer

  def get_weighted_sum(self, input):
    answer = []
    # iterate over all layers
    for a in range( len(input) - (len(self.kernel) - 1) ):

      layer_output = []

      # iterate over all the elements in the layer
      for b in range( len(input[a]) - (len(self.kernel[0]) - 1) ):
        dot_product = 0

        # iterate over the kernel layers
        for c in range(len(self.kernel)):

          # iterate over the kernel elements
          for d in range(len(self.kernel[c])):

            # apply the kernel to the input
            dot_product += self.kernel[c][d] * input[a+c][b+d]

        layer_output.append(dot_product + self.bias)

      answer.append(layer_output[:])
    return answer

class Dense:
  def __init__(self, neurons, activation, **kwargs):
    """
    Dense (fully connected layer)
    -----
      Fully connected perceptron layer, previously reffered as 'feedforward layer', accepts and returns 1D arrays.
    -----
    Args
    -----
    - neurons                   (int)     : the number of neurons in the layer
    - activation                (string)  : the activation function
    - (Optional) initialization (string)  : intialization of the weights, defaults to Glorot uniform
    - (Optional) learnable      (boolean) : whether or not the layer is learnable, defaults to True
    - (Optional) name           (string)  : the name of the layer
    -----
    Activation functions
    - ReLU
    - Softplus
    - Mish
    - Swish
    - Leaky ReLU
    - ELU
    - GELU
    - SELU
    - ReEU
    - None
    - TANDIP

    Normalization functions
    - Binary Step
    - Softsign
    - Sigmoid
    - Tanh
    
    Initialization
    - Xavier uniform in
    - Xavier uniform out
    - He uniform
    - Glorot uniform
    - LeCun uniform
    - He normal
    - Glorot normal
    - LeCun normal
    - Default
    - None
    """
    self.output_shape = neurons
    self.activation = activation.lower()
    self.input_shape = kwargs.get('input_shape', 0)
    self.initialization = kwargs.get('initialization', 'glorot uniform')

    neuron = {
      'weights': [0],
      'bias': 0
      }

    self.neurons = [neuron for _ in range(neurons)]

    self.learnable = kwargs.get('learnable', True)
    self.name = kwargs.get('name', 'dense')

  def reshape_input_shape(self, input_shape, output_shape):
    
    self.neurons = [
      {
      'weights': [Key.INITIALIZATION[self.initialization](input_shape, output_shape) for _ in range(input_shape)],
      'bias': Key.INITIALIZATION[self.initialization](input_shape, output_shape)
      }
      for _ in range(self.output_shape)
    ]

  def apply(self, input: list):
    self.input = input
    answer = []

    if type(input) != list:
      raise TypeError("input must be a 1D array list")
    if type(input[0]) not in (int, float):
      raise TypeError("input must be a 1D array list \nuse the built-in 'Flatten' layer before a neural network layer")

    # iterate over all the neurons
    for _neuron in self.neurons:
      dot_product = sum([input[i] * _neuron['weights'][i] for i in range(len(input))])

      answer.append(Key.ACTIVATION[self.activation](dot_product + _neuron['bias']))
    return answer

  def get_weighted_sum(self, input: list):
    self.input = input[:]
    answer = []

    if type(input) != list:
      raise TypeError("input must be a 1D array list")
    if type(input[0]) not in (int, float):
      raise TypeError("input must be a 1D array list \nuse the built-in 'Flatten' layer before a neural network layer")

    # iterate over all the neurons
    for _neuron in self.neurons:
      dot_product = sum([input[i] * _neuron['weights'][i] for i in range(len(input))])

      answer.append(dot_product + _neuron['bias'])
    return answer

class Localunit:
  def __init__(self, receptive_field, activation, **kwargs):
    """
    Local Unit
    -----
      Locally connected perceptron layer, accepts and returns 1D arrays.
      a constituent layer within a LCN (Locally connected networks), also known as a 'LCN' (Locally connected neurons).
    -----
    Args
    -----
    - receptive_field         (int)     : the receptive field of the layer
    - activation              (string)  : the activation function
    - (Optional) learnable    (boolean) : whether or not the layer is learnable, defaults to True
    - (Optional) name         (string)  : the name of the layer
    -----
    Activation functions
    - ReLU
    - Softplus
    - Mish
    - Swish
    - Leaky ReLU
    - ELU
    - GELU
    - SELU
    - ReEU
    - None
    - TANDIP

    Normalization functions
    - Binary Step
    - Softsign
    - Sigmoid
    - Tanh
    
    Initialization
    - Xavier uniform in
    - Xavier uniform out
    - He uniform
    - Glorot uniform
    - LeCun uniform
    - He normal
    - Glorot normal
    - LeCun normal
    - Default
    - None
    """
    self.receptive_field = receptive_field
    self.activation = activation
    self.name = kwargs.get('name', 'local unit')
    self.learnable = kwargs.get('learnable', True)
    self.initialization = kwargs.get('initialization', 'glorot uniform')

    self.input_shape = kwargs.get('input_shape', 0)
    self.output_shape = 0 # will get calibrated soon

    neuron = {
      'weights': [0],
      'bias': 0
      }

    self.neurons = [neuron] # will be calibrated soon

    self.learnable = kwargs.get('learnable', True)
    self.name = kwargs.get('name', 'unnamed_network')

  def reshape_input_shape(self, input_shape, next_shape):

    self.output_shape = input_shape - (self.receptive_field - 1)
    
    self.neurons = [
      {
      'weights': [Key.INITIALIZATION[self.initialization](input_shape, next_shape) for _ in range(self.receptive_field)],
      'bias': Key.INITIALIZATION[self.initialization](input_shape, next_shape)
      }
      for _ in range(self.output_shape)
    ]

  def apply(self, input: list):
    self.input = input
    answer = []

    if type(input) != list:
      raise TypeError("input must be a 1D array list")
    if type(input[0]) not in (int, float):
      raise TypeError("input must be a 1D array list \nuse the built-in 'Flatten' layer before a neural network layer")

    # iterate over all the neurons
    for a in range((len(self.neurons) - self.receptive_field) + 1):
      dot_product = 0

      # iterate over the input
      for b in range(self.receptive_field):

        dot_product += input[a + b] * self.neurons[a]['weights'][b]

      answer.append(Key.ACTIVATION[self.activation](dot_product + self.neurons[a]['bias']))
    return answer

  def get_weighted_sum(self, input: list):
    self.input = input
    answer = []

    if type(input) != list:
      raise TypeError("input must be a 1D array list")
    if type(input[0]) not in (int, float):
      raise TypeError("input must be a 1D array list \nuse the built-in 'Flatten' layer before a neural network layer")

    # iterate over all the neurons
    for a in range((len(self.neurons) - self.receptive_field) + 1):
      dot_product = 0

      # iterate over the input
      for b in range(self.receptive_field):

        dot_product += input[a + b] * self.neurons[a]['weights'][b]

      answer.append(dot_product + self.neurons[a]['bias'])
    return answer

class AFS:
  def __init__(self, activation, **kwargs):
    """
    Adaptive Feature Scaler
    -----
      Experimental layer, use a 'None' activation function for a traditional AFS layer, else its no longer
      a scaler layer anymore.
    -----
    Args
    -----
    - activation           (string)  : the activation function to use for the attention layer
    - (Optional) bias      (boolean) : whether or not to use bias
    - (Optional) learnable (boolean) : whether or not to learn
    - (Optional) name      (string)  : the name of the layer
    -----
      (Activation functions)
      - ReLU
      - Softplus
      - Mish
      - Swish
      - Leaky ReLU
      - ELU
      - GELU
      - SELU
      - ReEU
      - None
      - TANDIP

      (Normalization functions)
      - Binary Step
      - Softsign
      - Sigmoid
      - Tanh
      
      Initialization
      - Xavier uniform in
      - Xavier uniform out
      - He uniform
      - Glorot uniform
      - LeCun uniform
      - He normal
      - Glorot normal
      - LeCun normal
      - Default
      - None
    """
    self.activation = activation.lower()
    self.name = kwargs.get('name', 'feature scaler')
    self.learnable = kwargs.get('learnable', True)
    self.use_bias = kwargs.get('bias', True)
    self.initialization = kwargs.get('initialization', 'default')

    neuron = {
      'weight': 0,
      'bias': 0
      }
    self.neurons = [neuron]

  def reshape_input_shape(self, input_shape, output_shape):

    self.neurons = [
      {
      'weight': Key.INITIALIZATION[self.initialization](input_shape, output_shape),
      'bias': Key.INITIALIZATION[self.initialization](input_shape, output_shape)
      }
      for _ in range(input_shape)
    ]

  def apply(self, input):

    return [Key.ACTIVATION[self.activation](self.neurons[i]['weight'] * input[i] + self.neurons[i]['bias']) for i in range(len(input))]

  def get_weighted_sum(self, input: list):
    self.input = input[:]
    answer = []

    if type(input) != list:
      raise TypeError("input must be a 1D array list")
    if type(input[0]) not in (int, float):
      raise TypeError("input must be a 1D array list \nuse the built-in 'Flatten' layer before an AFS layer")

    return [self.neurons[i]['weight'] * input[i] + self.neurons[i]['bias'] for i in range(len(input))]

class Recurrent:
  def __init__(self, activation, **kwargs):
    """
    Recurrent Unit
    -----
      Primary block within RNNs (Recurrent Neural Networks). it is only compatible with other Recurrent Unit.
    -----
    Args
    -----
    - activation           (string)  : the activation function to use for the attention layer
    - (Optional) input     (boolean) : accept an input during propagation, on by default
    - (Optional) output    (boolean) : return anything during propagation, on by default
    - (Optional) learnable (boolean) : whether or not to learn, on by default
    - (Optional) name      (string)  : the name of the layer
    """
    self.activation = activation
    
    self.accept_input = kwargs.get('input', True)
    self.return_output = kwargs.get('output', True)
    
    self.name = kwargs.get('name', 'recurrent')
    self.learnable = kwargs.get('learnable', True)
    
    self.input_weight = random.uniform(0.1, 1)
    self.carry_weight = random.uniform(0.1, 1)
    self.bias = random.uniform(-0.5, 0.5)
  
  def apply(self, input, carry):
    
    return [Key.ACTIVATION[self.activation]((input * self.input_weight) + (carry * self.carry_weight) + self.bias)] * 2

  def get_weighted_sum(self, input, carry):
    
    return [(input * self.input_weight) + (carry * self.carry_weight) + self.bias] * 2

class LTSM: # unimplimented
  ...
  
class GRU: # unimplimented
  ...

# Functional layers

class Maxpooling:
  def __init__(self, size, stride, **kwargs):
    """
    Meanpooling
    -----
      Scales down any 2D array by pooling, accepts and returns 2D arrays.
    -----
    Args
    -----
    - size            (int)  : the size of the pooling window
    - stride          (int)  : the stride of the pooling window
    - (Optional) name (string) : the name of the layer
    """
    self.size = size
    self.stride = stride
    self.name = kwargs.get('name', 'maxpooling')
    self.input_size = 0

  def apply(self, input):
    answer = []
    self.input_size = arraytools.shape(input)

    # iterate over all the layers
    for a in range(0, len(input), self.stride):

      layer_output = []

      # iterate over all the elements in the layer
      for b in range(0, len(input[a]), self.stride):
        pool = []

        # control the vertical
        for c in range(self.size):

          # control the horizontal
          for d in range(self.size):

            if a+c < len(input) and b+d < len(input[a]):
              pool.append(input[a+c][b+d])

        layer_output.append( max(pool) )

      answer.append(layer_output[:])
    return answer

class Meanpooling:
  def __init__(self, size, stride, **kwargs):
    """
    Meanpooling
    -----
      Scales down any 2D array by pooling, accepts and returns 2D arrays.
    -----
    Args
    -----
    - size            (int)  : the size of the pooling window
    - stride          (int)  : the stride of the pooling window
    - (optional) name (string) : the name of the layer
    """
    self.size = size
    self.stride = stride
    self.name = kwargs.get('name', 'meanpooling')
    self.input_size = 0

  def apply(self, input):
    answer = []
    self.input_size = arraytools.shape(input)

    # iterate over all the layers
    for a in range(0, len(input), self.stride):

      layer_output = []

      # iterate over all the elements in the layer
      for b in range(0, len(input[a]), self.stride):
        pool = []

        # control the vertical
        for c in range(self.size):

          # control the horizontal
          for d in range(self.size):

            if a+c < len(input) and b+d < len(input[a]):
              pool.append(input[a+c][b+d])

        layer_output.append( sum(pool) / len(pool) )

      answer.append(layer_output[:])
    return answer

class Flatten:
  def __init__(self, **kwargs):
    """
    Flatten
    -----
      Flattens any 2D array into a 1D array, use this layer before a neural network layer (Dense layer)
    -----
    Args
    -----
    - (optional) name (string): the name of the layer
    """
    self.name = kwargs.get('name', 'flatten')
    self.neurons = [0]
    self.input_shape = 0

  def apply(self, input):
    self.input_shape = arraytools.shape(input)
    return arraytools.flatten(input)

  def set_length(self, length):
    self.neurons = [0 for _ in range(length)]

class Reshape:
  def __init__(self, width, height, **kwargs):
    """
    Reshape
    -----
      Reshapes an array, accepts and returns 2D arrays.
    -----
    Args
    -----
    - height (int)  : the height of the output
    - width (int)   : the width of the output
    - (Optional) name (string) : the name of the layer
    -----
    negative values means that the shape will be inferred, essencially counting backwards just
    like how the python range function works
    """
    self.height = height
    self.width = width
    self.name = kwargs.get('name', 'reshape')
    self.input_shape = 0

  def apply(self, input):
    self.input_shape = arraytools.shape(input)
    return arraytools.reshape(input, self.width, self.height)

class Operation:
  def __init__(self, operation, **kwargs):
    """
    Operation
    -----
      Operational layer for functions or normalizations.
    -----
    Args
    -----
    - operation    (string) : the scaler to use
    - (Optional) name   (string) : the name of the layer
    -----
    Available Operations:

      (Scalers)
      - standard scaler
      - min max scaler
      - max abs scaler
      - robust scaler

      (Activation functions)
      - ReLU
      - Softplus
      - Mish
      - Swish
      - Leaky ReLU
      - ELU
      - GELU
      - SELU
      - ReEU
      - None
      - TANDIP

      (Normalization functions)
      - Binary Step
      - Softsign
      - Sigmoid
      - Tanh
      - Softmax
    """
    self.operation = operation.lower()
    self.name = kwargs.get('name', 'operation')

    self.minimum = 0
    self.maximum = 0
    self.q1 = 0
    self.q3 = 0
    self.std = 0

    self.neurons = [0]

  def reshape_input_shape(self, input_shape, output_shape):

    self.neurons = [0 for _ in range(input_shape)]

  def apply(self, input):

    if self.operation == 'dropout':
      # Placeholder for dropout mechanism
      pass

    # activation functions
    if self.operation in Key.ACTIVATION:
      return [Key.ACTIVATION[self.operation](a) for a in input]

    # if its a minmax scaler
    if self.operation == 'min max scaler':

      self.minimum = min(input)
      self.maximum = max(input)

      return scaler.min_max_scaler(input, min=self.minimum, max=self.maximum)

    # defaults to a scaler
    if self.operation in Key.SCALER:

      self.q1 = float(np.percentile(input, 25))
      self.q3 = float(np.percentile(input, 75))
      self.std = float(np.std(input))

      return Key.SCALER[self.operation](input)

  def get_weighted_sum(self, x):
    return x



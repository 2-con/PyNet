"""
NetFlash API
=====
  A high-level API for sequential models of networks. Unlike NetCore, NetFlash is intended to be built for speed and efficiency
  by using JAX and other libraries to speed up the training process. This API is still under development.
-----
Provides
-----

  N/A

"""
#######################################################################################################
#                                    File Information and Handling                                    #
#######################################################################################################

__version__ = "1.0.0"

__package__ = "pynet"

if __name__ == "__main__":
  print("""
        This file is not meant to be run as a main file.
        More information can be found about PyNet's NetFlash API on the documentation.
        system > 'docs.txt' or the GitHub repository at https://github.com/2-con/PyNet
        """)
  exit()

#######################################################################################################
#                                               Imports                                               #
#######################################################################################################

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random

import jax
import jax.numpy as jnp

from tools import utility
from core.flash import loss, optimizer, metric, derivative
from core.flash.layers import *
from core.flash.callback import Callback
from core.vanilla.utility import do_nothing
from system.config import *

#######################################################################################################
#                                               Extra                                                 #
#######################################################################################################

""" Notes:

jax.grad and jax.value_and_grad dosn't work for some reason, it keeps breaking for no reason at all.
chatgpt says its a new bug and that i should be reported. whatever the cause, those are off the table.

yes, i've tried to fix it for hours and it still doesn't work.
"""

#######################################################################################################
#                                         Internal Functions                                          #
#######################################################################################################

class Key:

  OPTIMIZER = {
    "adam": optimizer.Adam,
    "rmsprop": optimizer.RMSprop,
    "adagrad": optimizer.Adagrad,
    "amsgrad": optimizer.Amsgrad,
    "adadelta": optimizer.Adadelta,
    "gradclip": optimizer.Gradclip,
    "adamax": optimizer.Adamax,
    "sgnd": optimizer.SGND,
    "default": optimizer.Default,
    "rprop": optimizer.Rprop,
    "momentum": optimizer.Momentum,
    "novograd": optimizer.Novograd,
  }
  
  OPTIMIZER_INITIALIZER = {
    # ps: parameter shape
    # pd: parameter dtype
    
    "default": lambda ps, pd: (), # No state needed for SGD
    "gradclip": lambda ps, pd: (), # No state needed
    "sgnd": lambda ps, pd: (), # No state needed
    
    "amsgrad": lambda ps, pd: (
        jnp.zeros(ps, dtype=pd),  # m
        jnp.zeros(ps, dtype=pd),  # v
        jnp.zeros(ps, dtype=pd)   # v_hat_max
    ),
    "momentum": lambda ps, pd: (
        jnp.zeros(ps, dtype=pd),  # velocity
    ),
    "rmsprop": lambda ps, pd: (
        jnp.zeros(ps, dtype=pd),  # avg_sq_grad
    ),
    "adagrad": lambda ps, pd: (
        jnp.zeros(ps, dtype=pd),  # sum_sq_grad
    ),
    "novograd": lambda ps, pd: (
        jnp.zeros(ps, dtype=pd),  # m
        jnp.zeros(ps, dtype=pd)   # v
    ),
    "adam": lambda ps, pd: (
        jnp.zeros(ps, dtype=pd),  # m
        jnp.zeros(ps, dtype=pd)   # v
    ),
    "adadelta": lambda ps, pd: (
        jnp.zeros(ps, dtype=pd),  # avg_sq_grad
        jnp.zeros(ps, dtype=pd)   # avg_sq_delta
    ),
    "adamax": lambda ps, pd: (
        jnp.zeros(ps, dtype=pd),  # m
        jnp.zeros(ps, dtype=pd)   # u_inf (max of past gradients)
    ),
    "rprop": lambda ps, pd: (
        jnp.zeros(ps, dtype=pd),              # prev_grad
        jnp.full(ps, 0.01, dtype=pd)  # step_size (often initialized to a small constant)
    ),
  }

  METRICS = {

    # classification metrics
    "accuracy": metric.Accuracy,
    "precision": metric.Precision,
    "recall": metric.Recall,
    "f1 score": metric.F1_score,
    "roc auc": metric.ROC_AUC,
    "r2 score": metric.R2_Score,
    
    #regression
    "mean squared error": loss.Mean_squared_error,
    "Root mean squared error": loss.Root_mean_squared_error,
    "mean absolute error": loss.Mean_absolute_error,
    "total absolute error": loss.Total_absolute_error,
    "total squared error": loss.Total_squared_error,
    "l1 loss": loss.L1_loss,
    
    # classification
    "categorical crossentropy": loss.Categorical_crossentropy,
    "sparse categorical crossentropy": loss.Sparse_categorical_crossentropy,
    "binary crossentropy": loss.Binary_crossentropy,

  }
  
  LOSS = {

    #regression
    "mean squared error": loss.Mean_squared_error,
    "Root mean squared error": loss.Root_mean_squared_error,
    "mean absolute error": loss.Mean_absolute_error,
    "total absolute error": loss.Total_absolute_error,
    "total squared error": loss.Total_squared_error,
    "l1 loss": loss.L1_loss,
    
    # classification
    "categorical crossentropy": loss.Categorical_crossentropy,
    "sparse categorical crossentropy": loss.Sparse_categorical_crossentropy,
    "binary crossentropy": loss.Binary_crossentropy,
  }
  
  LOSS_DERIVATIVE = {
    "mean squared error": derivative.Mean_squared_error_derivative,
    "Root mean squared error": derivative.Root_mean_squared_error_derivative,
    "mean absolute error": derivative.Mean_absolute_error_derivative,
    "total absolute error": derivative.Total_absolute_error_derivative,
    "total squared error": derivative.Total_squared_error_derivative,
    "l1 loss": derivative.L1_loss_derivative,
    
    # classification
    "categorical crossentropy": derivative.Categorical_crossentropy_derivative,
    "sparse categorical crossentropy": derivative.Sparse_categorical_crossentropy_derivative,
    "binary crossentropy": derivative.Binary_crossentropy_derivative,
  }
  
#######################################################################################################
#                                          Sequential Model                                           #
#######################################################################################################

class Sequential:
  # pre-processing

  def __init__(self, *args):
    """
    Sequential
    ======
      Sequential model where layers are processed sequentially.

      Must contain NetFlash layers to be added to the model. either directly through the constructor or through the add() method
    -----

    Available layers:
    - Convolution
    - Maxpooling
    - Meanpooling
    - Flatten
    - Reshape
    - Dense
    - Operation
    - Localunit

    Recurrent layers:
    - Recurrent
    - LSTM
    - GRU

    Parallelization layers:
    - Parallel
    - Merge

    Refer to the documentation for more information.
    """
    self.layers = list(args) if args is not None else []

    self.error_logs = []
    self.validation_error_logs = []
    self.metric_logs = []

  def add(self, layer):
    """
    Add
    -----
      Adds a specified layer to the network.
    -----
    Args
    -----
    - layer (NetFlash object) : the layer to add to the model
    """

    self.layers.append(layer)

  def compile(self, input_shape:tuple[int, ...], optimizer:str, loss:str, learning_rate:float, epochs:int, metrics:list, validation_split:float=0, batch_size:int=1, verbose:int=1, logging:int=1, *args, **kwargs):
    """
    Compile
    -----
      Compiles the model to be ready for training, but unlike NetCore, the input shape needs to be defined here
      so the model can be compiled faster. Custom callables can be passed as a parameter as long as they are compatible with JAX (JNP-based functions).
    -----
    Args
    -----
    - input_shape   (tuple[int, ...]) : shape of the input data, include channels for image data and features for tabular data.
    - loss          (str)             : loss function to use
    - learning_rate (float)           : learning rate to use
    - epochs        (int)             : number of epochs to train for
    - metrics       (list)            : metrics to use
    - (Optional) batch_size    (int)  : batch size to use
    - (Optional) verbose       (int)  : verbosity level
    - (Optional) logging       (int)  : logging level
    - (Optional) callbacks     (list) : logging level
    """
    self.input_shape = input_shape
    self.learning_rate = learning_rate
    self.epochs = epochs + 1
    self.batchsize = batch_size
    self.verbose = verbose
    self.logging = logging
    
    self.error_logs = []
    self.validation_error_logs = []
    
    ############################################################################################
    #                        Initialize model parameters and connections                       #
    ############################################################################################
    
    self.params_pytree = {} # Initialize the main parameters dictionary here
    sizes = []
    
    # set sizes while ignoring fans
    for layer_index, layer in enumerate(self.layers):
      if layer_index == 0:
        _, layer_size = layer.calibrate(input_shape,1) if type(layer) in (Dense, '') else layer.calibrate(input_shape,(1,))
      else:
        _, layer_size = layer.calibrate(sizes[layer_index-1],1) if type(layer) in (Dense, '') else layer.calibrate(sizes[layer_index-1],(1,))
      
      sizes.append(layer_size)
    
    for layer_index, layer in enumerate(self.layers):
      # Pass the individual key to the layer's init_params method
      
      if len(sizes) == 1:
        layer_params, _ = layer.calibrate(input_shape, sizes[layer_index])
      
      elif layer_index == 0: # If this is the first layer, use the input shape
        layer_params, _ = layer.calibrate(input_shape, sizes[layer_index+1])
      
      elif layer_index == len(self.layers) - 1: # If this is the last layer, use the output shape of the previous layer
        layer_params, _ = layer.calibrate(sizes[layer_index-1], sizes[layer_index])
      
      else:
        layer_params, _ = layer.calibrate(sizes[layer_index-1], sizes[layer_index+1])
      
      if layer_params: # if layer_params is not empty
        
        if type(layer_params) is not dict:
          raise TypeError(
            f"Layer {layer_index} ({type(layer).__name__})'s init_params method returned parameters of type "
            f"{type(layer_params)}. Expected a dictionary of parameters for learnable layers."
          )
        
        self.params_pytree[f'layer_{layer_index}'] = layer_params # Store layer's params under a unique key

    ############################################################################################
    #                                  Initialize optimizer                                    #
    ############################################################################################
    
    if optimizer not in Key.OPTIMIZER:
      raise ValueError(f"Optimizer '{optimizer}' not supported.")
    
    self.optimizer = Key.OPTIMIZER[optimizer]
    self.optimizer_hyperparams = {} # Store all remaining kwargs as optimizer hyperparams

    self.opt_state = jax.tree_util.tree_map(
      lambda p: Key.OPTIMIZER_INITIALIZER[optimizer](p.shape, p.dtype),
      self.params_pytree
    )
    
    ############################################################################################
    #                                 Initialize loss function                                 #
    ############################################################################################
    
    if loss not in Key.LOSS:
      raise ValueError(f"Loss function '{loss}' not supported.")
    
    self.loss_derivative = jax.jit(Key.LOSS_DERIVATIVE[loss.lower()])
    self.loss_function = jax.jit(Key.LOSS[loss.lower()])
    self.loss_derivative = jax.jit(Key.LOSS_DERIVATIVE[loss.lower()])
    
    ############################################################################################
    #                                 Initialize metrics                                       #
    ############################################################################################
    
    self.metrics = []
    
    for metric in metrics:
      
      if type(metric) == str:
        
        if metric not in Key.METRICS:
          raise ValueError(f"Metric '{metric}' not supported. Available: {list(Key.METRICS.keys())}")
        
        self.metrics.append(Key.METRICS[metric])
          
      else: # if its a func
        
        self.metrics.append(metric)   
        
    self.metrics = tuple(self.metrics)   

    self.is_compiled = True
    
    ############################################################################################
    #                          Initialize Valdation (if applicable)                            #
    ############################################################################################
    
    if 0 <= validation_split < 1:
      self.validation_split = validation_split
    else:
      raise ValueError("validation_split must be between [0,1)")
    
    ############################################################################################
    #                                        Callbacks                                         #
    ############################################################################################
    
    self.callback = kwargs.get('callback', Callback)
    
  def fit(self, features:jnp.ndarray, targets:jnp.ndarray):
    """
    Fit
    -----
      Trains the model on the given data. Ideally, the data given should be of JNP format, but any mismatching
      data types will be converted to JNP arrays.
    -----
    Args
    -----
    - features (JNP array) : the features to use
    - targets  (JNP array) : the corresponding targets to use
    """
    #############################################################################################
    #                                  Error pre-checks                                         #
    #############################################################################################
    
    if not self.is_compiled:
      raise RuntimeError("Model is not compiled. Call .compile() first.")

    if not isinstance(features, jnp.ndarray) or not isinstance(targets, jnp.ndarray):
      raise TypeError("features and targets must be NumPy arrays.")
    if features.shape[0] == 0 or targets.shape[0] == 0:
      raise ValueError("features or targets must not be empty.")
    if features.shape[0] != targets.shape[0]:
      raise ValueError("features and targets must have the same number of samples.")
    if features.ndim == 1:
      features = features[:, None]
    if targets.ndim == 1:
      targets = targets[:, None]

    print() if self.verbose >= 1 else do_nothing()
    
    #############################################################################################
    #                                        Functions                                          #
    #############################################################################################

    @jax.jit
    def propagate(features:jnp.ndarray, parameters:dict) -> tuple[jnp.ndarray, jnp.ndarray]:
      
      activations   = [features]
      weighted_sums = []
      
      x = features
      for i, layer in enumerate(self.layers):
        
        layer_params = parameters.get(f'layer_{i}', {})
        x, weighted_sum = layer.apply(layer_params, x)
        
        activations.append(x)
        weighted_sums.append(weighted_sum)
        
      return {
        'activations': activations,
        'weighted_sums': weighted_sums
      }
    
    def step(
      layers_tuple:tuple, backward_func_tuple:tuple,
      error:jnp.ndarray, parameters_pytree:dict, opt_state:dict, activations:jnp.ndarray, weighted_sums:jnp.ndarray, 
      timestep:int, optimizer_hyperparams:dict, learning_rate) -> tuple[dict, dict]:
      
      for layer_index in reversed(range(len(layers_tuple))):
        layer = self.layers[layer_index]
        
        layer_params = parameters_pytree.get(f'layer_{layer_index}', {})
        
        error, gradients = backward_func_tuple[layer_index](layer_params, activations[layer_index], error, weighted_sums[layer_index])
        
        if type(layer) in (Flatten, MaxPooling, MeanPooling, Operation, Dropout, Reshape):
          # this is just to skip layers that don't have parameters
          # their backward methods will still be called above to propagate the error correctly
          
          continue
        
        if type(layer) in (Recurrent, LSTM, GRU, Attention):
          new_layer_params = {}
          new_opt_state = {}

          for cell_key, cell_params in layer_params.items():
            cell_grads = gradients[cell_key]
            new_cell_params = {}
            new_cell_opt_state = {}

            for param_name, param_value in cell_params.items():
              updated_param, updated_opt_state = self.optimizer(
                learning_rate,
                param_value,
                cell_grads[param_name],
                opt_state[f'layer_{layer_index}'][cell_key][param_name],
                timestep=timestep,
                **optimizer_hyperparams
              )
              new_cell_params[param_name] = updated_param
              new_cell_opt_state[param_name] = updated_opt_state

            new_layer_params[cell_key] = new_cell_params
            new_opt_state[cell_key] = new_cell_opt_state

          parameters_pytree[f'layer_{layer_index}'] = new_layer_params
          opt_state[f'layer_{layer_index}'] = new_opt_state
          
        elif type(layer) in (Dense, Convolution, Deconvolution, Localunit):
          layer_params_weights, opt_state[f'layer_{layer_index}']['weights'] = self.optimizer(
            learning_rate,
            layer_params['weights'],
            gradients['weights'],
            opt_state[f'layer_{layer_index}']['weights'],
            timestep=timestep,
            **optimizer_hyperparams
          )
          
          layer_params_biases, opt_state[f'layer_{layer_index}']['biases'] = self.optimizer(
            learning_rate,
            layer_params['biases'],
            gradients['biases'],
            opt_state[f'layer_{layer_index}']['biases'],
            timestep=timestep,
            **optimizer_hyperparams
          )
        
          parameters_pytree[f'layer_{layer_index}'] = {
            'weights': layer_params_weights,
            'biases': layer_params_biases
          }
        
        else:
          raise NotImplementedError(f"Backpropagation and update systems are not implemented for layer of type \"{type(layer).__name__}\"")
        
      return parameters_pytree, opt_state
    
    #############################################################################################
    #                                        Variables                                          #
    #############################################################################################
    
    self.is_trained = True

    features = jnp.asarray(features[0:int(len(features)*(1-self.validation_split))])
    targets = jnp.asarray(targets[0:int(len(targets)*(1-self.validation_split))])
    
    validation_features = jnp.asarray(features[int(len(features)*self.validation_split):]) if self.validation_split > 0 else jnp.asarray([])
    validation_targets = jnp.asarray(targets[int(len(targets)*self.validation_split):]) if self.validation_split > 0 else jnp.asarray([])

    current_params = self.params_pytree
    current_opt_state = self.opt_state
    timestep = 0
    
    update_step = jax.jit(step, static_argnums=(0,1)) # static argnums dosent work for JIT wrappers apparently...
    backward_func_tuple = tuple(layer.backward for layer in self.layers)
    learning_rate = jnp.float32(self.learning_rate)
    
    # self.loss_derivative = lambda y_true, y_pred: y_true - y_pred
    
    callback = self.callback()
    callback.initialization(**locals())
    
    #############################################################################################
    #                                           Main                                            #
    #############################################################################################
    
    for epoch in (utility.progress_bar(range(self.epochs), "> Training", "Complete", decimals=2, length=100, empty=' ') if self.verbose == 1 else range(self.epochs)):

      callback.before_epoch(**locals())
      epoch_loss = 0.0

      for base_index in range(0, len(features), self.batchsize):
        key = jax.random.PRNGKey(random.randint(0, 2**32))  

        num_samples = features.shape[0]
        shuffled_indices = jax.random.permutation(key, num_samples)

        randomized_features = features[shuffled_indices]
        randomized_targets = targets[shuffled_indices]

        batch_features = randomized_features[base_index : base_index + self.batchsize]
        batch_targets = randomized_targets[base_index : base_index + self.batchsize]
        
        activations_and_weighted_sums = propagate(batch_features, self.params_pytree)

        epoch_loss += self.loss_function(batch_targets, activations_and_weighted_sums['activations'][-1])
        initial_error = self.loss_derivative(batch_targets, activations_and_weighted_sums['activations'][-1])

        callback.before_update(**locals())
        
        timestep += 1
        current_params, current_opt_state = update_step(
          tuple(self.layers),
          backward_func_tuple,
          initial_error,  
          self.params_pytree,
          self.opt_state,
          activations_and_weighted_sums['activations'],
          activations_and_weighted_sums['weighted_sums'],
          timestep,
          self.optimizer_hyperparams,
          learning_rate
        )

        callback.after_update(**locals())
        self.params_pytree = current_params
        self.opt_state = current_opt_state
      
      validation_activations_and_weighted_sums = propagate(validation_features, self.params_pytree) if len(validation_features) > 0 else do_nothing()
      validation_loss = self.loss_function(validation_targets, validation_activations_and_weighted_sums['activations'][-1]) if len(validation_features) > 0 else do_nothing()
      
      epoch_loss /= len(features)
      validation_loss /= len(features)
      
      self.error_logs.append(epoch_loss)
      self.validation_error_logs.append(validation_loss) if len(validation_features) > 0 else do_nothing()
      
      ############ post training
      
      callback.after_epoch(**locals())

      if epoch % self.logging == 0 and self.verbose >= 2:
        
        prefix              = f"Epoch {epoch+1 if epoch == 0 else epoch}/{self.epochs-1} ({round( ((epoch+1)/self.epochs)*100 , 2)}%)"
        pad                 = ' ' * ( len(f"Epoch {self.epochs}/{self.epochs-1} (100.0%)") - len(prefix))
        suffix              = f" ┃ Loss: {str(epoch_loss):22}"
        rate                = f" ┃ ROC: {str(epoch_loss - self.error_logs[epoch-self.logging] if epoch >= self.logging else 0):23}"
        
        if len(validation_features) > 0:
          validation_suffix = f" ┃ Validation: {str(validation_loss):22}"
          validation_rate   = f" ┃ VROC: {validation_loss - self.validation_error_logs[epoch-self.logging] if epoch >= self.logging else 0}"
        
        else:
          validation_suffix = f" ┃ Validation: {str('Unavailable'):12}"
          validation_rate   = f" ┃ VROC: Unavailable"
        
        if self.verbose == 3:
          print(prefix + pad + suffix)
        elif self.verbose == 4:
          print(prefix + pad + suffix + rate)
        elif self.verbose == 5:
          print(prefix + pad + suffix + rate + validation_suffix)
        elif self.verbose == 6:
          print(prefix + pad + suffix + rate + validation_suffix + validation_rate)

    self.params_pytree = current_params
    
    callback.end(**locals())

  def evaluate(self, features, targets, **kwargs) -> None:
    raise NotImplementedError("The evaluate method is not implemented in NetFlash. Please use the push method to get predictions.")

  def push(self, inputs:jnp.ndarray) -> jnp.ndarray:
    """
    Propagates the input through the entire model, excluding dropout layers (if any).
    weights will not be updated.
    """
    x = inputs
    for i, layer in enumerate(self.layers):
      
      if type(layer) is Dropout:
        continue
      
      layer_params = self.params_pytree.get(f'layer_{i}', {})
      x, weighted_sum = layer.apply(layer_params, x)
      
    return x


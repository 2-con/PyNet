"""
NetFlash API
=====
  A high-level API for sequential models of networks. Unlike NetCore, NetFlash is intended to be built for speed and efficiency
  by using JAX and other libraries to speed up the training process. This API is still under development.
-----
Provides
-----
  (Learnable layers)
  1. Convolution
  2. Deconvolution
  3. Dense
  4. Localunit
  5. Multiheaded self-attention

  (Utility layers)
  1. Maxpooling
  2. Meanpooling
  3. Flatten
  4. Reshape
  5. Operation (normalization and activation functions)
  
  (Recurrent units)
  1. Recurrent
  2. LSTM
  3. GRU
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
from core.flash import losses, metrics, optimizers
from core.flash.layers import *
from core.flash.callbacks import Callback
from core.static.utility import do_nothing
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
    "adam": optimizers.Adam,
    "rmsprop": optimizers.RMSprop,
    "adagrad": optimizers.Adagrad,
    "amsgrad": optimizers.AMSgrad,
    "adadelta": optimizers.Adadelta,
    "gradclip": optimizers.Gradclip,
    "adamax": optimizers.Adamax,
    "sgnd": optimizers.SGND,
    "default": optimizers.Default,
    "rprop": optimizers.Rprop,
    "momentum": optimizers.Momentum,
    "novograd": optimizers.Novograd,
  }

  METRICS = {

    # classification metrics
    "accuracy": metrics.Accuracy,
    "precision": metrics.Precision,
    "recall": metrics.Recall,
    "f1 score": metrics.F1_score,
    "roc auc": metrics.ROC_AUC,
    "r2 score": metrics.R2_Score,
    
    #regression
    "mean squared error": losses.Mean_squared_error().forward,
    "Root mean squared error": losses.Root_mean_squared_error().forward,
    "mean absolute error": losses.Mean_absolute_error().forward,
    "total absolute error": losses.Total_absolute_error().forward,
    "total squared error": losses.Total_squared_error().forward,
    "l1 loss": losses.L1_loss().forward,
    
    # classification
    "categorical crossentropy": losses.Categorical_crossentropy().forward,
    "sparse categorical crossentropy": losses.Sparse_categorical_crossentropy().forward,
    "binary crossentropy": losses.Binary_crossentropy().forward,

  }
  
  LOSS = {

    #regression
    "mean squared error": losses.Mean_squared_error,
    "Root mean squared error": losses.Root_mean_squared_error,
    "mean absolute error": losses.Mean_absolute_error,
    "total absolute error": losses.Total_absolute_error,
    "total squared error": losses.Total_squared_error,
    "l1 loss": losses.L1_loss,
    
    # classification
    "categorical crossentropy": losses.Categorical_crossentropy,
    "sparse categorical crossentropy": losses.Sparse_categorical_crossentropy,
    "binary crossentropy": losses.Binary_crossentropy,
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

  def compile(self, input_shape:tuple[int, ...], optimizer:str, loss:str, learning_rate:float, epochs:int, metrics:list=[], validation_split:float=0, batch_size:int=1, verbose:int=1, logging:int=1, *args, **kwargs):
    """
    Compile
    -----
      Compiles the model to be ready for training, but unlike NetCore, the input shape needs to be defined here
      so the model can be compiled faster. Custom callables can be passed as a parameter as long as they are compatible with JAX (JNP-based functions).
    -----
    Args
    -----
    - input_shape                 (tuple[int, ...])              : shape of the input data, include channels for image data and features for tabular data.
    - loss                        (str)                          : loss function to use
    - learning_rate               (float)                        : learning rate to use
    - epochs                      (int)                          : number of epochs to train for
    - (Optional) metrics          (list)                         : metrics to use
    - (Optional) batch_size       (int)                          : batch size to use
    - (Optional) verbose          (int)                          : verbosity level
    - (Optional) logging          (int)                          : how ofter to report if the verbosity is at least 3
    - (Optional) early_stopping   (bool)                         : whether or not to use early stopping, Evaluates based on the validation set. Defaults to False
    - (Optional) patience         (int)                          : how many epochs to wait before early stopping, defaults to 5
    - (Optional) callbacks        (core.flash.callback instance) : call a custom callback class during training with access to all local variables, read more in the documentation.
    - (Optional) validation_split (float)                        : fraction of the data to use for validation, must be between [0, 1). Default is 0 (no validation).
    - (Optional) regularization   (tuple[str, float])            : type of regularization to use, position 0 is the type ("L1" or "L2"), position 1 is the lambda value. Default is None (no regularization).
    
    Verbosity Levels
    -----
    - 0 : None
    - 1 : Progress bar of the whole training process
    - 2 : (Numerical output) Loss
    - 3 : (Numerical output) Loss and V Loss (Validation Loss)
    - 4 : (Numerical output) Loss, V Loss (Validation Loss) and the 1st metric in the 'metrics' list
    
    Optimizers
    -----
    - Amsgrad
    - Default
    - Gradclip
    - SGND
    - Momentum
    - RMSprop
    - Adagrad
    - Novograd
    - ADAM
    - Adadelta
    - Adamax
    - Rprop

    Losses
    -----
    - Mean Squared Error
    - Root Mean Squared Error
    - Mean Abseloute Error
    - Total Squared Error
    - Total Abseloute Error
    - L1 Loss
    - Categorical Crossentropy
    - Sparse Categorical Crossentropy
    - Binary Cross Entropy

    Metrics
    -----
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - ROC AUC
    - R2 Score
    """
    self.input_shape = input_shape
    self.learning_rate = jnp.float32(learning_rate)
    self.epochs = epochs
    self.batchsize = batch_size
    self.verbose = verbose
    self.logging = logging
    
    self.error_logs = []
    self.validation_error_logs = []
    self.metrics_logs = []
    
    ############################################################################################
    #                        Initialize model parameters and connections                       #
    ############################################################################################
    
    self.params_pytree = {} # Initialize the main parameters dictionary here
    sizes = []
    
    # set sizes while ignoring fans
    for layer_index, layer in enumerate(self.layers):
      if layer_index == 0:
        _, layer_size = layer.calibrate(fan_in_shape=input_shape, fan_out_shape=1) if type(layer) in (Dense, '') else layer.calibrate(fan_in_shape=input_shape, fan_out_shape=(1,))
      else:
        _, layer_size = layer.calibrate(fan_in_shape=sizes[layer_index-1], fan_out_shape=1) if type(layer) in (Dense, '') else layer.calibrate(fan_in_shape=sizes[layer_index-1], fan_out_shape=(1,))
      
      sizes.append(layer_size)
    
    for layer_index, layer in enumerate(self.layers):
      # Pass the individual key to the layer's init_params method
      
      if len(sizes) == 1:
        layer_params, _ = layer.calibrate(fan_in_shape=input_shape, fan_out_shape=sizes[layer_index])
      
      elif layer_index == 0: # If this is the first layer, use the input shape
        layer_params, _ = layer.calibrate(fan_in_shape=input_shape, fan_out_shape=sizes[layer_index+1])
      
      elif layer_index == len(self.layers) - 1: # If this is the last layer, use the output shape of the previous layer
        layer_params, _ = layer.calibrate(fan_in_shape=sizes[layer_index-1], fan_out_shape=sizes[layer_index])
      
      else:
        layer_params, _ = layer.calibrate(fan_in_shape=sizes[layer_index-1], fan_out_shape=sizes[layer_index+1])
      
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
    
    if optimizer.lower() not in Key.OPTIMIZER:
      raise ValueError(f"Optimizer '{optimizer}' not supported.")
    
    self.optimizer_fn = Key.OPTIMIZER[optimizer.lower()].update
    self.optimizer_hyperparams = {} # Store all remaining kwargs as optimizer hyperparams

    self.opt_state = jax.tree_util.tree_map(
      lambda p: Key.OPTIMIZER[optimizer.lower()].initialize(p.shape, p.dtype),
      self.params_pytree
    )
     
    ############################################################################################
    #                                 Initialize loss function                                 #
    ############################################################################################
    
    if loss not in Key.LOSS:
      raise ValueError(f"Loss function '{loss}' not supported.")
    
    self.loss = Key.LOSS[loss.lower()]
    
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
    
    #############################################################################################
    #                                     Regularization                                        #
    #############################################################################################
    
    self.regularization = kwargs.get('regularization', [None, 0])
    if self.regularization[0] not in (None, "L1", "L2"):
      raise ValueError("regularization type must be either None, 'L1' or 'L2'")
    if type(self.regularization[1]) not in (int, float) or self.regularization[1] < 0:
      raise ValueError("regularization lambda must be a non-negative number")
    
    #############################################################################################
    #                                     early stopping                                        #
    #############################################################################################
    
    self.enable_early_stopping = kwargs.get("early_stopping", False)
    self.patience = kwargs.get("patience", 5)
    
    if self.validation_split == 0 and self.enable_early_stopping:
      raise SystemError("Validation split cannot be 0 when early stopping is enabled.")
    
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
    if len(features) < self.batchsize:
      raise ValueError("batchsize cannot be larger than the number of samples.")
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
        x, weighted_sum = layer.forward(layer_params, x)
        
        activations.append(x)
        weighted_sums.append(weighted_sum)
        
      return {
        'activations': activations,
        'weighted_sums': weighted_sums
      }
    
    def step(
      layers_tuple:tuple,
      error:jnp.ndarray, parameters_pytree:dict, opt_state:dict, activations:jnp.ndarray, weighted_sums:jnp.ndarray, 
      timestep:int, optimizer_hyperparams:dict) -> tuple[dict, dict]:
      
      for layer_index in reversed(range(len(layers_tuple))):
        layer = self.layers[layer_index]
        
        layer_params = parameters_pytree.get(f'layer_{layer_index}', {})
        
        error, gradients = layer.backward(layer_params, activations[layer_index], error, weighted_sums[layer_index])
        
        gradients = losses.Loss_calculator.regularized_grad(layer_params, gradients, self.regularization[1], self.regularization[0], ignore_list=['bias', 'biases'])
        
        if hasattr(layer, "update"):
          parameters_pytree[f'layer_{layer_index}'], opt_state[f'layer_{layer_index}'] = layer.update(
            self.optimizer_fn,
            self.learning_rate,
            layer_params,
            gradients,
            opt_state[f'layer_{layer_index}'],
            timestep=timestep, 
            **optimizer_hyperparams
          )
          
        else:
          # this is just to skip layers that don't have parameters
          # their backward methods will still be called above to propagate the error correctly
          
          continue
        
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
    
    update_step = jax.jit(step, static_argnums=(0,)) # static argnums dosent work for JIT wrappers apparently...
    
    callback = self.callback()
    callback.initialization(**locals())
    
    patience_window = 0
    
    #############################################################################################
    #                                           Main                                            #
    #############################################################################################
    
    for epoch in (utility.progress_bar(range(self.epochs), "> Training", "Complete", decimals=2, length=50, empty=' ') if self.verbose == 1 else range(self.epochs)):

      callback.before_epoch(**locals())
      epoch_loss = 0.0

      for base_index in range(0, len(features), self.batchsize):
        if base_index + self.batchsize > len(features):
          continue
        
        key = jax.random.PRNGKey(random.randint(0, 2**32))
        
        shuffled_indices = jax.random.permutation(key, features.shape[0])

        batch_features = features[shuffled_indices][base_index : base_index + self.batchsize]
        batch_targets = targets[shuffled_indices][base_index : base_index + self.batchsize]
        
        # actual training method
        
        activations_and_weighted_sums = propagate(batch_features, self.params_pytree)
        epoch_loss += losses.Loss_calculator.forward_loss(batch_targets, activations_and_weighted_sums['activations'][-1], self.loss, self.regularization[1], self.regularization[0], self.params_pytree)
        initial_error = self.loss.backward(batch_targets, activations_and_weighted_sums['activations'][-1])

        callback.before_update(**locals())
        
        timestep += 1
        current_params, current_opt_state = update_step(
          tuple(self.layers),
          initial_error,  
          self.params_pytree,
          self.opt_state,
          activations_and_weighted_sums['activations'],
          activations_and_weighted_sums['weighted_sums'],
          timestep,
          self.optimizer_hyperparams,
        )

        callback.after_update(**locals())
        self.params_pytree = current_params
        self.opt_state = current_opt_state
      
      extra_activations_and_weighted_sums = propagate(validation_features, self.params_pytree) if len(validation_features) > 0 else do_nothing()
      validation_loss = self.loss_function(validation_targets, extra_activations_and_weighted_sums['activations'][-1]) if len(validation_features) > 0 else do_nothing()
      
      metric_stats = [metric_fn(validation_targets, extra_activations_and_weighted_sums['activations'][-1]) for metric_fn in self.metrics] if len(self.metrics) > 0 else do_nothing()
      self.metrics_logs.append(metric_stats)
      
      epoch_loss /= len(features)
      validation_loss /= len(features)
      
      if self.enable_early_stopping and epoch > self.patience:
        if validation_loss < self.validation_error_logs[-1]:
          patience_window = epoch + self.patience
      
        if epoch > patience_window:
          break
      
      self.error_logs.append(epoch_loss)
      self.validation_error_logs.append(validation_loss) if len(validation_features) > 0 else do_nothing()
      
      ############ post training
      
      callback.after_epoch(**locals())

      if (epoch % self.logging == 0 and self.verbose >= 2) or epoch == 0:
        
        lossROC       = 0 if epoch == 0 else epoch_loss      - self.error_logs[epoch-self.logging]
        validationROC = 0 if epoch < self.logging else validation_loss - self.validation_error_logs[epoch-self.logging] if self.validation_split > 0 else 0
        metricROC     = 0 if epoch < self.logging else metric_stats[0] - self.metrics_logs[epoch-self.logging][0] if len(self.metrics) > 0 else 0
        
        prefix = f"\033[1mEpoch {epoch}/{self.epochs}\033[0m ({round( ((epoch)/self.epochs)*100 , 2)}%)"
        prefix += ' ' * (25 + len(f"{self.epochs}") * 2 - len(prefix))
        
        print_loss = f"Loss: {epoch_loss:.2E}" if epoch_loss > 1000 or epoch_loss < 0.00001 else f"Loss: {epoch_loss:.5f}"
        print_loss = f"┃ \033[32m{print_loss:16}\033[0m" if lossROC < 0 else f"┃ \033[31m{print_loss:16}\033[0m" if lossROC > 0 else f"┃ {print_loss:16}"
        
        if self.verbose == 2:
          print(prefix + print_loss)
        
        elif self.verbose == 3:
          print_validation = f"V Loss: {validation_loss:.2E}" if validation_loss > 1000 or validation_loss < 0.00001 else f"V Loss: {validation_loss:.5f}" if self.validation_split > 0 else f"V Loss: N/A"
          print_validation = f"┃ \033[32m{print_validation:16}\033[0m" if validationROC < 0 else f"┃ \033[31m{print_validation:16}\033[0m" if validationROC > 0 else f"┃ {print_validation:16}"
          print(prefix + print_loss + print_validation)
        
        elif self.verbose == 4:
          print_metric = f"{self.metrics[0].__name__}: {metric_stats[0]:.5f}" if len(self.metrics) >= 1 else "Metrics N/A"
          print_metric = f"┃ \033[32m{print_metric:16}\033[0m" if metricROC > 0 else f"┃ \033[31m{print_metric:16}\033[0m" if metricROC < 0 else f"┃ {print_metric:16}"
          
          print_validation = f"V Loss: {validation_loss:.2E}" if validation_loss > 1000 or validation_loss < 0.00001 else f"V Loss: {validation_loss:.5f}" if self.validation_split > 0 else f"V Loss: N/A"
          print_validation = f"┃ \033[32m{print_validation:16}\033[0m" if validationROC < 0 else f"┃ \033[31m{print_validation:16}\033[0m" if validationROC > 0 else f"┃ {print_validation:16}"
          
          print(prefix + print_loss + print_validation + print_metric)
            
    self.params_pytree = current_params
    
    callback.end(**locals())

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
      x, _ = layer.apply(layer_params, x)
      
    return x


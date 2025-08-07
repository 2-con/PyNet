import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
import random

from system.config import *
import core.flash.activation as activation
import core.flash.initializer as initializer
import core.flash.derivative as derivative
import core.flash.scaler as scaler

class Key:

  ACTIVATION = {
    
    # normalization functions
    "sigmoid": activation.Sigmoid,
    "tanh": activation.Tanh,
    "binary step": activation.Binary_step,
    "softsign": activation.Softsign,
    "softmax": activation.Softmax,
    
    # rectifiers
    "relu": activation.ReLU,
    "softplus": activation.Softplus,
    "mish": activation.Mish,
    "swish": activation.Swish,
    "leaky relu": activation.Leaky_ReLU,
    "gelu": activation.GELU,
    "identity": activation.Linear,
    "reeu": activation.ReEU,
    "retanh": activation.ReTanh,
    
    # parametric functions
    'elu': activation.ELU,
    "selu": activation.SELU,
    "prelu": activation.PReLU,
    "silu": activation.SiLU,
  }
  
  ACTIVATION_DERIVATIVE = {
    
    # normalization functions
    "sigmoid": derivative.Sigmoid_derivative,
    "tanh": derivative.Tanh_derivative,
    "binary step": derivative.Binary_step_derivative,
    "softsign": derivative.Softsign_derivative,
    "softmax": derivative.Softmax_derivative,
    
    # rectifiers
    "relu": derivative.ReLU_derivative,
    "softplus": derivative.Softplus_derivative,
    "mish": derivative.Mish_derivative,
    "swish": derivative.Swish_derivative,
    "leaky relu": derivative.Leaky_ReLU_derivative,
    "gelu": derivative.GELU_derivative,
    "identity": derivative.Linear_derivative,
    "reeu": derivative.ReEU_derivative,
    "retanh": derivative.ReTanh_derivative,
    
    # parametric functions
    'elu': derivative.ELU_derivative,
    "selu": derivative.SELU_derivative,
    "prelu": derivative.PReLU_derivative,
    "silu": derivative.SiLU_derivative
  }
  
  SCALER = {
    "standard scaler": scaler.Standard_Scaler,
    "min max scaler": scaler.Min_Max_Scaler,
    "max abs scaler": scaler.Max_Abs_Scaler,
    "robust scaler": scaler.Robust_Scaler,
    "softmax": activation.Softmax
  }
  
  SCALER_DERIVATIVE = {
    "standard scaler": derivative.Standard_Scaler_Derivative,
    "min max scaler": derivative.Min_Max_Scaler_derivative,
    "max abs scaler": derivative.Max_Abs_Scaler_derivative,
    "robust scaler": derivative.Robust_Scaler_derivative,
    "softmax": derivative.Softmax_derivative
  }
  
  INITIALIZER = {
    "glorot uniform": initializer.Glorot_uniform,
    "glorot normal": initializer.Glorot_normal,
    "he uniform": initializer.He_uniform,
    "he normal": initializer.He_normal,
    "lecun uniform": initializer.Lecun_uniform,
    "lecun normal": initializer.Lecun_normal,
    "xavier uniform in": initializer.Xavier_uniform_in,
    "xavier uniform out": initializer.Xavier_uniform_out,
    "default": initializer.Default
  }
  
"""
JNP shapes are determined by (Y,X) or (Z,Y,X)
its the opposite of NetCore.

input format
  (batch_size, input_size) (Y,X)
  
output format
  (batch_size, output_size) (Y,X)
"""

class Dense:
  def __init__(self, neurons: int, activation, **kwargs):
    self.neuron_amount = neurons

    self.activation_name = activation.lower()
    
    # in case the user passed some doohickey panini for activation
    if self.activation_name not in Key.ACTIVATION:
      raise ValueError(f"Unknown activation: '{activation}'. Available: {list(Key.ACTIVATION.keys())}")
    
    self.activation_fn = Key.ACTIVATION[self.activation_name]
    self.activation_derivative_fn = Key.ACTIVATION_DERIVATIVE[self.activation_name]

    self.initializer_fn = Key.INITIALIZER['default']
    if self.activation_name in rectifiers:
      self.initializer_fn = Key.INITIALIZER['he normal']
    elif self.activation_name in normalization:
      self.initializer_fn = Key.INITIALIZER['glorot normal']

    if 'initializer' in kwargs:
      initializer_name = kwargs['initializer'].lower()
      if initializer_name not in Key.INITIALIZER:
        raise ValueError(f"Unknown initializer: '{initializer_name}'. Available: {list(Key.INITIALIZER.keys())}")
      self.initializer_fn = Key.INITIALIZER[initializer_name]

    self.input_size = None

  def calibrate(self, fan_in:tuple[int, ...], fan_out:int) -> tuple[dict, int]:

    self.input_size = fan_in[0]
    
    weights = self.initializer_fn((self.neuron_amount, fan_in[0]), fan_in[0], fan_out)
    biases = jnp.zeros((self.neuron_amount, 1))
    
    return {'weights': weights, 'biases': biases}, (self.neuron_amount,)

  def apply(self, params:dict, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    
    weighted_sums = params['weights'] @ inputs + params['biases']
    activated_output = self.activation_fn(weighted_sums)
    
    return activated_output, weighted_sums 
  
  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    
    # grads_z will have shape: (output_features, batch_size)
    grads_z = self.activation_derivative_fn(error, weighted_sums) # dE/dz

    # (output_features, input_features)
    grads_weights = jnp.einsum('ob,ib->oi', grads_z, inputs)

    grads_biases = jnp.sum(grads_z, axis=1, keepdims=True) 

    upstream_gradient = params['weights'].T @ grads_z

    param_grads = {
      'weights': grads_weights,
      'biases': grads_biases
    }

    return upstream_gradient, param_grads

class Flatten:
  def __init__(self, **kwargs):
    pass

  def calibrate(self, fan_in_shape: tuple[int, ...], batch_size: int) -> tuple[dict, int]:
    flattened_size = 1
    for dim in fan_in_shape:
      flattened_size *= dim
        
    self.input_shape = fan_in_shape
    
    # Flatten layer has no parameters
    return {}, (flattened_size,)

  def apply(self, params: dict, inputs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    # since all inputs are transposed, the input to this layer will be goofy if its composed of 2d images.
    
    flattened_output = inputs.T.reshape(inputs.shape[2], -1)
    
    # For a Flatten layer, we return the original inputs for the backward pass
    return flattened_output.T, inputs 

  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    #(self, params: dict, inputs_original_shape: jnp.ndarray, error: jnp.ndarray) -> tuple[jnp.ndarray, dict]:
  
    upstream_gradient = error.reshape(inputs.shape)

    param_grads = {} 

    return upstream_gradient, param_grads


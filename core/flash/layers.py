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
from core.vanilla.utility import do_nothing

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
  
=====
calibrate

returns:
  parameters, output_shape
"""

""" Todo:

(Learnable layers)
  1. Convolution
  2. Dense
  3. Localunit

  (Utility layers)
  1. Maxpooling
  2. Meanpooling
  3. Flatten
  4. Reshape
  5. Operation (normalization and activation functions)
  
  (Architectural layers)
  1. RecurrentBlock
  
  (Recurrent units)
  1. Recurrent
  2. LSTM
  3. GRU

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
      if kwargs['initializer'].lower() not in Key.INITIALIZER:
        raise ValueError(f"Unknown initializer: '{kwargs['initializer'].lower()}'. Available: {list(Key.INITIALIZER.keys())}")
      self.initializer_fn = Key.INITIALIZER[kwargs['initializer'].lower()]

    self.input_size = None

  def calibrate(self, fan_in:tuple[int, ...], fan_out_shape:int) -> tuple[dict, int]:

    self.input_size = fan_in[0]
    
    weights = self.initializer_fn((self.neuron_amount, fan_in[0]), fan_in[0], fan_out_shape)
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

class Convolution:
  def __init__(self, kernel:tuple[int, int], channels:int, activation:str, stride:tuple[int, int] = (1, 1), **kwargs):
    self.kernel = kernel
    self.channels = channels
    self.stride = stride
    self.activation_function = Key.ACTIVATION[activation]
    self.activation_derivative = Key.ACTIVATION_DERIVATIVE[activation]

    self.params = {} # only used for initialization
    self.input_shape = None # Will be set during calibration
    self.output_shape = None # Will be set during calibration
    
    if 'initializer' in kwargs and kwargs['initializer'].lower() not in Key.INITIALIZER:
      raise ValueError(f"Unknown initializer: '{kwargs['initializer'].lower()}'. Available: {list(Key.INITIALIZER.keys())}") 

  def calibrate(self, fan_in_shape: tuple[int, int, int], fan_out_shape:int) -> tuple[dict, tuple[int, ...]]:
    # The fan_in_shape here is (C_in, H, W).

    input_channels = fan_in_shape[0]

    # kernel shape = (output_channels, input_channels, kernel_height, kernel_width)
    weight_initializer = (Key.INITIALIZER['he normal'] if self.activation_function in rectifiers else Key.INITIALIZER['glorot normal']) if self.activation_function in normalization + rectifiers else Key.INITIALIZER['default']
    
    self.params['weights'] = weight_initializer((self.channels, input_channels, *self.kernel), fan_in_shape[-2] * fan_in_shape[-1], fan_out_shape)

    # bias shape = (output_channels,)
    self.params['biases'] = jnp.zeros(self.channels)
    
    # Calculate the output shape
    out_H = jnp.floor((fan_in_shape[-2] - self.kernel[0]) / self.stride[0]) + 1
    out_W = jnp.floor((fan_in_shape[-1] - self.kernel[1]) / self.stride[1]) + 1
    self.output_shape = (self.channels, out_H, out_W)
    
    return self.params, self.output_shape

  def apply(self, params: dict, inputs: jnp.ndarray) -> jnp.ndarray:
    
    # N: Batch size
    # C: Channels
    # H: image Height
    # W: image Width
    # O: Output channels
    # I: Input channels
    
    if len(inputs.T.shape) != 4:
      inputs = jnp.expand_dims(inputs.T, axis=1)
    
    # input  format NCHW (batch_size, channels, height, width)
    # kernel format OIHW (output_channels, input_channels, kernel_height, kernel_width)
    convolved = jax.lax.conv_general_dilated(
      lhs=inputs,# LHS = image
      rhs=params['weights'],# RHS = kernel
      window_strides=self.stride,
      padding='VALID',
      dimension_numbers=('NCHW', 'OIHW', 'NCHW') 
      # input,                            kernel,                                  output
      # (batch, channels, height, width), (out_channels, channels, height, width), (batch, channels, height, width)
    )
    
    # Add bias to each output channel
    bias = params['biases'][jnp.newaxis, :, jnp.newaxis, jnp.newaxis]
    output = convolved + bias

    return self.activation_function(output), output

  def backward(self, params: dict, inputs: jnp.ndarray, upstream_error: jnp.ndarray, weighted_sums: jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    # both jax.lax.conv_transpose and jax.lax.conv_general_dilated dosn't work for some reason.
    # so manual implementation is needed.
    # input  format NCHW (batch_size, channels, height, width)
    # kernel format OIHW (output_channels, input_channels, kernel_height, kernel_width)
    
    # derive the incoming errors (dE/dz)    
    d_WS = self.activation_derivative(upstream_error, weighted_sums)
    
    # derive the bias and sum it up per kernel
    grad_bias = jnp.sum(d_WS, axis=(0, 2, 3))

    # add a channel dimension to the input in case its 2d
    inputs = jnp.expand_dims(inputs.T, axis=1) if len(inputs.T.shape) != 4 else inputs.T
  
    # calculate the kernel gradient
    @jax.jit
    def correlate_2d_slices(input_slice, error_slice):
      return jax.scipy.signal.correlate(input_slice, error_slice, mode='valid')

    channel_correlations_per_batch = jax.vmap(correlate_2d_slices, in_axes=(1, None))
    all_correlations = jax.vmap(channel_correlations_per_batch, in_axes=(None, 1))
    grad_weights = jnp.sum(all_correlations(inputs, d_WS), axis=2)
    
    # calculate the upstream gradient
    N, C_out, H_out, W_out = d_WS.shape
      
    # dilate incoming error
    dilated_H = H_out + (H_out - 1) * (self.stride[0] - 1)
    dilated_W = W_out + (W_out - 1) * (self.stride[1] - 1)
    dilated_d_WS = jnp.zeros((N, C_out, dilated_H, dilated_W), dtype=d_WS.dtype)
    dilated_d_WS = dilated_d_WS.at[:, :, ::self.stride[0], ::self.stride[1]].set(d_WS)
    
    @jax.jit
    def convolve_image(error_slice, kernel_slice):
      return jax.scipy.signal.convolve(error_slice, kernel_slice, mode='full')
    
    # transposed convolve
    convolve_over_channels = jax.vmap(convolve_image, in_axes=(None, 1))
    all_convolutions = jax.vmap(convolve_over_channels, in_axes=(0, None))
    
    flipped_weights = jnp.flip(params['weights'], axis=(2, 3))
    upstream_gradient = jnp.sum(all_convolutions(dilated_d_WS, flipped_weights), axis=2)
    
    param_gradients = {
      'weights': grad_weights,
      'biases': grad_bias
    }

    return upstream_gradient, param_gradients

class Localunit:
  def __init__(self, receptive_field:int, activation:str, stride:int = 1, padding:str = 'VALID', **kwargs):
    self.receptive_field = receptive_field
    self.stride = stride
    self.padding = padding
    
    self.activation_name = activation.lower()
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
      if kwargs['initializer'].lower() not in Key.INITIALIZER:
        raise ValueError(f"Unknown initializer: '{kwargs['initializer'].lower()}'. Available: {list(Key.INITIALIZER.keys())}")
      self.initializer_fn = Key.INITIALIZER[kwargs['initializer'].lower()]

  def calibrate(self, fan_in_shape:tuple[int, ...], fan_out_shape:tuple[int, ...]) -> tuple[dict, tuple[int, ...]]:
    L_in = fan_in_shape[0]
    
    if self.padding.upper() == 'VALID':
        L_out = (L_in - self.receptive_field) // self.stride + 1
    else: # assuming 'SAME'
        L_out = (L_in + self.stride - 1) // self.stride

    # The weight shape is now (L_out, L_k), as there's no channel dimension.
    weights_shape = (L_out, self.receptive_field)
    self.output_shape = (L_out,)
    self.input_shape = fan_in_shape
    
    # fan_in is the size of the receptive field.
    weights = self.initializer_fn(weights_shape, self.receptive_field, 1)
    biases = jnp.zeros((L_out, 1))

    return {'weights': weights, 'biases': biases}, self.output_shape

  def apply(self, params:dict, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    # inputs are (N, L_in)
    weights = params['weights']
    N, L_in = inputs.shape
    L_out, L_k = weights.shape
    
    # Unfold the input tensor to get all receptive field patches.
    unfolded_inputs = jnp.lib.stride_tricks.as_strided(
      inputs,
      shape=(N, L_out, L_k),
      strides=(inputs.strides[0], self.stride * inputs.strides[1], inputs.strides[1])
    )
    
    # Perform the local matrix multiplication for weighted sums.
    weighted_sums = jnp.einsum('nLk, Lk->nL', unfolded_inputs, weights) + params['biases'].T
    activated_output = self.activation_fn(weighted_sums)

    return activated_output, weighted_sums

  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    weights = params['weights']
    N, L_in = inputs.shape
    L_out, L_k = weights.shape

    grads_z = self.activation_derivative_fn(error, weighted_sums)

    unfolded_inputs = jnp.lib.stride_tricks.as_strided(
      inputs,
      shape=(N, L_out, L_k),
      strides=(inputs.strides[0], self.stride * inputs.strides[1], inputs.strides[1])
    )
    
    # Calculate gradients for weights and biases.
    grads_weights = jnp.einsum('nL, nLk->Lk', grads_z, unfolded_inputs)
    grads_biases = jnp.sum(grads_z, axis=0, keepdims=True).T

    # Calculate the upstream gradient.
    upstream_gradient = jnp.lib.stride_tricks.as_strided(
      jnp.zeros_like(inputs),
      shape=(N, L_in, L_k),
      strides=(inputs.strides[0], inputs.strides[1], inputs.strides[1])
    )
    
    # Perform a manual transposed convolution-like operation with the error and flipped weights.
    flipped_weights = jnp.flip(weights, axis=1)
    for i in range(L_out):
      upstream_gradient = upstream_gradient.at[:, i * self.stride:i * self.stride + L_k, :].set(
        upstream_gradient[:, i * self.stride:i * self.stride + L_k, :] + grads_z[:, i:i+1].T * flipped_weights[i:i+1, :]
      )
    
    upstream_gradient = jnp.sum(upstream_gradient, axis=2)

    param_grads = {
      'weights': grads_weights,
      'biases': grads_biases
    }
    
    return upstream_gradient, param_grads






# functional layers



class MaxPooling:
  def __init__(self, pool_size:tuple[int, int] = (2, 2), strides:tuple[int, int] = (2, 2), **kwargs):
    self.pool_size = pool_size
    self.strides = strides

  def calibrate(self, fan_in_shape:tuple[int, ...], fan_out_shape:tuple[int,int]) -> tuple[dict, tuple[int, ...]]:
    # input shape = (C, H, W)
    C, H, W = fan_in_shape
    
    # Calculate the output dimensions after pooling
    pooled_H = (H - self.pool_size[0]) // self.strides[0] + 1
    pooled_W = (W - self.pool_size[1]) // self.strides[1] + 1
    
    self.input_shape = fan_in_shape
    
    return {}, (C, pooled_H, pooled_W)

  def apply(self, params:dict, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    # input format: (batch, channels, height, width)
    
    if len(inputs.T.shape) != 4:
      inputs = jnp.expand_dims(inputs.T, axis=1)
    
    # The forward pass uses reduce_window to get the maximum value in each pooling window.
    pooled_output = jax.lax.reduce_window(
      inputs,
      init_value=-jnp.inf,
      computation=jax.lax.max,
      window_dimensions=(1, 1, *self.pool_size),
      window_strides=(1, 1, *self.strides),
      padding='VALID'
    )

    # activations and WS
    return pooled_output, inputs

  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    N, C, H, W = inputs.shape
    _, _, pooled_H, pooled_W = error.shape
    
    @jax.jit
    def unpool_single_channel(input_slice, error_slice):
      input_window_shape = (H // self.strides[0], self.strides[0], W // self.strides[1], self.strides[1])
      input_reshaped = input_slice.reshape(input_window_shape)
      
      # rearrange dimensions to group windows
      input_windows = jnp.transpose(input_reshaped, (0, 2, 1, 3))
      input_windows = input_windows.reshape(-1, self.pool_size[0] * self.pool_size[1])

      # argmax to find max values for each window
      max_indices = jnp.argmax(input_windows, axis=1)

      # one-hot the indices
      one_hot_mask = jax.nn.one_hot(max_indices, self.pool_size[0] * self.pool_size[1])

      # reshape the gradients to match the encoded mask
      error_reshaped = error_slice.flatten()
      unpooled_gradient_flattened = one_hot_mask * jnp.expand_dims(error_reshaped, axis=1)
      
      # reshape unpooled gradient back to the original shape
      unpooled_gradient_reshaped = unpooled_gradient_flattened.reshape(H // self.strides[0], W // self.strides[1], self.strides[0], self.strides[1])
      unpooled_gradient_reshaped = jnp.transpose(unpooled_gradient_reshaped, (0, 2, 1, 3))

      return unpooled_gradient_reshaped.reshape(H, W)

    # Vectorize the operation over the batch and channels
    unpool_over_batch = jax.vmap(unpool_single_channel, in_axes=(0, 0))
    upstream_gradient = jax.vmap(unpool_over_batch, in_axes=(1, 1))(inputs, error)

    # vmap outputs (C, N, H, W) -> (N, C, H, W)
    upstream_gradient = jnp.transpose(upstream_gradient, (1, 0, 2, 3))

    return upstream_gradient, {}

class MeanPooling:
  def __init__(self, pool_size: tuple[int, int] = (2, 2), strides: tuple[int, int] = (2, 2), **kwargs):
    self.pool_size = pool_size
    self.strides = strides

  def calibrate(self, fan_in_shape: tuple[int, ...], fan_out_shape: tuple[int, int]) -> tuple[dict, tuple[int, ...]]:
    # input shape = (C, H, W)
    C, H, W = fan_in_shape
    
    # Calculate the output dimensions after pooling
    pooled_H = (H - self.pool_size[0]) // self.strides[0] + 1
    pooled_W = (W - self.pool_size[1]) // self.strides[1] + 1
    
    self.input_shape = fan_in_shape
    
    return {}, (C, pooled_H, pooled_W)

  def apply(self, params: dict, inputs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    # The apply method now performs a mean reduction instead of a max reduction.
    # It's computationally the same but uses a different primitive.
    
    if len(inputs.T.shape) != 4:
      inputs = jnp.expand_dims(inputs.T, axis=1)

    # The forward pass uses reduce_window to get the average value in each pooling window.
    pooled_output = jax.lax.reduce_window(
      inputs,
      init_value=0.,
      computation=jax.lax.add,
      window_dimensions=(1, 1, *self.pool_size),
      window_strides=(1, 1, *self.strides),
      padding='VALID'
    )
    
    # average the results
    pool_area = self.pool_size[0] * self.pool_size[1]
    pooled_output /= pool_area

    # activation and WS
    return pooled_output, inputs

  def backward(self, params: dict, inputs: jnp.ndarray, error: jnp.ndarray, weighted_sums: jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    N, C, H, W = inputs.shape
    _, _, pooled_H, pooled_W = error.shape
    pool_area = self.pool_size[0] * self.pool_size[1]

    @jax.jit
    def distribute_gradient(error_value):
      return jnp.full(self.pool_size, error_value / pool_area)

    distribute_spatial_gradients = jax.vmap(distribute_gradient)
    
    # reshapehe error tensor to (N * C, H_out * W_out)
    error_reshaped = error.reshape(N * C, pooled_H, pooled_W)

    # apply the function to the error.
    distributed_gradients = jax.vmap(distribute_spatial_gradients)(error_reshaped) # shape (N * C, H_out, W_out, H_k, W_k).
    
    final_shape = (N, C, H, W) # The new shape is (N, C, H, W).
    upstream_gradient = distributed_gradients.reshape(final_shape)

    return upstream_gradient, {}

class Flatten:
  def __init__(self, **kwargs):
    pass

  def calibrate(self, fan_in_shape: tuple[int, ...], fan_out_shape:int) -> tuple[dict, int]:
    flattened_size = 1
    for dim in fan_in_shape:
      flattened_size *= dim
        
    self.input_shape = fan_in_shape
    
    # Flatten layer has no parameters
    return {}, (int(flattened_size),)

  def apply(self, params: dict, inputs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    # since all inputs are transposed, the input to this layer will be goofy if its composed of 2d images.
    # input format: (batch, channels, height, width)
    
    flattened_output = inputs.reshape(inputs.shape[0], -1)
    
    # For a Flatten layer, we return the original inputs for the backward pass
    return flattened_output.T, inputs 

  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    #(self, params: dict, inputs_original_shape: jnp.ndarray, error: jnp.ndarray) -> tuple[jnp.ndarray, dict]:

    upstream_gradient = error.T.reshape(inputs.shape)

    return upstream_gradient, {}


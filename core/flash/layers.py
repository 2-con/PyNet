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

from tools.visual import dictionary_display as display

class Dense:
  def __init__(self, neurons:int, activation, **kwargs):
    """
    Dense
    -----
      A fully connected layer that connects the previous layer to the next layer. Accepts and returns 1D arrays (excludes batch dimension), so input_shape should be of the form
      (input_size,), anything after the 1st dimention will be ignored.
    -----
    Args
    -----
    - neurons     (int)     : the number of neurons in the layer
    - activation  (string)  : the activation function
    
    -----
    Activation functions
    - ReLU
    - Softplus
    - Mish
    - Swish
    - Leaky ReLU
    - GELU
    - ReEU
    - None
    - ReTanh

    Normalization functions
    - Softmax
    - Binary Step
    - Softsign
    - Sigmoid
    - Tanh
    
    Parametric functions
    - ELU
    - SELU
    - PReLU
    - SiLU
    
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
    self.neuron_amount = neurons
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

    self.input_size = None

  def calibrate(self, fan_in:tuple[int, ...], fan_out_shape:int):
    self.input_size = fan_in[0]
    weights = self.initializer_fn((self.input_size, self.neuron_amount), fan_in[0], fan_out_shape)
    biases = jnp.zeros((self.neuron_amount,))
    return {'weights': weights, 'biases': biases}, (self.neuron_amount,)

  def apply(self, params:dict, inputs:jnp.ndarray):
    # inputs: (batch, in_features), weights: (in_features, out_features)
    weighted_sums = inputs @ params['weights'] + params['biases']
    activated_output = self.activation_fn(weighted_sums)
    return activated_output, weighted_sums

  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray):
    # error: (batch, out_features), inputs: (batch, in_features)
    grads_z = self.activation_derivative_fn(error, weighted_sums)

    grads_weights = jnp.einsum("bi,bj->ij", inputs, grads_z)  # (in_features, out_features)
    grads_biases = jnp.sum(grads_z, axis=0)  # (out_features,)
    upstream_gradient = grads_z @ params['weights'].T  # (batch, in_features)

    param_grads = {
      'weights': grads_weights,
      'biases': grads_biases
    }

    return upstream_gradient, param_grads

class Convolution:
  def __init__(self, kernel: tuple[int, int], channels: int, activation: str, stride: tuple[int, int] = (1, 1), **kwargs):
    """
    Convolution
    -----
      Convolution that is fixed with a valid padding and no dilation. Accepts and returns 3D arrays (excludes batch dimension), so input_shape should be of the form
      (Image Height, Image Width, Channels).
    -----
    Args
    -----
    - neurons     (int)     : the number of neurons in the layer
    - activation  (string)  : the activation function
    
    -----
    Activation functions
    - ReLU
    - Softplus
    - Mish
    - Swish
    - Leaky ReLU
    - GELU
    - ReEU
    - None
    - ReTanh

    Normalization functions
    - Softmax
    - Binary Step
    - Softsign
    - Sigmoid
    - Tanh
    
    Parametric functions
    - ELU
    - SELU
    - PReLU
    - SiLU
    
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
    
    self.kernel = kernel
    self.channels = channels
    self.stride = stride

    if activation.lower() not in Key.ACTIVATION:
      raise ValueError(f"Unknown activation: '{activation}'. Available: {list(Key.ACTIVATION.keys())}")

    self.activation_function = Key.ACTIVATION[activation.lower()]
    self.activation_derivative = Key.ACTIVATION_DERIVATIVE[activation.lower()]

    self.params = {}
    self.input_shape = None
    self.output_shape = None

    if "initializer" in kwargs:
      if kwargs["initializer"].lower() not in Key.INITIALIZER:
        raise ValueError(
          f"Unknown initializer: '{kwargs['initializer'].lower()}'. "
          f"Available: {list(Key.INITIALIZER.keys())}"
        )
      self.initializer_fn = Key.INITIALIZER[kwargs["initializer"].lower()]
    else:
      # He for rectifiers, Glorot for normalizers, else default
      if self.activation_function in rectifiers:
        self.initializer_fn = Key.INITIALIZER["he normal"]
      elif self.activation_function in normalization:
        self.initializer_fn = Key.INITIALIZER["glorot normal"]
      else:
        self.initializer_fn = Key.INITIALIZER["default"]

  def calibrate(self, fan_in_shape: tuple[int, int, int], fan_out_shape: int):
    # fan_in_shape = (C_in, H, W)
    C_in, H, W = fan_in_shape

    self.params["weights"] = self.initializer_fn(
      (self.channels, C_in, *self.kernel),
      C_in * self.kernel[0] * self.kernel[1],
      fan_out_shape,
    )
    self.params["biases"] = jnp.zeros((self.channels,))

    # output dims (VALID padding)
    out_H = (H - self.kernel[0]) // self.stride[0] + 1
    out_W = (W - self.kernel[1]) // self.stride[1] + 1
    self.output_shape = (self.channels, out_H, out_W)

    return self.params, self.output_shape

  def apply(self, params: dict, inputs: jnp.ndarray):
    """
    inputs: (N, C_in, H, W)
    weights: (C_out, C_in, kH, kW)
    bias: (C_out,)
    """
    if inputs.ndim != 4:
      inputs = jnp.expand_dims(inputs, axis=1)

    convolved = jax.lax.conv_general_dilated(
      lhs=inputs,
      rhs=params["weights"],
      window_strides=self.stride,
      padding="VALID",
      dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )

    bias = params["biases"][jnp.newaxis, :, jnp.newaxis, jnp.newaxis]
    WS = convolved + bias
    activated = self.activation_function(WS)
    return activated, WS

  def backward(self, params: dict, inputs: jnp.ndarray, upstream_error: jnp.ndarray, weighted_sums: jnp.ndarray):
    if inputs.ndim != 4:
      inputs = jnp.expand_dims(inputs, axis=1)

    d_WS = self.activation_derivative(upstream_error, weighted_sums)  # (N, C_out, H_out, W_out)

    # bias gradients
    grad_bias = jnp.sum(d_WS, axis=(0, 2, 3))  # (C_out,)

    def correlate(inputs, errors, kernel_shape, strides):
      N, C_in, H_in, W_in = inputs.shape
      _, C_out, H_out, W_out = errors.shape
      kH, kW = kernel_shape
      sH, sW = strides

      grad_weights = jnp.zeros((C_out, C_in, kH, kW))

      # Loop over the batches
      for n in range(N):
        # Loop over the output spatial dimensions
        for h_out in range(H_out):
          for w_out in range(W_out):
            # Calculate the slice for the input patch
            h_start, w_start = h_out * sH, w_out * sW
            input_patch = jax.lax.slice(
              inputs[n],
              (0, h_start, w_start),
              (C_in, h_start + kH, w_start + kW)
            )

            # Get the error for the current output position
            error_patch = jax.lax.slice(
              errors[n],
              (0, h_out, w_out),
              (C_out, h_out + 1, w_out + 1)
            ).reshape(C_out, 1, 1, 1)

            # Compute the outer product and add to the gradient
            # error_patch: (C_out, 1, 1, 1)
            # input_patch: (C_in, kH, kW)
            # The result has shape (C_out, C_in, kH, kW)
            grad_weights += error_patch * jnp.expand_dims(input_patch, axis=0)

      return grad_weights

    def transposed_convolution(errors, weights, stride=(1,1)):
      N, C_out, H_out, W_out = errors.shape
      C_out_w, C_in, kH, kW = weights.shape
      sH, sW = stride

      # Compute input dimensions
      H_in = (H_out - 1) * sH + kH
      W_in = (W_out - 1) * sW + kW

      upstream_gradient = jnp.zeros((N, C_in, H_in, W_in))

      # Flip weights on spatial dimensions for transposed convolution
      flipped_weights = weights[:, :, ::-1, ::-1]  # shape: (C_out, C_in, kH, kW)

      # Loop over batches, output channels, and spatial positions
      for n in range(N):
        for co in range(C_out):
          for i in range(H_out):
            for j in range(W_out):
              h_start = i * sH
              w_start = j * sW
              # Broadcast the multiplication across input channels
              upstream_gradient = upstream_gradient.at[n, :, h_start:h_start+kH, w_start:w_start+kW].add(
                flipped_weights[co] * errors[n, co, i, j]
              )
      return upstream_gradient

    # Cross-correlation between input and error
    grad_weights = correlate(
      inputs=inputs,
      errors=d_WS,
      kernel_shape=self.kernel,
      strides=self.stride
    )

    # Conv transpose to propagate error back
    upstream_gradient = transposed_convolution(
      errors=d_WS, 
      weights=params["weights"], 
      stride=self.stride
    )

    param_gradients = {"weights": grad_weights, "biases": grad_bias}
    return upstream_gradient, param_gradients

class Recurrent:
  def __init__(self, cells:int, activation:str, input_sequence:tuple[int,...]=None, output_sequence:tuple[int,...]=None, **kwargs):
    self.cells = cells

    assert activation in Key.ACTIVATION, f"Unknown activation: '{activation}'. Available: {list(Key.ACTIVATION.keys())}"

    self.activation_fn = Key.ACTIVATION[activation.lower()]
    self.activation_derivative_fn = Key.ACTIVATION_DERIVATIVE[activation.lower()]

    self.initializer_fn = Key.INITIALIZER['default']
    if activation in rectifiers:
      self.initializer_fn = Key.INITIALIZER['he normal']
    elif activation in normalization:
      self.initializer_fn = Key.INITIALIZER['glorot normal']

    if 'initializer' in kwargs:
      if kwargs['initializer'].lower() not in Key.INITIALIZER:
        raise ValueError(f"Unknown initializer: '{kwargs['initializer'].lower()}'. Available: {list(Key.INITIALIZER.keys())}")
      self.initializer_fn = Key.INITIALIZER[kwargs['initializer'].lower()]

    self.input_sequence = input_sequence
    self.output_sequence = output_sequence

  def calibrate(self, fan_in_shape:tuple[int, ...], fan_out_shape:tuple[int,int]):
    features, sequence_length = fan_in_shape
    if self.input_sequence is None:
      self.input_sequence = tuple([_ for _ in range(features)]) 
    if self.output_sequence is None:
      self.output_sequence = tuple([_ for _ in range(self.cells)])

    params = {}
    for cell_index in range(self.cells):
      params[f'cell_{cell_index}'] = {
        'input_weights': self.initializer_fn((sequence_length,), features, fan_out_shape[0]),
        'carry_weights': self.initializer_fn((sequence_length,), features, fan_out_shape[0]),
        'bias': jnp.zeros(sequence_length)
      }
    return params, self.output_sequence

  def apply(self, params:dict, inputs:jnp.ndarray):
    # inputs: (batch, seq_len, features)
    batches, seq_len, features = inputs.shape
    per_batch_output = []
    per_batch_WS = []

    for n in range(batches):
      input_carry = jnp.zeros(features)
      weighted_sums = []
      outputs = []

      for cell_index in range(self.cells):
        input_carry = jnp.zeros(features) if cell_index == 0 else output_carry
        cell_params = params[f'cell_{cell_index}']

        input_vector = jnp.zeros(features)
        if cell_index in self.input_sequence:
          input_feature_idx = self.input_sequence.index(cell_index)
          input_vector = inputs[n, input_feature_idx, :]  # take entire sequence

        # print(input_vector)
        # print(cell_params['input_weights'])
        # exit()

        weighted_input = jnp.dot(input_vector, cell_params['input_weights'])
        
        # print(input_carry)
        # print(cell_params['carry_weights'])
        # exit()
        
        weighted_carry = jnp.dot(input_carry, cell_params['carry_weights'])
        WS = weighted_input + weighted_carry + cell_params['bias']

        output_carry = self.activation_fn(WS)
        weighted_sums.append(WS)

        if cell_index in self.output_sequence:
          outputs.append(output_carry)

      per_batch_output.append(outputs)
      per_batch_WS.append(weighted_sums)

    return jnp.array(per_batch_output), jnp.array(per_batch_WS)

  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray):
    batches, seq_len, features = inputs.shape
    grads = {k: {
      "input_weights": jnp.zeros_like(v["input_weights"]),
      "carry_weights": jnp.zeros_like(v["carry_weights"]),
      "bias": jnp.zeros_like(v["bias"])
    } for k, v in params.items()}

    input_grads = jnp.zeros_like(inputs)

    for n in range(batches):
      
      grad_carry = jnp.zeros(features)
      for cell_index in reversed(range(self.cells)):
        
        cell_params = params[f'cell_{cell_index}']
        WS = weighted_sums[n, cell_index]

        local_error = error[n, cell_index] + grad_carry
        
        # print(local_error)
        # print(WS)
        # exit()
        
        delta = self.activation_derivative_fn(local_error, WS)

        if cell_index in self.input_sequence:
          input_feature_idx = self.input_sequence.index(cell_index)
          input_vector = inputs[n, input_feature_idx, :]
        else:
          input_vector = jnp.zeros(seq_len)

        prev_carry = jnp.zeros(features) if cell_index == 0 else weighted_sums[n, cell_index-1]

        # print(delta)
        # print(prev_carry)
        # print(params[f'cell_{cell_index}']['carry_weights'])
        # exit()
        
        grads[f'cell_{cell_index}']["input_weights"] += jnp.dot(input_vector, delta)
        grads[f'cell_{cell_index}']["carry_weights"] += jnp.dot(prev_carry, delta)
        grads[f'cell_{cell_index}']["bias"] += delta

        if cell_index in self.input_sequence:
          
          input_grads = input_grads.at[n, :, input_feature_idx].add(delta @ cell_params['input_weights'].T)

        grad_carry = delta @ cell_params['carry_weights'].T

    return input_grads, grads
















# functional layers

class MaxPooling:
  def __init__(self, pool_size:tuple[int, int] = (2, 2), strides:tuple[int, int] = (2, 2), **kwargs):
    self.pool_size = pool_size
    self.strides = strides

  def calibrate(self, fan_in_shape:tuple[int, ...], fan_out_shape:tuple[int,int]) -> tuple[dict, tuple[int, ...]]:
    C, H, W = fan_in_shape
    pooled_H = (H - self.pool_size[0]) // self.strides[0] + 1
    pooled_W = (W - self.pool_size[1]) // self.strides[1] + 1
    self.input_shape = fan_in_shape
    return {}, (C, pooled_H, pooled_W)

  def apply(self, params:dict, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    if len(inputs.T.shape) != 4:
      inputs = jnp.expand_dims(inputs.T, axis=1)

    pooled_output = jax.lax.reduce_window(
      inputs,
      init_value=-jnp.inf,
      computation=jax.lax.max,
      window_dimensions=(1, 1, *self.pool_size),
      window_strides=(1, 1, *self.strides),
      padding='VALID'
    )
    return pooled_output, inputs   # WS is just inputs here

  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    # Inputs are already NCHW at this point
    N, C, H, W = inputs.shape
    _, _, pooled_H, pooled_W = error.shape

    pool_H, pool_W = self.pool_size
    stride_H, stride_W = self.strides

    def grad_single(x, grad_out):
      """Backprop a single (H,W) map with pooled grads."""
      grad_in = jnp.zeros_like(x)
      for i in range(pooled_H):
        for j in range(pooled_W):
          h_start, h_end = i * stride_H, i * stride_H + pool_H
          w_start, w_end = j * stride_W, j * stride_W + pool_W
          window = x[h_start:h_end, w_start:w_end]
          mask = window == jnp.max(window)
          grad_in = grad_in.at[h_start:h_end, w_start:w_end].add(mask * grad_out[i, j])
      return grad_in

    grad_fn = jax.vmap(jax.vmap(grad_single, in_axes=(0,0)), in_axes=(0,0))
    upstream_gradient = grad_fn(inputs, error)
    return upstream_gradient, {}

class MeanPooling:
  def __init__(self, pool_size: tuple[int, int] = (2, 2), strides: tuple[int, int] = (2, 2), **kwargs):
    self.pool_size = pool_size
    self.strides = strides

  def calibrate(self, fan_in_shape: tuple[int, ...], fan_out_shape: tuple[int, int]) -> tuple[dict, tuple[int, ...]]:
    C, H, W = fan_in_shape
    pooled_H = (H - self.pool_size[0]) // self.strides[0] + 1
    pooled_W = (W - self.pool_size[1]) // self.strides[1] + 1
    self.input_shape = fan_in_shape
    return {}, (C, pooled_H, pooled_W)

  def apply(self, params: dict, inputs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    if len(inputs.T.shape) != 4:
      inputs = jnp.expand_dims(inputs.T, axis=1)

    pooled_output = jax.lax.reduce_window(
      inputs,
      init_value=0.,
      computation=jax.lax.add,
      window_dimensions=(1, 1, *self.pool_size),
      window_strides=(1, 1, *self.strides),
      padding='VALID'
    )
    pool_area = self.pool_size[0] * self.pool_size[1]
    pooled_output /= pool_area
    return pooled_output, inputs

  def backward(self, params: dict, inputs: jnp.ndarray, error: jnp.ndarray, weighted_sums: jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    N, C, H, W = inputs.shape
    _, _, pooled_H, pooled_W = error.shape
    pool_H, pool_W = self.pool_size
    stride_H, stride_W = self.strides
    pool_area = pool_H * pool_W

    def grad_single(grad_out):
      grad_in = jnp.zeros((H, W))
      for i in range(pooled_H):
        for j in range(pooled_W):
          h_start, h_end = i * stride_H, i * stride_H + pool_H
          w_start, w_end = j * stride_W, j * stride_W + pool_W
          grad_in = grad_in.at[h_start:h_end, w_start:w_end].add(grad_out[i, j] / pool_area)
      return grad_in

    grad_fn = jax.vmap(jax.vmap(grad_single, in_axes=0), in_axes=0)
    upstream_gradient = grad_fn(error)
    return upstream_gradient, {}

class Flatten:
  def __init__(self, **kwargs):
    pass

  def calibrate(self, fan_in_shape: tuple[int, ...], fan_out_shape:int) -> tuple[dict, int]:
    flattened_size = 1
    for dim in fan_in_shape:
      flattened_size *= dim
    self.input_shape = fan_in_shape
    return {}, (int(flattened_size),)

  def apply(self, params: dict, inputs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    # Flatten batch to (batch, features)
    flattened_output = inputs.reshape(inputs.shape[0], -1)
    return flattened_output.T, inputs  # Keep inputs for backprop

  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    upstream_gradient = error.T.reshape(inputs.shape)
    return upstream_gradient, {}


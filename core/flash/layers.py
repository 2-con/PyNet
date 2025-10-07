import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
import typing
import random

from system.config import *
import core.flash.activation as activation
import core.flash.initializer as initializer
import core.flash.derivative as derivative
import core.flash.scaler as scaler
import core.flash.encoder as encoder
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
    "robust scaler": scaler.Robust_Scaler
  }
  
  SCALER_DERIVATIVE = {
    "standard scaler": derivative.Standard_Scaler_Derivative,
    "min max scaler": derivative.Min_Max_Scaler_derivative,
    "max abs scaler": derivative.Max_Abs_Scaler_derivative,
    "robust scaler": derivative.Robust_Scaler_derivative,
  }
  
  ENCODER = {
    "sinusoidal positional": encoder.SinusoidalEmbedding
  }
  
  ENCODER_DERIVATIVE = {
    "sinusoidal positional": lambda x: x
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

impliment parametric stuff in here,
instead of **KWARGS make it *ARGS so that the user can pass in anything they want

convert the forward and backward func into class methods to make it easier to integrate parametric functions

"""

class Dense:
  def __init__(self, neurons:int, activation, name:str="Null", *args, **kwargs):
    """
    Dense
    -----
      A fully connected layer that connects the previous layer to the next layer. Accepts and returns 1D arrays (excludes batch dimension), so input_shape should be of the form
      (input_size,), anything after the 1st dimention will be ignored.
    -----
    Args
    -----
    - neurons         (int)     : the number of neurons in the layer
    - activation      (string)  : the activation function
    - (Optional) name (string)  : the name of the layer
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
    self.name = name
    
    if activation.lower() not in Key.ACTIVATION:
      raise ValueError(f"Unknown activation: '{activation}'. Available: {list(Key.ACTIVATION.keys())}")
    
    self.activation_fn = Key.ACTIVATION[activation.lower()]
    self.activation_derivative_fn = Key.ACTIVATION_DERIVATIVE[activation.lower()]

    self.initializer_fn = Key.INITIALIZER['default']
    if activation.lower() in rectifiers:
      self.initializer_fn = Key.INITIALIZER['he normal']
    elif activation.lower() in normalization:
      self.initializer_fn = Key.INITIALIZER['glorot normal']

    if 'initializer' in kwargs:
      if kwargs['initializer'].lower() not in Key.INITIALIZER:
        raise ValueError(f"Unknown initializer: '{kwargs['initializer'].lower()}'. Available: {list(Key.INITIALIZER.keys())}")
      self.initializer_fn = Key.INITIALIZER[kwargs['initializer'].lower()]

    self.input_size = None

  def calibrate(self, fan_in:tuple[int, ...], fan_out_shape:int) -> tuple[dict, tuple[int, ...]]:
    self.input_size = fan_in[0]
    weights = self.initializer_fn((self.input_size, self.neuron_amount), fan_in[0], fan_out_shape)
    biases = jnp.zeros((self.neuron_amount,))
    
    
    return {'weights': weights, 'biases': biases}, (self.neuron_amount,)

  def apply(self, params:dict, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    # inputs: (batch, in_features), weights: (in_features, out_features)
    weighted_sums = inputs @ params['weights'] + params['biases']
    activated_output = self.activation_fn(weighted_sums)
    return activated_output, weighted_sums

  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> tuple[jnp.ndarray, dict]:
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

class Localunit:
  def __init__(self, receptive_field:int, activation, name:str="Null", *args, **kwargs):
    """
    LocalUnit (Locally Connected Layer)
    -----
      A locally connected layer that connects the previous layer to the next layer. Accepts and returns 1D arrays (excludes batch dimension), so input_shape should be of the form
      (input_size,), anything after the 1st dimention will be ignored.
    -----
    Args
    -----
    - receptive_field (int)    : the size of the receptive field for each neuron
    - activation      (string) : the activation function
    - (Optional) name (string) : the name of the layer
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
    self.receptive_field = receptive_field
    self.name = name
    
    if activation.lower() not in Key.ACTIVATION:
      raise ValueError(f"Unknown activation: '{activation}'. Available: {list(Key.ACTIVATION.keys())}")
    
    self.activation_fn = Key.ACTIVATION[activation.lower()]
    self.activation_derivative_fn = Key.ACTIVATION_DERIVATIVE[activation.lower()]

    self.initializer_fn = Key.INITIALIZER['default']
    if activation.lower() in rectifiers:
      self.initializer_fn = Key.INITIALIZER['he normal']
    elif activation.lower() in normalization:
      self.initializer_fn = Key.INITIALIZER['glorot normal']

    if 'initializer' in kwargs:
      if kwargs['initializer'].lower() not in Key.INITIALIZER:
        raise ValueError(f"Unknown initializer: '{kwargs['initializer'].lower()}'. Available: {list(Key.INITIALIZER.keys())}")
      self.initializer_fn = Key.INITIALIZER[kwargs['initializer'].lower()]

    self.input_size = None

  def calibrate(self, fan_in:tuple[int, ...], fan_out_shape:int) -> tuple[dict, tuple[int, ...]]:
    
    # generating the slide pattern
    ans = []
    height = (fan_in[0] - self.receptive_field) + 1
    
    if height < 1:
      raise ValueError("Field size must be less than or equal to width.")
    if self.receptive_field < 0 or fan_in[0] < 0:
      raise ValueError("Width or field size must be non-negative.")
    
    for i in range(height):
      row = [0 for _ in range(fan_in[0])]
      
      for j in range(fan_in[0]):
        
        if j+i < fan_in[0] and j+i >= i and j < self.receptive_field:
          row[j+i] = 1
      
      ans.append(row)

    # permanent localunit weight mask
    self.mask = jnp.array(ans).T
    
    self.input_size = fan_in[0]
    weights = self.initializer_fn((self.mask.shape[0], self.mask.shape[1]), fan_in[0], fan_out_shape)
    biases = jnp.zeros((self.mask.shape[1],))
    
    return {'weights': weights, 'biases': biases}, (self.mask.shape[0],)

  def apply(self, params:dict, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    # inputs: (batch, in_features), weights: (in_features, out_features)
    weighted_sums = inputs @ (params['weights'] * self.mask) + params['biases']
    activated_output = self.activation_fn(weighted_sums)
    return activated_output, weighted_sums

  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    # error: (batch, out_features), inputs: (batch, in_features)
    grads_z = self.activation_derivative_fn(error, weighted_sums)

    grads_weights = jnp.einsum("bi,bj->ij", inputs, grads_z) * self.mask  # (in_features, out_features)
    
    grads_biases = jnp.sum(grads_z, axis=0)  # (out_features,)
    upstream_gradient = grads_z @ params['weights'].T  # (batch, in_features)

    param_grads = {
      'weights': grads_weights,
      'biases': grads_biases
    }

    return upstream_gradient, param_grads

class Convolution:
  def __init__(self, kernel:tuple[int, int], channels:int, activation:str, stride:tuple[int, int], name:str="Null", *args, **kwargs):
    """
    Convolution
    -----
      Convolution that is fixed with a valid padding and no dilation. Accepts and returns 3D arrays (excludes batch dimension), so input_shape should be of the form
      (Image Height, Image Width, Channels).
    -----
    Args
    -----
    - kernel          (tuple[int, int]) : the kernel dimensions to apply, automatically generated
    - channels        (int)             : the number of channels in the kernel
    - stride          (tuple[int, int]) : the stride to apply to the kernel
    - activation      (string)          : the activation function
    - (Optional) name (string)          : the name of the layer
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
    self.name = name

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

  def calibrate(self, fan_in_shape:tuple[int, int, int], fan_out_shape:int) -> tuple[dict, tuple[int, ...]]:
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

    return self.params, (self.channels, out_H, out_W)

  def apply(self, params:dict, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
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

  def backward(self, params:dict, inputs:jnp.ndarray, upstream_error:jnp.ndarray, weighted_sums:jnp.ndarray) -> tuple[jnp.ndarray, dict]:
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

class Deconvolution:
  def __init__(self, kernel:tuple[int, int], channels:int, activation:str, stride:tuple[int, int], name:str="Null", *args, **kwargs):
    """
    Deconvolution
    -----
      a Deconvolution layer within the context of deep learning is actually a transposed convolution. Accepts and returns 3D arrays (excludes batch dimension), so input_shape should be of the form
      (Image Height, Image Width, Channels).
    -----
    Args
    -----
    - kernel          (tuple[int, int]) : the kernel dimensions to apply, must be of the form (kernel_height, kernel_width)
    - channels        (int)             : the number of channels in the kernel
    - stride          (tuple[int, int]) : the stride to apply to the kernel
    - activation      (string)          : the activation function
    - (Optional) name (string)          : the name of the layer
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
    self.name = name
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

  def calibrate(self, fan_in_shape:tuple[int, int, int], fan_out_shape:int) -> tuple[dict, tuple[int, ...]]:
    # fan_in_shape = (C_in, H, W)
    C_in, H, W = fan_in_shape
    
    sH, sW = self.stride
    kH, kW = self.kernel

    self.params["weights"] = self.initializer_fn(
      (self.channels, C_in, *self.kernel),
      C_in * kH * kW,
      fan_out_shape,
    )
    out_H = (H + kH) * sH - 1
    out_W = (W + kW) * sW- 1

    self.params["biases"] = jnp.zeros((self.channels, out_H, out_W))

    return self.params, (self.channels, out_H, out_W)

  def apply(self, params:dict, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    if inputs.ndim != 4:
      inputs = jnp.expand_dims(inputs, axis=1)
      
    N, C_in, H_in, W_in = inputs.shape
    sH, sW = self.stride
    kH, kW = self.kernel
    
    H_out = (H_in - 1) * sH + kH
    W_out = (W_in - 1) * sW + kW
    
    upscaled = jnp.zeros((N, C_in, H_out, W_out))
    
    for n in range(N): # batch
      for co in range(self.channels): # channels out
        for i in range(H_in): # input height
          for j in range(W_in): # input width
            for ci in range(C_in): # channels in
              upscaled = upscaled.at[n,co,i*sH:i*sH+kH,j*sW:j*sW+kW].add(self.params["weights"][co,ci,:,:]*inputs[n,ci,i,j])
              
    WS = upscaled + params["biases"]
    activated = self.activation_function(WS)
    return activated, WS

  def backward(self, params: dict, inputs: jnp.ndarray, upstream_error: jnp.ndarray, weighted_sums: jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    if inputs.ndim != 4:
      inputs = jnp.expand_dims(inputs, axis=1)

    d_WS = self.activation_derivative(upstream_error, weighted_sums)  # (N, C_out, H_out, W_out)

    grad_bias = jnp.sum(d_WS, axis=(0, 2, 3))  # (C_out, H_out, W_out) → reduce to (C_out,)

    def correlate_deconv(inputs, errors, kernel_shape, stride):
      N, C_in, H_in, W_in = inputs.shape
      N_err, C_out, H_out, W_out = errors.shape
      
      kH, kW = kernel_shape
      sH, sW = stride

      grad_weights = jnp.zeros((C_out, C_in, kH, kW))

      for n in range(N):
        for h_in in range(H_in):
          for w_in in range(W_in):
            h_start = h_in * sH
            w_start = w_in * sW
            h_end, w_end = h_start + kH, w_start + kW

            error_patch = errors[n, :, h_start:h_end, w_start:w_end]  # (C_out, kH, kW)
            
            for ci in range(C_in):
              grad_weights = grad_weights.at[:, ci].add(inputs[n, ci, h_in, w_in] * error_patch)
              
      return grad_weights

    def transposed_to_input(errors, weights, stride):
      N, C_out, H_out, W_out = errors.shape
      C_out_w, C_in, kH, kW = weights.shape
      sH, sW = stride

      H_in = (H_out - kH) // sH + 1
      W_in = (W_out - kW) // sW + 1

      upstream_gradient = jnp.zeros((N, C_in, H_in, W_in))

      # No flip for deconv: it's already transposed
      for n in range(N):
        for co in range(C_out):
          for i in range(H_in):
            for j in range(W_in):
              h_start, w_start = i * sH, j * sW
              h_end, w_end = h_start + kH, w_start + kW

              upstream_gradient = upstream_gradient.at[n, :, i, j].add(jnp.sum(weights[co] * errors[n, co, h_start:h_end, w_start:w_end], axis=(1, 2)))
              
      return upstream_gradient

    grad_weights = correlate_deconv(
      inputs=inputs,
      errors=d_WS,
      kernel_shape=self.kernel,
      stride=self.stride
    )

    upstream_gradient = transposed_to_input(
      errors=d_WS,
      weights=params["weights"],
      stride=self.stride
    )

    param_gradients = {"weights": grad_weights, "biases": grad_bias}
    return upstream_gradient, param_gradients

class Recurrent:
  def __init__(self, cells:int, activation:str, input_sequence:tuple[int,...]=None, output_sequence:tuple[int,...]=None, name:str="Null", *args, **kwargs):
    """
    Recurrent
    -----
      Recurrent layer that processes sequences of data, the input format should be a 2D array (excludes batch dimension), so input_shape should be of the form
      (sequence_length, features). The layer processes the input sequence sequence_length by sequence_length, maintaining a hidden state of size features.
    -----
    Args
    -----
    - Cells                       (int)           : the number of cells in the layer
    - activation                  (string)        : the activation function for the layer
    - (Optional) input_sequence   (tuple of int)  : indices of cells that receive input from the input sequence, all cells receive input by default
    - (Optional) output_sequence  (tuple of int)  : indices of cells that output to the next layer, all cells output by default
    - (Optional) name             (string)        : the name of the layer
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
    
    self.cells = cells
    self.name = name

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

  def calibrate(self, fan_in_shape:tuple[int, ...], fan_out_shape:tuple[int,int]) -> tuple[dict, tuple[int,int]]:
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
        'final_weights': self.initializer_fn((sequence_length * 2, sequence_length), features, fan_out_shape[0]),
        'final_bias': jnp.zeros(sequence_length * 2)
      }
    return params, (len(self.output_sequence),sequence_length)

  def apply(self, params:dict, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
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

        weighted_input = input_vector * cell_params['input_weights']
        weighted_carry = input_carry  * cell_params['carry_weights']

        merged = jnp.concatenate((weighted_carry, weighted_input)) + cell_params['final_bias']

        output_carry = self.activation_fn(merged @ cell_params['final_weights'])
        weighted_sums.append([
          merged, merged @ cell_params['final_weights']
          ])

        outputs.append(output_carry) if cell_index in self.output_sequence else do_nothing()

      per_batch_output.append(outputs)
      per_batch_WS.append(weighted_sums)

    return jnp.array(per_batch_output), per_batch_WS

  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    batches, seq_len, features = inputs.shape
    grads = {k: {
      "input_weights": jnp.zeros_like(v["input_weights"]),
      "carry_weights": jnp.zeros_like(v["carry_weights"]),
      "final_weights": jnp.zeros_like(v["final_weights"]),
      "final_bias": jnp.zeros_like(v["final_bias"])
    } for k, v in params.items()}

    input_grads = jnp.zeros_like(inputs)

    for n in range(batches):
      
      grad_carry = jnp.zeros(features)
      for cell_index in reversed(range(self.cells)):
        
        cell_params = params[f'cell_{cell_index}']
        final_input = weighted_sums[n][cell_index][0]
        WS = weighted_sums[n][cell_index][1]

        local_error = error[n, cell_index] + grad_carry
        
        grads_z = self.activation_derivative_fn(local_error, WS)
        
        grads_weights = jnp.outer(final_input, grads_z)  # (in_features, out_features)
        grads_biases = jnp.sum(grads_z, axis=0)  # (out_features,)
        upstream_gradient = grads_z @ cell_params['final_weights'].T  # (batch, in_features)

        if cell_index in self.input_sequence:
          input_feature_idx = self.input_sequence.index(cell_index)
          input_vector = inputs[n, input_feature_idx, :]
        else:
          input_vector = jnp.zeros(seq_len)

        prev_carry = jnp.zeros(features) if cell_index == 0 else weighted_sums[n][cell_index-1]
        
        grads[f'cell_{cell_index}']["input_weights"] += upstream_gradient[:len(grads[f'cell_{cell_index}']["input_weights"])]
        grads[f'cell_{cell_index}']["carry_weights"] += upstream_gradient[len(grads[f'cell_{cell_index}']["input_weights"]):]
        grads[f'cell_{cell_index}']["final_weights"] += grads_weights
        grads[f'cell_{cell_index}']["final_bias"]    += grads_biases

        if cell_index in self.input_sequence:
          input_grads = input_grads.at[n, :, input_feature_idx].add(grads_z @ cell_params['input_weights'].T)

        grad_carry = grads_z @ cell_params['carry_weights'].T

    return input_grads, grads

class LSTM:
  def __init__(self, cells:int, activation:str, input_sequence:tuple[int,...]=None, output_sequence:tuple[int,...]=None, name:str="Null", *args, **kwargs):
    """
    LSTM (Long Short-Term Memory)
    -----
      a Long Short-Term Memory layer that processes sequences of data, the input format should be a 2D array (excludes batch dimension), so input_shape should be of the form
      (sequence_length, features). The layer processes the input sequence sequence_length by sequence_length, maintaining a hidden state of size features.
    -----
    Args
    -----
    - Cells                       (int)           : the number of cells in the layer
    - activation                  (string)        : the activation function for the layer
    - (Optional) input_sequence   (tuple of int)  : indices of cells that receive input from the input sequence, all cells receive input by default
    - (Optional) output_sequence  (tuple of int)  : indices of cells that output to the next layer, all cells output by default
    - (Optional) name             (string)        : the name of the layer
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
    
    self.cells = cells
    self.name = name

    assert activation in Key.ACTIVATION, f"Unknown activation: '{activation}'. Available: {list(Key.ACTIVATION.keys())}"

    # final head activation (pointwise) and derivative signature same as your framework
    self.activation_fn = Key.ACTIVATION[activation.lower()]
    self.activation_derivative_fn = Key.ACTIVATION_DERIVATIVE[activation.lower()]

    # initializer selection matching your Recurrent
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

  def calibrate(self, fan_in_shape:tuple[int, ...], fan_out_shape:tuple[int,int]) -> tuple[dict, tuple[int, ...]]:

    sequence_length, features = fan_in_shape

    if sequence_length != self.cells:
      raise ValueError(f"calibrate mismatch: sequence_length={sequence_length} but self.cells={self.cells}")

    if self.input_sequence is None:
      self.input_sequence = tuple([_ for _ in range(self.cells)])
    if self.output_sequence is None:
      self.output_sequence = tuple([_ for _ in range(self.cells)])

    concat_size = features * 2  # merged = concat(weighted_carry, weighted_input)

    params = {}
    for cell_index in range(self.cells):
      params[f'cell_{cell_index}'] = {
        # elementwise multipliers (preserve length)
        'input_weights': self.initializer_fn((features,), features, features),
        'carry_weights': self.initializer_fn((features,), features, features),

        'forget_weights': self.initializer_fn((concat_size, features), concat_size, features),
        'forget_bias': jnp.zeros((features,)),

        'input_gate_weights': self.initializer_fn((concat_size, features), concat_size, features),
        'input_gate_bias': jnp.zeros((features,)),

        'output_gate_weights': self.initializer_fn((concat_size, features), concat_size, features),
        'output_gate_bias': jnp.zeros((features,)),

        'candidate_weights': self.initializer_fn((concat_size, features), concat_size, features),
        'candidate_bias': jnp.zeros((features,)),

        'final_weights': self.initializer_fn((concat_size, features), concat_size, features),
        'final_bias': jnp.zeros((concat_size,))
      }
    return params, (len(self.output_sequence),sequence_length)

  def apply(self, params:dict, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    batches, cells_dim, features = inputs.shape
    if cells_dim != self.cells:
      raise ValueError(f"apply mismatch: inputs has cells_dim={cells_dim} but self.cells={self.cells}")

    per_batch_output = []
    per_batch_WS = []

    for n in range(batches):
      # initial carry / hidden are zeros (preserve-length)
      input_carry = jnp.zeros((features,))
      c_t = jnp.zeros((features,))    # cell state (kept for completeness, used in gating)
      h_t = jnp.zeros((features,))

      weighted_sums = []
      outputs = []

      for cell_index in range(self.cells):
        cell_params = params[f'cell_{cell_index}']

        # elementwise weighted inputs
        if cell_index in self.input_sequence:
          input_feature_idx = self.input_sequence.index(cell_index)
          input_vector = inputs[n, input_feature_idx, :]
        else:
          input_vector = jnp.zeros((features,))

        weighted_input = input_vector * cell_params['input_weights']   # elementwise
        weighted_carry = input_carry * cell_params['carry_weights']    # elementwise

        # merged concat (carry first, input second) + final_bias (matches your Recurrent idea)
        merged = jnp.concatenate((weighted_carry, weighted_input)) + cell_params['final_bias']  # (2*features,)

        # gates: merged -> features (fully connected preserving length)
        z_f = jnp.dot(merged, cell_params['forget_weights']) + cell_params['forget_bias']
        z_i = jnp.dot(merged, cell_params['input_gate_weights']) + cell_params['input_gate_bias']
        z_g = jnp.dot(merged, cell_params['candidate_weights']) + cell_params['candidate_bias']
        z_o = jnp.dot(merged, cell_params['output_gate_weights']) + cell_params['output_gate_bias']

        f_t = Key.ACTIVATION['sigmoid'](z_f)
        i_t = Key.ACTIVATION['sigmoid'](z_i)
        o_t = Key.ACTIVATION['sigmoid'](z_o)
        g_t = Key.ACTIVATION['tanh'](z_g)

        # cell + hidden update (standard LSTM equations but using merged->gates)
        c_prev = c_t
        h_prev = h_t

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * Key.ACTIVATION['tanh'](c_t)

        # final FC head (merged -> features) then activation_fn (pointwise)
        act_WS = jnp.dot(merged, cell_params['final_weights'])   # (features,)
        output_carry = self.activation_fn(act_WS)

        # stash all needed values for backward
        WS = {
          'merged': merged, 
          'act_WS': act_WS,
          
          'z_f': z_f,                                                                                                                                               
          'z_i': z_i, 
          'z_o': z_o, 
          'z_g': z_g,
          
          'f_t': f_t, 
          'i_t': i_t, 
          'o_t': o_t, 
          'g_t': g_t,
          
          'c_prev': c_prev, 
          'c_t': c_t,
          
          'h_prev': h_prev, 
          'h_t': h_t,
          
          'input_vector': input_vector,
          'weighted_input': weighted_input, 'weighted_carry': weighted_carry
        }
        weighted_sums.append(WS)

        # collect outputs in order of output_sequence (same as your Recurrent)
        if cell_index in self.output_sequence:
          outputs.append(output_carry)

        # next cell's carry / hidden
        input_carry = output_carry   # keep the same naming as your Recurrent

      per_batch_output.append(jnp.stack(outputs, axis=0) if len(outputs) > 0 else jnp.zeros((0, features)))
      per_batch_WS.append(weighted_sums)

    return jnp.stack(per_batch_output, axis=0), per_batch_WS

  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:list) -> tuple[jnp.ndarray, dict]:
    batches, cells_dim, features = inputs.shape
    if cells_dim != self.cells:
      raise ValueError(f"backward mismatch: inputs has cells_dim={cells_dim} but self.cells={self.cells}")

    # prepare grads per-cell
    grads = {k: {
      "input_weights": jnp.zeros_like(v["input_weights"]),
      "carry_weights": jnp.zeros_like(v["carry_weights"]),
      
      "forget_weights": jnp.zeros_like(v["forget_weights"]),
      "forget_bias": jnp.zeros_like(v["forget_bias"]),
      
      "input_gate_weights": jnp.zeros_like(v["input_gate_weights"]),
      "input_gate_bias": jnp.zeros_like(v["input_gate_bias"]),
      
      "output_gate_weights": jnp.zeros_like(v["output_gate_weights"]),
      "output_gate_bias": jnp.zeros_like(v["output_gate_bias"]),
      
      "candidate_weights": jnp.zeros_like(v["candidate_weights"]),
      "candidate_bias": jnp.zeros_like(v["candidate_bias"]),
      
      "final_weights": jnp.zeros_like(v["final_weights"]),
      "final_bias": jnp.zeros_like(v["final_bias"]),
    } for k, v in params.items()}

    input_grads = jnp.zeros_like(inputs)

    for n in range(batches):
      # gradient carrying along the chain (for hidden / carry)
      dh_next = jnp.zeros((features,))   # gradient w.r.t. next hidden (from future cell)
      dc_next = jnp.zeros((features,))   # gradient w.r.t. next cell state

      # iterate cells backward
      for cell_index in reversed(range(self.cells)):
        cell_key = f'cell_{cell_index}'
        cell_params = params[cell_key]
        WS = weighted_sums[n][cell_index]

        merged = WS['merged']        # 2 * features
        act_WS = WS['act_WS']        # features
        
        z_f = WS['z_f']; z_i = WS['z_i']; z_o = WS['z_o']; z_g = WS['z_g']
        f_t = WS['f_t']; i_t = WS['i_t']; o_t = WS['o_t']; g_t = WS['g_t']
        
        c_prev = WS['c_prev']; c_t = WS['c_t']
        h_prev = WS['h_prev']; h_t = WS['h_t']
        
        input_vector = WS['input_vector']

        # get error slice if this cell produced an output
        if cell_index in self.output_sequence:
          k = self.output_sequence.index(cell_index)
          d_y = error[n, k, :]
        else:
          d_y = jnp.zeros((features,))

        # grads_z corresponds to delta in feature space after activation
        grads_z = self.activation_derivative_fn(d_y, act_WS)   # (features,)
        # final_weights gradient
        grads[cell_key]['final_weights'] += jnp.outer(merged, grads_z)   # (2*features, features)

        # gradient flowing back into merged from final head
        upstream_from_final = jnp.dot(grads_z, cell_params['final_weights'].T)  # (2*features,)

        grads[cell_key]['final_bias'] += upstream_from_final

        # compute derivative wrt o, i, f, g using standard LSTM formulas
        tanh_c = Key.ACTIVATION['tanh'](c_t)
        # output gate
        do = dh_next * tanh_c
        do_raw = do * (o_t * (1.0 - o_t))   # σ'(z_o) with o_t

        # cell state derivative
        dc = dh_next * o_t * (1.0 - tanh_c ** 2) + dc_next

        di_raw = (dc * g_t) * (i_t * (1.0 - i_t))   # σ'(z_i)
        df_raw = (dc * c_prev) * (f_t * (1.0 - f_t))   # σ'(z_f)
        dg_raw = (dc * i_t) * (1.0 - g_t ** 2)       # tanh'(z_g) via g_t

        # accumulate gradients for gate FCs (merged -> features)
        grads[cell_key]['forget_weights'] += jnp.outer(merged, df_raw)
        grads[cell_key]['forget_bias'] += df_raw

        grads[cell_key]['input_gate_weights'] += jnp.outer(merged, di_raw)
        grads[cell_key]['input_gate_bias'] += di_raw

        grads[cell_key]['output_gate_weights'] += jnp.outer(merged, do_raw)
        grads[cell_key]['output_gate_bias'] += do_raw

        grads[cell_key]['candidate_weights'] += jnp.outer(merged, dg_raw)
        grads[cell_key]['candidate_bias'] += dg_raw

        # gradient wrt merged coming from gate paths (via their weight matrices)
        d_concat_from_gates = (
          jnp.dot(df_raw, cell_params['forget_weights'].T) +
          jnp.dot(di_raw, cell_params['input_gate_weights'].T) +
          jnp.dot(do_raw, cell_params['output_gate_weights'].T) +
          jnp.dot(dg_raw, cell_params['candidate_weights'].T)
        )   # shape (2*features,)

        # total gradient into merged = from final head + from gates
        d_merged_total = upstream_from_final + d_concat_from_gates   # (2*features,)

        # split into carry and input halves
        d_weighted_carry = d_merged_total[:features]
        d_weighted_input = d_merged_total[features:]

        # weighted_input = input_vector * input_weights  -> d input_weights = input_vector * d_weighted_input
        grads[cell_key]['input_weights'] += input_vector * d_weighted_input
        grads[cell_key]['carry_weights'] += h_prev * d_weighted_carry

        if cell_index in self.input_sequence:
          input_feature_idx = self.input_sequence.index(cell_index)
          # derivative: weighted_input = input_vector * input_weights -> dx = d_weighted_input * input_weights
          input_grads = input_grads.at[n, input_feature_idx, :].add(d_weighted_input * cell_params['input_weights'])

        dh_prev_from_merged = d_weighted_carry * cell_params['carry_weights']

        dh_next = dh_prev_from_merged
        dc_next = dc * f_t

    return input_grads, grads

class GRU:
  def __init__(self, cells:int, activation:str, input_sequence:tuple[int,...]=None, output_sequence:tuple[int,...]=None, name:str="Null", *args, **kwargs):
    """
    GRU (Gated Recurrent Unit)
    -----
      A Gated Recurrent Unit layer that processes sequences of data, the input format should be a 2D array (excludes batch dimension), so input_shape should be of the form
      (sequence_length, features). The layer processes the input sequence sequence_length by sequence_length, maintaining a hidden state of size features.
    -----
    Args
    -----
    - cells                       (int)           : the number of cells in the layer
    - activation                  (string)        : the activation function for the layer
    - (Optional) input_sequence   (tuple of int)  : indices of cells that receive input from the input sequence, all cells receive input by default
    - (Optional) output_sequence  (tuple of int)  : indices of cells that output to the next layer, all cells output by default
    - (Optional) name             (string)        : the name of the layer
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
    self.cells = cells
    self.name = name

    assert activation in Key.ACTIVATION, f"Unknown activation: '{activation}'. Available: {list(Key.ACTIVATION.keys())}"

    self.activation_fn = Key.ACTIVATION[activation.lower()]
    self.activation_derivative_fn = Key.ACTIVATION_DERIVATIVE[activation.lower()]

    # default init
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

  def calibrate(self, fan_in_shape:tuple[int,...], fan_out_shape:tuple[int,int]) -> tuple[dict, tuple[int, ...]]:
    
    features, sequence_length = fan_in_shape
    if self.input_sequence is None:
      self.input_sequence = tuple([_ for _ in range(features)]) 
      
    if self.output_sequence is None:
      self.output_sequence = tuple([_ for _ in range(self.cells)])

    params = {}
    for cell_index in range(self.cells):
      params[f'cell_{cell_index}'] = {
        # reset gate
        "W_r": self.initializer_fn((sequence_length,), features, fan_out_shape[0]),
        "U_r": self.initializer_fn((sequence_length,), features, fan_out_shape[0]),
        "b_r": jnp.zeros(sequence_length),

        # update gate
        "W_z": self.initializer_fn((sequence_length,), features, fan_out_shape[0]),
        "U_z": self.initializer_fn((sequence_length,), features, fan_out_shape[0]),
        "b_z": jnp.zeros(sequence_length),

        # candidate hidden
        "W_h": self.initializer_fn((sequence_length,), features, fan_out_shape[0]),
        "U_h": self.initializer_fn((sequence_length,), features, fan_out_shape[0]),
        "b_h": jnp.zeros(sequence_length),

        # final fully-connected layer before output
        "final_weights": self.initializer_fn((sequence_length * 2, sequence_length), features, fan_out_shape[0]),
        "final_bias": jnp.zeros(sequence_length * 2)
      }
      
    return params, (len(self.output_sequence),sequence_length)

  def apply(self, params:dict, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    # inputs: (batch, seq_len, features)
    batches, seq_len, features = inputs.shape
    per_batch_output = []
    per_batch_WS = []

    for n in range(batches):
      
      h_t = jnp.zeros(features)
      weighted_sums = []
      outputs = []

      for cell_index in range(self.cells):
        cell_params = params[f'cell_{cell_index}']

        # pick input feature
        x_t = jnp.zeros(features)
        if cell_index in self.input_sequence:
          input_feature_idx = self.input_sequence.index(cell_index)
          x_t = inputs[n, input_feature_idx, :]

        # reset gate
        r_t = jax.nn.sigmoid(x_t * cell_params["W_r"] + h_t * cell_params["U_r"] + cell_params["b_r"])

        # update gate
        z_t = jax.nn.sigmoid(x_t * cell_params["W_z"] + h_t * cell_params["U_z"] + cell_params["b_z"])

        # candidate hidden
        h_hat = self.activation_fn(x_t * cell_params["W_h"] + (r_t * h_t) * cell_params["U_h"] + cell_params["b_h"])

        # final hidden state
        h_t = (1 - z_t) * h_t + z_t * h_hat

        # fully connected before output
        merged = jnp.concatenate((h_t, x_t)) + cell_params['final_bias']
        output_carry = self.activation_fn(merged @ cell_params['final_weights'])

        weighted_sums.append([r_t, z_t, h_hat, h_t, merged, merged @ cell_params['final_weights']])

        if cell_index in self.output_sequence:
          outputs.append(output_carry)

      per_batch_output.append(outputs)
      per_batch_WS.append(weighted_sums)

    return jnp.array(per_batch_output), per_batch_WS

  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    batches, seq_len, features = inputs.shape
    grads = {k: {
      "W_r": jnp.zeros_like(v["W_r"]), 
      "U_r": jnp.zeros_like(v["U_r"]), 
      "b_r": jnp.zeros_like(v["b_r"]),
      
      "W_z": jnp.zeros_like(v["W_z"]), 
      "U_z": jnp.zeros_like(v["U_z"]), 
      "b_z": jnp.zeros_like(v["b_z"]),
      
      "W_h": jnp.zeros_like(v["W_h"]), 
      "U_h": jnp.zeros_like(v["U_h"]), 
      "b_h": jnp.zeros_like(v["b_h"]),
      
      "final_weights": jnp.zeros_like(v["final_weights"]),
      "final_bias":    jnp.zeros_like(v["final_bias"])
    } for k, v in params.items()}

    input_grads = jnp.zeros_like(inputs)

    for n in range(batches):
      
      grad_h = jnp.zeros(features)
      
      for cell_index in reversed(range(self.cells)):
        cell_params = params[f'cell_{cell_index}']
        r_t, z_t, h_hat, h_t, final_input, WS = weighted_sums[n][cell_index]

        # combine error + upstream
        local_error = error[n, cell_index] + grad_h

        # backprop through final fully-connected
        grads_z = self.activation_derivative_fn(local_error, WS)
        grads_weights = jnp.outer(final_input, grads_z)
        grads_biases = jnp.sum(grads_z, axis=0)
        upstream = grads_z @ cell_params['final_weights'].T

        grads[f'cell_{cell_index}']["final_weights"] += grads_weights
        grads[f'cell_{cell_index}']["final_bias"] += grads_biases

        # split back to h_t and x_t contributions
        grad_h = upstream[:features]
        grad_x = upstream[features:]

        if cell_index in self.input_sequence:
          input_feature_idx = self.input_sequence.index(cell_index)
          
          # print(input_grads)
          # print(input_grads[n, input_feature_idx, :])
          # print(grad_x)
          # exit()
          
          input_grads = input_grads.at[n, input_feature_idx, :].add(grad_x)

        # TODO: properly expand into r_t, z_t, and candidate grads
        # (same structure as GRU math but with FCs)

    return input_grads, grads

class Attention:
  def __init__(self, heads:int, activation:str, name:str="Null", *args, **kwargs):
    """
    Multiheaded Self-Attention
    -----
      Primary block within Transformer networks, the amount of attention heads is configurable. 
      It accepts data with shape (batch_size, sequence_length, features) simmilar to RNNs.
    -----
    Args
    -----
    - heads           (int)     : the number of attention heads
    - activation      (string)  : the activation function applied to the output
    - (Optional) name (string)  : the name of the layer
    """
    self.heads = heads
    self.name = name

    # validate activation
    assert activation in Key.ACTIVATION, f"Unknown activation: '{activation}'. Available: {list(Key.ACTIVATION.keys())}"
    self.activation_fn = Key.ACTIVATION[activation.lower()]
    self.activation_derivative_fn = Key.ACTIVATION_DERIVATIVE[activation.lower()]

    # initializer selection
    self.initializer_fn = Key.INITIALIZER['default']
    if activation in rectifiers:
      self.initializer_fn = Key.INITIALIZER['he normal']
    elif activation in normalization:
      self.initializer_fn = Key.INITIALIZER['glorot normal']

    if 'initializer' in kwargs:
      if kwargs['initializer'].lower() not in Key.INITIALIZER:
        raise ValueError(f"Unknown initializer: '{kwargs['initializer'].lower()}'. Available: {list(Key.INITIALIZER.keys())}")
      self.initializer_fn = Key.INITIALIZER[kwargs['initializer'].lower()]

  def calibrate(self, fan_in_shape:tuple[int,...], fan_out_shape:tuple[int,int]) -> tuple[dict, tuple[int,int]]:
    """
    fan_in_shape: (features, sequence_length)
    fan_out_shape: (features_out, sequence_length_out) - we preserve length
    """
    features, sequence_length = fan_in_shape
    params = {}

    # each head has its own Q, K, V
    for head in range(self.heads):
      params[f'head_{head}'] = {
        "W_Q": self.initializer_fn((sequence_length, sequence_length), features, fan_out_shape[0]),
        "W_K": self.initializer_fn((sequence_length, sequence_length), features, fan_out_shape[0]),
        "W_V": self.initializer_fn((sequence_length, sequence_length), features, fan_out_shape[0]),
        "b_Q": jnp.zeros(sequence_length),
        "b_K": jnp.zeros(sequence_length),
        "b_V": jnp.zeros(sequence_length),
      }

    # final projection after concat
    params["final"] = {
      "W_O": self.initializer_fn((sequence_length * self.heads, sequence_length), features, fan_out_shape[0]),
      "b_O": jnp.zeros(sequence_length)
    }

    return params, (features, sequence_length)

  def apply(self, params:dict, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    batch_size, features, seq_len = inputs.shape
    per_batch_outputs = []
    per_batch_WS = []

    for n in range(batch_size):  # process one batch at a time
      batch_input = inputs[n]  # (seq_len, features)
      head_outputs = []
      head_ws = []

      for head in range(self.heads):
        head_params = params[f'head_{head}']

        # fully-connected projections
        Q = batch_input @ head_params["W_Q"] + head_params["b_Q"]  # (S, F)
        K = batch_input @ head_params["W_K"] + head_params["b_K"]  # (S, F)
        V = batch_input @ head_params["W_V"] + head_params["b_V"]  # (S, F)

        # scaled dot-product attention
        attn_scores = (Q @ K.T) / jnp.sqrt(seq_len)     # (S, S)
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)  # (S, S)
        head_out = attn_weights @ V                      # (S, F)

        head_outputs.append(head_out)
        head_ws.append((Q, K, V, attn_scores, attn_weights, head_out))

      # concat all heads along feature dim → (S, F * heads)
      concat_out = jnp.concatenate(head_outputs, axis=-1)
      
      # final FC projection (maps back to (S, F))
      merged = jax.vmap(lambda token: token @ params["final"]["W_O"] + params["final"]["b_O"])(concat_out)
      out = self.activation_fn(merged)  # (S, F)

      per_batch_outputs.append(out)
      per_batch_WS.append((head_ws, concat_out, merged, out))

    return jnp.array(per_batch_outputs), per_batch_WS

  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:dict) -> tuple[jnp.ndarray, dict]:
    batch_size, features, seq_len = inputs.shape

    # initialize grads
    grads = {}
    for head in range(self.heads):
      grads[f'head_{head}'] = {
        "W_Q": jnp.zeros_like(params[f'head_{head}']["W_Q"]),
        "b_Q": jnp.zeros_like(params[f'head_{head}']["b_Q"]),
        "W_K": jnp.zeros_like(params[f'head_{head}']["W_K"]),
        "b_K": jnp.zeros_like(params[f'head_{head}']["b_K"]),
        "W_V": jnp.zeros_like(params[f'head_{head}']["W_V"]),
        "b_V": jnp.zeros_like(params[f'head_{head}']["b_V"]),
      }

    grads["final"] = {
      "W_O": jnp.zeros_like(params["final"]["W_O"]),
      "b_O": jnp.zeros_like(params["final"]["b_O"]),
    }

    input_grads = jnp.zeros_like(inputs)

    # process per batch
    for n in range(batch_size):
      batch_input = inputs[n]    # (S, F)
      batch_error = error[n]     # (S, F)
      head_ws, concat_out, merged, out = weighted_sums[n]

      # ---- backprop through final FC + activation ----
      d_out = self.activation_derivative_fn(batch_error, merged)   # (S, F)
      grads["final"]["W_O"] += jnp.einsum("bi,bj->ij", concat_out, d_out)                # (F*H, F)
      grads["final"]["b_O"] += jnp.sum(d_out, axis=0)              # (F,)
      d_concat = d_out @ params["final"]["W_O"].T                  # (S, F*H)

      # ---- split gradient across heads ----
      split_d = jnp.split(d_concat, self.heads, axis=-1)

      # accumulate grads per-head + input grads
      d_input_total = jnp.zeros_like(batch_input)

      for head, d_head_out in enumerate(split_d):
        Q, K, V, attn_scores, attn_weights, head_out = head_ws[head]
        head_params = params[f'head_{head}']

        # grad wrt V (from attn_weights @ V)
        dV = attn_weights.T @ d_head_out  # (S, F)

        # grad wrt attn_weights
        d_attn_weights = d_head_out @ V.T  # (S, S)

        # grad wrt attn_scores (softmax backprop)
        # softmax jacobian simplified: dL/dS = (grad - sum(grad*W, axis=-1)) * W
        d_attn_scores = d_attn_weights * attn_weights - attn_weights * jnp.sum(d_attn_weights * attn_weights, axis=-1, keepdims=True)

        # grad wrt Q and K
        dQ = d_attn_scores @ K / jnp.sqrt(seq_len)
        dK = d_attn_scores.T @ Q / jnp.sqrt(seq_len)

        # propagate through FCs: (batch_input @ W + b)
        grads[f'head_{head}']["W_Q"] += batch_input.T @ dQ
        grads[f'head_{head}']["b_Q"] += jnp.sum(dQ, axis=0)

        grads[f'head_{head}']["W_K"] += batch_input.T @ dK
        grads[f'head_{head}']["b_K"] += jnp.sum(dK, axis=0)

        grads[f'head_{head}']["W_V"] += batch_input.T @ dV
        grads[f'head_{head}']["b_V"] += jnp.sum(dV, axis=0)

        # accumulate gradient wrt inputs
        d_input_total += dQ @ head_params["W_Q"].T
        d_input_total += dK @ head_params["W_K"].T
        d_input_total += dV @ head_params["W_V"].T

      # assign input grads for this batch
      input_grads = input_grads.at[n].set(d_input_total)

    return input_grads, grads

# functional layers

class MaxPooling:
  def __init__(self, pool_size:tuple[int, int], strides:tuple[int, int], name:str="Null", *args, **kwargs):
    """
    Max Pooling
    -----
      A layer that performs max pooling on a 2D input while adjusting to channel dimensions. make sure to set the input shape in the format (Channels, Height, Width).
    -----
    Args
    -----
    - pool_size       (tuple of int)  : the size of the pooling window (height, width)
    - strides         (tuple of int)  : the stride of the pooling window (height, width)
    - (Optional) name (string)        : the name of the layer
    """
    self.pool_size = pool_size
    self.strides = strides
    self.name = name

  def calibrate(self, fan_in_shape:tuple[int, ...], fan_out_shape:tuple[int,int]) -> tuple[dict, tuple[int, ...]]:
    C, H, W = fan_in_shape
    pooled_H = (H - self.pool_size[0]) // self.strides[0] + 1
    pooled_W = (W - self.pool_size[1]) // self.strides[1] + 1
    self.input_shape = fan_in_shape
    
    return {}, (C, pooled_H, pooled_W)

  def apply(self, params:dict, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    if len(inputs.shape) != 4:
      inputs = jnp.expand_dims(inputs, axis=1)

    pooled_output = jax.lax.reduce_window(
      inputs,
      init_value=-jnp.inf,
      computation=jax.lax.max,
      window_dimensions=(1, 1, *self.pool_size),
      window_strides=(1, 1, *self.strides),
      padding='VALID'
    )
    
    return pooled_output, inputs

  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    if len(inputs.shape) != 4:
      inputs = jnp.expand_dims(inputs, axis=1)
      
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
  def __init__(self, pool_size:tuple[int, int] = (2, 2), strides:tuple[int, int] = (2, 2), name:str="Null", *args, **kwargs):
    """
    Mean Pooling
    -----
      A layer that performs mean pooling on a 2D input while adjusting to channel dimensions. make sure to set the input shape in the format (Channels, Height, Width).
    -----
    Args
    -----
    - pool_size       (tuple of int)  : the size of the pooling window (height, width)
    - strides         (tuple of int)  : the stride of the pooling window (height, width)
    - (Optional) name (string)        : the name of the layer
    """
    self.pool_size = pool_size
    self.strides = strides
    self.name = name

  def calibrate(self, fan_in_shape:tuple[int, ...], fan_out_shape:tuple[int, int]) -> tuple[dict, tuple[int, ...]]:
    C, H, W = fan_in_shape
    pooled_H = (H - self.pool_size[0]) // self.strides[0] + 1
    pooled_W = (W - self.pool_size[1]) // self.strides[1] + 1
    self.input_shape = fan_in_shape
    return {}, (C, pooled_H, pooled_W)

  def apply(self, params:dict, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    if len(inputs.shape) != 4:
      inputs = jnp.expand_dims(inputs, axis=1)

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

  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> tuple[jnp.ndarray, dict]:
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
  def __init__(self, name:str="Null", *args, **kwargs):
    """
    Flatten
    -----
      A layer that flattens any ndim input into a 1D input while adjusting to batch dimensions. 
      make sure to set the input shape in the format (Channels, Height, Width).
    -----
    Args
    -----
    - (Optional) name (string)        : the name of the layer
    """
    
    self.name = name

  def calibrate(self, fan_in_shape:tuple[int, ...], fan_out_shape:int) -> tuple[dict, int]:
    flattened_size = jnp.prod(jnp.array(fan_in_shape))
    self.input_shape = fan_in_shape
    
    return {}, (int(flattened_size),)

  def apply(self, params:dict, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    # Flatten batch to (batch, features)
    flattened_output = inputs.reshape(inputs.shape[0], -1)
    
    return flattened_output, inputs  # Keep inputs for backprop

  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    upstream_gradient = error.reshape(inputs.shape)
    return upstream_gradient, {}

class Operation:
  def __init__(self, operation:typing.Union[str, callable], operation_derivative:callable=do_nothing, name:str="Null", *args, **kwargs):
    """
    Operation
    -----
      A layer that performs an operation on any ndim input while preserving shape. 
      this layer auto-adjusts and does not need to a fixed input shape, but make sure to set the input shape in the format that the operation expects.
    -----
    Args
    -----
    - operation             (str or callable) : the operation to perform on the input
    - operation_derivative  (callable)        : the derivative of the operation if a callable is provided, make sure the backwards function has the format of FUNCTION(error, weighted_sums)
    - (Optional) name       (string)          : the name of the layer
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

    Scaler functions
    - Standard Scaler
    - Min Max Scaler
    - Max Abs Scaler
    - Robust Scaler
    """
    
    if type(operation) == str:
      if operation in Key.ACTIVATION:
        self.operation_fn = Key.ACTIVATION[operation.lower()]
        self.operation_derivative_fn = Key.ACTIVATION_DERIVATIVE[operation.lower()]
        
      if operation in Key.SCALER:
        self.operation_fn = Key.SCALER[operation.lower()]
        self.operation_derivative_fn = Key.SCALER_DERIVATIVE[operation.lower()]
        
      if operation in Key.ENCODER:
        self.operation_fn = Key.ENCODER[operation.lower()]
        self.operation_derivative_fn = Key.ENCODER_DERIVATIVE[operation.lower()]

    else:
      self.operation_fn = operation
      
      if not callable(operation_derivative):
        raise TypeError("operation_derivative must be a callable function")
      if operation_derivative is do_nothing:
        raise ValueError("operation_derivative must be provided as a callable function")
      
      self.operation_derivative_fn = operation_derivative
    
    self.name = name
  
  def calibrate(self, fan_in_shape:tuple[int, ...], fan_out_shape:tuple[int, ...]) -> tuple[dict, tuple[int, ...]]:
    
    if self.operation_fn(jnp.zeros(fan_in_shape)).shape != fan_in_shape:
      raise SystemError(f"Operation layer '{self.name}' operation function does not preserve shape: {fan_in_shape} -> {self.operation_fn(jnp.zeros_like(fan_in_shape)).shape}")
    
    return {}, fan_in_shape
  
  def apply(self, params:dict, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    return self.operation_fn(inputs), inputs  # WS is just inputs for backprop
  
  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    return self.operation_derivative_fn(error, weighted_sums), {}

class Dropout:
  def __init__(self, rate:float, mode:str, name:str="Null", *args, **kwargs):
    if not (0.0 <= rate < 1.0):
      raise ValueError("Dropout rate must be in the range [0.0, 1.0)")
    if mode.lower() not in ('random', 'fixed'):
      raise ValueError("Dropout mode must be 'random' or 'fixed'")
    
    self.mode = mode.lower() 
    self.rate = rate
    self.name = name
  
  def calibrate(self, fan_in_shape:tuple[int, ...], fan_out_shape:tuple[int, ...]) -> tuple[dict, tuple[int, ...]]:
    return {}, fan_in_shape
  
  def apply(self, params:dict, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    if self.rate == 0.0:
      return inputs, jnp.ones_like(inputs)
    
    if self.mode == 'random':
      mask = jax.random.bernoulli(jax.random.PRNGKey(random.randint(1,1000)), p=1.0 - self.rate, shape=inputs.shape)

    else:
      drop_rate = int(jnp.ones((inputs.shape)).flatten().shape[0] * self.rate)
      ones = jnp.ones((inputs.shape)).flatten()[0:drop_rate]
      zeros = jnp.zeros((inputs.shape)).flatten()[drop_rate:]

      mask = jax.random.permutation(jax.random.PRNGKey(random.randint(1,1000)), jnp.concatenate((ones, zeros))).reshape(inputs.shape)

    return inputs * mask, mask
  
  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    if self.rate == 0.0:
      return error, {}
    
    return error * weighted_sums / (1.0 - self.rate), {}

class Reshape:
  def __init__(self, target_shape:tuple[int, ...], name:str="Null", *args, **kwargs):
    self.target_shape = target_shape
    self.name = name
  
  def calibrate(self, fan_in_shape:tuple[int, ...], fan_out_shape:tuple[int, ...]) -> tuple[dict, tuple[int, ...]]:
    input_size = jnp.prod(jnp.array(fan_in_shape[1:]))  # exclude batch dimension
    target_size = jnp.prod(jnp.array(self.target_shape))
    
    if input_size != target_size:
      raise ValueError(f"Reshape layer '{self.name}' cannot reshape from {fan_in_shape} to {self.target_shape} due to size mismatch ({input_size} -> {target_size})")
    
    return {}, self.target_shape
  
  def apply(self, params:dict, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    batch_size = inputs.shape[0]
    reshaped_output = inputs.reshape((batch_size, *self.target_shape))
    
    return reshaped_output, inputs  # WS is just inputs for backprop
  
  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    upstream_gradient = error.reshape(inputs.shape)
    return upstream_gradient, {}
  

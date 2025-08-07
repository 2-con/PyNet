# normally jax.grad takes care of this, but there is an issue with the gradient calculation where
# jax detects a tuple at pos 1 when no tuple is found anywhere else in the code. This is a workaround
# for that issue.

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import jax.numpy as jnp
import jax

from system.defaults import *
from core.flash.activation import Sigmoid

_EPSILON = 1e-8 # Ensure this matches your original definition

def Standard_Scaler_Derivative(std):
  return jnp.where(std != 0, 1.0 / (std + epsilon_default), 0.0)

def Min_Max_Scaler_derivative(min_val, max_val):
  range_val = max_val - min_val
  return jnp.where(range_val != 0, 1.0 / (range_val + epsilon_default), 0.0)

def Max_Abs_Scaler_derivative(max_abs_val):
  return jnp.where(max_abs_val != 0, 1.0 / (max_abs_val + epsilon_default), 0.0)

def Robust_Scaler_derivative(q1, q3):
  iqr = q3 - q1
  return jnp.where(iqr != 0, 1.0 / (iqr + epsilon_default), 0.0)

# normalization

def Sigmoid_derivative(grads_output, y_forward):
  return grads_output * y_forward * (1.0 - y_forward)

def Tanh_derivative(grads_output, y_forward):
  return grads_output * (1.0 - y_forward**2)

def Binary_step_derivative(grads_output, x_input):
  return jnp.zeros_like(x_input)

def Softsign_derivative(grads_output, x_input):
  return grads_output / (1.0 + jnp.abs(x_input))**2

def Softmax_derivative(grads_output, y_forward): # y_forward is the output of softmax
  sum_of_grads_times_y = jnp.sum(grads_output * y_forward, axis=-1, keepdims=True)
  return y_forward * (grads_output - sum_of_grads_times_y)

# Rectifier
def ReLU_derivative(grads_output, x_input):
  return grads_output * jnp.where(x_input > 0, 1.0, 0.0)

def Softplus_derivative(grads_output, x_input):
  return grads_output * Sigmoid(x_input)

def Mish_derivative(grads_output, x_input):
  softplus_x = jnp.log(1.0 + jnp.exp(x_input))
  tanh_softplus_x = jnp.tanh(softplus_x)
  sigmoid_x = Sigmoid(x_input)
  dmish_dx = tanh_softplus_x + x_input * (1.0 - tanh_softplus_x**2) * sigmoid_x
  return grads_output * dmish_dx

def Swish_derivative(grads_output, x_input):
  sigmoid_x = Sigmoid(x_input)
  dswish_dx = sigmoid_x + x_input * sigmoid_x * (1.0 - sigmoid_x)
  return grads_output * dswish_dx

def Leaky_ReLU_derivative(grads_output, x_input):
  return grads_output * jnp.where(x_input > 0, 1.0, 0.1)

def GELU_derivative(grads_output, x_input):
  phi_x = 0.5 * (1.0 + jax.scipy.special.erf(x_input / jnp.sqrt(2.0)))
  pdf_x = jax.scipy.stats.norm.pdf(x_input)
  dgelu_dx = phi_x + x_input * pdf_x
  return grads_output * dgelu_dx

def Linear_derivative(grads_output, x_input):
  return grads_output

def ReEU_derivative(grads_output, x_input):
  dydx = jnp.zeros_like(x_input)
  
  dydx = jnp.where(x_input > 10.0, 1.0, dydx)
  
  is_in_default_region = jnp.logical_and(x_input >= -10.0, x_input <= 10.0)
  
  val_exp_x = jnp.exp(x_input)
  val_max_1_x_plus_1 = jnp.maximum(1.0, x_input + 1.0)
  
  deriv_exp_x = jnp.exp(x_input)
  deriv_max_1_x_plus_1 = jnp.where(x_input > 0.0, 1.0, 0.0) # Derivative of x+1 is 1 if x+1 > 1 (i.e. x > 0), else derivative of 1 is 0
  
  # Derivative within the default region for min(A, B)
  dydx_default_inner = jnp.where(val_exp_x < val_max_1_x_plus_1, deriv_exp_x, deriv_max_1_x_plus_1)
  
  dydx = jnp.where(is_in_default_region, dydx_default_inner, dydx)
  
  return grads_output * dydx

def ReTanh_derivative(grads_output, x_input):
  tanh_plus_1 = jnp.tanh(x_input + 1.0)
  dretanh_dx = (tanh_plus_1 + 1.0) / 2.0 + x_input * (1.0 - tanh_plus_1**2) / 2.0
  return grads_output * dretanh_dx

def ELU_derivative(grads_output, x_input, alpha=ELU_alpha_default):
  dydx = jnp.zeros_like(x_input)
  
  dydx = jnp.where(x_input > 0.0, 1.0, dydx)
  
  default_region_mask = jnp.logical_and(x_input <= 0.0, x_input >= -10.0)
  dydx = jnp.where(default_region_mask, alpha * jnp.exp(x_input), dydx)
  
  return grads_output * dydx

def SELU_derivative(grads_output, x_input, alpha=SELU_alpha_default, beta=SELU_beta_default):
  dydx_inner = jnp.where(x_input > 0.0, 1.0, alpha * jnp.exp(x_input))
  return grads_output * beta * dydx_inner

def PReLU_derivative(grads_output, x_input, alpha=PReLU_alpha_default):
  dydx = jnp.where(x_input > 0.0, 1.0, alpha)
  return grads_output * dydx

def SiLU_derivative(grads_output, x_input, alpha=SiLU_alpha_default):
  sigmoid_ax = Sigmoid(alpha * x_input) # Reuses the Sigmoid function
  dsilux_dx = sigmoid_ax + x_input * alpha * sigmoid_ax * (1.0 - sigmoid_ax)
  return grads_output * dsilux_dx

# loss

def Mean_squared_error_derivative(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
  N = y_true.size # Total number of elements, equivalent to batch_size * output_dim
  return 2.0 * (y_pred - y_true) / N

def Root_mean_squared_error_derivative(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
  N = y_true.size # Total number of elements
  
  mse = jnp.mean(jnp.square(y_true - y_pred))
  rmse = jnp.sqrt(mse + _EPSILON) # Add epsilon for stability if RMSE is zero

  return (y_pred - y_true) / (N * rmse)

def Mean_absolute_error_derivative(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
  N = y_true.size # Total number of elements
  
  return jnp.sign(y_pred - y_true) / N

def Total_squared_error_derivative(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
  return y_pred - y_true

def Total_absolute_error_derivative(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
  return jnp.sign(y_pred - y_true)

def L1_loss_derivative(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
  return Mean_absolute_error_derivative(y_true, y_pred)

def Categorical_crossentropy_derivative(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
  epsilon = 1e-15 # Matches your forward function
  y_pred_clipped = jnp.clip(y_pred, epsilon, 1.0 - epsilon)
  
  N = y_true.shape[0] # Batch size
  
  return - (y_true / y_pred_clipped) / N

def Sparse_categorical_crossentropy_derivative(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
  epsilon = 1e-15 # Matches your forward function
  y_pred_clipped = jnp.clip(y_pred, epsilon, 1.0 - epsilon)
  
  N = y_true.shape[0] # Batch size
  num_classes = y_pred.shape[-1]
  
  y_true_one_hot = jax.nn.one_hot(y_true, num_classes)
  gradient_unnormalized = y_true_one_hot / y_pred_clipped
  
  return - gradient_unnormalized / N


def Binary_crossentropy_derivative(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
  epsilon = 1e-15 # Matches your forward function
  y_pred_clipped = jnp.clip(y_pred, epsilon, 1.0 - epsilon)
  
  N = y_true.size # Total number of elements
  
  return (y_pred_clipped - y_true) / (y_pred_clipped * (1.0 - y_pred_clipped)) / N


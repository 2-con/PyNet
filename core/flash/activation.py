import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import jax.numpy as jnp
import jax
from system.defaults import *

# normalization

def Sigmoid(x, *parametric):
  return 1.0 / (1.0 + jnp.exp(-x))

def Tanh(x, *parametric):
  return jnp.tanh(x)

def Binary_step(x, *parametric):
  return jnp.where(x > 0, 1.0, 0.0)

def Softsign(x, *parametric):
  return x / (1.0 + jnp.abs(x))

def Softmax(x, *parametric):
  exp_x = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
  return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)

# rectifier

def ReLU(x, *parametric):
  return jnp.maximum(0.0, x)

def Softplus(x, *parametric):
  return jnp.log(1.0 + jnp.exp(x))

def Mish(x, *parametric):
  return x * jnp.tanh(jnp.log(1.0 + jnp.exp(x)))

def Swish(x, *parametric):
  return x * Sigmoid(x)

def Leaky_ReLU(x, *parametric):
  return jnp.maximum(0.1 * x, x)

def GELU(x, *parametric):
  return jax.nn.gelu(x)

def Linear(x, *parametric):
  return x

def ReEU(x, *parametric):
  conditions = [x > 10, x < -10]
  choices = [x, 0.0]
  return jnp.select(conditions, choices, default=jnp.minimum(jnp.exp(x), jnp.maximum(1.0, x + 1.0)))

def ReTanh(x, *parametric):
  return x * (jnp.tanh(x + 1.0) + 1.0) / 2.0

# parametrics

def ELU(x, *parametric):
  conditions = [x > 0, x < -10]
  choices = [x, -1.0]
  return jnp.select(conditions, choices, default=parametric[0] * (jnp.exp(x) - 1.0))

def SELU(x, *parametric):
  return jnp.where(x > 0, x, parametric[0] * (jnp.exp(x) - 1.0))

def PReLU(x, *parametric):
  return jnp.maximum(parametric[0] * x, x)

def SiLU(x, *parametric):
  return x * Sigmoid(parametric[0] * x)

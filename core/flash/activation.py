import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import jax.numpy as jnp
import jax
from system.defaults import *

# normalization

def Sigmoid(x, **kwargs):
  return 1.0 / (1.0 + jnp.exp(-x))

def Tanh(x, **kwargs):
  return jnp.tanh(x)

def Binary_step(x, **kwargs):
  return jnp.where(x > 0, 1.0, 0.0)

def Softsign(x, **kwargs):
  return x / (1.0 + jnp.abs(x))

def Softmax(x, **kwargs):
  exp_x = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
  return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)

# rectifier

def ReLU(x, **kwargs):
  return jnp.maximum(0.0, x)

def Softplus(x, **kwargs):
  return jnp.log(1.0 + jnp.exp(x))

def Mish(x, **kwargs):
  return x * jnp.tanh(jnp.log(1.0 + jnp.exp(x)))

def Swish(x, **kwargs):
  return x * Sigmoid(x)

def Leaky_ReLU(x, **kwargs):
  return jnp.maximum(0.1 * x, x)

def GELU(x, **kwargs):
  return jax.nn.gelu(x)

def Linear(x, **kwargs):
  return x

def ReEU(x, **kwargs):
  conditions = [x > 10, x < -10]
  choices = [x, 0.0]
  return jnp.select(conditions, choices, default=jnp.minimum(jnp.exp(x), jnp.maximum(1.0, x + 1.0)))

def ReTanh(x, **kwargs):
  return x * (jnp.tanh(x + 1.0) + 1.0) / 2.0

def ELU(x, alpha=ELU_alpha_default, **kwargs):
  conditions = [x > 0, x < -10]
  choices = [x, -1.0]
  return jnp.select(conditions, choices, default=alpha * (jnp.exp(x) - 1.0))

def SELU(x, alpha=SELU_alpha_default, beta=SELU_beta_default, **kwargs):
  return beta * jnp.where(x > 0, x, alpha * (jnp.exp(x) - 1.0))

def PReLU(x, alpha=PReLU_alpha_default, **kwargs):
  return jnp.maximum(alpha * x, x)

def SiLU(x, alpha=SiLU_alpha_default, **kwargs):
  return x * Sigmoid(alpha * x)

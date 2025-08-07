import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import jax.numpy as jnp
from system.defaults import *

def Standard_Scaler(x: jnp.ndarray) -> jnp.ndarray:
  average = jnp.mean(x, axis=0)
  standard_deviation = jnp.std(x, axis=0)

  scaled_x = jnp.where(
    standard_deviation != 0,
    (x - average) / (standard_deviation + epsilon_default),
    0.0
  )
  return scaled_x

def Min_Max_Scaler(x: jnp.ndarray, min_val: float, max_val: float) -> jnp.ndarray:
  range_val = max_val - min_val
  
  scaled_x = jnp.where(
    range_val != 0,
    (x - min_val) / (range_val + epsilon_default),
    0.0 
  )
  return scaled_x

def Max_Abs_Scaler(x: jnp.ndarray) -> jnp.ndarray:
  max_abs_val = jnp.max(jnp.abs(x), axis=0)

  scaled_x = jnp.where(
    max_abs_val != 0,
    x / (max_abs_val + epsilon_default),
    0.0
  )
  return scaled_x

def Robust_Scaler(x: jnp.ndarray) -> jnp.ndarray:
  q1 = jnp.quantile(x, 0.25, axis=0)
  q3 = jnp.quantile(x, 0.75, axis=0)
  iqr = q3 - q1

  scaled_x = jnp.where(
    iqr != 0,
    (x - q1) / (iqr + epsilon_default), 
    0.0 
  )
  return scaled_x

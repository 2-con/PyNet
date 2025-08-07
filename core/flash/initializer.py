import jax.numpy as jnp
from jax import random
import random as rand

def Glorot_uniform(shape: tuple, fan_in: int, fan_out_size: int):
  limit = jnp.sqrt(2 / (fan_in + fan_out_size))
  return random.uniform(random.PRNGKey(rand.randint(1,1000)), shape, minval=-limit, maxval=limit)

def Glorot_normal(shape: tuple, fan_in: int, fan_out_size: int):
  std_dev = jnp.sqrt(2 / (fan_in + fan_out_size))
  return random.normal(random.PRNGKey(rand.randint(1,1000)), shape)

def He_uniform(shape: tuple, fan_in: int, fan_out_size: int):
  limit = jnp.sqrt(6 / fan_in)
  return random.uniform(random.PRNGKey(rand.randint(1,1000)), shape, minval=-limit, maxval=limit)

def He_normal(shape: tuple, fan_in: int, fan_out_size: int):
  std_dev = jnp.sqrt(2 / fan_in)
  return random.normal(random.PRNGKey(rand.randint(1,1000)), shape)

def Lecun_uniform(shape: tuple, fan_in: int, fan_out_size: int):
  limit = jnp.sqrt(3 / fan_in)
  return random.uniform(random.PRNGKey(rand.randint(1,1000)), shape, minval=-limit, maxval=limit)

def Lecun_normal(shape: tuple, fan_in: int, fan_out_size: int):
  std_dev = jnp.sqrt(1 / fan_in)
  return random.normal(random.PRNGKey(rand.randint(1,1000)), shape)

def Xavier_uniform_in(shape: tuple, fan_in: int, fan_out_size: int):
  limit = jnp.sqrt(6 / fan_in)
  return random.uniform(random.PRNGKey(rand.randint(1,1000)), shape, minval=-limit, maxval=limit)

def Xavier_uniform_out(shape: tuple, fan_in: int, fan_out_size: int):
  limit = jnp.sqrt(6 / fan_out_size)
  return random.uniform(random.PRNGKey(rand.randint(1,1000)), shape, minval=-limit, maxval=limit)

def Default(shape: tuple, fan_in: int, fan_out_size: int):
  return random.uniform(random.PRNGKey(rand.randint(1,1000)), shape, minval=-1, maxval=1)
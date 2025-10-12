import jax.numpy as jnp
from jax import random
import random as rand

class Glorot_Uniform:
  def __call__(self, shape: tuple, fan_in: int, fan_out_size: int):
    limit = jnp.sqrt(2 / (fan_in + fan_out_size))
    return random.uniform(random.PRNGKey(rand.randint(1,1000)), shape, minval=-limit, maxval=limit)

class Glorot_Normal:
  def __call__(self, shape: tuple, fan_in: int, fan_out_size: int):
    std_dev = jnp.sqrt(2 / (fan_in + fan_out_size))
    return random.normal(random.PRNGKey(rand.randint(1,1000)), shape)

class He_Uniform:
  def __call__(self, shape: tuple, fan_in: int, fan_out_size: int):
    limit = jnp.sqrt(6 / fan_in)
    return random.uniform(random.PRNGKey(rand.randint(1,1000)), shape, minval=-limit, maxval=limit)

class He_Normal:
  def __call__(self, shape: tuple, fan_in: int, fan_out_size: int):
    std_dev = jnp.sqrt(2 / fan_in)
    return random.normal(random.PRNGKey(rand.randint(1,1000)), shape)

class Lecun_Uniform:
  def __call__(self, shape: tuple, fan_in: int, fan_out_size: int):
    limit = jnp.sqrt(3 / fan_in)
    return random.uniform(random.PRNGKey(rand.randint(1,1000)), shape, minval=-limit, maxval=limit)

class Lecun_Normal:
  def __call__(self, shape: tuple, fan_in: int, fan_out_size: int):
    std_dev = jnp.sqrt(1 / fan_in)
    return random.normal(random.PRNGKey(rand.randint(1,1000)), shape)

class Xavier_Uniform_In:
  def __call__(self, shape: tuple, fan_in: int, fan_out_size: int):
    limit = jnp.sqrt(6 / fan_in)
    return random.uniform(random.PRNGKey(rand.randint(1,1000)), shape, minval=-limit, maxval=limit)

class Xavier_Uniform_Out:
  def __call__(self, shape: tuple, fan_in: int, fan_out_size: int):
    limit = jnp.sqrt(6 / fan_out_size)
    return random.uniform(random.PRNGKey(rand.randint(1,1000)), shape, minval=-limit, maxval=limit)

class Default:
  def __call__(self, shape: tuple, fan_in: int, fan_out_size: int):
    return random.uniform(random.PRNGKey(rand.randint(1,1000)), shape, minval=-1, maxval=1)
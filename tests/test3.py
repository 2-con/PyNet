import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax.numpy as jnp

a = jnp.array([
  [1,1,1],
  [-1,-1,-1],])

b = jnp.array([0.1,0.2,0.3])

def PReLU(x, *parametric):
  return jnp.maximum(parametric[0] * x, x)

print(PReLU(a, b))
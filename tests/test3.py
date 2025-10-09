import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax.numpy as jnp

a = {
  "one": 1,
  "two": 2,
  "three": 3
}

for key, value in a.items():
  print(f"{key} - {value}")
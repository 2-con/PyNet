import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax.numpy as jnp
import jax
from api.netflash import Sequential
from core.flash.layers import *
from tools.arraytools import generate_random_array
from tools.visual import array_display
import time

# Example Usage
model = Sequential(
  Recurrent(3, 'identity', output_sequence=(0,1,2))
)

# Compile the model
model.compile(
  input_shape=(3,2),
  optimizer='adam',
  loss='mean squared error',
  learning_rate=0.01,
  epochs=100,
  metrics=['accuracy'], 
  batch_size=2,
  verbose=3,
  logging=1
)

# some dummy data for training
features = jnp.array([
  # [1,1,1],
  # [2,2,2],
  # [1,1],
  # [2,2]
  
[[1,1],
 [2,1],
 [3,1]],

[[4,1],
 [5,1],
 [6,1]],
], dtype=jnp.float32)

targets = jnp.array([ 
  # [2,2],
  # [4,4],
  
[[2,1],
 [3,1],
 [4,1]],

[[5,1],
 [6,1],
 [7,1]],
])

# Fit the model
start = time.perf_counter()

model.fit(features, targets) 

print(f"""
      Finished training in {time.perf_counter() - start} seconds
      """)

array_display(model.push(features).tolist())
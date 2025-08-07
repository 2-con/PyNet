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
  Flatten(),
  Dense(2, 'leaky relu'),
)

# Compile the model
model.compile(
  input_shape=(2,2),
  optimizer='default',
  loss='mean squared error',
  learning_rate=0.001,
  epochs=1000,
  metrics=['accuracy'],
  batch_size=4,
  verbose=3,
  logging=100
)

# Generate some dummy data for training
features = jnp.array([
  # [1,1],
  # [2,2],
  # [3,3],
  # [4,4]
  
[[1,2],
 [3,4]],

[[5,6],
 [7,8]],

[[9,10],
 [11,12]],

[[13,14],
 [15,16]]
])
targets = jnp.array([
  [1,1],
  [2,2],
  [3,3],
  [4,4]
])

# Fit the model
start = time.perf_counter()

model.fit(features, targets)

print(f"""
      Finished training in {time.perf_counter() - start} seconds
      """)

# array_display(model.push(features).tolist())
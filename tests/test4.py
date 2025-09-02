import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax.numpy as jnp
from api.netflash import Sequential
from core.flash.layers import *
from tools.visual import array_display
import time

# Example Usage
model = Sequential(
  LSTM(3, activation="identity"),
)

# Compile the model
model.compile(
  input_shape=(3,2),
  optimizer='adam',
  loss='mean squared error',
  learning_rate=0.01,
  epochs=700,
  metrics=['accuracy'], 
  batch_size=2,
  verbose=3,
  logging=100
)

# some dummy data for training
features = jnp.array([ 
  # [1,1,1],
  # [2,2,2],
  # [1,1],
  # [2,2]
  
[[1,6],
 [2,5],
 [3,4]],

[[4,3],
 [5,2],
 [6,1]],
], dtype=jnp.float32)

targets = jnp.array([ 
#   [2,2],
#   [4,4],
  
[[2,7],
 [3,6],
 [4,5]],

[[5,4],
 [6,3],
 [7,2]],
])

# Fit the model
start = time.perf_counter()

model.fit(features, targets) 

print(f"""
      Finished training in {time.perf_counter() - start} seconds
      """)

array_display(model.push(features).tolist())
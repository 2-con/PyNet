import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax.numpy as jnp
from api.netflash import Sequential
from core.flash.layers import *
from tools.visual import array_display
import time
import matplotlib.pyplot as plt
from core.flash.callback import Callback

# Example Usage
model = Sequential(
  Convolution((3,3), 1, "prelu", (1,1)),
  Flatten(),
  Dense(3, "prelu"),
)

# Compile the model
model.compile(
  input_shape=(1,5,5),
  optimizer='adam',
  loss='mean squared error',
  learning_rate=0.001,
  epochs=100,
  metrics=['accuracy'], 
  batch_size=2,
  verbose=4,
  logging=1,
)

# some dummy data for training
features = jax.random.uniform(key=jax.random.key(random.randint(1,1000)), minval=0, maxval=10, shape=(20,5,5))
targets = jax.random.uniform(key=jax.random.key(random.randint(1,1000)), minval=0, maxval=10, shape=(20,3))

# jnp.array([[0,0],[0,1],[1,0],[1,1]])
# jnp.array([[0,1],[1,0],[1,0],[0,1]])

# Fit the model
start = time.perf_counter()

model.fit(features, targets) 

print(f"""
      Finished training in {time.perf_counter() - start} seconds
      """)
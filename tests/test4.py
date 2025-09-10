import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax.numpy as jnp
from api.netflash import Sequential
from core.flash.layers import *
from tools.visual import array_display
import time

# Example Usage
model = Sequential(
  Dense(10, 'elu'),
  Dense(6, 'elu'),
  Dense(4, 'elu'),
)

# Compile the model
model.compile(
  input_shape=(10,),
  optimizer='rprop',
  loss='mean squared error',
  learning_rate=0.01,
  epochs=100,
  metrics=['accuracy'], 
  batch_size=2,
  verbose=3,
  logging=1
)

# some dummy data for training
features = jax.random.uniform(key=jax.random.key(random.randint(1,1000)), minval=0, maxval=10, shape=(20,10))

jnp.array([ 
  # [1,1,1],
  # [2,2,2],
  # [1,1],
  # [2,2]
  
# [[1,6],
#  [3,4]],

# [[4,3],
#  [6,1]],
], dtype=jnp.float32)

targets = jax.random.uniform(key=jax.random.key(random.randint(1,1000)), minval=0, maxval=10, shape=(20,4))

jnp.array([ 
  [2,2],
  [4,4],


# [[2,7],
#  [4,5]],

# [[5,4],
#  [7,2]],
])

# Fit the model
start = time.perf_counter()

model.fit(features, targets) 

print(f"""
      Finished training in {time.perf_counter() - start} seconds
      """)
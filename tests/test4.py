import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.netflash import Sequential
from core.flash.layers import *
import time

# Example Usage
model = Sequential(
  LSTM(5, "relu")
 )

# Compile the model
model.compile(
  input_shape=(5,6),
  optimizer='adam',
  loss='mean squared error',
  learning_rate=0.001,
  epochs=1000,
  batch_size=51,
  verbose=2,
  logging=99,
)

# some dummy data for training
features = jax.random.uniform(key=jax.random.key(1), minval=0, maxval=10, shape=(100,5,6))
targets = jax.random.uniform(key=jax.random.key(1), minval=0, maxval=10, shape=(100,5,6))

# jnp.array([[0,0],[0,1],[1,0],[1,1]])
# jnp.array([[0,1],[1,0],[1,0],[0,1]])

# Fit the model
start = time.perf_counter()

model.fit(features, targets) 

print(f"""
      Finished training in {time.perf_counter() - start} seconds
      """)
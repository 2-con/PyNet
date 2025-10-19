import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.standardnet import Sequential
from core.standard import layers, optimizers, losses
import time
import jax

# Example Usage
model = Sequential()
model.add(layers.Dense(2, "identity"))

# Compile the model
model.compile(
  input_shape=(2,),
  optimizer=optimizers.Adam,
  loss=losses.Mean_Squared_Error,
  validation_split=0.2,
  learning_rate=0.01,
  epochs=100,
  batch_size=1,
  verbose=3,
  logging=1,
)

# some dummy data for training
features = jax.random.uniform(key=jax.random.key(1), minval=0, maxval=10, shape=(100,2))
targets = jax.random.uniform(key=jax.random.key(1), minval=0, maxval=10, shape=(100,2))

# jnp.array([[0,0],[0,1],[1,0],[1,1]])
# jnp.array([[0,1],[1,0],[1,0],[0,1]])

# Fit the model
start = time.perf_counter()

model.fit(features, targets) 
model.push(features)

print(f"""
      Finished training in {time.perf_counter() - start} seconds
      """)
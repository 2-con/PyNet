import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import api.netcore as net
import time
from tools.arraytools import generate_random_array

features = generate_random_array(10,1000)
targets = generate_random_array(10,1000, min=0, max=100)

model = net.Sequential(
  
  net.RecurrentBlock(
    net.GRU(),
    net.GRU(),
    net.GRU(),
    net.GRU(),
    net.GRU(),
    net.GRU(),
    net.GRU(),
    net.GRU(),
    net.GRU(),
    net.GRU(),
  ),
  
  net.Dense(10, 'leaky relu')
)

model.compile(
  optimizer='momentum',
  loss='mean squared error',
  learning_rate=0.01,
  epochs=100,
  metrics=['mean squared error'],
  validation_split=0.2,
  logging=10,
  verbose=6,
  optimize=True
)

start_time = time.perf_counter()
model.fit(features, targets)
end_time = time.perf_counter()
duration = end_time - start_time
print(f"""
      finished training in {duration} seconds
      """)

model.evaluate(
  features,
  targets,
  logging=True
)
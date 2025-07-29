import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import api.netcore as net
import time
from tools.arraytools import generate_random_array

features = generate_random_array(100,100)
targets = generate_random_array(10,100)

model = net.Sequential(
  # net.Dense(64, 'relu'),
  # net.Dense(64, 'relu'),
  net.Dense(32, 'elu'),
  net.Dense(10, 'elu'),
)

model.compile(
  optimizer='none',
  loss='mean squared error',
  learning_rate=0.1,
  epochs=100,
  metrics=['mean squared error'],
  logging=10,
  verbose=1,
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
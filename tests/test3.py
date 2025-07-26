import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import api.multinet as net
import time
from tools.arraytools import generate_random_array

features = generate_random_array(100,1000)
targets = generate_random_array(10,1000)

model = net.Sequential(
  # net.Dense(64, 'relu'),
  # net.Dense(64, 'relu'),
  net.Dense(32, 'relu'),
  net.Dense(10, 'relu'),
)

model.compile(
  optimizer='none',
  loss='mean squared error',
  learning_rate=0.01,
  epochs=100,
  metrics=['accuracy'],
  logging=10,
  verbose=3
)

start_time = time.perf_counter()
model.fit(features, targets)
end_time = time.perf_counter()
duration = end_time - start_time
print(f"""
      finished training in {duration} seconds
      """)
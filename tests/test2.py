import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import api.netcore as net
import time
from tools.arraytools import generate_random_array

features = generate_random_array(784,100)
targets = generate_random_array(10,100)

model = net.Sequential(
  net.Dense(64, 'relu'),
  net.Dense(64, 'relu'),
  net.Dense(10, 'relu'),
  # net.Operation('softmax'),
)

model.compile(
  optimizer='none',
  loss='mean squared error',
  learning_rate=1.0,
  epochs=10,
  metrics=['mean squared error'],
  logging=1,
  verbose=2,
  optimize=False
)

start_time = time.perf_counter()

model.fit(features, targets)

print(f"""
      finished training in {time.perf_counter() - start_time} seconds
      """)

model.evaluate(
  features, 
  targets,
  logging=True
)
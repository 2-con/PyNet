import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import api.alpha as alpha
from tools.arraytools import generate_random_array
from core.loss import Mean_squared_error as mse
from core.activation import *
from core.derivative import *
from tools.utility import progress_bar
import time

features = generate_random_array(100,100)
targets = generate_random_array(10,100)

model = alpha.initialize(100,32,10)

start_time = time.perf_counter()

for epoch in progress_bar(range(100), "> Training", "%  Complete") if True else range(100):
  for feature, target in zip(features, targets):
    activations, weighted_sums = alpha.propagate(model, feature, ELU)
    error = alpha.backpropegate(model, activations, weighted_sums, target, ELU_derivative)
    
    alpha.update(model, activations, error, 0.001)

  # if epoch % 10 == 0:
  #   print(f"Epoch {epoch:5} | Error {mse(target, activations[-1])}")

end_time = time.perf_counter()
duration = end_time - start_time
print(f"""
      finished training in {duration} seconds
      """)

all_error = []

for feature, target in zip(features, targets):
  activations, _ = alpha.propagate(model, feature, ReLU)
  all_error.append(mse(target, activations[-1]))

print(f"Average error: {sum(all_error) / len(all_error)}")

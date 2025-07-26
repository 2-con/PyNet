import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import api.flash as flash
from tools.arraytools import generate_random_array
from core.loss import Mean_squared_error as mse
from core.activation import ReLU
from core.derivative import ReLU_derivative
from tools.utility import progress_bar
import time

features = generate_random_array(100,600)
targets = generate_random_array(10,600)

model = flash.initialize(100,32,10)

start_time = time.perf_counter()

for epoch in progress_bar(range(100), "> Training", "%  Complete") if False else range(100):
  for feature, target in zip(features, targets):
    activations, weighted_sums = flash.propagate(model, feature, ReLU)
    error = flash.backpropegate(model, activations, weighted_sums, target, ReLU_derivative)
    
    flash.update(model, activations, error, 0.01)

  if epoch % 10 == 0:
    print(f"Epoch {epoch:5} | Error {mse(target, activations[-1])}")

end_time = time.perf_counter()
duration = end_time - start_time
print(f"""
      finished training in {duration} seconds
      """)
